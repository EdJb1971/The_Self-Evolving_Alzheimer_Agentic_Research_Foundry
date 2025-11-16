import time
import json
import os
import requests
import logging
import random
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas
from datetime import datetime, timedelta
from alznexus_audit_trail.schemas import MAX_METADATA_SIZE_BYTES

AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
ADWORKBENCH_PROXY_URL = os.getenv("ADWORKBENCH_PROXY_URL")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")
AGENT_REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL")
AGENT_REGISTRY_API_KEY = os.getenv("AGENT_REGISTRY_API_KEY")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

MAX_RESULT_DATA_SIZE_BYTES = 1 * 1024 * 1024

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")
if not AGENT_REGISTRY_URL or not AGENT_REGISTRY_API_KEY:
    raise ValueError("AGENT_REGISTRY_URL or AGENT_REGISTRY_API_KEY environment variables not set.")

logger = logging.getLogger(__name__)

def calculate_backoff_with_jitter(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter_factor: float = 0.1) -> float:
    """Calculate exponential backoff delay with jitter to prevent thundering herd."""
    # Exponential backoff: base_delay * (2 ^ attempt)
    delay = base_delay * (2 ** attempt)

    # Add jitter: randomize delay by ±jitter_factor
    jitter = delay * jitter_factor * (2 * random.random() - 1)  # -jitter_factor to +jitter_factor
    delay_with_jitter = delay + jitter

    # Ensure delay is within reasonable bounds
    return min(max(delay_with_jitter, 0.1), max_delay)

def log_audit_event(entity_type: str, entity_id: str, event_type: str, description: str, metadata: dict = None):
    headers = {"X-API-Key": AUDIT_API_KEY, "Content-Type": "application/json"}
    payload = {
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "event_type": event_type,
        "description": description,
        "metadata": metadata if metadata is not None else {}
    }
    try:
        response = requests.post(f"{AUDIT_TRAIL_URL}/audit/log", headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Audit log successful: {event_type} for {entity_type}:{entity_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to log audit event: {str(e)}", exc_info=True)

@celery_app.task(bind=True, name="match_collaboration_task")
def match_collaboration_task(self, agent_task_id: int):
    max_retries = 3
    attempt = 0
    while attempt <= max_retries:
        try:
            db: Session = SessionLocal()
            db_agent_task = crud.get_agent_task(db, agent_task_id)
            if not db_agent_task:
                raise ValueError(f"Agent task with ID {agent_task_id} not found.")

            agent_id = db_agent_task.agent_id

            crud.update_agent_task_status(db, agent_task_id, "IN_PROGRESS")
            crud.update_agent_state(db, agent_id, current_task_id=agent_task_id)
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="COLLABORATION_MATCHMAKING_STARTED",
                description=f"Agent {agent_id} started collaboration matchmaking task {agent_task_id}: {db_agent_task.task_description}",
                metadata=db_agent_task.model_dump()
            )

            # COMP-016: Execute collaboration matchmaking by identifying research problems and suggesting optimal agent collaborations
            adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}
            registry_headers = {"X-API-Key": AGENT_REGISTRY_API_KEY}

            # 1. Query AD Workbench for research problems and data gaps
            query_text = f"Retrieve complex research problems or data gaps related to: {db_agent_task.task_description}"
            adworkbench_query_payload = {"query_text": query_text}

            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="ADWORKBENCH_QUERY_INITIATED",
                description=f"Agent {agent_id} querying AD Workbench for research problems: {query_text}",
                metadata={"query_text": query_text}
            )

            adworkbench_response = requests.post(
                f"{ADWORKBENCH_PROXY_URL}/adworkbench/query",
                headers=adworkbench_headers,
                json=adworkbench_query_payload
            )
            adworkbench_response.raise_for_status()
            query_status_response = adworkbench_response.json()
            adworkbench_query_id = query_status_response["id"]

            # TODO: CQ-CM-001: Replace with actual asynchronous calls/polling mechanism for AD Workbench query. The blocking time.sleep() has been removed.
            # A proper async implementation would involve storing adworkbench_query_id and having a separate task or service check for completion.
            query_result_response = requests.get(
                f"{ADWORKBENCH_PROXY_URL}/adworkbench/query/{adworkbench_query_id}/status",
                headers=adworkbench_headers
            )
            query_result_response.raise_for_status()
            final_query_status = query_result_response.json()

            if final_query_status["status"] != "COMPLETED":
                raise Exception(f"AD Workbench query for research problems failed or timed out: {final_query_status['status']}")
            
            result_data_str = final_query_status["result_data"]
            if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
                raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
            raw_problem_data = json.loads(result_data_str)
            
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="ADWORKBENCH_QUERY_COMPLETED",
                description=f"Agent {agent_id} received research problem data from AD Workbench query {adworkbench_query_id}.",
                metadata={"adworkbench_query_id": adworkbench_query_id, "problem_summary": raw_problem_data.get("data", [])[:1]}
            )

            # 2. Query Agent Registry for available agents and their capabilities
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="AGENT_REGISTRY_QUERY_INITIATED",
                description=f"Agent {agent_id} querying Agent Registry for available agents.",
                metadata={"registry_url": AGENT_REGISTRY_URL}
            )

            registry_response = requests.get(f"{AGENT_REGISTRY_URL}/registry/agents", headers=registry_headers)
            registry_response.raise_for_status()
            registered_agents = registry_response.json().get("agents", [])

            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="AGENT_REGISTRY_QUERY_COMPLETED",
                description=f"Agent {agent_id} received {len(registered_agents)} agents from Agent Registry.",
                metadata={"num_registered_agents": len(registered_agents)}
            )

            # CQ-CM-002: Implement actual complex matching logic using LLM and agent capabilities
            problem_statement = raw_problem_data.get("data", [{}])[0].get("problem_description", db_agent_task.task_description)

            # Analyze agent capabilities and match to problem
            agent_capabilities = []
            for agent in registered_agents:
                agent_capabilities.append({
                    "agent_id": agent["agent_id"],
                    "capabilities": agent.get("capabilities", []),
                    "api_endpoint": agent.get("api_endpoint", "")
                })

            # Use LLM to analyze problem and suggest optimal collaborations
            collaboration_prompt = f"""Analyze this research problem and suggest optimal agent collaborations:

Research Problem: {problem_statement}

Available Agents and Capabilities:
{json.dumps(agent_capabilities, indent=2)}

Task: {db_agent_task.task_description}

Please suggest 2-3 optimal agent teams that would work together effectively on this problem. Consider:
1. Complementary capabilities that cover different aspects of the problem
2. Agent specializations that align with research needs
3. Potential for interdisciplinary insights

Format your response as a JSON object with this structure:
{{
    "suggested_teams": [
        {{
            "team_id": "TEAM-001",
            "agents": [
                {{"agent_id": "agent_name", "role": "Specific role description"}}
            ],
            "rationale": "Why this team composition is optimal",
            "expected_outcomes": ["outcome1", "outcome2"]
        }}
    ],
    "external_experts": [
        {{"name": "Expert Name", "expertise": "Field", "affiliation": "Institution"}}
    ],
    "matchmaking_summary": "Overall analysis summary"
}}"""

            llm_payload = {
                "model_name": "gemini-1.5-flash",
                "prompt": collaboration_prompt,
                "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
            }
            llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}

            llm_response = requests.post(
                f"{LLM_SERVICE_URL}/llm/structured-output",
                headers=llm_headers,
                json={
                    "model_name": "gemini-1.5-flash",
                    "prompt": collaboration_prompt,
                    "response_format": {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "suggested_teams": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "team_id": {"type": "string"},
                                            "agents": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "agent_id": {"type": "string"},
                                                        "role": {"type": "string"}
                                                    },
                                                    "required": ["agent_id", "role"]
                                                }
                                            },
                                            "rationale": {"type": "string"},
                                            "expected_outcomes": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["team_id", "agents", "rationale"]
                                    }
                                },
                                "external_experts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "expertise": {"type": "string"},
                                            "affiliation": {"type": "string"}
                                        }
                                    }
                                },
                                "matchmaking_summary": {"type": "string"}
                            },
                            "required": ["suggested_teams", "external_experts", "matchmaking_summary"]
                        }
                    },
                    "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
                }
            )
            llm_response.raise_for_status()
            llm_result = llm_response.json()

            # Parse the structured LLM response
            try:
                collaboration_analysis = llm_result["structured_output"]
                suggested_collaborations = {
                    "research_problem": problem_statement,
                    "suggested_teams": collaboration_analysis.get("suggested_teams", []),
                    "external_experts": collaboration_analysis.get("external_experts", []),
                    "matchmaking_summary": collaboration_analysis.get("matchmaking_summary", f"Analysis completed with {len(registered_agents)} available agents."),
                    "llm_analysis": llm_result.get("response_text", "")
                }
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Failed to parse LLM collaboration response: {e}")
                # Fallback to basic structure
                suggested_collaborations = {
                    "research_problem": problem_statement,
                    "suggested_teams": [],
                    "external_experts": [],
                    "matchmaking_summary": f"LLM analysis failed, but {len(registered_agents)} agents are available.",
                    "error": str(e)
                }

            # 4. Publish collaboration suggestions as an insight
            insight_name_val = f"Collaboration Suggestion: {db_agent_task.task_description}"
            insight_publish_request_obj = schemas.InsightPublishRequest(
                insight_name=insight_name_val,
                insight_description=f"Automatically identified collaboration opportunities for research problem: {db_agent_task.task_description}.",
                data_source_ids=[f"adworkbench_query_{adworkbench_query_id}", "agent_registry_scan"],
                payload=suggested_collaborations,
                tags=["collaboration_matchmaking", "team_formation", agent_id]
            )
            insight_payload = insight_publish_request_obj.model_dump_json()

            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="PUBLISHING_COLLABORATION_INSIGHT",
                description=f"Agent {agent_id} publishing collaboration matchmaking insight.",
                metadata={"insight_name": insight_name_val}
            )

            publish_response = requests.post(
                f"{ADWORKBENCH_PROXY_URL}/adworkbench/publish-insight",
                headers=adworkbench_headers,
                data=insight_payload
            )
            publish_response.raise_for_status()
            publish_result = publish_response.json()

            mock_result = {
                "status": "success",
                "agent_output": f"Collaboration matchmaking completed for task {agent_task_id}.",
                "collaboration_suggestions": suggested_collaborations,
                "published_insight_id": publish_result.get("insight_id")
            }

            crud.update_agent_task_status(db, agent_task_id, "COMPLETED", mock_result)
            crud.update_agent_state(db, agent_id, current_task_id=None)
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="COLLABORATION_MATCHMAKING_COMPLETED",
                description=f"Agent {agent_id} completed collaboration matchmaking task {agent_task_id} and published insight.",
                metadata={"task_result": mock_result}
            )
            return {"agent_task_id": agent_task_id, "status": "COMPLETED", "result": mock_result}
        except requests.exceptions.RequestException as e:
            error_message = f"AD Workbench Proxy or Agent Registry API call failed: {e}"
            crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message})
            crud.update_agent_state(db, agent_id, current_task_id=None)
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="COLLABORATION_MATCHMAKING_FAILED",
                description=f"Agent {agent_id} failed to match collaborations for task {agent_task_id}: {error_message}",
                metadata={"error": error_message}
            )
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
            raise
        except Exception as e:
            error_message = str(e)
            crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message})
            crud.update_agent_state(db, agent_id, current_task_id=None)
            log_audit_event(
                entity_type="AGENT",
                entity_id=f"{agent_id}-{agent_task_id}",
                event_type="COLLABORATION_MATCHMAKING_FAILED",
                description=f"Agent {agent_id} failed to match collaborations for task {agent_task_id}: {error_message}",
                metadata={"error": error_message}
            )
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
            raise
        finally:
            db.close()
        attempt += 1
        if attempt <= max_retries:
            delay = calculate_backoff_with_jitter(attempt, base_delay=1.0, jitter_factor=0.3)
            logger.warning(f"Task {agent_task_id} failed (attempt {attempt}/{max_retries}), retrying in {delay:.2f} seconds: {str(e)}")
            time.sleep(delay)
        else:
            logger.error(f"Task {agent_task_id} failed after {max_retries} attempts: {str(e)}")
            raise

@celery_app.task(bind=True, name="perform_reflection_task")
def perform_reflection_task(self, agent_id: str, reflection_metadata: dict):
    db: Session = SessionLocal()
    try:
        crud.update_agent_state(db, agent_id, last_reflection_at=datetime.utcnow(), metadata_json={"reflection_status": "IN_PROGRESS"})
        log_audit_event(
            entity_type="AGENT",
            entity_id=agent_id,
            event_type="REFLECTION_STARTED",
            description=f"Agent {agent_id} initiated self-reflection.",
            metadata=reflection_metadata
        )

        recent_tasks = crud.get_recent_agent_tasks(db, agent_id, days_ago=7)
        task_summary = {"total_tasks": len(recent_tasks), "completed": 0, "failed": 0, "pending": 0}
        for task in recent_tasks:
            if task.status == "COMPLETED":
                task_summary["completed"] += 1
            elif task.status == "FAILED":
                task_summary["failed"] += 1
            else:
                task_summary["pending"] += 1

        audit_history_response = requests.get(
            f"{AUDIT_TRAIL_URL}/audit/history/AGENT/{agent_id}",
            headers={"X-API-Key": AUDIT_API_KEY}
        )
        audit_history_response.raise_for_status()
        audit_history = audit_history_response.json().get("history", [])
        recent_audit_events = [e for e in audit_history if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > (datetime.utcnow() - timedelta(days=7))]

        # CQ-CM-003: Perform comprehensive analysis of collaboration matchmaking performance using LLM
        performance_data = {
            "task_summary": task_summary,
            "recent_audit_events": recent_audit_events[:10],  # Limit to prevent token overflow
            "agent_id": agent_id,
            "reflection_metadata": reflection_metadata
        }

        reflection_prompt = f"""As a collaboration matchmaker agent, analyze your recent performance in identifying research problems and suggesting optimal collaborations.

Performance Data:
- Total tasks in last 7 days: {task_summary['total_tasks']}
- Completed: {task_summary['completed']}
- Failed: {task_summary['failed']}
- Pending: {task_summary['pending']}
- Recent audit events: {len(recent_audit_events)}

Recent Audit Events Summary:
{json.dumps([{"event_type": e.get("event_type", ""), "description": e.get("description", "")[:100]} for e in recent_audit_events[:5]], indent=2)}

Agent Reflection Metadata:
{json.dumps(reflection_metadata, indent=2)}

Please provide:
1. A detailed analysis of your effectiveness in matching agents to research problems and facilitating collaborations
2. Assessment of collaboration rationale quality and success rates
3. Identification of gaps in agent discovery or matching algorithms
4. Specific recommendations for improving collaboration matchmaking capabilities
5. Evaluation of how well collaborations align with Alzheimer's research goals

Structure your response as a JSON object with keys: 'performance_analysis', 'collaboration_effectiveness', 'identified_gaps', 'recommendations', 'research_alignment'."""

        llm_payload = {
            "model_name": "gemini-1.5-flash",
            "prompt": reflection_prompt,
            "metadata": {"agent_id": agent_id, "reflection_type": "collaboration_matchmaker_performance"}
        }
        llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}

        llm_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/chat",
            headers=llm_headers,
            json=llm_payload
        )
        llm_response.raise_for_status()

        # Safe JSON parsing of LLM response
        try:
            llm_result = llm_response.json()
        except json.JSONDecodeError as e:
            print(f"LLM response JSON parsing failed: {e}")
            print(f"LLM response text: {llm_response.text[:500]}...")
            raise Exception(f"Invalid JSON response from LLM service: {e}")

        # Validate LLM response structure
        if "response_text" not in llm_result:
            raise Exception(f"LLM response missing 'response_text' field: {llm_result.keys()}")

        reflection_text = llm_result["response_text"]
        if not reflection_text or not reflection_text.strip():
            raise Exception("LLM returned empty reflection analysis")

        # Parse structured reflection from LLM response
        try:
            reflection_analysis = json.loads(reflection_text)
        except json.JSONDecodeError:
            # Fallback: extract key sections from text response
            reflection_analysis = {
                "performance_analysis": reflection_text,
                "collaboration_effectiveness": "Unable to parse structured analysis",
                "identified_gaps": ["Improve LLM response parsing"],
                "recommendations": ["Enhance collaboration algorithms"],
                "research_alignment": "Analysis provided but structure unclear"
            }

        proposed_adjustments = reflection_analysis.get("recommendations", [
            "Enhance querying of Agent Registry for more granular capabilities.",
            "Develop more sophisticated algorithms for matching agents to complex problems.",
            "Improve rationale generation for suggested collaborations."
        ])

        # Ensure recommendations are specific to collaboration matchmaking
        collaboration_specific_recommendations = []
        for rec in proposed_adjustments:
            if "collaborat" not in rec.lower():
                rec = f"For collaboration matchmaking: {rec}"
            collaboration_specific_recommendations.append(rec)

        analysis_outcome = reflection_analysis.get("performance_analysis",
            f"Agent {agent_id} reviewed {task_summary['total_tasks']} tasks in the last 7 days. " \
            f"Completed: {task_summary['completed']}, Failed: {task_summary['failed']}. " \
            "LLM-powered analysis identified opportunities for improved collaboration matchmaking processes.")

        reflection_result = {
            "analysis_summary": analysis_outcome,
            "performance_analysis": reflection_analysis.get("performance_analysis", ""),
            "collaboration_effectiveness": reflection_analysis.get("collaboration_effectiveness", ""),
            "identified_gaps": reflection_analysis.get("identified_gaps", []),
            "proposed_adjustments": collaboration_specific_recommendations,
            "research_alignment": reflection_analysis.get("research_alignment", ""),
            "task_performance_summary": task_summary,
            "recent_audit_event_count": len(recent_audit_events),
            "llm_reflection_used": True,
            "original_reflection_metadata": reflection_metadata
        }

        crud.update_agent_state(db, agent_id, metadata_json={"reflection_status": "COMPLETED", "last_reflection_result": reflection_result})
        log_audit_event(
            entity_type="AGENT",
            entity_id=agent_id,
            event_type="REFLECTION_COMPLETED",
            description=f"Agent {agent_id} completed self-reflection. Analysis: {analysis_outcome}",
            metadata=reflection_result
        )
        return {"agent_id": agent_id, "status": "COMPLETED", "reflection_result": reflection_result}
    except requests.exceptions.RequestException as e:
        error_message = f"Audit Trail Service call failed during reflection: {str(e)}"
        crud.update_agent_state(db, agent_id, metadata_json={"reflection_status": "FAILED", "error": error_message})
        log_audit_event(
            entity_type="AGENT",
            entity_id=agent_id,
            event_type="REFLECTION_FAILED",
            description=f"Agent {agent_id} reflection failed: {error_message}",
            metadata={"error": error_message, "original_reflection_metadata": reflection_metadata}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    except Exception as e:
        error_message = str(e)
        # Note: agent_task_id is not available in reflection task context, use agent_id for logging failure
        # crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message}) # This line is incorrect for reflection task
        crud.update_agent_state(db, agent_id, metadata_json={"reflection_status": "FAILED", "error": error_message})
        log_audit_event(
            entity_type="AGENT",
            entity_id=agent_id,
            event_type="REFLECTION_FAILED",
            description=f"Agent {agent_id} reflection failed: {error_message}",
            metadata={"error": error_message, "original_reflection_metadata": reflection_metadata}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
