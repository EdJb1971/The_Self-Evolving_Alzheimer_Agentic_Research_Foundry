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
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

MAX_RESULT_DATA_SIZE_BYTES = 1 * 1024 * 1024

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")

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

@celery_app.task(bind=True, name="bridge_literature_task")
def bridge_literature_task(self, agent_task_id: int):
    max_retries = 3
    attempt = 0
    while attempt <= max_retries:
        try:
            db: Session = SessionLocal()
            try:
                db_agent_task = crud.get_agent_task(db, agent_task_id)
                if not db_agent_task:
                    raise ValueError(f"Agent task with ID {agent_task_id} not found.")

                agent_id = db_agent_task.agent_id

                crud.update_agent_task_status(db, agent_task_id, "IN_PROGRESS")
                crud.update_agent_state(db, agent_id, current_task_id=agent_task_id)
                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="LITERATURE_BRIDGING_STARTED",
                    description=f"Agent {agent_id} started literature bridging task {agent_task_id}: {db_agent_task.task_description}",
                    metadata=db_agent_task.model_dump()
                )

                # COMP-015: Simulate scanning scientific literature and synthesizing connections
                adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

                query_text = f"Retrieve scientific literature for bridging connections related to: {db_agent_task.task_description}"
                adworkbench_query_payload = {"query_text": query_text}

                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="ADWORKBENCH_QUERY_INITIATED",
                    description=f"Agent {agent_id} querying AD Workbench for literature data: {query_text}",
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

                # TODO: CQ-LB-001: Replace with actual asynchronous calls/polling mechanism for AD Workbench query. The blocking time.sleep() has been removed.
                # A proper async implementation would involve storing adworkbench_query_id and having a separate task or service check for completion.
                query_result_response = requests.get(
                    f"{ADWORKBENCH_PROXY_URL}/adworkbench/query/{adworkbench_query_id}/status",
                    headers=adworkbench_headers
                )
                query_result_response.raise_for_status()
                final_query_status = query_result_response.json()

                if final_query_status["status"] != "COMPLETED":
                    raise Exception(f"AD Workbench query for literature data failed or timed out: {final_query_status['status']}")
                
                result_data_str = final_query_status["result_data"]
                if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
                    raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
                raw_data_summary = json.loads(result_data_str)
        
                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="ADWORKBENCH_QUERY_COMPLETED",
                    description=f"Agent {agent_id} received literature data from AD Workbench query {adworkbench_query_id}.",
                    metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data_summary.get("data", [])[:1]}
                )

                # CQ-LB-002: Implement actual literature analysis and synthesis using LLM
                analysis_prompt = f"""Analyze the following scientific literature data and identify meaningful connections between different research areas related to Alzheimer's disease.

Literature Data: {json.dumps(raw_data_summary)}

Task: {db_agent_task.task_description}

Please identify:
1. Key research areas represented in the data
2. Potential connections or relationships between these areas
3. Novel insights that emerge from bridging these areas
4. Supporting evidence or references

Format your response as a JSON object with the following structure:
{{
    "bridged_areas": [
        {{
            "area_a": "Research Area 1",
            "area_b": "Research Area 2", 
            "connection": "Description of the connection",
            "references": ["ref1", "ref2"],
            "strength": "high/medium/low"
        }}
    ],
    "key_insights": ["insight1", "insight2"],
    "summary": "Overall summary of connections found"
}}"""

                llm_payload = {
                    "model_name": "gemini-1.5-flash",
                    "prompt": analysis_prompt,
                    "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
                }
                llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}

                llm_response = requests.post(
                    f"{LLM_SERVICE_URL}/llm/structured-output",
                    headers=llm_headers,
                    json={
                        "model_name": "gemini-1.5-flash",
                        "prompt": analysis_prompt,
                        "response_format": {
                            "type": "json_object",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "bridged_areas": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "area_a": {"type": "string"},
                                                "area_b": {"type": "string"},
                                                "connection": {"type": "string"},
                                                "references": {"type": "array", "items": {"type": "string"}},
                                                "strength": {"type": "string", "enum": ["high", "medium", "low"]}
                                            },
                                            "required": ["area_a", "area_b", "connection"]
                                        }
                                    },
                                    "key_insights": {"type": "array", "items": {"type": "string"}},
                                    "summary": {"type": "string"}
                                },
                                "required": ["bridged_areas", "key_insights", "summary"]
                            }
                        },
                        "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
                    }
                )
                llm_response.raise_for_status()
                llm_result = llm_response.json()

                # Parse the structured LLM response
                try:
                    literature_analysis = llm_result["structured_output"]
                    literature_connections = {
                        "topic": db_agent_task.task_description,
                        "bridged_areas": literature_analysis.get("bridged_areas", []),
                        "key_insights": literature_analysis.get("key_insights", []),
                        "summary": literature_analysis.get("summary", f"Analysis completed for {len(raw_data_summary.get('data', []))} literature entries."),
                        "llm_analysis": llm_result.get("response_text", "")
                    }
                except (KeyError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    # Fallback to basic structure
                    literature_connections = {
                        "topic": db_agent_task.task_description,
                        "bridged_areas": [],
                        "key_insights": ["LLM analysis failed to parse"],
                        "summary": f"Retrieved {len(raw_data_summary.get('data', []))} literature entries but analysis failed.",
                        "error": str(e)
                    }

                insight_name_val = f"Literature Bridge: {db_agent_task.task_description}"
                insight_publish_request_obj = schemas.InsightPublishRequest(
                    insight_name=insight_name_val,
                    insight_description=f"Automatically synthesized connections between disparate research areas for: {db_agent_task.task_description}.",
                    data_source_ids=[f"adworkbench_query_{adworkbench_query_id}"],
                    payload=literature_connections,
                    tags=["literature_bridging", "research_synthesis", agent_id]
                )
                insight_payload = insight_publish_request_obj.model_dump_json()

                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="PUBLISHING_LITERATURE_INSIGHT",
                    description=f"Agent {agent_id} publishing literature bridging insight.",
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
                    "agent_output": f"Literature bridging completed for task {agent_task_id}.",
                    "literature_connections": literature_connections,
                    "published_insight_id": publish_result.get("insight_id")
                }

                crud.update_agent_task_status(db, agent_task_id, "COMPLETED", mock_result)
                crud.update_agent_state(db, agent_id, current_task_id=None)
                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="LITERATURE_BRIDGING_COMPLETED",
                    description=f"Agent {agent_id} completed literature bridging task {agent_task_id} and published insight.",
                    metadata={"task_result": mock_result}
                )
                return {"agent_task_id": agent_task_id, "status": "COMPLETED", "result": mock_result}
            except requests.exceptions.RequestException as e:
                error_message = f"AD Workbench Proxy or external API call failed: {e}"
                crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message})
                crud.update_agent_state(db, agent_id, current_task_id=None)
                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="LITERATURE_BRIDGING_FAILED",
                    description=f"Agent {agent_id} failed to bridge literature for task {agent_task_id}: {error_message}",
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
                    event_type="LITERATURE_BRIDGING_FAILED",
                    description=f"Agent {agent_id} failed to bridge literature for task {agent_task_id}: {error_message}",
                    metadata={"error": error_message}
                )
                self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
                raise
            finally:
                db.close()
        except Exception as e:
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

        # CQ-LB-003: Implement LLM-powered literature bridging analysis
        reflection_prompt = f"""Analyze the recent performance of an Alzheimer's disease literature bridging agent and provide insights for improvement.

Agent Performance Data:
- Total tasks in last 7 days: {task_summary['total_tasks']}
- Completed tasks: {task_summary['completed']}
- Failed tasks: {task_summary['failed']}
- Pending tasks: {task_summary['pending']}
- Recent audit events: {len(recent_audit_events)}

Recent Tasks Summary:
{chr(10).join([f"- {task.task_description[:100]}... (Status: {task.status})" for task in recent_tasks[:5]])}

Recent Audit Events:
{chr(10).join([f"- {event.event_type}: {event.description[:100]}..." for event in recent_audit_events[:5]])}

Agent specializes in: Literature bridging, connecting disparate research findings, identifying knowledge gaps, and synthesizing connections across Alzheimer's disease research domains.

Please analyze this agent's performance and provide:

1. **Literature Analysis Quality**:
   - Effectiveness of literature search and retrieval
   - Quality of connection synthesis between studies
   - Identification of knowledge gaps and research opportunities
   - Cross-domain integration capabilities

2. **Bridging Methodology**:
   - Sophistication of connection algorithms
   - Handling of conflicting evidence
   - Temporal trend analysis in literature
   - Citation network analysis effectiveness

3. **Research Synthesis**:
   - Quality of evidence integration
   - Identification of emerging patterns
   - Predictive power of literature trends
   - Clinical translation potential assessment

4. **Improvement Recommendations**:
   - Enhanced literature search strategies
   - Better connection algorithms
   - Improved knowledge gap identification
   - More sophisticated synthesis methods

5. **Future Research Directions**:
   - Novel literature analysis approaches
   - Integration with other research modalities
   - Predictive modeling from literature trends
   - Clinical impact assessment methods

For Alzheimer's disease literature context, consider:
- Biomarker research integration challenges
- Treatment development pipeline analysis
- Epidemiological study connections
- Translational research gaps
- Regulatory and clinical trial literature

Format your response as a JSON object with this structure:
{{
    "literature_analysis": {{
        "search_effectiveness": "High/Medium/Low",
        "connection_quality": 0.85,
        "gap_identification": "Assessment of gap finding",
        "cross_domain_integration": "Quality of interdisciplinary connections"
    }},
    "bridging_methodology": {{
        "algorithm_sophistication": "Assessment of bridging methods",
        "conflict_resolution": "Handling of contradictory findings",
        "temporal_analysis": "Trend identification quality",
        "network_analysis": "Citation network effectiveness"
    }},
    "research_synthesis": {{
        "evidence_integration": "Quality of synthesis",
        "pattern_recognition": "Emerging pattern identification",
        "predictive_accuracy": 0.75,
        "clinical_translation": "Clinical relevance assessment"
    }},
    "improvement_recommendations": {{
        "search_strategies": ["Strategy 1", "Strategy 2"],
        "connection_algorithms": ["Algorithm 1", "Algorithm 2"],
        "gap_analysis": ["Method 1", "Method 2"],
        "synthesis_methods": ["Method 1", "Method 2"]
    }},
    "future_directions": {{
        "novel_approaches": ["Approach 1", "Approach 2"],
        "integration_opportunities": ["Integration 1", "Integration 2"],
        "predictive_modeling": ["Model 1", "Model 2"],
        "clinical_assessment": ["Assessment 1", "Assessment 2"]
    }},
    "reflection_summary": {{
        "key_insights": ["Insight 1", "Insight 2"],
        "action_items": ["Action 1", "Action 2"],
        "expected_impact": "Description of expected improvements"
    }}
}}"""

        llm_reflection_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/structured-output",
            headers={"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"},
            json={
                "model_name": "gemini-1.5-flash",
                "prompt": reflection_prompt,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "literature_analysis": {
                                "type": "object",
                                "properties": {
                                    "search_effectiveness": {"type": "string"},
                                    "connection_quality": {"type": "number"},
                                    "gap_identification": {"type": "string"},
                                    "cross_domain_integration": {"type": "string"}
                                }
                            },
                            "bridging_methodology": {
                                "type": "object",
                                "properties": {
                                    "algorithm_sophistication": {"type": "string"},
                                    "conflict_resolution": {"type": "string"},
                                    "temporal_analysis": {"type": "string"},
                                    "network_analysis": {"type": "string"}
                                }
                            },
                            "research_synthesis": {
                                "type": "object",
                                "properties": {
                                    "evidence_integration": {"type": "string"},
                                    "pattern_recognition": {"type": "string"},
                                    "predictive_accuracy": {"type": "number"},
                                    "clinical_translation": {"type": "string"}
                                }
                            },
                            "improvement_recommendations": {
                                "type": "object",
                                "properties": {
                                    "search_strategies": {"type": "array", "items": {"type": "string"}},
                                    "connection_algorithms": {"type": "array", "items": {"type": "string"}},
                                    "gap_analysis": {"type": "array", "items": {"type": "string"}},
                                    "synthesis_methods": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "future_directions": {
                                "type": "object",
                                "properties": {
                                    "novel_approaches": {"type": "array", "items": {"type": "string"}},
                                    "integration_opportunities": {"type": "array", "items": {"type": "string"}},
                                    "predictive_modeling": {"type": "array", "items": {"type": "string"}},
                                    "clinical_assessment": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "reflection_summary": {
                                "type": "object",
                                "properties": {
                                    "key_insights": {"type": "array", "items": {"type": "string"}},
                                    "action_items": {"type": "array", "items": {"type": "string"}},
                                    "expected_impact": {"type": "string"}
                                }
                            }
                        },
                        "required": ["literature_analysis", "bridging_methodology", "research_synthesis", "improvement_recommendations", "future_directions", "reflection_summary"]
                    }
                },
                "metadata": {"agent_id": agent_id, "reflection_type": "literature_bridging_analysis"}
            }
        )
        llm_reflection_response.raise_for_status()
        reflection_analysis = llm_reflection_response.json()["structured_output"]

        # Generate analysis outcome and proposed adjustments from LLM insights
        analysis_outcome = f"Agent {agent_id} completed comprehensive LLM-powered literature analysis. " \
                          f"Search effectiveness: {reflection_analysis['literature_analysis']['search_effectiveness']}. " \
                          f"Connection quality: {reflection_analysis['literature_analysis']['connection_quality']:.1%}. " \
                          f"Key insights: {', '.join(reflection_analysis['reflection_summary']['key_insights'][:2])}."

        proposed_adjustments = reflection_analysis['improvement_recommendations']['search_strategies'] + \
                              reflection_analysis['improvement_recommendations']['connection_algorithms'] + \
                              reflection_analysis['improvement_recommendations']['gap_analysis']

        reflection_result = {
            "analysis_summary": analysis_outcome,
            "proposed_adjustments": proposed_adjustments,
            "task_performance_summary": task_summary,
            "recent_audit_event_count": len(recent_audit_events),
            "llm_powered_analysis": reflection_analysis,
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
