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

                # TODO: CQ-LB-002: Replace with actual literature analysis and synthesis (LLM interaction). The blocking time.sleep() has been removed.
                literature_connections = {
                    "topic": db_agent_task.task_description,
                    "bridged_areas": [
                        {"area_a": "Neuroinflammation", "area_b": "Gut Microbiome", "connection": "Emerging evidence links gut dysbiosis to neuroinflammatory pathways in AD.", "references": ["PMID:12345", "PMID:67890"]},
                        {"area_a": "Amyloid Beta", "area_b": "Sleep Disorders", "connection": "Poor sleep quality accelerates amyloid-beta accumulation and impairs clearance.", "references": ["PMID:11223", "PMID:44556"]}
                    ],
                    "summary": f"Synthesized connections between {len(raw_data_summary.get('data', []))} literature entries."
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

        # TODO: CQ-LB-003: Replace with actual analysis/LLM interaction. The blocking time.sleep() has been removed.
        analysis_outcome = f"Agent {agent_id} reviewed {task_summary['total_tasks']} tasks in the last 7 days. " \
                           f"Completed: {task_summary['completed']}, Failed: {task_summary['failed']}. " \
                           "Identified potential for improving literature search strategies and connection synthesis."
        
        proposed_adjustments = [
            "Refine keywords for literature search to improve relevance.",
            "Explore new NLP models for extracting nuanced connections.",
            "Increase cross-referencing with external knowledge graphs."
        ]

        reflection_result = {
            "analysis_summary": analysis_outcome,
            "proposed_adjustments": proposed_adjustments,
            "task_performance_summary": task_summary,
            "recent_audit_event_count": len(recent_audit_events),
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
