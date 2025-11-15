import time
import json
import os
import requests
import logging
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

def poll_adworkbench_query_status(adworkbench_query_id: str, headers: dict, proxy_url: str, max_retries: int = 10, delay_seconds: int = 2):
    for i in range(max_retries):
        query_result_response = requests.get(
            f"{proxy_url}/adworkbench/query/{adworkbench_query_id}/status",
            headers=headers
        )
        query_result_response.raise_for_status()
        final_query_status = query_result_response.json()

        if final_query_status["status"] == "COMPLETED":
            return final_query_status
        elif final_query_status["status"] == "FAILED":
            raise Exception(f"AD Workbench query failed: {final_query_status.get('error', 'Unknown error')}")
        
        logger.info(f"Polling AD Workbench query {adworkbench_query_id}. Status: {final_query_status['status']}. Retry {i+1}/{max_retries}")
        time.sleep(delay_seconds)
    
    raise Exception(f"AD Workbench query {adworkbench_query_id} timed out after {max_retries * delay_seconds} seconds.")

@celery_app.task(bind=True, name="validate_hypothesis_task")
def validate_hypothesis_task(self, agent_task_id: int):
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
            event_type="HYPOTHESIS_VALIDATION_STARTED",
            description=f"Agent {agent_id} started hypothesis validation task {agent_task_id}: {db_agent_task.task_description}",
            metadata=db_agent_task.model_dump()
        )

        # STORY-306: Simulate hypothesis validation by querying AD Workbench Proxy and publishing insights
        adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

        query_text = f"Retrieve supporting data for hypothesis: {db_agent_task.task_description}"
        adworkbench_query_payload = {"query_text": query_text}

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_INITIATED",
            description=f"Agent {agent_id} querying AD Workbench for hypothesis data: {query_text}",
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

        # CQ-SPRINT12-001: Replaced time.sleep with actual asynchronous polling mechanism
        final_query_status = poll_adworkbench_query_status(adworkbench_query_id, adworkbench_headers, ADWORKBENCH_PROXY_URL)

        result_data_str = final_query_status["result_data"]
        if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
            raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
        raw_data_summary = json.loads(result_data_str)
        
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_COMPLETED",
            description=f"Agent {agent_id} received hypothesis data from AD Workbench query {adworkbench_query_id}.",
            metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data_summary.get("data", [])[:1]}
        )

        # CQ-SPRINT12-002: Placeholder for actual complex hypothesis evaluation (LLM interaction)
        # TODO: Implement the actual LLM interaction and complex evaluation logic required for hypothesis validation.
        # This should include robust prompt engineering, response parsing, and error handling.
        # For now, a simulated outcome is generated.
        # time.sleep(10)

        validation_result_status = "SUPPORTED"
        validation_reasoning = f"Hypothesis '{db_agent_task.task_description}' is strongly supported by {len(raw_data_summary.get('data', []))} patient records showing significant correlation with biomarker X and Y."
        if "contradictory" in db_agent_task.task_description.lower():
            validation_result_status = "REJECTED"
            validation_reasoning = f"Hypothesis '{db_agent_task.task_description}' is contradicted by recent findings in dataset Z."

        hypothesis_validation_report = {
            "hypothesis_text": db_agent_task.task_description,
            "validation_status": validation_result_status,
            "reasoning": validation_reasoning,
            "supporting_evidence_count": len(raw_data_summary.get('data', [])),
            "contradictory_evidence_found": (validation_result_status == "REJECTED"),
            "data_sources_consulted": [f"adworkbench_query_{adworkbench_query_id}"]
        }

        insight_name_val = f"Hypothesis Validation: {db_agent_task.task_description[:50]}..."
        insight_publish_request_obj = schemas.InsightPublishRequest(
            insight_name=insight_name_val,
            insight_description=f"Validation report for hypothesis: {db_agent_task.task_description}. Status: {validation_result_status}.",
            data_source_ids=[f"adworkbench_query_{adworkbench_query_id}"],
            payload=hypothesis_validation_report,
            tags=["hypothesis_validation", validation_result_status.lower(), agent_id]
        )
        insight_payload = insight_publish_request_obj.model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_HYPOTHESIS_VALIDATION_INSIGHT",
            description=f"Agent {agent_id} publishing hypothesis validation insight.",
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
            "agent_output": f"Hypothesis validation completed for task {agent_task_id}.",
            "validation_report": hypothesis_validation_report,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", mock_result)
        crud.update_agent_state(db, agent_id, current_task_id=None)
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="HYPOTHESIS_VALIDATION_COMPLETED",
            description=f"Agent {agent_id} completed hypothesis validation task {agent_task_id} and published insight.",
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
            event_type="HYPOTHESIS_VALIDATION_FAILED",
            description=f"Agent {agent_id} failed to validate hypothesis for task {agent_task_id}: {error_message}",
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
            event_type="HYPOTHESIS_VALIDATION_FAILED",
            description=f"Agent {agent_id} failed to validate hypothesis for task {agent_task_id}: {error_message}",
            metadata={"error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()

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

        # CQ-SPRINT12-003: Placeholder for actual analysis/LLM interaction
        # TODO: Implement the actual logic for analyzing recent tasks and audit events, potentially using LLMs
        # for generating insights and proposed adjustments during self-reflection.
        # For now, a simulated outcome is generated.
        # time.sleep(5)
        analysis_outcome = f"Agent {agent_id} reviewed {task_summary['total_tasks']} tasks in the last 7 days. " \
                           f"Completed: {task_summary['completed']}, Failed: {task_summary['failed']}. " \
                           "Identified potential for improved data source selection and more robust error handling in hypothesis validation."
        
        proposed_adjustments = [
            "Refine statistical models for hypothesis testing.",
            "Expand search for contradictory evidence across more diverse datasets.",
            "Improve prompt engineering for LLM-based plausibility checks."
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
        # Note: The original code had crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message}) here.
        # This is incorrect as agent_task_id is not available in reflection task scope. Corrected to update agent state.
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
