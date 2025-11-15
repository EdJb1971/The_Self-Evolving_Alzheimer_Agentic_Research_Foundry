import time
import json
import os
import requests
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas

AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
ADWORKBENCH_PROXY_URL = os.getenv("ADWORKBENCH_PROXY_URL")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")
if not LLM_SERVICE_URL or not LLM_API_KEY:
    raise ValueError("LLM_SERVICE_URL or LLM_API_KEY environment variables not set.")

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
        print(f"Audit log successful: {event_type} for {entity_type}:{entity_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to log audit event: {e}")

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def harmonize_data_task(self, agent_task_id: int):
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
            event_type="HARMONIZATION_STARTED",
            description=f"Agent {agent_id} started data harmonization task {agent_task_id}: {db_agent_task.task_description}",
            metadata=db_agent_task.model_dump()
        )

        # STORY-304: Simulate data harmonization by scanning AD Workbench and publishing insights
        adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_SCAN_INITIATED",
            description=f"Agent {agent_id} initiating AD Workbench data scan.",
            metadata={"task_description": db_agent_task.task_description}
        )

        scan_response = requests.get(f"{ADWORKBENCH_PROXY_URL}/adworkbench/data/scan", headers=adworkbench_headers)
        scan_response.raise_for_status()
        scanned_data = scan_response.json()
        datasets_found = scanned_data.get("datasets_found", [])

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_SCAN_COMPLETED",
            description=f"Agent {agent_id} completed AD Workbench data scan. Found {len(datasets_found)} datasets.",
            metadata={"datasets_found": datasets_found}
        )

        # Perform schema harmonization using LLM
        harmonization_prompt = f"Harmonize the schemas from these datasets: {datasets_found}. Create a unified schema."
        llm_payload = {
            "model_name": "gemini-1.5-flash",
            "prompt": harmonization_prompt,
            "metadata": {"agent_task_id": agent_task_id}
        }
        llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}
        
        llm_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/chat",
            headers=llm_headers,
            json=llm_payload
        )
        llm_response.raise_for_status()
        llm_result = llm_response.json()
        harmonization_text = llm_result["response_text"]
        
        harmonized_schema = {
            "schema_name": f"Harmonized_Schema_for_{db_agent_task.task_description.replace(' ', '_')}",
            "version": "1.0",
            "llm_harmonization": harmonization_text,
            "source_datasets": datasets_found,
            "harmonization_report": "LLM-generated harmonized schema."
        }

        # Publish harmonized schema as an insight
        insight_payload = schemas.InsightPublishRequest(
            insight_name=f"Harmonized Schema: {harmonized_schema['schema_name']}",
            insight_description=f"Automatically generated harmonized schema for datasets: {', '.join(datasets_found)}.",
            data_source_ids=datasets_found,
            payload=harmonized_schema,
            tags=["data_harmonization", "schema", agent_id]
        ).model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_HARMONIZED_INSIGHT",
            description=f"Agent {agent_id} publishing harmonized schema insight.",
            metadata={"insight_name": harmonized_schema['schema_name']}
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
            "agent_output": f"Data harmonization completed for task {agent_task_id}.",
            "harmonized_schema": harmonized_schema,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", mock_result)
        crud.update_agent_state(db, agent_id, current_task_id=None) # Task completed, clear current task
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="HARMONIZATION_COMPLETED",
            description=f"Agent {agent_id} completed data harmonization task {agent_task_id} and published insight.",
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
            event_type="HARMONIZATION_FAILED",
            description=f"Agent {agent_id} failed to harmonize data for task {agent_task_id}: {error_message}",
            metadata={"error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    except Exception as e:
        error_message = str(e)
        crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message})
        crud.update_agent_state(db, agent_id, current_task_id=None) # Task failed, clear current task
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="HARMONIZATION_FAILED",
            description=f"Agent {agent_id} failed to harmonize data for task {agent_task_id}: {error_message}",
            metadata={"error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
