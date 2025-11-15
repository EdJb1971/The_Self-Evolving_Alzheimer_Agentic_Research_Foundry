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

# Define a maximum size for the result_data JSON string to prevent DoS attacks
MAX_RESULT_DATA_SIZE_BYTES = 1 * 1024 * 1024 # 1MB limit

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")

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
        # SEC-001-DSA: Log only string representation of exception to prevent information exposure
        print(f"Failed to log audit event: {str(e)}")

@celery_app.task(bind=True, name="screen_drugs_task")
def screen_drugs_task(self, agent_task_id: int):
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
            event_type="DRUG_SCREENING_STARTED",
            description=f"Agent {agent_id} started drug screening task {agent_task_id}: {db_agent_task.task_description}",
            metadata=db_agent_task.model_dump()
        )

        # STORY-303: Simulate drug screening by querying AD Workbench Proxy and publishing insights
        adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

        # 1. Simulate querying AD Workbench for relevant data (e.g., disease pathways, target profiles)
        query_text = f"Retrieve disease pathway data for drug screening related to: {db_agent_task.task_description}"
        adworkbench_query_payload = {"query_text": query_text}

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_INITIATED",
            description=f"Agent {agent_id} querying AD Workbench for drug screening data: {query_text}",
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

        # Simulate waiting for AD Workbench query to complete
        time.sleep(8) 
        query_result_response = requests.get(
            f"{ADWORKBENCH_PROXY_URL}/adworkbench/query/{adworkbench_query_id}/status",
            headers=adworkbench_headers
        )
        query_result_response.raise_for_status()
        final_query_status = query_result_response.json()

        if final_query_status["status"] != "COMPLETED":
            raise Exception(f"AD Workbench query for drug screening data failed or timed out: {final_query_status['status']}")
        
        # SEC-002-DSA: Validate size of result_data before parsing to prevent DoS
        result_data_str = final_query_status["result_data"]
        if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
            raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
        raw_data_summary = json.loads(result_data_str)
        
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_COMPLETED",
            description=f"Agent {agent_id} received drug screening data from AD Workbench query {adworkbench_query_id}.",
            metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data_summary.get("data", [])[:1]}
        )

        # 2. Simulate complex in-silico drug screening or literature review
        time.sleep(10) # Simulate complex screening work

        drug_candidates = [
            {
                "candidate_id": "DRUG-001",
                "name": "Compound Alpha",
                "target": "Beta-secretase 1",
                "mechanism_of_action": "BACE1 inhibitor",
                "rationale": "High binding affinity to BACE1, low off-target effects based on in-silico models."
            },
            {
                "candidate_id": "DRUG-002",
                "name": "Molecule Beta",
                "target": "Tau protein aggregation",
                "mechanism_of_action": "Tau aggregation inhibitor",
                "rationale": "Literature review suggests strong evidence for reducing Tau pathology in preclinical models."
            }
        ]

        # 3. Publish the identified drug candidates as an insight
        # CQ-001-DSA: Capture insight_name before serialization for audit log metadata
        insight_name_val = f"Potential Drug Candidates: {db_agent_task.task_description.replace(' ', '_')}"
        insight_publish_request_obj = schemas.InsightPublishRequest(
            insight_name=insight_name_val,
            insight_description=f"Automatically identified drug candidates for: {db_agent_task.task_description}.",
            data_source_ids=[f"adworkbench_query_{adworkbench_query_id}"],
            payload={"candidates": drug_candidates},
            tags=["drug_screening", "drug_discovery", agent_id]
        )
        insight_payload = insight_publish_request_obj.model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_DRUG_CANDIDATES_INSIGHT",
            description=f"Agent {agent_id} publishing identified drug candidates insight.",
            metadata={"insight_name": insight_name_val} # CQ-001-DSA: Use the captured insight_name_val
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
            "agent_output": f"Drug screening completed for task {agent_task_id}.",
            "identified_drug_candidates": drug_candidates,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", mock_result)
        crud.update_agent_state(db, agent_id, current_task_id=None) # Task completed, clear current task
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="DRUG_SCREENING_COMPLETED",
            description=f"Agent {agent_id} completed drug screening task {agent_task_id} and published insight.",
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
            event_type="DRUG_SCREENING_FAILED",
            description=f"Agent {agent_id} failed to screen drugs for task {agent_task_id}: {error_message}",
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
            event_type="DRUG_SCREENING_FAILED",
            description=f"Agent {agent_id} failed to screen drugs for task {agent_task_id}: {error_message}",
            metadata={"error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
