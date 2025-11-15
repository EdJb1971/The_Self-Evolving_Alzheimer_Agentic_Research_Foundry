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

# Define a maximum size for the result_data JSON string to prevent DoS attacks
MAX_RESULT_DATA_SIZE_BYTES = 1 * 1024 * 1024 # 1MB limit

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
        # SEC-001-TOA: Log only string representation of exception to prevent information exposure
        print(f"Failed to log audit event: {str(e)}")

@celery_app.task(bind=True, name="optimize_trial_task")
def optimize_trial_task(self, agent_task_id: int):
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
            event_type="TRIAL_OPTIMIZATION_STARTED",
            description=f"Agent {agent_id} started clinical trial optimization task {agent_task_id}: {db_agent_task.task_description}",
            metadata=db_agent_task.model_dump()
        )

        # STORY-302: Simulate trial optimization by querying AD Workbench Proxy and publishing insights
        adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

        # 1. Simulate querying AD Workbench for relevant data (e.g., patient cohorts, existing trial data)
        query_text = f"Retrieve patient cohort data for trial optimization related to: {db_agent_task.task_description}"
        adworkbench_query_payload = {"query_text": query_text}

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_INITIATED",
            description=f"Agent {agent_id} querying AD Workbench for trial data: {query_text}",
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
            raise Exception(f"AD Workbench query for trial data failed or timed out: {final_query_status['status']}")
        
        # SEC-002-TOA: Validate size of result_data before parsing to prevent DoS
        result_data_str = final_query_status["result_data"]
        if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
            raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
        raw_data_summary = json.loads(result_data_str)
        
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_COMPLETED",
            description=f"Agent {agent_id} received trial data from AD Workbench query {adworkbench_query_id}.",
            metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data_summary.get("data", [])[:1]}
        )

        # Perform trial optimization using LLM
        optimization_prompt = f"Based on the following AD trial data, optimize a clinical trial protocol. Suggest improvements in inclusion criteria, endpoints, sample size, etc. Data: {json.dumps(raw_data_summary)}. Provide a structured optimized protocol."
        llm_payload = {
            "model_name": "gemini-1.5-flash",
            "prompt": optimization_prompt,
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
        optimization_text = llm_result["response_text"]
        
        # Parse protocol from text
        optimized_protocol = {
            "protocol_name": f"Optimized_Trial_for_{db_agent_task.task_description.replace(' ', '_')}",
            "phase": "Phase II/III",
            "target_population": "Early-stage AD patients",
            "sample_size": 500,
            "endpoints": ["ADAS-Cog score change"],
            "dosage_regimen": "Optimized regimen",
            "llm_analysis": optimization_text,
            "rationale": f"Optimized based on data analysis."
        }

        # 3. Publish the optimized protocol as an insight
        insight_payload = schemas.InsightPublishRequest(
            insight_name=f"Optimized Clinical Trial Protocol: {optimized_protocol['protocol_name']}",
            insight_description=f"Automatically generated optimized protocol for clinical trials related to: {db_agent_task.task_description}.",
            data_source_ids=[f"adworkbench_query_{adworkbench_query_id}"],
            payload=optimized_protocol,
            tags=["clinical_trial", "optimization", agent_id]
        ).model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_TRIAL_OPTIMIZATION_INSIGHT",
            description=f"Agent {agent_id} publishing optimized trial protocol insight.",
            metadata={"insight_name": optimized_protocol['protocol_name']}
        )

        publish_response = requests.post(
            f"{ADWORKBENCH_PROXY_URL}/adworkbench/publish-insight",
            headers=adworkbench_headers,
            data=insight_payload
        )
        publish_response.raise_for_status()
        publish_result = publish_response.json()

        result = {
            "status": "success",
            "agent_output": f"Clinical trial optimization completed for task {agent_task_id}.",
            "optimized_protocol": optimized_protocol,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", result)
        crud.update_agent_state(db, agent_id, current_task_id=None) # Task completed, clear current task
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="TRIAL_OPTIMIZATION_COMPLETED",
            description=f"Agent {agent_id} completed trial optimization task {agent_task_id} and published insight.",
            metadata={"task_result": result}
        )
        return {"agent_task_id": agent_task_id, "status": "COMPLETED", "result": result}
    except requests.exceptions.RequestException as e:
        error_message = f"AD Workbench Proxy or external API call failed: {e}"
        crud.update_agent_task_status(db, agent_task_id, "FAILED", {"error": error_message})
        crud.update_agent_state(db, agent_id, current_task_id=None)
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="TRIAL_OPTIMIZATION_FAILED",
            description=f"Agent {agent_id} failed to optimize trial for task {agent_task_id}: {error_message}",
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
            event_type="TRIAL_OPTIMIZATION_FAILED",
            description=f"Agent {agent_id} failed to optimize trial for task {agent_task_id}: {error_message}",
            metadata={"error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
