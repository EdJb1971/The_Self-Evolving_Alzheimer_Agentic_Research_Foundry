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
from alznexus_audit_trail.schemas import MAX_METADATA_SIZE_BYTES # SEC-001

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

logger = logging.getLogger(__name__) # CQ-003

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
        logger.info(f"Audit log successful: {event_type} for {entity_type}:{entity_id}") # CQ-003
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to log audit event: {str(e)}", exc_info=True) # CQ-003

@celery_app.task(bind=True)
def execute_agent_task(self, agent_task_id: int):
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
                    event_type="TASK_EXECUTION_STARTED",
                    description=f"Agent {agent_id} started executing task {agent_task_id}: {db_agent_task.task_description}",
                    metadata=db_agent_task.model_dump()
                )

                # STORY-301: Simulate biomarker identification by querying AD Workbench Proxy
                query_text = f"Retrieve data for biomarker analysis related to: {db_agent_task.task_description}"
                adworkbench_query_payload = {"query_text": query_text}
                adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="ADWORKBENCH_QUERY_INITIATED",
                    description=f"Agent {agent_id} querying AD Workbench for data: {query_text}",
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

                # Poll for AD Workbench query completion asynchronously
                max_polls = 60  # 5 minutes max
                poll_count = 0
                while poll_count < max_polls:
                    time.sleep(5)  # Poll every 5 seconds
                    query_result_response = requests.get(
                        f"{ADWORKBENCH_PROXY_URL}/adworkbench/query/{adworkbench_query_id}/status",
                        headers=adworkbench_headers
                    )
                    query_result_response.raise_for_status()
                    final_query_status = query_result_response.json()
                    
                    if final_query_status["status"] == "COMPLETED":
                        break
                    elif final_query_status["status"] == "FAILED":
                        raise Exception(f"AD Workbench query failed: {final_query_status.get('message', 'Unknown error')}")
                    poll_count += 1
                
                if final_query_status["status"] != "COMPLETED":
                    raise Exception("AD Workbench query timed out")
                
                result_data_str = final_query_status["result_data"]
                if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
                    raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
                
                # Safe JSON parsing with fallback
                try:
                    raw_data = json.loads(result_data_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed for AD Workbench result: {e}")
                    print(f"Raw data preview: {result_data_str[:200]}...")
                    # Fallback to basic structure
                    raw_data = {"error": "JSON parsing failed", "raw_data": result_data_str[:1000]}
                
                # Apply basic differential privacy: add noise to numerical data
                if "data" in raw_data and isinstance(raw_data["data"], list):
                    for item in raw_data["data"]:
                        for key, value in item.items():
                            if isinstance(value, (int, float)):
                                noise = random.gauss(0, 0.1 * abs(value))  # 10% noise
                                item[key] = value + noise
                
                # SEC-001: Truncate raw_data.get("message") to prevent metadata size overflow
                adworkbench_message = raw_data.get("message", "")
                # Estimate available space for the message, allowing for other metadata fields
                # A simple heuristic: reserve half the MAX_METADATA_SIZE_BYTES for the message
                # The actual size will depend on other fields in mock_result, but this provides a safe upper bound.
                max_message_len_bytes = MAX_METADATA_SIZE_BYTES // 2 
                truncated_message = adworkbench_message
                if len(adworkbench_message.encode('utf-8')) > max_message_len_bytes:
                    # Truncate and add an ellipsis
                    truncated_message = adworkbench_message.encode('utf-8')[:max_message_len_bytes - 3].decode('utf-8', 'ignore') + "..."
                    logger.warning(f"Truncated AD Workbench message for audit log metadata due to size constraint. Original size: {len(adworkbench_message.encode('utf-8'))} bytes.")

                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="ADWORKBENCH_QUERY_COMPLETED",
                    description=f"Agent {agent_id} received data from AD Workbench query {adworkbench_query_id}.",
                    metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data.get("data", [])[:1]}
                )

                # Perform data analysis using LLM
                analysis_prompt = f"Analyze the following AD Workbench data for potential novel biomarkers related to Alzheimer's disease. Data: {json.dumps(raw_data)}. Identify any promising biomarkers and explain why."
                llm_payload = {
                    "model_name": "gemini-1.5-flash",  # Easily swappable
                    "prompt": analysis_prompt,
                    "metadata": {"agent_task_id": agent_task_id}
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
                
                analysis_text = llm_result["response_text"]
                if not analysis_text or not analysis_text.strip():
                    raise Exception("LLM returned empty analysis text")
                
                # Parse biomarkers from analysis (simple extraction, could be improved)
                identified_biomarkers = []
                if "biomarker" in analysis_text.lower():
                    # Simple regex or split to extract
                    lines = analysis_text.split('\n')
                    for line in lines:
                        if "biomarker" in line.lower():
                            identified_biomarkers.append(line.strip())
                if not identified_biomarkers:
                    identified_biomarkers = ["No specific biomarkers identified from analysis"]

                result = {
                    "status": "success",
                    "agent_output": f"Processed task {agent_task_id} for {agent_id}. Analysis completed.",
                    "identified_biomarkers": identified_biomarkers,
                    "adworkbench_query_id": adworkbench_query_id,
                    "llm_analysis": analysis_text,
                    "adworkbench_query_result_summary": truncated_message
                }

                crud.update_agent_task_status(db, agent_task_id, "COMPLETED", result)
                crud.update_agent_state(db, agent_id, current_task_id=None) # Task completed, clear current task
                log_audit_event(
                    entity_type="AGENT",
                    entity_id=f"{agent_id}-{agent_task_id}",
                    event_type="BIOMARKER_IDENTIFICATION_COMPLETED",
                    description=f"Agent {agent_id} completed task {agent_task_id} and identified biomarkers.",
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
                    event_type="TASK_EXECUTION_FAILED",
                    description=f"Agent {agent_id} failed to execute task {agent_task_id}: {error_message}",
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
                    event_type="TASK_EXECUTION_FAILED",
                    description=f"Agent {agent_id} failed to execute task {agent_task_id}: {error_message}",
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
def perform_reflection_task(self, agent_id: str, reflection_metadata: dict):
    db: Session = SessionLocal()
    try:
        # STORY-502: Incorporate a 'reflection' mechanism
        crud.update_agent_state(db, agent_id, last_reflection_at=datetime.utcnow(), metadata_json={"reflection_status": "IN_PROGRESS"})
        log_audit_event(
            entity_type="AGENT",
            entity_id=agent_id,
            event_type="REFLECTION_STARTED",
            description=f"Agent {agent_id} initiated self-reflection.",
            metadata=reflection_metadata
        )

        # 1. Fetch recent tasks for this agent
        recent_tasks = crud.get_recent_agent_tasks(db, agent_id, days_ago=7)
        task_summary = {"total_tasks": len(recent_tasks), "completed": 0, "failed": 0, "pending": 0}
        for task in recent_tasks:
            if task.status == "COMPLETED":
                task_summary["completed"] += 1
            elif task.status == "FAILED":
                task_summary["failed"] += 1
            else:
                task_summary["pending"] += 1

        # 2. Fetch recent audit history for this agent
        audit_history_response = requests.get(
            f"{AUDIT_TRAIL_URL}/audit/history/AGENT/{agent_id}",
            headers={"X-API-Key": AUDIT_API_KEY}
        )
        audit_history_response.raise_for_status()
        audit_history = audit_history_response.json().get("history", [])
        recent_audit_events = [e for e in audit_history if datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > (datetime.utcnow() - timedelta(days=7))]

        # 3. Simulate analysis of past performance and outcomes against ethical guidelines/research goals
        # Perform data analysis using LLM (no sleep needed)
        analysis_outcome = f"Agent {agent_id} reviewed {task_summary['total_tasks']} tasks in the last 7 days. " \
                           f"Completed: {task_summary['completed']}, Failed: {task_summary['failed']}. " \
                           "Identified potential for improved data source selection and more robust error handling."
        
        # 4. Simulate proposing adjustments
        proposed_adjustments = [
            "Prioritize data sources with higher data quality scores.",
            "Implement retry mechanisms for AD Workbench API calls.",
            "Refine task descriptions for clarity and specificity."
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
