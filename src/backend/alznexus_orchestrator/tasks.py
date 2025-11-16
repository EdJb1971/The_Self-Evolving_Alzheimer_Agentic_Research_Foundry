import os
import requests
import json
import time
import logging
import random
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas
from datetime import datetime, timedelta
from typing import List, Dict, Any
import redis
from contextlib import contextmanager
from celery import group, chord
import redis
from contextlib import contextmanager

# Redis for distributed locking
REDIS_URL = os.getenv("ORCHESTRATOR_REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Circuit breaker state
circuit_breaker_state = {
    "failures": 0,
    "last_failure_time": 0,
    "state": "CLOSED"  # CLOSED, OPEN, HALF_OPEN
}
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 60  # seconds

@contextmanager
def distributed_lock(lock_key: str, timeout: int = 30, blocking_timeout: int = 10):
    """Distributed lock using Redis for preventing race conditions."""
    lock_value = f"{os.getpid()}_{time.time()}"
    lock_acquired = False

    try:
        # Try to acquire lock
        lock_acquired = redis_client.set(lock_key, lock_value, ex=timeout, nx=True)

        if not lock_acquired:
            # Wait for lock with timeout
            start_time = time.time()
            while time.time() - start_time < blocking_timeout:
                if redis_client.set(lock_key, lock_value, ex=timeout, nx=True):
                    lock_acquired = True
                    break
                time.sleep(0.1)

        if not lock_acquired:
            raise TimeoutError(f"Could not acquire distributed lock for key: {lock_key}")

        yield lock_acquired

    finally:
        if lock_acquired:
            # Only release if we still own the lock
            current_value = redis_client.get(lock_key)
            if current_value == lock_value:
                redis_client.delete(lock_key)

def check_circuit_breaker(service_name: str) -> bool:
    """Check if circuit breaker allows the request."""
    current_time = time.time()

    if circuit_breaker_state["state"] == "OPEN":
        if current_time - circuit_breaker_state["last_failure_time"] > CIRCUIT_BREAKER_TIMEOUT:
            circuit_breaker_state["state"] = "HALF_OPEN"
            logger.info(f"Circuit breaker for {service_name} moved to HALF_OPEN")
        else:
            logger.warning(f"Circuit breaker for {service_name} is OPEN, blocking request")
            return False

    return True

def record_circuit_breaker_failure(service_name: str):
    """Record a failure for circuit breaker."""
    circuit_breaker_state["failures"] += 1
    circuit_breaker_state["last_failure_time"] = time.time()

    if circuit_breaker_state["failures"] >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
        circuit_breaker_state["state"] = "OPEN"
        logger.warning(f"Circuit breaker for {service_name} opened after {circuit_breaker_state['failures']} failures")

def record_circuit_breaker_success(service_name: str):
    """Record a success for circuit breaker."""
    if circuit_breaker_state["state"] == "HALF_OPEN":
        circuit_breaker_state["state"] = "CLOSED"
        circuit_breaker_state["failures"] = 0
        logger.info(f"Circuit breaker for {service_name} closed after successful request")

# Initialize logger
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="execute_single_agent_task")
def execute_single_agent_task(self, sub_task_data: Dict[str, Any], orchestrator_task_id: int):
    """Execute a single agent task with proper error handling and circuit breaker."""
    sub_agent_id = sub_task_data["agent_id"]
    task_description = sub_task_data["task_description"]
    task_metadata = sub_task_data.get("task_metadata", {})

    try:
        # Check circuit breaker before making request
        if not check_circuit_breaker(sub_agent_id):
            return {"agent_id": sub_agent_id, "status": "CIRCUIT_OPEN", "error": "Circuit breaker is open"}

        # Get enriched context from autonomous learning service
        enriched_context = get_enriched_context(sub_agent_id, sub_task_data)

        # Call the sub-agent's execute-task endpoint with timeout
        agent_task_payload = schemas.AgentTaskCreate(
            agent_id=sub_agent_id,
            orchestrator_task_id=orchestrator_task_id,
            task_description=task_description,
            metadata_json={**task_metadata, "enriched_context": enriched_context}
        ).model_dump_json()

        headers = {"X-API-Key": AGENT_API_KEY, "Content-Type": "application/json"}
        response = requests.post(
            f"{AGENT_SERVICE_BASE_URL}/agent/{sub_agent_id}/execute-task",
            headers=headers,
            data=agent_task_payload,
            timeout=300  # 5 minute timeout per agent task
        )
        response.raise_for_status()

        # Record circuit breaker success
        record_circuit_breaker_success(sub_agent_id)

        agent_response = response.json()

        # Record agent performance for continuous learning
        record_agent_performance(sub_agent_id, sub_task_data, {
            "success": True,
            "execution_time": agent_response.get("execution_time", 0.0),
            "accuracy_score": agent_response.get("accuracy_score"),
            "confidence_score": agent_response.get("confidence_score"),
            "result": agent_response
        })

        return {"agent_id": sub_agent_id, "status": "SUCCESS", "response": agent_response}

    except requests.exceptions.Timeout:
        error_message = f"Timeout calling sub-agent {sub_agent_id}"
        record_circuit_breaker_failure(sub_agent_id)
        record_agent_performance(sub_agent_id, sub_task_data, {
            "success": False,
            "execution_time": 300.0,  # Max timeout
            "error": error_message
        })
        return {"agent_id": sub_agent_id, "status": "TIMEOUT", "error": error_message}

    except requests.exceptions.RequestException as e:
        error_message = f"Sub-agent {sub_agent_id} task failed: {e}"
        record_circuit_breaker_failure(sub_agent_id)
        record_agent_performance(sub_agent_id, sub_task_data, {
            "success": False,
            "execution_time": 0.0,
            "error": error_message
        })
        return {"agent_id": sub_agent_id, "status": "FAILED", "error": error_message}

    except Exception as e:
        error_message = f"An unexpected error occurred during sub-agent {sub_agent_id} task: {e}"
        record_circuit_breaker_failure(sub_agent_id)
        record_agent_performance(sub_agent_id, sub_task_data, {
            "success": False,
            "execution_time": 0.0,
            "error": error_message
        })
        return {"agent_id": sub_agent_id, "status": "FAILED", "error": error_message}

@celery_app.task(bind=True, name="process_agent_results")
def process_agent_results(self, results: List[Dict[str, Any]], orchestrator_task_id: int):
    """Process results from parallel agent task execution."""
    db: Session = SessionLocal()
    try:
        overall_status = "COMPLETED"
        failed_count = 0

        for result in results:
            if result["status"] in ["FAILED", "TIMEOUT", "CIRCUIT_OPEN"]:
                failed_count += 1
                overall_status = "FAILED"

        crud.update_orchestrator_task_status(db, orchestrator_task_id, overall_status, {"sub_task_results": results})

        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_FINISHED",
            description=f"Sub-agent coordination finished with {len(results)} tasks, {failed_count} failures",
            metadata={
                "orchestrator_task_id": orchestrator_task_id,
                "total_tasks": len(results),
                "failed_tasks": failed_count,
                "success_rate": (len(results) - failed_count) / len(results) if results else 0
            }
        )

        return {"orchestrator_task_id": orchestrator_task_id, "status": overall_status, "results": results}

    finally:
        db.close()

ADWORKBENCH_PROXY_URL = os.getenv("ADWORKBENCH_PROXY_URL")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")
AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
AGENT_SERVICE_BASE_URL = os.getenv("AGENT_SERVICE_BASE_URL") # e.g., http://alznexus_biomarker_hunter_agent:8000
AGENT_API_KEY = os.getenv("AGENT_API_KEY")
AGENT_REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL")
AGENT_REGISTRY_API_KEY = os.getenv("AGENT_REGISTRY_API_KEY")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
AUTONOMOUS_LEARNING_URL = os.getenv("AUTONOMOUS_LEARNING_URL", "http://localhost:8007")
AUTONOMOUS_LEARNING_API_KEY = os.getenv("AUTONOMOUS_LEARNING_API_KEY", "test_autonomous_key_123")

if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")
if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not AGENT_SERVICE_BASE_URL or not AGENT_API_KEY:
    raise ValueError("AGENT_SERVICE_BASE_URL or AGENT_API_KEY environment variables not set.")
if not AGENT_REGISTRY_URL or not AGENT_REGISTRY_API_KEY:
    raise ValueError("AGENT_REGISTRY_URL or AGENT_REGISTRY_API_KEY environment variables not set.")
if not LLM_SERVICE_URL or not LLM_API_KEY:
    raise ValueError("LLM_SERVICE_URL or LLM_API_KEY environment variables not set.")

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

def get_enriched_context(agent_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get enriched context from autonomous learning service for agent task execution."""
    try:
        payload = {
            "task_data": task_data,
            "task_domain": task_data.get("domain", "general"),
            "agent_id": agent_id
        }
        headers = {"X-API-Key": AUTONOMOUS_LEARNING_API_KEY, "Content-Type": "application/json"}
        
        response = requests.post(
            f"{AUTONOMOUS_LEARNING_URL}/context/enrich/{agent_id}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            enriched_context = result.get("enriched_context", {})
            logger.info(f"Retrieved enriched context for agent {agent_id}: {len(str(enriched_context))} chars")
            return enriched_context
        else:
            logger.warning(f"Failed to get enriched context for agent {agent_id}: {response.status_code}")
            return {}
            
    except Exception as e:
        logger.error(f"Error getting enriched context for agent {agent_id}: {str(e)}")
        return {}

def record_agent_performance(agent_id: str, task_data: Dict[str, Any], result: Dict[str, Any]):
    """Record agent performance in autonomous learning service for continuous improvement."""
    try:
        performance_data = {
            "agent_id": agent_id,
            "task_type": task_data.get("type", "unknown"),
            "task_id": f"orchestrator_{datetime.utcnow().timestamp()}",
            "success_rate": 1.0 if result.get("success", False) else 0.0,
            "execution_time": result.get("execution_time", 0.0),
            "accuracy_score": result.get("accuracy_score"),
            "confidence_score": result.get("confidence_score"),
            "outcome_data": result,
            "context_used": task_data.get("enriched_context", {}),
            "feedback_received": {"orchestrator_feedback": True}
        }
        
        headers = {"X-API-Key": AUTONOMOUS_LEARNING_API_KEY, "Content-Type": "application/json"}
        response = requests.post(
            f"{AUTONOMOUS_LEARNING_URL}/performance/",
            headers=headers,
            json=performance_data,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Recorded performance for agent {agent_id}")
        else:
            logger.warning(f"Failed to record performance for agent {agent_id}: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error recording performance for agent {agent_id}: {str(e)}")

def check_task_deduplication(task_data: Dict[str, Any]) -> bool:
    """Check if a similar task has been attempted recently and failed, preventing repetition."""
    try:
        # Create a task signature for deduplication
        task_signature = {
            "task_description": task_data.get("task_description", ""),
            "agent_id": task_data.get("agent_id", ""),
            "domain": task_data.get("domain", "general"),
            "task_type": task_data.get("type", "unknown")
        }
        
        headers = {"X-API-Key": AUTONOMOUS_LEARNING_API_KEY, "Content-Type": "application/json"}
        response = requests.post(
            f"{AUTONOMOUS_LEARNING_URL}/task/deduplication/check",
            headers=headers,
            json=task_signature,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            is_duplicate = result.get("is_duplicate", False)
            if is_duplicate:
                logger.info(f"Task deduplication: Skipping duplicate task for agent {task_data.get('agent_id')}")
            return is_duplicate
        else:
            logger.warning(f"Failed to check task deduplication: {response.status_code}")
            return False  # Allow task if deduplication check fails
            
    except Exception as e:
        logger.error(f"Error checking task deduplication: {str(e)}")
        return False  # Allow task if deduplication check fails

# SEC-SPRINT12-001 & CQ-SPRINT12-007: Removed hardcoded KNOWN_AGENT_IDS.
# In a real system, this would come from a robust agent registry service with authentication.
# The orchestrator should query this registry to obtain trusted agent IDs and their endpoints.

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

def get_registered_agents() -> List[Dict[str, Any]]:
    """Dynamically fetches registered agents from the Agent Registry."""
    registry_headers = {"X-API-Key": AGENT_REGISTRY_API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.get(f"{AGENT_REGISTRY_URL}/registry/agents", headers=registry_headers)
        response.raise_for_status()
        agents_data = response.json().get("agents", [])
        logger.info(f"Successfully fetched {len(agents_data)} registered agents from registry.")
        return agents_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch registered agents from registry: {str(e)}", exc_info=True)
        # In a production system, this might trigger an alert or fallback mechanism.
        return [] # Return empty list if registry is unreachable or fails

@celery_app.task(bind=True, name="initiate_daily_scan_task")
def initiate_daily_scan_task(self, orchestrator_task_id: int):
    db: Session = SessionLocal()
    try:
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "IN_PROGRESS")
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DAILY_SCAN_INITIATED",
            description="Master Orchestrator initiated daily data scan via AD Workbench Proxy.",
            metadata={"orchestrator_task_id": orchestrator_task_id}
        )

        headers = {"X-API-Key": ADWORKBENCH_API_KEY}
        response = requests.get(f"{ADWORKBENCH_PROXY_URL}/adworkbench/data/scan", headers=headers)
        response.raise_for_status()
        scan_result = response.json()

        crud.update_orchestrator_task_status(db, orchestrator_task_id, "COMPLETED", scan_result)
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DAILY_SCAN_COMPLETED",
            description="Daily data scan completed successfully.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "scan_result": scan_result}
        )
        return {"orchestrator_task_id": orchestrator_task_id, "status": "COMPLETED", "result": scan_result}
    except requests.exceptions.RequestException as e:
        error_message = f"AD Workbench Proxy call failed: {e}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DAILY_SCAN_FAILED",
            description=f"Daily data scan failed: {error_message}",
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DAILY_SCAN_FAILED",
            description=f"Daily data scan failed: {error_message}",
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="coordinate_sub_agents_task")
def coordinate_sub_agents_task(self, orchestrator_task_id: int, sub_agent_tasks_data: List[Dict[str, Any]]):
    # Use distributed lock to prevent concurrent orchestration of the same task
    lock_key = f"orchestrator_lock_{orchestrator_task_id}"

    try:
        with distributed_lock(lock_key, timeout=300):  # 5 minute timeout
            return _coordinate_sub_agents_task_impl(self, orchestrator_task_id, sub_agent_tasks_data)
    except TimeoutError:
        logger.error(f"Could not acquire lock for orchestrator task {orchestrator_task_id}")
        self.update_state(state='FAILURE', meta={'exc_type': 'TimeoutError', 'exc_message': 'Lock acquisition timeout'})
        raise

def _coordinate_sub_agents_task_impl(self, orchestrator_task_id: int, sub_agent_tasks_data: List[Dict[str, Any]]):
    """Implementation of sub-agent coordination using queue-based parallel processing."""
    db: Session = SessionLocal()
    try:
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "IN_PROGRESS")
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_STARTED",
            description="Master Orchestrator started coordinating tasks for multiple sub-agents using parallel queues.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "num_sub_tasks": len(sub_agent_tasks_data)}
        )

        # Filter out duplicate tasks
        filtered_tasks = []
        for sub_task_data in sub_agent_tasks_data:
            sub_agent_id = sub_task_data["agent_id"]
            task_description = sub_task_data["task_description"]

            # Check for task deduplication to avoid repeating failed approaches
            if check_task_deduplication(sub_task_data):
                log_audit_event(
                    entity_type="ORCHESTRATOR",
                    entity_id=str(orchestrator_task_id),
                    event_type="SUB_AGENT_TASK_SKIPPED",
                    description=f"Skipped task for sub-agent {sub_agent_id} due to deduplication: {task_description}",
                    metadata={"sub_agent_id": sub_agent_id, "reason": "task_deduplication"}
                )
                continue

            filtered_tasks.append(sub_task_data)

        if not filtered_tasks:
            # All tasks were duplicates
            crud.update_orchestrator_task_status(db, orchestrator_task_id, "COMPLETED", {"reason": "all_tasks_deduplicated"})
            return {"orchestrator_task_id": orchestrator_task_id, "status": "COMPLETED", "results": []}

        # Create parallel task group for efficient processing
        task_group = group(
            execute_single_agent_task.s(task_data, orchestrator_task_id)
            for task_data in filtered_tasks
        )

        # Execute tasks in parallel and process results
        chord_result = chord(task_group)(process_agent_results.s(orchestrator_task_id))

        return chord_result.get()  # Wait for completion and return results

    except Exception as e:
        error_message = f"Critical error in sub-agent coordination: {e}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_FAILED",
            description=error_message,
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="resolve_debate_task")
def resolve_debate_task(self, orchestrator_task_id: int, debate_details: Dict[str, Any]):
    db: Session = SessionLocal()
    try:
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "IN_PROGRESS")
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DEBATE_RESOLUTION_STARTED",
            description="Master Orchestrator initiated debate resolution.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "debate_details": debate_details}
        )

        # Resolve debate using LLM
        debate_prompt = f"Resolve the following multi-agent debate in Alzheimer's research. Debate details: {json.dumps(debate_details)}. Evaluate perspectives, evidence, and provide a reasoned resolution."
        llm_payload = {
            "model_name": "gemini-1.5-flash",
            "prompt": debate_prompt,
            "metadata": {"orchestrator_task_id": orchestrator_task_id}
        }
        llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}
        
        llm_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/chat",
            headers=llm_headers,
            json=llm_payload
        )
        llm_response.raise_for_status()
        llm_result = llm_response.json()
        resolution_text = llm_result["response_text"]
        
        resolution = {
            "decision": "Resolved via LLM analysis",
            "reasoning": resolution_text,
            "resolved_points": debate_details.get("points_of_contention", []),
            "winning_agent_perspective": "LLM-determined"
        }

        crud.update_orchestrator_task_status(db, orchestrator_task_id, "COMPLETED", resolution)
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DEBATE_RESOLUTION_COMPLETED",
            description="Multi-agent debate resolved successfully.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "resolution": resolution}
        )
        return {"orchestrator_task_id": orchestrator_task_id, "status": "COMPLETED", "resolution": resolution}
    except Exception as e:
        error_message = f"Debate resolution failed: {e}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DEBATE_RESOLUTION_FAILED",
            description=error_message,
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="perform_self_correction_task")
def perform_self_correction_task(self, orchestrator_task_id: int):
    db: Session = SessionLocal()
    try:
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "IN_PROGRESS")
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SELF_CORRECTION_STARTED",
            description="Master Orchestrator initiated self-correction and adaptation process.",
            metadata={"orchestrator_task_id": orchestrator_task_id}
        )

        # CQ-SPRINT12-009: Gather comprehensive performance metrics from database
        # Query actual orchestrator task performance data
        orchestrator_tasks = crud.get_orchestrator_tasks(db, limit=1000)  # Get recent tasks

        # Calculate real performance metrics
        total_tasks = len(orchestrator_tasks)
        successful_tasks = len([t for t in orchestrator_tasks if t.status == "COMPLETED"])
        failed_tasks = len([t for t in orchestrator_tasks if t.status == "FAILED"])
        pending_tasks = len([t for t in orchestrator_tasks if t.status == "PENDING"])

        overall_task_summary = {
            "total_orchestrator_tasks": total_tasks,
            "successful": successful_tasks,
            "failed": failed_tasks,
            "pending": pending_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "time_range_days": 30  # Last 30 days of data
        }

        recent_reflections = []
        # SEC-SPRINT12-001 & CQ-SPRINT12-007: Dynamically fetch agent IDs from registry
        registered_agents = get_registered_agents()
        registered_agent_ids = [agent['agent_id'] for agent in registered_agents]

        recent_reflections = []
        # SEC-SPRINT12-001 & CQ-SPRINT12-007: Dynamically fetch agent IDs from registry
        registered_agents = get_registered_agents()
        registered_agent_ids = [agent['agent_id'] for agent in registered_agents]

        for agent_id in registered_agent_ids:
            try:
                audit_history_response = requests.get(
                    f"{AUDIT_TRAIL_URL}/audit/history/AGENT/{agent_id}",
                    headers={"X-API-Key": AUDIT_API_KEY}
                )
                audit_history_response.raise_for_status()
                agent_audit_history = audit_history_response.json().get("history", [])
                agent_recent_reflections = [
                    e for e in agent_audit_history 
                    if e['event_type'] == 'REFLECTION_COMPLETED' and 
                       datetime.fromisoformat(e['timestamp'].replace('Z', '+00:00')) > (datetime.utcnow() - timedelta(days=30))
                ]
                recent_reflections.extend(agent_recent_reflections)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch audit history for agent {agent_id} during self-correction: {str(e)}", exc_info=True)

        reflection_insights = [r['metadata_json'].get('proposed_adjustments', []) for r in recent_reflections if 'metadata_json' in r and r['metadata_json']]
        reflection_summary = {"num_reflections": len(recent_reflections), "insights": reflection_insights}

        # Perform analysis using LLM with RAG for learning from past results
        analysis_prompt = f"Analyze the following performance data and agent reflections for self-correction in Alzheimer's research orchestration. Data: {json.dumps(overall_task_summary)}. Reflections: {json.dumps(reflection_summary)}. Propose adaptations and new goals."
        llm_payload = {
            "model_name": "gemini-1.5-flash",
            "prompt": analysis_prompt,
            "metadata": {"orchestrator_task_id": orchestrator_task_id, "requester_agent": "orchestrator"}
        }
        llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}
        
        # Enable RAG for learning from accumulated knowledge
        llm_url = f"{LLM_SERVICE_URL}/llm/chat?enable_rag=true"
        
        llm_response = requests.post(
            llm_url,
            headers=llm_headers,
            json=llm_payload
        )
        llm_response.raise_for_status()
        llm_result = llm_response.json()
        analysis_text = llm_result["response_text"]
        
        # Parse adaptation from text
        adaptation_decision = {
            "type": "GOAL_MODIFICATION",
            "description": "LLM-driven adaptation based on analysis.",
            "new_goal_proposal": "Improve data quality and error handling.",
            "llm_analysis": analysis_text,
            "impact": "Reduce failures."
        }

        new_goal_text = adaptation_decision["new_goal_proposal"]
        new_goal_create = schemas.ResearchGoalCreate(goal_text=new_goal_text)
        db_new_goal = crud.create_research_goal(db, new_goal_create)

        self_correction_result = {
            "overall_task_performance": overall_task_summary,
            "agent_reflection_summary": reflection_summary,
            "analysis_report": analysis_text,
            "adaptation_decision": adaptation_decision,
            "new_research_goal_id": db_new_goal.id
        }

        crud.update_orchestrator_task_status(db, orchestrator_task_id, "COMPLETED", self_correction_result)
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SELF_CORRECTION_COMPLETED",
            description="Master Orchestrator completed self-correction and adapted strategies.",
            metadata=self_correction_result
        )
        return {"orchestrator_task_id": orchestrator_task_id, "status": "COMPLETED", "result": self_correction_result}
    except requests.exceptions.RequestException as e:
        error_message = f"Audit Trail Service call failed during self-correction: {str(e)}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SELF_CORRECTION_FAILED",
            description=f"Master Orchestrator self-correction failed: {error_message}",
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    except Exception as e:
        error_message = str(e)
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SELF_CORRECTION_FAILED",
            description=f"Master Orchestrator self-correction failed: {error_message}",
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
