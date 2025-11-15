import os
import requests
import json
import time
import logging
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas
from datetime import datetime, timedelta
from typing import List, Dict, Any

ADWORKBENCH_PROXY_URL = os.getenv("ADWORKBENCH_PROXY_URL")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")
AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
AGENT_SERVICE_BASE_URL = os.getenv("AGENT_SERVICE_BASE_URL") # e.g., http://alznexus_biomarker_hunter_agent:8000
AGENT_API_KEY = os.getenv("AGENT_API_KEY")
AGENT_REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL")
AGENT_REGISTRY_API_KEY = os.getenv("AGENT_REGISTRY_API_KEY")

if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")
if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not AGENT_SERVICE_BASE_URL or not AGENT_API_KEY:
    raise ValueError("AGENT_SERVICE_BASE_URL or AGENT_API_KEY environment variables not set.")
if not AGENT_REGISTRY_URL or not AGENT_REGISTRY_API_KEY:
    raise ValueError("AGENT_REGISTRY_URL or AGENT_REGISTRY_API_KEY environment variables not set.")

logger = logging.getLogger(__name__)

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
    db: Session = SessionLocal()
    overall_status = "COMPLETED"
    results = []
    try:
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "IN_PROGRESS")
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_STARTED",
            description="Master Orchestrator started coordinating tasks for multiple sub-agents.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "num_sub_tasks": len(sub_agent_tasks_data)}
        )

        for i, sub_task_data in enumerate(sub_agent_tasks_data):
            sub_agent_id = sub_task_data["agent_id"]
            task_description = sub_task_data["task_description"]
            task_metadata = sub_task_data.get("task_metadata", {})
            
            log_audit_event(
                entity_type="ORCHESTRATOR",
                entity_id=str(orchestrator_task_id),
                event_type="DISPATCHING_SUB_AGENT_TASK",
                description=f"Dispatching task to sub-agent {sub_agent_id}: {task_description}",
                metadata={"sub_agent_id": sub_agent_id, "task_description": task_description, "orchestrator_task_id": orchestrator_task_id}
            )

            try:
                # Call the sub-agent's execute-task endpoint
                agent_task_payload = schemas.AgentTaskCreate(
                    agent_id=sub_agent_id,
                    orchestrator_task_id=orchestrator_task_id,
                    task_description=task_description,
                    metadata_json=task_metadata
                ).model_dump_json()

                headers = {"X-API-Key": AGENT_API_KEY, "Content-Type": "application/json"}
                # Assuming a generic AGENT_SERVICE_BASE_URL and agent_id is part of the path
                response = requests.post(f"{AGENT_SERVICE_BASE_URL}/agent/{sub_agent_id}/execute-task", 
                                         headers=headers, data=agent_task_payload)
                response.raise_for_status()
                agent_response = response.json()
                results.append({"agent_id": sub_agent_id, "status": "SUCCESS", "response": agent_response})
                log_audit_event(
                    entity_type="ORCHESTRATOR",
                    entity_id=str(orchestrator_task_id),
                    event_type="SUB_AGENT_TASK_COMPLETED",
                    description=f"Sub-agent {sub_agent_id} completed its task.",
                    metadata={"sub_agent_id": sub_agent_id, "agent_response": agent_response}
                )
            except requests.exceptions.RequestException as e:
                error_message = f"Sub-agent {sub_agent_id} task failed: {e}"
                results.append({"agent_id": sub_agent_id, "status": "FAILED", "error": error_message})
                overall_status = "FAILED"
                log_audit_event(
                    entity_type="ORCHESTRATOR",
                    entity_id=str(orchestrator_task_id),
                    event_type="SUB_AGENT_TASK_FAILED",
                    description=error_message,
                    metadata={"sub_agent_id": sub_agent_id, "error": error_message}
                )
            except Exception as e:
                error_message = f"An unexpected error occurred during sub-agent {sub_agent_id} task: {e}"
                results.append({"agent_id": sub_agent_id, "status": "FAILED", "error": error_message})
                overall_status = "FAILED"
                log_audit_event(
                    entity_type="ORCHESTRATOR",
                    entity_id=str(orchestrator_task_id),
                    event_type="SUB_AGENT_TASK_FAILED",
                    description=error_message,
                    metadata={"sub_agent_id": sub_agent_id, "error": error_message}
                )
        
        crud.update_orchestrator_task_status(db, orchestrator_task_id, overall_status, {"sub_task_results": results})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_FINISHED",
            description=f"Master Orchestrator finished coordinating sub-agent tasks with overall status: {overall_status}.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "overall_status": overall_status, "results_summary": results}
        )
        return {"orchestrator_task_id": orchestrator_task_id, "overall_status": overall_status, "results": results}
    except Exception as e:
        error_message = f"Overall coordination task failed: {e}"
        crud.update_orchestrator_task_status(db, orchestrator_task_id, "FAILED", {"error": error_message})
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="SUB_AGENT_COORDINATION_FAILED",
            description=error_message,
            metadata={"orchestrator_task_id": orchestrator_task_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
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

        # CQ-SPRINT12-008: Placeholder for actual complex reasoning/LLM interaction
        # TODO: Implement the actual logic for evaluating conflicting agent perspectives and evidence,
        # potentially leveraging LLMs for sophisticated debate resolution and decision-making.
        # For now, a simulated outcome is generated.
        # time.sleep(7)

        mock_resolution = {
            "decision": "Adopt Agent A's proposal with modifications from Agent B's evidence.",
            "reasoning": "After evaluating conflicting findings on biomarker X, Agent A's methodology was deemed more robust for early-stage detection, but Agent B provided critical evidence regarding confounding factors that need to be integrated.",
            "resolved_points": debate_details["points_of_contention"],
            "winning_agent_perspective": debate_details["conflicting_agents"][0] # Mocking a winner
        }

        crud.update_orchestrator_task_status(db, orchestrator_task_id, "COMPLETED", mock_resolution)
        log_audit_event(
            entity_type="ORCHESTRATOR",
            entity_id=str(orchestrator_task_id),
            event_type="DEBATE_RESOLUTION_COMPLETED",
            description="Multi-agent debate resolved successfully.",
            metadata={"orchestrator_task_id": orchestrator_task_id, "resolution": mock_resolution}
        )
        return {"orchestrator_task_id": orchestrator_task_id, "status": "COMPLETED", "resolution": mock_resolution}
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

        # CQ-SPRINT12-009: Placeholder for actual data gathering/querying
        # TODO: Implement the actual data retrieval mechanisms to gather comprehensive performance metrics,
        # audit logs, and other relevant data for the self-correction analysis.
        # For now, a simulated outcome is generated.
        # time.sleep(3)
        overall_task_summary = {"total_orchestrator_tasks": 100, "successful": 85, "failed": 15}

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

        # CQ-SPRINT12-010: Placeholder for actual complex analysis/LLM interaction
        # TODO: Implement the actual analytical and LLM-driven reasoning processes to derive insights
        # from performance data and agent reflections, leading to concrete adaptation decisions and new goal proposals.
        # For now, a simulated outcome is generated.
        # time.sleep(7)
        analysis_report = (
            f"Overall orchestrator task success rate: {overall_task_summary['successful']}/{overall_task_summary['total_orchestrator_tasks']}. "
            f"Recent agent reflections ({reflection_summary['num_reflections']}) highlight common themes such as "
            "'need for better data source validation' and 'improved error handling'."
        )

        adaptation_decision = {
            "type": "GOAL_MODIFICATION",
            "description": "Adjusting future research goals to prioritize data quality and robust error handling based on self-correction analysis.",
            "new_goal_proposal": "Establish a data quality assurance sub-goal for all new data ingestion pipelines.",
            "impact": "Expected to reduce task failure rate by 5% over next quarter."
        }

        new_goal_text = adaptation_decision["new_goal_proposal"]
        new_goal_create = schemas.ResearchGoalCreate(goal_text=new_goal_text)
        db_new_goal = crud.create_research_goal(db, new_goal_create)

        self_correction_result = {
            "overall_task_performance": overall_task_summary,
            "agent_reflection_summary": reflection_summary,
            "analysis_report": analysis_report,
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
