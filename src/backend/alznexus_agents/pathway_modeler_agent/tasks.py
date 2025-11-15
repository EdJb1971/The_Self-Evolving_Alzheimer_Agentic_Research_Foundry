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
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

MAX_RESULT_DATA_SIZE_BYTES = 1 * 1024 * 1024

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
if not ADWORKBENCH_PROXY_URL or not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_PROXY_URL or ADWORKBENCH_API_KEY environment variables not set.")
if not LLM_SERVICE_URL or not LLM_API_KEY:
    raise ValueError("LLM_SERVICE_URL or LLM_API_KEY environment variables not set.")

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

@celery_app.task(bind=True, name="model_pathway_task")
def model_pathway_task(self, agent_task_id: int):
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
            event_type="PATHWAY_MODELING_STARTED",
            description=f"Agent {agent_id} started pathway modeling task {agent_task_id}: {db_agent_task.task_description}",
            metadata=db_agent_task.model_dump()
        )

        # STORY-305: Simulate pathway modeling by querying AD Workbench Proxy and publishing insights
        adworkbench_headers = {"X-API-Key": ADWORKBENCH_API_KEY, "Content-Type": "application/json"}

        query_text = f"Retrieve biological network data for pathway modeling related to: {db_agent_task.task_description}"
        adworkbench_query_payload = {"query_text": query_text}

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_INITIATED",
            description=f"Agent {agent_id} querying AD Workbench for pathway data: {query_text}",
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

        # CQ-SPRINT12-004: Replaced time.sleep with actual asynchronous polling mechanism
        final_query_status = poll_adworkbench_query_status(adworkbench_query_id, adworkbench_headers, ADWORKBENCH_PROXY_URL)

        result_data_str = final_query_status["result_data"]
        if len(result_data_str.encode('utf-8')) > MAX_RESULT_DATA_SIZE_BYTES:
            raise ValueError(f"AD Workbench query result_data size exceeds {MAX_RESULT_DATA_SIZE_BYTES} bytes.")
        raw_data_summary = json.loads(result_data_str)
        
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="ADWORKBENCH_QUERY_COMPLETED",
            description=f"Agent {agent_id} received pathway data from AD Workbench query {adworkbench_query_id}.",
            metadata={"adworkbench_query_id": adworkbench_query_id, "data_summary": raw_data_summary.get("data", [])[:1]}
        )

        # PM-001: Implement comprehensive disease pathway modeling with systems biology and mathematical modeling
        # Step 1: Analyze biological networks and construct pathway models
        pathway_modeling_prompt = f"""Construct a comprehensive disease progression model for Alzheimer's disease based on the available biological data:

Biological Data: {json.dumps(raw_data_summary)}
Modeling Task: {db_agent_task.task_description}

Perform comprehensive pathway modeling including:

1. **Network Analysis**:
   - Identify key molecular pathways and interactions
   - Map protein-protein interactions and signaling cascades
   - Analyze gene regulatory networks and epigenetic modifications

2. **Mathematical Modeling**:
   - Construct differential equation models for disease progression
   - Implement stochastic models for biomarker dynamics
   - Develop pharmacokinetic/pharmacodynamic models

3. **Systems Biology Integration**:
   - Integrate multi-omics data (genomics, proteomics, metabolomics)
   - Model cell-cell interactions and tissue-level effects
   - Incorporate environmental and lifestyle factors

4. **Disease Progression Simulation**:
   - Model temporal progression from preclinical to severe stages
   - Identify critical transition points and tipping mechanisms
   - Simulate biomarker trajectories over time

5. **Intervention Modeling**:
   - Model therapeutic intervention points and mechanisms
   - Simulate drug effects on pathway dynamics
   - Predict optimal intervention timing and combinations

6. **Uncertainty Quantification**:
   - Implement probabilistic models with confidence intervals
   - Assess model sensitivity to parameter variations
   - Provide uncertainty bounds for predictions

Format your response as a JSON object with this structure:
{{
    "pathway_model": {{
        "model_name": "AD_Pathway_Model_2025",
        "model_type": "Systems_Biology_Network",
        "version": "1.0",
        "temporal_scope": "Preclinical_to_Severe",
        "key_pathways": [
            {{
                "pathway_name": "Amyloid Beta Cascade",
                "components": ["APP", "BACE1", "PSEN1", "Aβ42", "Aβ40"],
                "interactions": ["APP→BACE1→Aβ", "PSEN1→γ-secretase→Aβ"],
                "critical_nodes": ["BACE1", "γ-secretase"],
                "biomarkers": ["CSF_Aβ42", "Plasma_Aβ"]
            }}
        ],
        "mathematical_model": {{
            "equations": [
                "d[Aβ]/dt = k_syn - k_deg*[Aβ] - k_clear*[Aβ]",
                "d[Tau_P]/dt = k_phos*[Tau] - k_dephos*[Tau_P]"
            ],
            "parameters": {{
                "k_syn": 0.1,
                "k_deg": 0.05,
                "k_clear": 0.02
            }},
            "initial_conditions": {{
                "Aβ": 500,
                "Tau_P": 0.1
            }}
        }},
        "progression_stages": [
            {{
                "stage": "Preclinical",
                "duration_years": 10,
                "biomarker_changes": ["Aβ↑", "Tau↑"],
                "cognitive_impairment": "None",
                "intervention_window": "Optimal"
            }}
        ],
        "intervention_points": [
            {{
                "target": "BACE1_Inhibition",
                "stage": "Early_Preclinical",
                "mechanism": "Reduce_Aβ_Production",
                "predicted_effect": "Delay_Onset_5_years",
                "confidence": 0.85,
                "biomarker_response": ["CSF_Aβ↓40%", "Plasma_Aβ↓30%"]
            }}
        ],
        "simulation_results": {{
            "baseline_trajectory": {{
                "time_points": [0, 5, 10, 15, 20],
                "mmse_scores": [30, 28, 25, 20, 15],
                "abeta_levels": [500, 600, 750, 900, 1100]
            }},
            "intervention_scenarios": [
                {{
                    "scenario": "Early_BACE_Inhibition",
                    "intervention_time": 5,
                    "outcome": "Delayed_Onset_3_years",
                    "biomarker_trajectory": [500, 520, 480, 450, 420]
                }}
            ]
        }},
        "uncertainty_analysis": {{
            "parameter_sensitivity": {{
                "most_influential": ["k_syn_Aβ", "clearance_rate"],
                "confidence_intervals": "±25%_for_predictions"
            }},
            "model_validation": {{
                "goodness_of_fit": 0.89,
                "predictive_accuracy": 0.82,
                "cross_validation_score": 0.78
            }}
        }},
        "clinical_implications": {{
            "diagnostic_markers": ["CSF_Aβ42/Aβ40_ratio", "Plasma_p-tau181"],
            "therapeutic_targets": ["BACE1", "Tau_phosphorylation", "Neuroinflammation"],
            "trial_endpoints": ["Change_from_baseline_CDR-SB", "Time_to_MCI_conversion"],
            "personalized_medicine": {{
                "biomarker_guided": true,
                "genetic_stratification": ["APOE4_status", "TREM2_variants"],
                "response_prediction": "70%_accuracy"
            }}
        }}
    }},
    "model_validation": {{
        "data_fit_quality": "Excellent",
        "predictive_performance": "Good",
        "biological_plausibility": "High",
        "clinical_relevance": "Strong"
    }},
    "recommendations": {{
        "research_priorities": ["Validate_key_interactions", "Longitudinal_biomarker_studies"],
        "clinical_applications": ["Early_diagnostic_algorithm", "Targeted_therapy_selection"],
        "future_directions": ["Multi-scale_modeling", "Real-time_monitoring_integration"]
    }}
}}"""

        llm_modeling_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/structured-output",
            headers={"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"},
            json={
                "model_name": "gemini-1.5-flash",
                "prompt": pathway_modeling_prompt,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "pathway_model": {
                                "type": "object",
                                "properties": {
                                    "model_name": {"type": "string"},
                                    "model_type": {"type": "string"},
                                    "version": {"type": "string"},
                                    "temporal_scope": {"type": "string"},
                                    "key_pathways": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "pathway_name": {"type": "string"},
                                                "components": {"type": "array", "items": {"type": "string"}},
                                                "interactions": {"type": "array", "items": {"type": "string"}},
                                                "critical_nodes": {"type": "array", "items": {"type": "string"}},
                                                "biomarkers": {"type": "array", "items": {"type": "string"}}
                                            }
                                        }
                                    },
                                    "mathematical_model": {
                                        "type": "object",
                                        "properties": {
                                            "equations": {"type": "array", "items": {"type": "string"}},
                                            "parameters": {"type": "object"},
                                            "initial_conditions": {"type": "object"}
                                        }
                                    },
                                    "progression_stages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "stage": {"type": "string"},
                                                "duration_years": {"type": "number"},
                                                "biomarker_changes": {"type": "array", "items": {"type": "string"}},
                                                "cognitive_impairment": {"type": "string"},
                                                "intervention_window": {"type": "string"}
                                            }
                                        }
                                    },
                                    "intervention_points": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "target": {"type": "string"},
                                                "stage": {"type": "string"},
                                                "mechanism": {"type": "string"},
                                                "predicted_effect": {"type": "string"},
                                                "confidence": {"type": "number"},
                                                "biomarker_response": {"type": "array", "items": {"type": "string"}}
                                            }
                                        }
                                    },
                                    "simulation_results": {"type": "object"},
                                    "uncertainty_analysis": {"type": "object"},
                                    "clinical_implications": {"type": "object"}
                                }
                            },
                            "model_validation": {"type": "object"},
                            "recommendations": {"type": "object"}
                        },
                        "required": ["pathway_model", "model_validation", "recommendations"]
                    }
                },
                "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
            }
        )
        llm_modeling_response.raise_for_status()
        modeling_results = llm_modeling_response.json()["structured_output"]

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PATHWAY_MODELING_COMPLETED",
            description=f"Agent {agent_id} completed comprehensive pathway modeling with mathematical simulation.",
            metadata={"model_name": modeling_results.get('pathway_model', {}).get('model_name')}
        )
        
        # Step 2: Create comprehensive pathway modeling report
        disease_model = {
            "model_name": modeling_results.get('pathway_model', {}).get('model_name', f"Disease_Progression_Model_for_{db_agent_task.task_description.replace(' ', '_')}"),
            "model_type": modeling_results.get('pathway_model', {}).get('model_type', 'Systems_Biology_Network'),
            "version": modeling_results.get('pathway_model', {}).get('version', '1.0'),
            "temporal_scope": modeling_results.get('pathway_model', {}).get('temporal_scope', 'Preclinical_to_Severe'),
            "key_pathways": modeling_results.get('pathway_model', {}).get('key_pathways', []),
            "mathematical_model": modeling_results.get('pathway_model', {}).get('mathematical_model', {}),
            "progression_stages": modeling_results.get('pathway_model', {}).get('progression_stages', []),
            "intervention_points": modeling_results.get('pathway_model', {}).get('intervention_points', []),
            "simulation_results": modeling_results.get('pathway_model', {}).get('simulation_results', {}),
            "uncertainty_analysis": modeling_results.get('pathway_model', {}).get('uncertainty_analysis', {}),
            "clinical_implications": modeling_results.get('pathway_model', {}).get('clinical_implications', {}),
            "model_validation": modeling_results.get('model_validation', {}),
            "recommendations": modeling_results.get('recommendations', {}),
            "modeling_summary": {
                "pathways_identified": len(modeling_results.get('pathway_model', {}).get('key_pathways', [])),
                "intervention_points_found": len(modeling_results.get('pathway_model', {}).get('intervention_points', [])),
                "mathematical_equations": len(modeling_results.get('pathway_model', {}).get('mathematical_model', {}).get('equations', [])),
                "simulation_scenarios": len(modeling_results.get('pathway_model', {}).get('simulation_results', {}).get('intervention_scenarios', [])),
                "validation_score": modeling_results.get('model_validation', {}).get('data_fit_quality', 'Unknown')
            }
        }

        insight_name_val = f"Disease Progression Model: {disease_model['model_name']}"
        insight_publish_request_obj = schemas.InsightPublishRequest(
            insight_name=insight_name_val,
            insight_description=f"Comprehensive systems biology disease progression model with mathematical simulation, intervention analysis, and clinical implications for: {db_agent_task.task_description}.",
            data_source_ids=[f"adworkbench_query_{adworkbench_query_id}"],
            payload=disease_model,
            tags=["pathway_modeling", "disease_progression", "systems_biology", "mathematical_modeling", "intervention_analysis", agent_id]
        )
        insight_payload = insight_publish_request_obj.model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_PATHWAY_MODEL_INSIGHT",
            description=f"Agent {agent_id} publishing comprehensive disease pathway model insight.",
            metadata={"insight_name": insight_name_val}
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
            "agent_output": f"Comprehensive pathway modeling completed for task {agent_task_id}.",
            "modeling_summary": {
                "model_name": disease_model['model_name'],
                "model_type": disease_model['model_type'],
                "pathways_identified": disease_model['modeling_summary']['pathways_identified'],
                "intervention_points": disease_model['modeling_summary']['intervention_points_found'],
                "validation_score": disease_model['modeling_summary']['validation_score'],
                "clinical_targets": len(disease_model.get('clinical_implications', {}).get('therapeutic_targets', []))
            },
            "disease_model": disease_model,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", result)
        crud.update_agent_state(db, agent_id, current_task_id=None)
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PATHWAY_MODELING_COMPLETED",
            description=f"Agent {agent_id} completed pathway modeling task {agent_task_id}.",
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
            event_type="PATHWAY_MODELING_FAILED",
            description=f"Agent {agent_id} failed to model pathway for task {agent_task_id}: {error_message}",
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
            event_type="PATHWAY_MODELING_FAILED",
            description=f"Agent {agent_id} failed to model pathway for task {agent_task_id}: {error_message}",
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

        # CQ-SPRINT12-006: Placeholder for actual analysis/LLM interaction
        # TODO: Implement the actual logic for analyzing recent tasks and audit events, potentially using LLMs
        # for generating insights and proposed adjustments during self-reflection.
        # For now, a simulated outcome is generated.
        # time.sleep(5)
        analysis_outcome = f"Agent {agent_id} reviewed {task_summary['total_tasks']} tasks in the last 7 days. " \
                           f"Completed: {task_summary['completed']}, Failed: {task_summary['failed']}. " \
                           "Identified potential for improved data source selection and more robust error handling in pathway modeling."
        
        proposed_adjustments = [
            "Refine biological network data parsing for edge cases.",
            "Explore alternative simulation algorithms for disease progression.",
            "Improve validation metrics for identified intervention points."
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
