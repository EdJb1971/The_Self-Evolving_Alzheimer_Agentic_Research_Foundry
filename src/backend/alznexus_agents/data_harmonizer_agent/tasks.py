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

        # DH-001: Implement comprehensive data harmonization with schema analysis, quality assessment, and ETL
        # Step 1: Analyze dataset schemas and metadata
        schema_analysis_prompt = f"""Analyze the following datasets and extract their schemas, data types, and relationships:

Datasets: {json.dumps(datasets_found)}

For each dataset, identify:
1. Column/field names and their data types
2. Primary keys and foreign keys
3. Data constraints and validation rules
4. Missing data patterns
5. Data quality issues

Format your response as a JSON object with this structure:
{{
    "datasets": [
        {{
            "dataset_id": "dataset_1",
            "schema": {{
                "columns": [
                    {{
                        "name": "column_name",
                        "data_type": "string|integer|float|date|boolean",
                        "nullable": true,
                        "constraints": ["unique", "not_null"],
                        "description": "field description"
                    }}
                ],
                "primary_key": ["column1"],
                "relationships": ["foreign_key_column -> referenced_table.column"]
            }},
            "data_quality": {{
                "completeness": 0.95,
                "consistency": 0.88,
                "accuracy": 0.92,
                "issues": ["missing values in column X", "inconsistent date formats"]
            }},
            "sample_data": ["sample1", "sample2"]
        }}
    ]
}}"""

        llm_schema_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/structured-output",
            headers=llm_headers,
            json={
                "model_name": "gemini-1.5-flash",
                "prompt": schema_analysis_prompt,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "datasets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "dataset_id": {"type": "string"},
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "columns": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "data_type": {"type": "string"},
                                                            "nullable": {"type": "boolean"},
                                                            "constraints": {"type": "array", "items": {"type": "string"}},
                                                            "description": {"type": "string"}
                                                        }
                                                    }
                                                },
                                                "primary_key": {"type": "array", "items": {"type": "string"}},
                                                "relationships": {"type": "array", "items": {"type": "string"}}
                                            }
                                        },
                                        "data_quality": {
                                            "type": "object",
                                            "properties": {
                                                "completeness": {"type": "number"},
                                                "consistency": {"type": "number"},
                                                "accuracy": {"type": "number"},
                                                "issues": {"type": "array", "items": {"type": "string"}}
                                            }
                                        },
                                        "sample_data": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        },
                        "required": ["datasets"]
                    }
                },
                "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
            }
        )
        llm_schema_response.raise_for_status()
        schema_analysis = llm_schema_response.json()["structured_output"]

        # Step 2: Perform semantic alignment using biomedical ontologies
        semantic_alignment_prompt = f"""Perform semantic alignment of the analyzed datasets using biomedical ontologies:

Dataset Analysis: {json.dumps(schema_analysis)}

Task: Create semantic mappings between datasets using these biomedical ontologies:
- SNOMED CT (clinical terms)
- LOINC (laboratory observations)
- RxNorm (medications)
- ICD-10/11 (diagnoses)
- HGNC (gene nomenclature)
- GO (gene ontology)
- MeSH (medical subject headings)

For Alzheimer's disease research, focus on:
- Clinical assessments (MMSE, CDR, ADAS-Cog)
- Biomarkers (amyloid-beta, tau, p-tau)
- Genetic markers (APOE, PSEN1, APP)
- Imaging data (MRI, PET)
- Drug treatments and clinical trials

Format your response as a JSON object with this structure:
{{
    "semantic_mappings": [
        {{
            "source_dataset": "dataset_1",
            "target_concept": "Alzheimer's Disease Assessment",
            "ontology_term": "SNOMED CT 371151006",
            "confidence": 0.95,
            "mapping_type": "exact|partial|related",
            "transformation_rules": ["normalize_scale_0_30", "handle_missing_as_null"]
        }}
    ],
    "unified_schema": {{
        "name": "Alzheimer_Research_Unified_Schema",
        "version": "1.0",
        "domains": ["clinical", "biomarker", "genetic", "imaging"],
        "columns": [
            {{
                "name": "patient_id",
                "data_type": "string",
                "semantic_type": "Patient Identifier",
                "ontology_mapping": "SNOMED CT 116154003",
                "validation_rules": ["not_null", "unique"],
                "description": "Unique patient identifier"
            }}
        ]
    }},
    "etl_pipeline": {{
        "transformations": [
            {{
                "step_name": "normalize_clinical_scores",
                "description": "Normalize MMSE scores to 0-30 scale",
                "input_columns": ["mmse_score", "mmse_total"],
                "output_column": "normalized_mmse",
                "transformation_type": "normalization",
                "parameters": {{"min_value": 0, "max_value": 30}}
            }}
        ],
        "data_quality_rules": [
            {{
                "rule_name": "age_range_check",
                "description": "Age must be between 18 and 120",
                "column": "age",
                "rule_type": "range",
                "parameters": {{"min": 18, "max": 120}}
            }}
        ]
    }}
}}"""

        llm_alignment_response = requests.post(
            f"{LLM_SERVICE_URL}/llm/structured-output",
            headers=llm_headers,
            json={
                "model_name": "gemini-1.5-flash",
                "prompt": semantic_alignment_prompt,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "semantic_mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source_dataset": {"type": "string"},
                                        "target_concept": {"type": "string"},
                                        "ontology_term": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "mapping_type": {"type": "string"},
                                        "transformation_rules": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            },
                            "unified_schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "version": {"type": "string"},
                                    "domains": {"type": "array", "items": {"type": "string"}},
                                    "columns": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "data_type": {"type": "string"},
                                                "semantic_type": {"type": "string"},
                                                "ontology_mapping": {"type": "string"},
                                                "validation_rules": {"type": "array", "items": {"type": "string"}},
                                                "description": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            },
                            "etl_pipeline": {
                                "type": "object",
                                "properties": {
                                    "transformations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "step_name": {"type": "string"},
                                                "description": {"type": "string"},
                                                "input_columns": {"type": "array", "items": {"type": "string"}},
                                                "output_column": {"type": "string"},
                                                "transformation_type": {"type": "string"},
                                                "parameters": {"type": "object"}
                                            }
                                        }
                                    },
                                    "data_quality_rules": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "rule_name": {"type": "string"},
                                                "description": {"type": "string"},
                                                "column": {"type": "string"},
                                                "rule_type": {"type": "string"},
                                                "parameters": {"type": "object"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["semantic_mappings", "unified_schema", "etl_pipeline"]
                    }
                },
                "metadata": {"agent_task_id": agent_task_id, "agent": agent_id}
            }
        )
        llm_alignment_response.raise_for_status()
        harmonization_result = llm_alignment_response.json()["structured_output"]

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="SEMANTIC_ALIGNMENT_COMPLETED",
            description=f"Agent {agent_id} completed semantic alignment and ETL pipeline generation.",
            metadata={"mappings_created": len(harmonization_result.get('semantic_mappings', []))}
        )
        
        # Step 3: Create comprehensive harmonization report
        harmonized_schema = {
            "schema_name": f"Harmonized_Schema_for_{db_agent_task.task_description.replace(' ', '_')}",
            "version": "1.0",
            "source_datasets": datasets_found,
            "schema_analysis": schema_analysis,
            "semantic_mappings": harmonization_result.get("semantic_mappings", []),
            "unified_schema": harmonization_result.get("unified_schema", {}),
            "etl_pipeline": harmonization_result.get("etl_pipeline", {}),
            "harmonization_report": {
                "datasets_analyzed": len(schema_analysis.get('datasets', [])),
                "semantic_mappings_created": len(harmonization_result.get('semantic_mappings', [])),
                "unified_columns": len(harmonization_result.get('unified_schema', {}).get('columns', [])),
                "etl_steps": len(harmonization_result.get('etl_pipeline', {}).get('transformations', [])),
                "quality_rules": len(harmonization_result.get('etl_pipeline', {}).get('data_quality_rules', []))
            }
        }

        # Publish comprehensive harmonization results as an insight
        insight_payload = schemas.InsightPublishRequest(
            insight_name=f"Comprehensive Data Harmonization: {harmonized_schema['schema_name']}",
            insight_description=f"Complete data harmonization analysis including schema analysis, semantic alignment, and ETL pipeline for {len(datasets_found)} datasets.",
            data_source_ids=datasets_found,
            payload=harmonized_schema,
            tags=["data_harmonization", "schema", "semantic_alignment", "etl", "biomedical_ontology", agent_id]
        ).model_dump_json()

        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="PUBLISHING_HARMONIZED_INSIGHT",
            description=f"Agent {agent_id} publishing comprehensive harmonization insight.",
            metadata={"insight_name": harmonized_schema['schema_name']}
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
            "agent_output": f"Comprehensive data harmonization completed for task {agent_task_id}.",
            "harmonization_summary": {
                "datasets_processed": len(datasets_found),
                "schema_analysis_completed": True,
                "semantic_mappings_created": len(harmonization_result.get('semantic_mappings', [])),
                "unified_schema_generated": True,
                "etl_pipeline_created": True,
                "quality_assessment_performed": True
            },
            "harmonized_schema": harmonized_schema,
            "published_insight_id": publish_result.get("insight_id")
        }

        crud.update_agent_task_status(db, agent_task_id, "COMPLETED", result)
        crud.update_agent_state(db, agent_id, current_task_id=None) # Task completed, clear current task
        log_audit_event(
            entity_type="AGENT",
            entity_id=f"{agent_id}-{agent_task_id}",
            event_type="HARMONIZATION_COMPLETED",
            description=f"Agent {agent_id} completed comprehensive data harmonization task {agent_task_id} and published insight.",
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
