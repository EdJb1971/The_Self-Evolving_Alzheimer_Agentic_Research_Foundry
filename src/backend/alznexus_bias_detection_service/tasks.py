import os
import requests
import logging
import time
import json
from sqlalchemy.orm import Session
from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas
from alznexus_llm_service.schemas import LLMChatRequest, LLMResponse # Import LLM schemas

AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL or AUDIT_API_KEY environment variables not set.")
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

@celery_app.task(bind=True, name="detect_bias_task")
def detect_bias_task(self, report_id: int, entity_type: str, entity_id: str, data_to_analyze: str, analysis_context: dict):
    db: Session = SessionLocal()
    try:
        # CQ-BIAS-001: Fetch and update the existing report, instead of creating a new one
        existing_report = crud.get_bias_detection_report(db, report_id)
        if not existing_report:
            raise ValueError(f"BiasDetectionReport with ID {report_id} not found.")

        # Update report status to IN_PROGRESS
        crud.update_bias_detection_report(db, report_id, schemas.BiasDetectionReportCreate(
            entity_type=entity_type, entity_id=entity_id, data_snapshot=data_to_analyze,
            detected_bias=False, analysis_summary="Analysis in progress.", metadata_json=analysis_context
        ))
        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(report_id),
            event_type="BIAS_DETECTION_STARTED",
            description=f"Bias detection started for {entity_type}:{entity_id}.",
            metadata={"report_id": report_id, "entity_type": entity_type, "entity_id": entity_id}
        )

        # Simulate bias detection using LLM Service (STORY-601)
        llm_headers = {"X-API-Key": LLM_API_KEY, "Content-Type": "application/json"}
        # SEC-BIAS-001: Instruct LLM to output structured JSON for bias analysis
        bias_detection_prompt = (
            f"Analyze the following text for potential biases, including but not limited to demographic, "
            f"selection, or confirmation bias. Provide a summary of any detected biases, their type, "
            f"severity (low, medium, high), and suggest concrete corrective actions. "
            f"Output your analysis as a JSON object with the following keys: "
            f"'detected_bias' (boolean), 'bias_type' (string, e.g., 'demographic', 'selection', 'none'), "
            f"'severity' (string, e.g., 'low', 'medium', 'high', 'none'), 'analysis_summary' (string), "
            f"'proposed_corrections' (list of strings). If no bias is detected, set 'detected_bias' to false "
            f"and 'bias_type' and 'severity' to 'none'. "
            f"Text to analyze: {data_to_analyze}. Context: {json.dumps(analysis_context)}"
        )
        llm_request = LLMChatRequest(model_name="grok-1", prompt=bias_detection_prompt, max_tokens=1000)

        llm_response_raw = requests.post(f"{LLM_SERVICE_URL}/llm/chat", headers=llm_headers, json=llm_request.model_dump())
        llm_response_raw.raise_for_status()
        llm_response_data = LLMResponse.model_validate(llm_response_raw.json())

        llm_analysis_text = llm_response_data.response_text

        # SEC-BIAS-001: Parse LLM's structured JSON bias analysis
        detected_bias = False
        bias_type = "none"
        severity = "none"
        analysis_summary = llm_analysis_text
        proposed_corrections = []

        try:
            bias_analysis_json = json.loads(llm_analysis_text)
            detected_bias = bias_analysis_json.get("detected_bias", False)
            bias_type = bias_analysis_json.get("bias_type", "none")
            severity = bias_analysis_json.get("severity", "none")
            analysis_summary = bias_analysis_json.get("analysis_summary", llm_analysis_text)
            proposed_corrections = bias_analysis_json.get("proposed_corrections", [])
        except json.JSONDecodeError:
            logger.warning(f"SEC-BIAS-001: LLM response was not valid JSON. Falling back to default parsing. Response: {llm_analysis_text[:200]}")
            # Fallback to basic keyword parsing if JSON is invalid (less secure, but prevents crash)
            detected_bias = "bias detected" in llm_analysis_text.lower() or "biased" in llm_analysis_text.lower()
            if detected_bias:
                if "demographic bias" in llm_analysis_text.lower(): bias_type = "demographic"
                elif "selection bias" in llm_analysis_text.lower(): bias_type = "selection"
                elif "confirmation bias" in llm_analysis_text.lower(): bias_type = "confirmation"
                else: bias_type = "general"

                if "high severity" in llm_analysis_text.lower(): severity = "high"
                elif "medium severity" in llm_analysis_text.lower(): severity = "medium"
                else: severity = "low"
                if "corrective actions" in llm_analysis_text.lower():
                    proposed_corrections = ["Review data sampling strategy", "Diversify data sources"]

        report_update_data = schemas.BiasDetectionReportCreate(
            entity_type=entity_type,
            entity_id=entity_id,
            data_snapshot=data_to_analyze,
            detected_bias=detected_bias,
            bias_type=bias_type,
            severity=severity,
            analysis_summary=analysis_summary,
            proposed_corrections={"actions": proposed_corrections},
            metadata_json=analysis_context
        )
        db_report = crud.update_bias_detection_report(db, report_id, report_update_data)
        if not db_report:
            raise ValueError(f"Failed to update BiasDetectionReport with ID {report_id}.")

        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(db_report.id),
            event_type="BIAS_DETECTED" if detected_bias else "NO_BIAS_DETECTED",
            description=f"Bias detection completed for {entity_type}:{entity_id}. Bias detected: {detected_bias}.",
            metadata=db_report.model_dump()
        )

        return {"report_id": db_report.id, "status": "COMPLETED", "detected_bias": detected_bias, "bias_type": bias_type, "severity": severity, "analysis_summary": analysis_summary, "proposed_corrections": proposed_corrections}
    except requests.exceptions.RequestException as e:
        error_message = f"LLM Service call failed during bias detection: {e}"
        # CQ-BIAS-001: Update existing report with failure status
        crud.update_bias_detection_report(db, report_id, schemas.BiasDetectionReportCreate(
            entity_type=entity_type, entity_id=entity_id, data_snapshot=data_to_analyze,
            detected_bias=False, analysis_summary=f"Failed to perform bias detection: {error_message}",
            metadata_json=analysis_context
        ))
        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(report_id),
            event_type="BIAS_DETECTION_FAILED",
            description=f"Bias detection failed for {entity_type}:{entity_id}: {error_message}",
            metadata={"report_id": report_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    except Exception as e:
        error_message = str(e)
        # CQ-BIAS-001: Update existing report with failure status
        crud.update_bias_detection_report(db, report_id, schemas.BiasDetectionReportCreate(
            entity_type=entity_type, entity_id=entity_id, data_snapshot=data_to_analyze,
            detected_bias=False, analysis_summary=f"Failed to perform bias detection: {error_message}",
            metadata_json=analysis_context
        ))
        log_audit_event(
            entity_type="BIAS_DETECTION_SERVICE",
            entity_id=str(report_id),
            event_type="BIAS_DETECTION_FAILED",
            description=f"Bias detection failed for {entity_type}:{entity_id}: {error_message}",
            metadata={"report_id": report_id, "error": error_message}
        )
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': error_message})
        raise
    finally:
        db.close()
