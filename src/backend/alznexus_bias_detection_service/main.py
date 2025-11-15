import os
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import detect_bias_task, log_audit_event
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI(
    title="AlzNexus Bias Detection Service",
    description="A dedicated service for continuously analyzing data inputs, agent reasoning, and generated outputs for potential biases.",
    version="1.0.0",
)

BIAS_DETECTION_API_KEY = os.getenv("BIAS_DETECTION_API_KEY")
BIAS_DETECTION_REDIS_URL = os.getenv("BIAS_DETECTION_REDIS_URL", "redis://localhost:6379")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")

if not BIAS_DETECTION_API_KEY:
    raise ValueError("BIAS_DETECTION_API_KEY environment variable not set.")
if not LLM_SERVICE_URL or not LLM_API_KEY:
    raise ValueError("LLM_SERVICE_URL or LLM_API_KEY environment variables not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == BIAS_DETECTION_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # Database schema migrations are handled externally (e.g., via Alembic in a CI/CD pipeline)
    # and are not performed by the application at startup.
    print("Bias Detection Service: Database schema migrations are managed externally.")
    redis_connection = redis.from_url(BIAS_DETECTION_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for Bias Detection Service.")

@app.post("/bias/detect", response_model=schemas.BiasDetectionResponse, status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def detect_bias(
    bias_request: schemas.BiasDetectionRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-601: Submits data or agent output for bias detection analysis asynchronously."""
    # Create an initial report entry with PENDING status
    initial_report = schemas.BiasDetectionReportCreate(
        entity_type=bias_request.entity_type,
        entity_id=bias_request.entity_id,
        data_snapshot=bias_request.data_to_analyze,
        detected_bias=False, # Default to false, will be updated by task
        analysis_summary="Bias detection initiated.",
        metadata_json=bias_request.analysis_context
    )
    db_report = crud.create_bias_detection_report(db, initial_report)

    # Dispatch the bias detection task to Celery
    detect_bias_task.delay(
        db_report.id,
        bias_request.entity_type,
        bias_request.entity_id,
        bias_request.data_to_analyze,
        bias_request.analysis_context.model_dump() if bias_request.analysis_context else {}
    )

    log_audit_event(
        entity_type="BIAS_DETECTION_SERVICE",
        entity_id=str(db_report.id),
        event_type="BIAS_DETECTION_INITIATED",
        description=f"Bias detection initiated for {bias_request.entity_type}:{bias_request.entity_id}. Processing asynchronously.",
        metadata=db_report.model_dump()
    )

    return schemas.BiasDetectionResponse(
        report_id=db_report.id,
        status="PENDING",
        message="Bias detection initiated. Processing asynchronously.",
        detected_bias=False
    )

@app.get("/bias/report/{report_id}", response_model=schemas.BiasDetectionReport, status_code=200,
         dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def get_bias_report(
    report_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves a detailed report on detected biases and proposed corrections."""
    db_report = crud.get_bias_detection_report(db, report_id)
    if db_report is None:
        raise HTTPException(status_code=404, detail="Bias report not found")
    return db_report
