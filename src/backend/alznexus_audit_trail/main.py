import os
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI(
    title="AlzNexus Audit Trail Service",
    description="Maintains a comprehensive, immutable log of all platform operations.",
    version="1.0.0",
)

AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
AUDIT_REDIS_URL = os.getenv("AUDIT_REDIS_URL", "redis://localhost:6379")

if not AUDIT_API_KEY:
    raise ValueError("AUDIT_API_KEY environment variable not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == AUDIT_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # In production, database schema migrations (e.g., using Alembic) should be used.
    print("Audit Trail Service: Schema management handled by migrations (not create_all).")
    redis_connection = redis.from_url(AUDIT_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for Audit Trail Service.")

@app.post("/audit/log", response_model=schemas.AuditLogEntry, status_code=201,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def log_event(
    log_entry: schemas.AuditLogCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-503: Records an event, decision, or reasoning step in the audit trail."""
    db_log = crud.create_audit_log_entry(db, log_entry)
    return db_log

@app.get("/audit/history/{entity_type}/{entity_id}", response_model=schemas.AuditHistoryResponse,
         dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def get_audit_history(
    entity_type: str,
    entity_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves the audit history for a specific entity (agent, task, or insight)."""
    history = crud.get_audit_history(db, entity_type, entity_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No audit history found for {entity_type}:{entity_id}")
    return schemas.AuditHistoryResponse(entity_type=entity_type, entity_id=entity_id, history=history)
