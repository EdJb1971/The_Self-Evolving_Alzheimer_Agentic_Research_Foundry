import os
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import screen_drugs_task, log_audit_event, perform_reflection_task
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from datetime import datetime, timezone
import requests

AGENT_ID = "drug_screener_agent_001"

app = FastAPI(
    title=f"AlzNexus Specialized Sub-Agent Service: {AGENT_ID}",
    description="A specialized sub-agent for identifying potential drug candidates.",
    version="1.0.0",
)

DRUG_SCREENER_AGENT_API_KEY = os.getenv("DRUG_SCREENER_AGENT_API_KEY")
DRUG_SCREENER_REDIS_URL = os.getenv("DRUG_SCREENER_REDIS_URL", "redis://localhost:6379")
ADWORKBENCH_PROXY_URL = os.getenv("ADWORKBENCH_PROXY_URL")
ADWORKBENCH_API_KEY = os.getenv("ADWORKBENCH_API_KEY")
AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
AGENT_REGISTRY_URL = os.getenv("AGENT_REGISTRY_URL")
AGENT_REGISTRY_API_KEY = os.getenv("AGENT_REGISTRY_API_KEY")
AGENT_EXTERNAL_API_ENDPOINT = os.getenv("AGENT_EXTERNAL_API_ENDPOINT")

if not DRUG_SCREENER_AGENT_API_KEY:
    raise ValueError("DRUG_SCREENER_AGENT_API_KEY environment variable not set.")
if not ADWORKBENCH_PROXY_URL:
    raise ValueError("ADWORKBENCH_PROXY_URL environment variable not set.")
if not ADWORKBENCH_API_KEY:
    raise ValueError("ADWORKBENCH_API_KEY environment variable not set.")
if not AUDIT_TRAIL_URL:
    raise ValueError("AUDIT_TRAIL_URL environment variable not set.")
if not AUDIT_API_KEY:
    raise ValueError("AUDIT_API_KEY environment variable not set.")
if not AGENT_REGISTRY_URL or not AGENT_REGISTRY_API_KEY:
    raise ValueError("AGENT_REGISTRY_URL or AGENT_REGISTRY_API_KEY environment variables not set.")
if not AGENT_EXTERNAL_API_ENDPOINT:
    raise ValueError("AGENT_EXTERNAL_API_ENDPOINT environment variable not set. This is required for agent registration.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == DRUG_SCREENER_AGENT_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

async def register_self_with_registry():
    """Registers this agent with the central Agent Registry Service."""
    registry_headers = {"X-API-Key": AGENT_REGISTRY_API_KEY, "Content-Type": "application/json"}
    registration_payload = schemas.AgentRegister(
        agent_id=AGENT_ID,
        capabilities={
            "description": "Identifies potential drug candidates by screening against disease pathways and target profiles.",
            "tools": ["ADWorkbenchQueryTool", "InSilicoScreeningTool", "LiteratureReviewTool"],
            "domain": "Drug Discovery"
        },
        api_endpoint=AGENT_EXTERNAL_API_ENDPOINT # Use the explicit external endpoint
    ).model_dump_json()

    try:
        log_audit_event(
            entity_type="AGENT",
            entity_id=AGENT_ID,
            event_type="AGENT_REGISTRATION_ATTEMPT",
            description=f"Agent {AGENT_ID} attempting to register with Agent Registry.",
            metadata={"registry_url": AGENT_REGISTRY_URL}
        )
        response = requests.post(f"{AGENT_REGISTRY_URL}/registry/register", headers=registry_headers, data=registration_payload)
        response.raise_for_status()
        log_audit_event(
            entity_type="AGENT",
            entity_id=AGENT_ID,
            event_type="AGENT_REGISTRATION_SUCCESS",
            description=f"Agent {AGENT_ID} successfully registered with Agent Registry.",
            metadata=response.json()
        )
        print(f"Agent {AGENT_ID} successfully registered with Agent Registry.")
    except requests.exceptions.RequestException as e:
        log_audit_event(
            entity_type="AGENT",
            entity_id=AGENT_ID,
            event_type="AGENT_REGISTRATION_FAILED",
            description=f"Agent {AGENT_ID} failed to register with Agent Registry: {str(e)}",
            metadata={"error": str(e), "registry_url": AGENT_REGISTRY_URL}
        )
        print(f"ERROR: Agent {AGENT_ID} failed to register with Agent Registry: {e}")

@app.on_event("startup")
async def startup_event():
    # Database schema migrations are handled externally (e.g., via Alembic in a CI/CD pipeline)
    # and are not performed by the application at startup.
    print(f"Agent Service ({AGENT_ID}): Database schema migrations are managed externally.")
    with Session(engine) as db:
        crud.get_or_create_agent_state(db, AGENT_ID)
        print(f"Agent state initialized for {AGENT_ID}.")
    redis_connection = redis.from_url(DRUG_SCREENER_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for Agent Service.")
    await register_self_with_registry() # Register agent with the registry

@app.post("/agent/{agent_id}/execute-task", response_model=schemas.AgentTask, status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def execute_task(
    agent_id: str,
    task: schemas.AgentTaskCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-303: Instructs the Drug Screener sub-agent to execute a drug screening task asynchronously."""
    if agent_id != AGENT_ID:
        raise HTTPException(status_code=400, detail=f"Invalid agent_id. This service is for {AGENT_ID}.")

    db_agent_task = crud.create_agent_task(db, task)

    screen_drugs_task.delay(db_agent_task.id)

    log_audit_event(
        entity_type="AGENT",
        entity_id=f"{agent_id}-{db_agent_task.id}",
        event_type="TASK_RECEIVED",
        description=f"Agent {agent_id} received task {db_agent_task.id}: {task.task_description}",
        metadata=db_agent_task.model_dump()
    )

    return db_agent_task

@app.get("/agent/{agent_id}/status", response_model=schemas.AgentStatusResponse,
         dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def get_agent_status(
    agent_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves the current status and progress of a sub-agent's task."""
    if agent_id != AGENT_ID:
        raise HTTPException(status_code=400, detail=f"Invalid agent_id. This service is for {AGENT_ID}.")

    agent_state = crud.get_or_create_agent_state(db, agent_id)
    current_task = None
    if agent_state.current_task_id:
        current_task = crud.get_agent_task(db, agent_state.current_task_id)

    status_message = "Idle" if not current_task else f"Working on task {current_task.id}: {current_task.status}"

    return schemas.AgentStatusResponse(
        agent_id=agent_id,
        status=status_message,
        current_task_id=agent_state.current_task_id,
        current_goal=agent_state.current_goal,
        message=f"Status for {agent_id}"
    )

@app.post("/agent/{agent_id}/reflect", status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def reflect_agent(
    agent_id: str,
    reflection_data: schemas.AgentStateUpdate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Triggers a self-reflection process for the agent on its recent actions/outcomes asynchronously."""
    if agent_id != AGENT_ID:
        raise HTTPException(status_code=400, detail=f"Invalid agent_id. This service is for {AGENT_ID}.")

    perform_reflection_task.delay(agent_id, reflection_data.model_dump(exclude_unset=True))

    log_audit_event(
        entity_type="AGENT",
        entity_id=agent_id,
        event_type="REFLECTION_INITIATED",
        description=f"Agent {agent_id} initiated self-reflection. Processing asynchronously.",
        metadata=reflection_data.model_dump(exclude_unset=True)
    )
    return {"message": f"Agent {agent_id} reflection initiated. Processing asynchronously."}
