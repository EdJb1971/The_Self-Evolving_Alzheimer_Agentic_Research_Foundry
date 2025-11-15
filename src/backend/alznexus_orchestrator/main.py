import os
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import initiate_daily_scan_task, coordinate_sub_agents_task, resolve_debate_task, perform_self_correction_task, log_audit_event
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI(
    title="AlzNexus Master Orchestrator Service",
    description="The brain of AlzNexus, coordinating sub-agents and managing research goals.",
    version="1.0.0",
)

ORCHESTRATOR_API_KEY = os.getenv("ORCHESTRATOR_API_KEY")
ORCHESTRATOR_REDIS_URL = os.getenv("ORCHESTRATOR_REDIS_URL", "redis://localhost:6379")
AGENT_SERVICE_BASE_URL = os.getenv("AGENT_SERVICE_BASE_URL")
AGENT_API_KEY = os.getenv("AGENT_API_KEY")

if not ORCHESTRATOR_API_KEY:
    raise ValueError("ORCHESTRATOR_API_KEY environment variable not set.")
if not AGENT_SERVICE_BASE_URL:
    raise ValueError("AGENT_SERVICE_BASE_URL environment variable not set.")
if not AGENT_API_KEY:
    raise ValueError("AGENT_API_KEY environment variable not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == ORCHESTRATOR_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # In production, database schema migrations (e.g., using Alembic) should be used.
    print("Orchestrator Service: Schema management handled by migrations (not create_all).")
    redis_connection = redis.from_url(ORCHESTRATOR_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for Orchestrator Service.")

@app.post("/orchestrator/set-goal", response_model=schemas.ResearchGoal, status_code=201,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def set_research_goal(
    goal: schemas.ResearchGoalCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-101: Sets a new overarching research goal for the agent swarm."""
    db_goal = crud.create_research_goal(db, goal)
    log_audit_event(
        entity_type="ORCHESTRATOR",
        entity_id=str(db_goal.id),
        event_type="GOAL_SET",
        description=f"New research goal set: {goal.goal_text}",
        metadata=db_goal.model_dump()
    )
    return db_goal

@app.post("/orchestrator/initiate-daily-scan", response_model=schemas.DailyScanInitiateResponse, status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def initiate_daily_scan(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-102: Triggers the daily data scanning process across AD Workbench asynchronously."""
    active_goal = crud.get_active_research_goal(db)
    if not active_goal:
        raise HTTPException(status_code=404, detail="No active research goal found to initiate scan.")

    # Create an orchestrator task record
    orchestrator_task_create = schemas.OrchestratorTaskCreate(
        goal_id=active_goal.id,
        task_type="DAILY_SCAN",
        description="Initiate daily data scan across AD Workbench."
    )
    db_orchestrator_task = crud.create_orchestrator_task(db, orchestrator_task_create)

    # Dispatch the daily scan task to Celery
    initiate_daily_scan_task.delay(db_orchestrator_task.id)

    return schemas.DailyScanInitiateResponse(
        orchestrator_task_id=db_orchestrator_task.id,
        status="PENDING",
        message="Daily data scan initiated. Processing asynchronously."
    )

@app.post("/orchestrator/coordinate-task", status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def coordinate_task(
    task_data: schemas.TaskCoordinationRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-105: Assigns a specific task to one or more sub-agents, enabling collaboration."""
    # Create an orchestrator task record for coordination
    orchestrator_task_create = schemas.OrchestratorTaskCreate(
        goal_id=task_data.goal_id,
        task_type="COORDINATE_SUB_AGENT",
        description=task_data.overall_description,
        metadata_json=task_data.coordination_metadata
    )
    db_orchestrator_task = crud.create_orchestrator_task(db, orchestrator_task_create)

    # Dispatch the coordination task to Celery, passing the sub-agent tasks
    coordinate_sub_agents_task.delay(db_orchestrator_task.id, [task.model_dump() for task in task_data.sub_agent_tasks])

    log_audit_event(
        entity_type="ORCHESTRATOR",
        entity_id=str(db_orchestrator_task.id),
        event_type="TASK_COORDINATION_INITIATED",
        description=f"Orchestrator initiated coordination for task: {task_data.overall_description}",
        metadata=db_orchestrator_task.model_dump()
    )
    return {"message": "Task coordination initiated.", "orchestrator_task_id": db_orchestrator_task.id}

@app.post("/orchestrator/resolve-debate", status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def resolve_debate(
    debate_data: schemas.DebateInitiateRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-104: Facilitates and resolves debates between conflicting sub-agents."""
    # Create an orchestrator task record for debate resolution
    orchestrator_task_create = schemas.OrchestratorTaskCreate(
        goal_id=debate_data.goal_id,
        task_type="RESOLVE_DEBATE",
        description=debate_data.description,
        metadata_json=debate_data.debate_metadata
    )
    db_orchestrator_task = crud.create_orchestrator_task(db, orchestrator_task_create)

    # Dispatch the debate resolution task to Celery
    resolve_debate_task.delay(db_orchestrator_task.id, debate_data.model_dump())

    log_audit_event(
        entity_type="ORCHESTRATOR",
        entity_id=str(db_orchestrator_task.id),
        event_type="DEBATE_RESOLUTION_INITIATED",
        description=f"Orchestrator initiated debate resolution for: {debate_data.description}",
        metadata=db_orchestrator_task.model_dump()
    )
    return {"message": "Debate resolution initiated.", "orchestrator_task_id": db_orchestrator_task.id}

@app.post("/orchestrator/initiate-self-correction", response_model=schemas.DailyScanInitiateResponse, status_code=202,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def initiate_self_correction(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """STORY-103: Triggers the continuous self-correction and adaptation mechanism asynchronously."""
    active_goal = crud.get_active_research_goal(db)
    if not active_goal:
        raise HTTPException(status_code=404, detail="No active research goal found to initiate self-correction.")

    orchestrator_task_create = schemas.OrchestratorTaskCreate(
        goal_id=active_goal.id,
        task_type="SELF_CORRECTION",
        description="Initiate continuous self-correction and adaptation."
    )
    db_orchestrator_task = crud.create_orchestrator_task(db, orchestrator_task_create)

    perform_self_correction_task.delay(db_orchestrator_task.id)

    log_audit_event(
        entity_type="ORCHESTRATOR",
        entity_id=str(db_orchestrator_task.id),
        event_type="SELF_CORRECTION_TRIGGERED",
        description="Master Orchestrator triggered self-correction process.",
        metadata=db_orchestrator_task.model_dump()
    )
    return schemas.DailyScanInitiateResponse(
        orchestrator_task_id=db_orchestrator_task.id,
        status="PENDING",
        message="Self-correction process initiated. Processing asynchronously."
    )
