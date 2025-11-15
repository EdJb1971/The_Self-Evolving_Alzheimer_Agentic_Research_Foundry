import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import log_audit_event
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI(
    title="AlzNexus Agent Registry Service",
    description="Dedicated service for dynamic registration and discovery of specialized sub-agents.",
    version="1.0.0",
)

REGISTRY_API_KEY = os.getenv("REGISTRY_API_KEY")
REGISTRY_REDIS_URL = os.getenv("REGISTRY_REDIS_URL", "redis://localhost:6379")

if not REGISTRY_API_KEY:
    raise ValueError("REGISTRY_API_KEY environment variable not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == REGISTRY_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # Database schema migrations are handled externally (e.g., via Alembic in a CI/CD pipeline)
    # and are not performed by the application at startup.
    print("Agent Registry Service: Database schema migrations are managed externally.")
    redis_connection = redis.from_url(REGISTRY_REDIS_URL, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    print("FastAPI-Limiter initialized for Agent Registry Service.")

@app.post("/registry/register", response_model=schemas.AgentDetails, status_code=201,
          dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def register_agent(
    agent_data: schemas.AgentRegister,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Registers a new sub-agent with its ID, capabilities, and API endpoint."""
    db_agent, is_new = crud.register_agent(db, agent_data)
    event_type = "AGENT_REGISTERED" if is_new else "AGENT_UPDATED"
    description = f"Agent {agent_data.agent_id} {'registered' if is_new else 'updated'} with capabilities and API endpoint."
    log_audit_event(
        entity_type="AGENT_REGISTRY",
        entity_id=agent_data.agent_id,
        event_type=event_type,
        description=description,
        metadata=db_agent.model_dump()
    )
    return db_agent

@app.get("/registry/agents", response_model=schemas.AgentListResponse,
         dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def get_all_agents(
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves a list of all currently registered sub-agents and their details."""
    agents = crud.get_all_agents(db)
    return schemas.AgentListResponse(agents=agents)

@app.get("/registry/agents/{agent_id}", response_model=schemas.AgentDetails,
         dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def get_agent_details(
    agent_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves details of a specific registered sub-agent."""
    agent = crud.get_agent_by_id(db, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "alznexus_agent_registry"}
