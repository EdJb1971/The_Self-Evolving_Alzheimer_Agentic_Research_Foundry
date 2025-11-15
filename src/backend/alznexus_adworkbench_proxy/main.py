import os
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import simulate_federated_query

app = FastAPI(
    title="AlzNexus AD Workbench API Proxy Service",
    description="Gateway for secure, privacy-preserving federated queries to AD Workbench.",
    version="1.0.0",
)

# Placeholder for API Key authentication
# In a real scenario, this would involve proper key management and validation against an identity provider.
API_KEY = os.getenv("ADWORKBENCH_API_KEY", "supersecretapikey") # For demonstration, use a strong key in production
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

@app.on_event("startup")
async def startup_event():
    # Removed Base.metadata.create_all(bind=engine) for production readiness.
    # In production, database schema migrations (e.g., using Alembic) should be used.
    print("Database startup event: Schema management handled by migrations (not create_all).")

@app.post("/adworkbench/query", response_model=schemas.QueryStatusResponse, status_code=202)
async def query_adworkbench(
    query: schemas.ADWorkbenchQueryCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key) # Added API Key authentication
):
    """Executes a federated query against AD Workbench data sources asynchronously."""
    db_query = crud.create_adworkbench_query(db, query)
    
    # Dispatch the federated query simulation task to Celery
    simulate_federated_query.delay(db_query.id)
    
    return schemas.QueryStatusResponse(
        id=db_query.id,
        status="PENDING",
        message="Query submitted. Processing asynchronously."
    )

@app.get("/adworkbench/query/{query_id}/status", response_model=schemas.QueryStatusResponse)
async def get_query_status(
    query_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key) # Added API Key authentication
):
    """Retrieves the current status and result of a previously submitted AD Workbench query."""
    db_query = crud.get_adworkbench_query(db, query_id)
    if db_query is None:
        raise HTTPException(status_code=404, detail="Query not found")
    
    return schemas.QueryStatusResponse(
        id=db_query.id,
        status=db_query.status,
        message=f"Query status: {db_query.status}",
        result_data=db_query.result_data
    )

@app.post("/adworkbench/publish-insight", status_code=200)
async def publish_insight(
    insight_data: schemas.InsightPublishRequest, # Added Pydantic schema for input validation
    api_key: str = Depends(get_api_key) # Added API Key authentication
):
    """Publishes a generated insight or finding back to AD Workbench. (Placeholder)"""
    # CQ-001: In a real scenario, this would interact with AD Workbench's publishing API.
    # This placeholder currently only validates input and logs it.
    print(f"Attempting to publish insight with validated data: {insight_data.model_dump_json()}")
    # Simulate interaction with AD Workbench API
    mock_insight_id = f"mock-insight-{hash(insight_data.insight_name) % 10000}"
    return {"message": "Insight publishing initiated (placeholder).", "insight_id": mock_insight_id}

@app.get("/adworkbench/data/scan", status_code=200)
async def scan_adworkbench_data(
    api_key: str = Depends(get_api_key) # Added API Key authentication
):
    """Scans for new or updated datasets within AD Workbench. (Placeholder)"""
    # CQ-002: In a real scenario, this would query AD Workbench for data updates.
    print("Initiating AD Workbench data scan (placeholder).")
    # Simulate scanning for data
    return {"message": "AD Workbench data scan initiated (placeholder).", "datasets_found": ["dataset_A", "dataset_B"]}
