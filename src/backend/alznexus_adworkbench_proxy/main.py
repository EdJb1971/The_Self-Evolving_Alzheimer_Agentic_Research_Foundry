import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import execute_federated_query

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
    
    # Dispatch the federated query task to Celery
    execute_federated_query.delay(db_query.id)
    
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

@app.post("/adworkbench/publish-insight", response_model=schemas.InsightPublishResponse, status_code=201)
async def publish_insight(
    insight_data: schemas.InsightPublishRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Publishes a generated insight or finding to AD Workbench and stores it locally."""
    try:
        # Store the insight in our local database
        db_insight = crud.create_adworkbench_insight(db, insight_data)

        # TODO: CQ-ADW-001: Implement real AD Workbench API integration
        # For now, we'll simulate publishing to AD Workbench
        # In production, this would make an actual API call to AD Workbench
        mock_workbench_id = f"wb-insight-{db_insight.id}"

        # Update the insight with the workbench ID
        db_insight.workbench_insight_id = mock_workbench_id
        db.commit()

        return schemas.InsightPublishResponse(
            insight_id=db_insight.id,
            message="Insight published successfully",
            workbench_insight_id=mock_workbench_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish insight: {str(e)}")

@app.get("/adworkbench/data/scan", status_code=200)
async def scan_adworkbench_data(
    api_key: str = Depends(get_api_key)
):
    """Scans for new or updated datasets within AD Workbench."""
    try:
        # TODO: CQ-ADW-002: Implement real AD Workbench data scanning
        # In production, this would query AD Workbench for available datasets
        # For now, return mock data that represents typical AD datasets

        mock_datasets = [
            {
                "dataset_id": "adni_clinical_v1",
                "name": "ADNI Clinical Data",
                "description": "Clinical assessments and biomarkers from ADNI study",
                "data_types": ["clinical", "biomarkers"],
                "record_count": 2500,
                "last_updated": "2024-01-15T10:30:00Z"
            },
            {
                "dataset_id": "adni_imaging_v2",
                "name": "ADNI Imaging Data",
                "description": "MRI and PET imaging data from ADNI participants",
                "data_types": ["imaging", "mri", "pet"],
                "record_count": 1800,
                "last_updated": "2024-01-10T14:20:00Z"
            },
            {
                "dataset_id": "adni_genomics_v1",
                "name": "ADNI Genomics Data",
                "description": "Genetic variants and expression data",
                "data_types": ["genomics", "snps", "expression"],
                "record_count": 1200,
                "last_updated": "2024-01-08T09:15:00Z"
            }
        ]

        return {
            "message": "AD Workbench data scan completed",
            "datasets_found": len(mock_datasets),
            "datasets": mock_datasets,
            "scan_timestamp": "2024-01-20T16:45:00Z"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data scan failed: {str(e)}")

@app.get("/adworkbench/insights", status_code=200)
async def list_insights(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Lists published insights."""
    insights = crud.get_adworkbench_insights(db, skip=skip, limit=limit)
    return {
        "insights": [
            {
                "id": insight.id,
                "insight_name": insight.insight_name,
                "insight_description": insight.insight_description,
                "tags": insight.tags,
                "created_at": insight.created_at.isoformat() if insight.created_at else None,
                "workbench_insight_id": insight.workbench_insight_id
            }
            for insight in insights
        ],
        "total": len(insights),
        "skip": skip,
        "limit": limit
    }

@app.get("/adworkbench/insights/{insight_id}", status_code=200)
async def get_insight(
    insight_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """Retrieves a specific insight by ID."""
    insight = crud.get_adworkbench_insight(db, insight_id)
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")

    return {
        "id": insight.id,
        "insight_name": insight.insight_name,
        "insight_description": insight.insight_description,
        "data_source_ids": insight.data_source_ids,
        "payload": insight.payload,
        "tags": insight.tags,
        "status": insight.status,
        "workbench_insight_id": insight.workbench_insight_id,
        "created_at": insight.created_at.isoformat() if insight.created_at else None,
        "updated_at": insight.updated_at.isoformat() if insight.updated_at else None
    }
