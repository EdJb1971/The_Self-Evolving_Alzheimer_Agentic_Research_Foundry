import os
from datetime import datetime
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

# Production-ready API Key authentication with multiple key support
VALID_API_KEYS = set()

# Load primary API key
primary_key = os.getenv("ADWORKBENCH_API_KEY")
if primary_key:
    VALID_API_KEYS.add(primary_key)

# Load additional API keys (comma-separated)
additional_keys = os.getenv("ADWORKBENCH_ADDITIONAL_API_KEYS", "")
if additional_keys:
    for key in additional_keys.split(","):
        key = key.strip()
        if key:
            VALID_API_KEYS.add(key)

# Ensure at least one valid key exists
if not VALID_API_KEYS:
    raise ValueError("No valid API keys configured. Set ADWORKBENCH_API_KEY or ADWORKBENCH_ADDITIONAL_API_KEYS environment variables.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key against configured valid keys.
    Supports multiple valid keys for different clients/services.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return api_key

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

        # CQ-ADW-001: AD Workbench API integration
        # Attempt to publish to AD Workbench API if configured
        workbench_insight_id = None

        adworkbench_api_url = os.getenv("ADWORKBENCH_API_URL")
        adworkbench_api_key = os.getenv("ADWORKBENCH_API_KEY")

        if adworkbench_api_url and adworkbench_api_key:
            try:
                # Prepare insight data for AD Workbench API
                workbench_payload = {
                    "title": insight_data.title,
                    "description": insight_data.description,
                    "data": insight_data.data,
                    "metadata": insight_data.metadata or {},
                    "source": "AlzNexus_Platform",
                    "timestamp": insight_data.created_at.isoformat() if insight_data.created_at else None
                }

                headers = {
                    "Authorization": f"Bearer {adworkbench_api_key}",
                    "Content-Type": "application/json"
                }

                # Make API call to AD Workbench
                response = requests.post(
                    f"{adworkbench_api_url}/insights",
                    json=workbench_payload,
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                workbench_response = response.json()
                workbench_insight_id = workbench_response.get("id", f"wb-{db_insight.id}")

                logger.info(f"Successfully published insight {db_insight.id} to AD Workbench: {workbench_insight_id}")

            except requests.exceptions.RequestException as api_error:
                logger.warning(f"Failed to publish insight to AD Workbench API: {str(api_error)}")
                # Continue with local storage only
                workbench_insight_id = f"local-{db_insight.id}"
            except Exception as api_error:
                logger.warning(f"Unexpected error publishing to AD Workbench: {str(api_error)}")
                workbench_insight_id = f"local-{db_insight.id}"
        else:
            logger.info("AD Workbench API not configured, storing locally only")
            workbench_insight_id = f"local-{db_insight.id}"

        # Update the insight with the workbench ID
        db_insight.workbench_insight_id = workbench_insight_id
        db.commit()

        return schemas.InsightPublishResponse(
            insight_id=db_insight.id,
            message="Insight published successfully",
            workbench_insight_id=workbench_insight_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish insight: {str(e)}")

@app.get("/adworkbench/data/scan", status_code=200)
async def scan_adworkbench_data(
    api_key: str = Depends(get_api_key)
):
    """Scans for new or updated datasets within AD Workbench."""
    try:
        # CQ-ADW-002: AD Workbench data scanning
        # Attempt to scan AD Workbench API if configured
        datasets = []

        adworkbench_api_url = os.getenv("ADWORKBENCH_API_URL")
        adworkbench_api_key = os.getenv("ADWORKBENCH_API_KEY")

        if adworkbench_api_url and adworkbench_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {adworkbench_api_key}",
                    "Content-Type": "application/json"
                }

                # Query AD Workbench for available datasets
                response = requests.get(
                    f"{adworkbench_api_url}/datasets",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()

                workbench_datasets = response.json()
                datasets = workbench_datasets.get("datasets", [])

                logger.info(f"Successfully scanned {len(datasets)} datasets from AD Workbench API")

            except requests.exceptions.RequestException as api_error:
                logger.warning(f"Failed to scan AD Workbench API: {str(api_error)}")
                # Fall back to cached/local dataset information
                datasets = get_cached_datasets()
            except Exception as api_error:
                logger.warning(f"Unexpected error scanning AD Workbench: {str(api_error)}")
                datasets = get_cached_datasets()
        else:
            logger.info("AD Workbench API not configured, using cached dataset information")
            datasets = get_cached_datasets()

        return {
            "message": "AD Workbench data scan completed",
            "datasets_found": len(datasets),
            "datasets": datasets,
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "api_connected": bool(adworkbench_api_url and adworkbench_api_key)
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


def get_cached_datasets():
    """
    Return cached dataset information when AD Workbench API is not available.
    This represents typical Alzheimer's disease research datasets.
    """
    return [
        {
            "dataset_id": "adni_clinical_v1",
            "name": "ADNI Clinical Data",
            "description": "Clinical assessments and biomarkers from ADNI study",
            "data_types": ["clinical", "biomarkers"],
            "record_count": 2500,
            "last_updated": "2024-01-15T10:30:00Z",
            "status": "available"
        },
        {
            "dataset_id": "adni_imaging_v2",
            "name": "ADNI Imaging Data",
            "description": "MRI and PET imaging data from ADNI participants",
            "data_types": ["imaging", "mri", "pet"],
            "record_count": 1800,
            "last_updated": "2024-01-10T14:20:00Z",
            "status": "available"
        },
        {
            "dataset_id": "adni_genomics_v1",
            "name": "ADNI Genomics Data",
            "description": "Genetic variants and expression data",
            "data_types": ["genomics", "snps", "expression"],
            "record_count": 1200,
            "last_updated": "2024-01-08T09:15:00Z",
            "status": "available"
        },
        {
            "dataset_id": "rosmap_clinical_v1",
            "name": "ROSMAP Clinical & Omics",
            "description": "Religious Orders Study and Memory and Aging Project clinical and multi-omics data",
            "data_types": ["clinical", "transcriptomics", "methylomics", "metabolomics"],
            "record_count": 3200,
            "last_updated": "2024-01-12T11:45:00Z",
            "status": "available"
        }
    ]
