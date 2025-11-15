from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

try:
    # Try relative imports first (when run as package)
    from .database import engine, Base
    from .routers.seeds import router as seeds_router
    from .routers.provenance import router as provenance_router
    from .routers.validation import router as validation_router
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database import engine, Base
    from routers.seeds import router as seeds_router
    from routers.provenance import router as provenance_router
    from routers.validation import router as validation_router

# Create database tables
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup on shutdown (if needed)

# Create FastAPI app
app = FastAPI(
    title="AlzNexus Reproducibility Service",
    description="Scientific reproducibility framework for Alzheimer's research agents",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    seeds_router,
    prefix="/api/v1",
    tags=["seeds"]
)

app.include_router(
    provenance_router,
    prefix="/api/v1",
    tags=["provenance"]
)

app.include_router(
    validation_router,
    prefix="/api/v1",
    tags=["validation"]
)

@app.get("/")
async def root():
    """
    Root endpoint with service information
    """
    return {
        "service": "AlzNexus Reproducibility Service",
        "version": "1.0.0",
        "description": "Ensuring scientific reproducibility in Alzheimer's research",
        "endpoints": {
            "seeds": "/api/v1/seeds",
            "provenance": "/api/v1/provenance",
            "validation": "/api/v1/validate"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "reproducibility-service",
        "timestamp": "2024-01-01T00:00:00Z"  # Would be dynamic in real implementation
    }

@app.get("/api/v1/status")
async def service_status():
    """
    Detailed service status
    """
    return {
        "service": "AlzNexus Reproducibility Service",
        "status": "operational",
        "capabilities": [
            "Random seed management",
            "Data provenance tracking",
            "Analysis snapshot creation",
            "Reproducibility validation",
            "Environment capture",
            "Code and data versioning"
        ],
        "supported_agents": [
            "biomarker_hunter_agent",
            "collaboration_matchmaker_agent",
            "data_harmonizer_agent",
            "drug_screener_agent",
            "hypothesis_validator_agent",
            "literature_bridger_agent",
            "pathway_modeler_agent",
            "trial_optimizer_agent"
        ]
    }