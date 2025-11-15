import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .database import engine, Base, get_db
from . import crud, models, schemas
from .routers import statistical_analysis, validation, reports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
STATISTICAL_API_KEY = os.getenv("STATISTICAL_API_KEY")
if not STATISTICAL_API_KEY:
    if os.getenv("ENV") != "production":
        logger.warning("STATISTICAL_API_KEY not set. Using default for development.")
        STATISTICAL_API_KEY = "dev_statistical_key_123"
    else:
        raise ValueError("STATISTICAL_API_KEY must be set in production")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AlzNexus Statistical Engine Service")
    Base.metadata.create_all(bind=engine)

    yield

    # Shutdown
    logger.info("Shutting down AlzNexus Statistical Engine Service")

app = FastAPI(
    title="AlzNexus Statistical Engine API",
    description="Statistical validation and analysis service for Alzheimer's research",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication middleware
@app.middleware("http")
async def api_key_auth(request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != STATISTICAL_API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"}
        )
    response = await call_next(request)
    return response

# Include routers
app.include_router(
    statistical_analysis.router,
    prefix="/api/v1/statistical",
    tags=["statistical_analysis"]
)

app.include_router(
    validation.router,
    prefix="/api/v1/validation",
    tags=["validation"]
)

app.include_router(
    reports.router,
    prefix="/api/v1/reports",
    tags=["reports"]
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "alznexus_statistical_engine"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AlzNexus Statistical Engine",
        "version": "1.0.0",
        "description": "Statistical validation and analysis for Alzheimer's research",
        "endpoints": {
            "statistical_analysis": "/api/v1/statistical",
            "validation": "/api/v1/validation",
            "reports": "/api/v1/reports",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True if os.getenv("ENV") != "production" else False
    )