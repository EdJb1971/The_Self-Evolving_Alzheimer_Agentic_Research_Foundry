import os
from fastapi import FastAPI, Depends, HTTPException, Security, Request, Response
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, get_db, Base
from .tasks import perform_bayesian_uncertainty_task, perform_monte_carlo_uncertainty_task, perform_pinn_modeling_task, perform_risk_assessment_task
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from datetime import datetime, timezone
import requests

UNCERTAINTY_SERVICE_ID = "alznexus_uncertainty_service_001"

app = FastAPI(
    title=f"AlzNexus Uncertainty Quantification Service: {UNCERTAINTY_SERVICE_ID}",
    description="Advanced uncertainty quantification and error bounds calculation for scientific research outputs.",
    version="1.0.0",
)

UNCERTAINTY_SERVICE_API_KEY = os.getenv("UNCERTAINTY_SERVICE_API_KEY")
UNCERTAINTY_REDIS_URL = os.getenv("UNCERTAINTY_REDIS_URL", "redis://localhost:6379")
AUDIT_TRAIL_URL = os.getenv("AUDIT_TRAIL_URL")
AUDIT_API_KEY = os.getenv("AUDIT_API_KEY")
STATISTICAL_ENGINE_URL = os.getenv("STATISTICAL_ENGINE_URL")
STATISTICAL_API_KEY = os.getenv("STATISTICAL_API_KEY")

if not UNCERTAINTY_SERVICE_API_KEY:
    raise ValueError("UNCERTAINTY_SERVICE_API_KEY environment variable not set.")
if not AUDIT_TRAIL_URL or not AUDIT_API_KEY:
    raise ValueError("AUDIT_TRAIL_URL and AUDIT_API_KEY environment variables not set.")
if not STATISTICAL_ENGINE_URL or not STATISTICAL_API_KEY:
    raise ValueError("STATISTICAL_ENGINE_URL and STATISTICAL_API_KEY environment variables not set.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == UNCERTAINTY_SERVICE_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API key")

@app.on_event("startup")
async def startup_event():
    redis_instance = redis.from_url(UNCERTAINTY_REDIS_URL)
    await FastAPILimiter.init(redis_instance)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": UNCERTAINTY_SERVICE_ID}

# Import routers
from .routers import bayesian, monte_carlo, pinn, risk_assessment

app.include_router(
    bayesian.router,
    prefix="/uncertainty/bayesian",
    tags=["Bayesian Uncertainty"],
    dependencies=[Depends(get_api_key), Depends(RateLimiter(times=10, seconds=60))]
)

app.include_router(
    monte_carlo.router,
    prefix="/uncertainty/monte-carlo",
    tags=["Monte Carlo Uncertainty"],
    dependencies=[Depends(get_api_key), Depends(RateLimiter(times=10, seconds=60))]
)

app.include_router(
    pinn.router,
    prefix="/uncertainty/pinn",
    tags=["Physics-Informed Neural Networks"],
    dependencies=[Depends(get_api_key), Depends(RateLimiter(times=5, seconds=60))]
)

app.include_router(
    risk_assessment.router,
    prefix="/uncertainty/risk",
    tags=["Risk Assessment"],
    dependencies=[Depends(get_api_key), Depends(RateLimiter(times=20, seconds=60))]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)