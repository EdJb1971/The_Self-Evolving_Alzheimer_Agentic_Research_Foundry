from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import pymc3 as pm
import arviz as az
from ..database import get_db
from ..models import UncertaintyAnalysis, BayesianModel
from ..schemas import (
    BayesianPredictionRequest,
    BayesianPredictionResponse,
    BayesianModelTrainRequest,
    BayesianModelTrainResponse
)
from ..tasks import perform_bayesian_uncertainty_task
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/predict", response_model=BayesianPredictionResponse)
async def bayesian_prediction(
    request: BayesianPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> BayesianPredictionResponse:
    """
    Perform Bayesian prediction with uncertainty quantification.

    Uses Bayesian neural networks to provide prediction intervals and uncertainty estimates.
    """
    try:
        # Get or create Bayesian model
        db_model = db.query(BayesianModel).filter(
            BayesianModel.model_name == request.model_name,
            BayesianModel.is_active == True
        ).first()

        if not db_model:
            raise HTTPException(
                status_code=404,
                detail=f"Active Bayesian model '{request.model_name}' not found"
            )

        # Perform Bayesian prediction
        result = await perform_bayesian_uncertainty_task(
            model_config=db_model.model_config,
            input_data=request.input_data,
            confidence_level=request.confidence_level or 0.95
        )

        # Store analysis results
        analysis = UncertaintyAnalysis(
            analysis_type="bayesian",
            model_name=request.model_name,
            input_data=request.input_data,
            results=result["prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            confidence_level=request.confidence_level or 0.95,
            computation_time=result.get("computation_time")
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return BayesianPredictionResponse(
            prediction_id=analysis.id,
            prediction=result["prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            confidence_level=analysis.confidence_level,
            metadata={
                "model_name": request.model_name,
                "computation_time": result.get("computation_time"),
                "method": "bayesian_neural_network"
            }
        )

    except Exception as e:
        logger.error(f"Bayesian prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bayesian prediction failed: {str(e)}")

@router.post("/train", response_model=BayesianModelTrainResponse)
async def train_bayesian_model(
    request: BayesianModelTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> BayesianModelTrainResponse:
    """
    Train a new Bayesian neural network model.

    This is a computationally intensive operation that runs asynchronously.
    """
    try:
        # Check if model already exists
        existing_model = db.query(BayesianModel).filter(
            BayesianModel.model_name == request.model_name
        ).first()

        if existing_model and request.overwrite == False:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{request.model_name}' already exists. Set overwrite=True to replace."
            )

        # Create or update model record
        if existing_model:
            existing_model.model_config = request.model_config
            existing_model.is_active = False  # Deactivate during training
            db.commit()
            model_id = existing_model.id
        else:
            new_model = BayesianModel(
                model_name=request.model_name,
                model_config=request.model_config,
                is_active=False
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)
            model_id = new_model.id

        # Start async training
        background_tasks.add_task(
            perform_bayesian_training_task,
            model_id=model_id,
            model_config=request.model_config,
            training_data=request.training_data
        )

        return BayesianModelTrainResponse(
            model_id=model_id,
            model_name=request.model_name,
            status="training_started",
            estimated_completion_time="2-4 hours"  # Bayesian training is slow
        )

    except Exception as e:
        logger.error(f"Bayesian model training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.get("/models")
async def list_bayesian_models(db: Session = Depends(get_db)):
    """List all available Bayesian models"""
    models = db.query(BayesianModel).all()
    return [
        {
            "id": model.id,
            "model_name": model.model_name,
            "is_active": model.is_active,
            "created_at": model.created_at,
            "performance_metrics": model.performance_metrics
        }
        for model in models
    ]

@router.get("/models/{model_name}")
async def get_bayesian_model(model_name: str, db: Session = Depends(get_db)):
    """Get details of a specific Bayesian model"""
    model = db.query(BayesianModel).filter(BayesianModel.model_name == model_name).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return {
        "id": model.id,
        "model_name": model.model_name,
        "model_config": model.model_config,
        "performance_metrics": model.performance_metrics,
        "is_active": model.is_active,
        "created_at": model.created_at,
        "updated_at": model.updated_at
    }