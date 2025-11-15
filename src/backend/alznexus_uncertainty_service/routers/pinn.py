from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import deepxde as dde
import torch
from datetime import datetime
from ..database import get_db
from ..models import UncertaintyAnalysis, PINNModel
from ..schemas import (
    PINNModelingRequest,
    PINNModelingResponse,
    PINNTrainRequest,
    PINNTrainResponse,
    PINNEvolutionRequest
)
from ..tasks import perform_pinn_modeling_task, perform_pinn_evolution_task
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/model", response_model=PINNModelingResponse)
async def pinn_disease_modeling(
    request: PINNModelingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> PINNModelingResponse:
    """
    Perform physics-informed neural network modeling for disease progression.

    Uses PINNs to model Alzheimer's disease progression with biological constraints.
    """
    try:
        # Get active PINN model for the disease
        db_model = db.query(PINNModel).filter(
            PINNModel.disease_model == request.disease_model,
            PINNModel.is_converged == True
        ).first()

        if not db_model:
            raise HTTPException(
                status_code=404,
                detail=f"No converged PINN model found for disease model '{request.disease_model}'"
            )

        # Perform PINN-based disease modeling
        result = await perform_pinn_modeling_task(
            model_config=db_model.neural_network_config,
            physics_constraints=db_model.physics_constraints,
            input_conditions=request.input_conditions,
            time_horizon=request.time_horizon
        )

        # Store analysis results
        analysis = UncertaintyAnalysis(
            analysis_type="pinn",
            model_name=f"pinn_{request.disease_model}",
            input_data={
                "input_conditions": request.input_conditions,
                "time_horizon": request.time_horizon
            },
            results=result["trajectory_prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            confidence_level=request.confidence_level or 0.95,
            computation_time=result.get("computation_time")
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return PINNModelingResponse(
            prediction_id=analysis.id,
            disease_trajectory=result["trajectory_prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            key_biomarkers=result.get("key_biomarkers", []),
            intervention_points=result.get("intervention_points", []),
            confidence_level=analysis.confidence_level,
            metadata={
                "disease_model": request.disease_model,
                "pinn_converged": True,
                "computation_time": result.get("computation_time"),
                "physics_constraints_satisfied": result.get("constraints_satisfied", True)
            }
        )

    except Exception as e:
        logger.error(f"PINN modeling failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PINN modeling failed: {str(e)}")

@router.post("/train", response_model=PINNTrainResponse)
async def train_pinn_model(
    request: PINNTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> PINNTrainResponse:
    """
    Train a new physics-informed neural network for disease modeling.

    Incorporates biological constraints (PDEs) into the neural network training.
    """
    try:
        # Check if model already exists
        existing_model = db.query(PINNModel).filter(
            PINNModel.model_name == request.model_name
        ).first()

        if existing_model and request.overwrite == False:
            raise HTTPException(
                status_code=409,
                detail=f"PINN model '{request.model_name}' already exists. Set overwrite=True to replace."
            )

        # Create or update PINN model record
        if existing_model:
            existing_model.neural_network_config = request.neural_network_config
            existing_model.physics_constraints = request.physics_constraints
            existing_model.is_converged = False  # Reset convergence status
            db.commit()
            model_id = existing_model.id
        else:
            new_model = PINNModel(
                model_name=request.model_name,
                disease_model=request.disease_model,
                neural_network_config=request.neural_network_config,
                physics_constraints=request.physics_constraints,
                is_converged=False
            )
            db.add(new_model)
            db.commit()
            db.refresh(new_model)
            model_id = new_model.id

        # Start async PINN training
        background_tasks.add_task(
            perform_pinn_modeling_task,
            model_id=model_id,
            neural_network_config=request.neural_network_config,
            physics_constraints=request.physics_constraints,
            training_data=request.training_data
        )

        return PINNTrainResponse(
            model_id=model_id,
            model_name=request.model_name,
            disease_model=request.disease_model,
            status="training_started",
            estimated_completion_time="4-8 hours"  # PINN training is very compute-intensive
        )

    except Exception as e:
        logger.error(f"PINN training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PINN training failed: {str(e)}")

@router.get("/models")
async def list_pinn_models(db: Session = Depends(get_db)):
    """List all available PINN models"""
    models = db.query(PINNModel).all()
    return [
        {
            "id": model.id,
            "model_name": model.model_name,
            "disease_model": model.disease_model,
            "is_converged": model.is_converged,
            "created_at": model.created_at,
            "convergence_metrics": model.convergence_metrics
        }
        for model in models
    ]

@router.get("/models/{model_name}")
async def get_pinn_model(model_name: str, db: Session = Depends(get_db)):
    """Get details of a specific PINN model"""
    model = db.query(PINNModel).filter(PINNModel.model_name == model_name).first()
    if not model:
        raise HTTPException(status_code=404, detail=f"PINN model '{model_name}' not found")

    return {
        "id": model.id,
        "model_name": model.model_name,
        "disease_model": model.disease_model,
        "neural_network_config": model.neural_network_config,
        "physics_constraints": model.physics_constraints,
        "training_history": model.training_history,
        "convergence_metrics": model.convergence_metrics,
        "is_converged": model.is_converged,
        "created_at": model.created_at,
        "updated_at": model.updated_at
    }

@router.post("/evolve", response_model=PINNTrainResponse)
async def evolve_pinn_model(
    request: PINNEvolutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> PINNTrainResponse:
    """
    Evolve an existing PINN model with new data and refined biological constraints.

    This enables continuous learning as new research findings become available.
    """
    try:
        # Get existing model
        db_model = db.query(PINNModel).filter(
            PINNModel.model_name == request.model_name,
            PINNModel.is_converged == True
        ).first()

        if not db_model:
            raise HTTPException(
                status_code=404,
                detail=f"No converged PINN model found for evolution: '{request.model_name}'"
            )

        # Merge new constraints with existing ones
        evolved_constraints = {**db_model.physics_constraints, **request.new_constraints}

        # Update model for evolution
        db_model.physics_constraints = evolved_constraints
        db_model.is_converged = False  # Mark for retraining
        db.commit()

        # Start evolution training
        background_tasks.add_task(
            perform_pinn_evolution_task,
            model_id=db_model.id,
            existing_config=db_model.neural_network_config,
            evolved_constraints=evolved_constraints,
            new_training_data=request.new_training_data,
            feedback_data=request.feedback_data
        )

        return PINNTrainResponse(
            model_id=db_model.id,
            model_name=request.model_name,
            disease_model=db_model.disease_model,
            status="evolving",
            estimated_completion_time="6-12 hours"  # Evolution takes time
        )

    except Exception as e:
        logger.error(f"PINN evolution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PINN evolution failed: {str(e)}")

@router.post("/feedback")
async def submit_pinn_feedback(
    model_name: str,
    prediction_accuracy: float,
    new_findings: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Submit feedback on PINN predictions to enable continuous learning.

    Agents and users can report prediction accuracy and new biological findings.
    """
    try:
        model = db.query(PINNModel).filter(PINNModel.model_name == model_name).first()
        if not model:
            raise HTTPException(status_code=404, detail=f"PINN model '{model_name}' not found")

        # Store feedback for future evolution
        feedback_record = {
            "prediction_accuracy": prediction_accuracy,
            "new_findings": new_findings,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback_type": "prediction_validation"
        }

        # Update model training history
        history = model.training_history or []
        history.append(feedback_record)
        model.training_history = history
        db.commit()

        # Trigger evolution if accuracy is low or new findings are significant
        should_evolve = (
            prediction_accuracy < 0.8 or  # Low accuracy
            len(new_findings.get("validated_hypotheses", [])) > 0  # New validated knowledge
        )

        if should_evolve:
            # Mark model for evolution
            model.is_converged = False
            db.commit()

            return {
                "status": "evolution_triggered",
                "reason": "low_accuracy" if prediction_accuracy < 0.8 else "new_findings",
                "next_steps": "Model will be retrained with new data"
            }

        return {
            "status": "feedback_recorded",
            "message": "Feedback stored for future model evolution"
        }

    except Exception as e:
        logger.error(f"PINN feedback submission failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")