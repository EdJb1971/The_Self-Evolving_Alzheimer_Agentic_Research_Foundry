from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import UncertaintyAnalysis
from ..schemas import MonteCarloEnsembleRequest, MonteCarloEnsembleResponse
from ..tasks import perform_monte_carlo_uncertainty_task
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ensemble", response_model=MonteCarloEnsembleResponse)
async def monte_carlo_ensemble_prediction(
    request: MonteCarloEnsembleRequest,
    db: Session = Depends(get_db)
) -> MonteCarloEnsembleResponse:
    """
    Perform Monte Carlo uncertainty quantification using model ensembles.

    Creates multiple model predictions and aggregates uncertainty estimates.
    """
    try:
        # Perform Monte Carlo ensemble prediction
        result = await perform_monte_carlo_uncertainty_task(
            model_configs=request.model_configs,
            input_data=request.input_data,
            n_samples=request.n_samples,
            confidence_level=request.confidence_level or 0.95
        )

        # Store analysis results
        analysis = UncertaintyAnalysis(
            analysis_type="monte_carlo",
            model_name=f"ensemble_{len(request.model_configs)}_models",
            input_data={
                "model_configs": request.model_configs,
                "input_data": request.input_data,
                "n_samples": request.n_samples
            },
            results=result["ensemble_prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            confidence_level=request.confidence_level or 0.95,
            computation_time=result.get("computation_time")
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return MonteCarloEnsembleResponse(
            prediction_id=analysis.id,
            ensemble_prediction=result["ensemble_prediction"],
            uncertainty_bounds=result["uncertainty_bounds"],
            individual_predictions=result["individual_predictions"],
            confidence_level=analysis.confidence_level,
            metadata={
                "n_models": len(request.model_configs),
                "n_samples": request.n_samples,
                "computation_time": result.get("computation_time"),
                "method": "monte_carlo_ensemble"
            }
        )

    except Exception as e:
        logger.error(f"Monte Carlo ensemble prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Monte Carlo prediction failed: {str(e)}")

@router.post("/dropout")
async def monte_carlo_dropout(
    model_config: dict,
    input_data: dict,
    n_samples: int = 100,
    confidence_level: float = 0.95,
    db: Session = Depends(get_db)
):
    """
    Perform Monte Carlo dropout uncertainty estimation.

    Uses dropout at inference time to estimate model uncertainty.
    """
    try:
        # Placeholder: Implement Monte Carlo dropout
        # This would use a neural network with dropout layers enabled during inference

        prediction = {
            "mean_prediction": 0.5,
            "uncertainty": 0.1,
            "confidence_interval": [0.3, 0.7]
        }

        analysis = UncertaintyAnalysis(
            analysis_type="monte_carlo",
            model_name="dropout_uncertainty",
            input_data={"model_config": model_config, "input_data": input_data, "n_samples": n_samples},
            results=prediction,
            uncertainty_bounds={"method": "monte_carlo_dropout", "confidence_level": confidence_level},
            confidence_level=confidence_level
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return {
            "prediction_id": analysis.id,
            "prediction": prediction,
            "method": "monte_carlo_dropout",
            "n_samples": n_samples
        }

    except Exception as e:
        logger.error(f"Monte Carlo dropout failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Monte Carlo dropout failed: {str(e)}")

@router.get("/methods")
async def get_monte_carlo_methods():
    """Get available Monte Carlo uncertainty quantification methods"""
    return {
        "methods": [
            {
                "name": "ensemble_prediction",
                "description": "Aggregate predictions from multiple models",
                "use_case": "Combining different model architectures"
            },
            {
                "name": "monte_carlo_dropout",
                "description": "Use dropout at inference time for uncertainty",
                "use_case": "Neural network uncertainty estimation"
            },
            {
                "name": "bootstrap_aggregation",
                "description": "Bootstrap sampling with model aggregation",
                "use_case": "Statistical uncertainty from limited data"
            }
        ]
    }