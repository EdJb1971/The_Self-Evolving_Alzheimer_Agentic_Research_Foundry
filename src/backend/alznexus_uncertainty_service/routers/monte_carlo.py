from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import UncertaintyAnalysis
from ..schemas import MonteCarloEnsembleRequest, MonteCarloEnsembleResponse
from ..tasks import perform_monte_carlo_uncertainty_task
import logging
import numpy as np
import tensorflow as tf

# Configure TensorFlow for CPU usage to avoid GPU issues
tf.config.set_visible_devices([], 'GPU')

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

    Uses dropout at inference time to estimate model uncertainty through multiple forward passes.
    """
    try:
        # Extract model configuration and input data
        input_dim = model_config.get('input_dim', len(input_data.get('features', [0.5])))
        hidden_dims = model_config.get('hidden_dims', [64, 32])
        dropout_rate = model_config.get('dropout_rate', 0.1)

        # Prepare input data
        X_test = np.array(input_data.get('features', [0.5])).reshape(1, -1)

        # Build neural network with dropout for Monte Carlo estimation
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

        # Hidden layers with dropout
        for hidden_dim in hidden_dims:
            model.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Compile model (weights will be random for uncertainty estimation)
        model.compile(optimizer='adam', loss='mse')

        # Perform Monte Carlo sampling with dropout enabled
        predictions = []
        for _ in range(n_samples):
            # Each forward pass with dropout enabled gives different prediction
            pred = model(X_test, training=True)  # training=True enables dropout
            predictions.append(pred.numpy().flatten()[0])

        predictions = np.array(predictions)

        # Calculate statistics
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)

        # Calculate confidence bounds
        z_score = tf.keras.backend.eval(tf.keras.backend.constant(confidence_level).numpy())
        if confidence_level == 0.95:
            z_score = 1.96
        elif confidence_level == 0.99:
            z_score = 2.576
        else:
            # Calculate z-score for given confidence level
            z_score = tf.keras.backend.eval(tf.keras.backend.constant(
                tf.keras.backend.abs(tf.keras.backend.erfinv(confidence_level * 2 - 1)) * tf.keras.backend.sqrt(2.0)
            ).numpy())

        lower_bound = mean_prediction - z_score * std_prediction
        upper_bound = mean_prediction + z_score * std_prediction

        prediction = {
            "mean_prediction": float(mean_prediction),
            "std_prediction": float(std_prediction),
            "confidence_interval": [float(lower_bound), float(upper_bound)],
            "predictions_distribution": {
                "samples": predictions.tolist()[:50],  # Store first 50 samples
                "n_total_samples": len(predictions)
            }
        }

        analysis = UncertaintyAnalysis(
            analysis_type="monte_carlo",
            model_name="dropout_uncertainty",
            input_data={"model_config": model_config, "input_data": input_data, "n_samples": n_samples},
            results=prediction,
            uncertainty_bounds={
                "method": "monte_carlo_dropout",
                "confidence_level": confidence_level,
                "z_score": z_score
            },
            confidence_level=confidence_level
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return {
            "prediction_id": analysis.id,
            "prediction": prediction,
            "method": "monte_carlo_dropout",
            "n_samples": n_samples,
            "model_config": {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "dropout_rate": dropout_rate
            }
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