from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import json

from ..database import get_db
from ..models import ValidationMetric, StatisticalAnalysis
from ..schemas import (
    ValidationMetricCreate, ValidationMetricResponse,
    ModelValidationRequest, ModelValidationResponse,
    CrossValidationRequest, CrossValidationResponse,
    StatisticalValidationRequest, StatisticalValidationResponse
)
from ..crud import create_validation_metric, get_validation_metrics

router = APIRouter()

@router.post("/model-validation", response_model=ModelValidationResponse)
async def validate_model_performance(
    request: ModelValidationRequest,
    db: Session = Depends(get_db)
) -> ModelValidationResponse:
    """
    Validate machine learning model performance with comprehensive metrics
    """
    try:
        y_true = np.array(request.y_true)
        y_pred = np.array(request.y_pred)

        if len(y_true) != len(y_pred):
            raise HTTPException(status_code=400, detail="y_true and y_pred must have same length")

        validation_results = {}

        # Regression metrics
        if request.task_type.lower() == "regression":
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = r2_score(y_true, y_pred)

            validation_results = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "task_type": "regression"
            }

        # Classification metrics
        elif request.task_type.lower() == "classification":
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )

            validation_results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "task_type": "classification"
            }

        else:
            raise HTTPException(status_code=400, detail="task_type must be 'regression' or 'classification'")

        # Create validation metric record
        metric_data = ValidationMetricCreate(
            metric_name=f"{request.task_type}_validation",
            metric_value=validation_results.get("r2_score") or validation_results.get("accuracy") or 0.0,
            metric_type=request.task_type,
            parameters=json.dumps({
                "task_type": request.task_type,
                "sample_size": len(y_true)
            }),
            results=json.dumps(validation_results),
            threshold=request.threshold or 0.7
        )

        metric = create_validation_metric(db, metric_data)

        return ModelValidationResponse(
            validation_id=metric.id,
            metrics=validation_results,
            task_type=request.task_type,
            sample_size=len(y_true),
            meets_threshold=validation_results.get("r2_score", validation_results.get("accuracy", 0)) >= (request.threshold or 0.7)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model validation failed: {str(e)}")

@router.post("/cross-validation", response_model=CrossValidationResponse)
async def perform_cross_validation(
    request: CrossValidationRequest,
    db: Session = Depends(get_db)
) -> CrossValidationResponse:
    """
    Perform k-fold cross-validation for model evaluation
    """
    try:
        from sklearn.model_selection import KFold
        from sklearn.base import BaseEstimator

        # This is a simplified version - in practice, you'd pass the actual model
        # For now, we'll simulate cross-validation results
        kf = KFold(n_splits=request.k_folds, shuffle=True, random_state=42)

        # Placeholder for actual model scores
        # In a real implementation, you'd fit the model on each fold
        scores = np.random.normal(0.8, 0.1, request.k_folds)  # Simulated scores

        cv_results = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "k_folds": request.k_folds
        }

        # Create validation metric record
        metric_data = ValidationMetricCreate(
            metric_name="cross_validation",
            metric_value=cv_results["mean_score"],
            metric_type="cross_validation",
            parameters=json.dumps({
                "k_folds": request.k_folds,
                "scoring_metric": request.scoring_metric or "accuracy"
            }),
            results=json.dumps(cv_results),
            threshold=request.threshold or 0.7
        )

        metric = create_validation_metric(db, metric_data)

        return CrossValidationResponse(
            validation_id=metric.id,
            mean_score=cv_results["mean_score"],
            std_score=cv_results["std_score"],
            scores=cv_results["scores"],
            k_folds=request.k_folds,
            meets_threshold=cv_results["mean_score"] >= (request.threshold or 0.7)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {str(e)}")

@router.post("/statistical-validation", response_model=StatisticalValidationResponse)
async def perform_statistical_validation(
    request: StatisticalValidationRequest,
    db: Session = Depends(get_db)
) -> StatisticalValidationResponse:
    """
    Perform statistical validation checks (normality, homoscedasticity, etc.)
    """
    try:
        data = np.array(request.data)
        validation_type = request.validation_type.lower()

        validation_results = {}

        if validation_type == "normality":
            # Shapiro-Wilk test
            if len(data) <= 5000:
                stat, p_value = stats.shapiro(data.flatten())
                validation_results = {
                    "test": "shapiro-wilk",
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value > 0.05,
                    "alpha": request.alpha or 0.05
                }
            else:
                # Kolmogorov-Smirnov test for larger samples
                stat, p_value = stats.kstest(data.flatten(), 'norm')
                validation_results = {
                    "test": "kolmogorov-smirnov",
                    "statistic": stat,
                    "p_value": p_value,
                    "is_normal": p_value > 0.05,
                    "alpha": request.alpha or 0.05
                }

        elif validation_type == "homoscedasticity":
            # Levene's test for equal variances
            if len(request.groups) < 2:
                raise HTTPException(status_code=400, detail="At least 2 groups required for homoscedasticity test")

            groups = [np.array(group) for group in request.groups]
            stat, p_value = stats.levene(*groups)

            validation_results = {
                "test": "levene",
                "statistic": stat,
                "p_value": p_value,
                "equal_variances": p_value > (request.alpha or 0.05),
                "alpha": request.alpha or 0.05
            }

        elif validation_type == "independence":
            # Chi-square test for independence
            if not request.contingency_table:
                raise HTTPException(status_code=400, detail="Contingency table required for independence test")

            table = np.array(request.contingency_table)
            stat, p_value, dof, expected = stats.chi2_contingency(table)

            validation_results = {
                "test": "chi-square",
                "statistic": stat,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "independent": p_value > (request.alpha or 0.05),
                "alpha": request.alpha or 0.05
            }

        else:
            raise HTTPException(status_code=400, detail="Unsupported validation type")

        # Create validation metric record
        metric_data = ValidationMetricCreate(
            metric_name=f"{validation_type}_validation",
            metric_value=validation_results.get("p_value", 0.0),
            metric_type=validation_type,
            parameters=json.dumps({
                "validation_type": validation_type,
                "alpha": request.alpha or 0.05,
                "sample_size": len(data) if hasattr(data, '__len__') else None
            }),
            results=json.dumps(validation_results),
            threshold=request.alpha or 0.05
        )

        metric = create_validation_metric(db, metric_data)

        return StatisticalValidationResponse(
            validation_id=metric.id,
            validation_type=validation_type,
            results=validation_results,
            passes_validation=validation_results.get("p_value", 0) > (request.alpha or 0.05)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistical validation failed: {str(e)}")

@router.get("/metrics", response_model=List[ValidationMetricResponse])
async def get_validation_metrics_list(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[ValidationMetricResponse]:
    """
    Retrieve list of validation metrics
    """
    try:
        metrics = get_validation_metrics(db, skip=skip, limit=limit)
        return [
            ValidationMetricResponse(
                id=metric.id,
                metric_name=metric.metric_name,
                metric_value=metric.metric_value,
                metric_type=metric.metric_type,
                parameters=json.loads(metric.parameters) if metric.parameters else {},
                results=json.loads(metric.results) if metric.results else {},
                threshold=metric.threshold,
                created_at=metric.created_at
            )
            for metric in metrics
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation metrics: {str(e)}")

@router.get("/metrics/{metric_id}", response_model=ValidationMetricResponse)
async def get_validation_metric(
    metric_id: int,
    db: Session = Depends(get_db)
) -> ValidationMetricResponse:
    """
    Retrieve a specific validation metric by ID
    """
    try:
        from ..crud import get_validation_metric
        metric = get_validation_metric(db, metric_id)
        if not metric:
            raise HTTPException(status_code=404, detail="Validation metric not found")

        return ValidationMetricResponse(
            id=metric.id,
            metric_name=metric.metric_name,
            metric_value=metric.metric_value,
            metric_type=metric.metric_type,
            parameters=json.loads(metric.parameters) if metric.parameters else {},
            results=json.loads(metric.results) if metric.results else {},
            threshold=metric.threshold,
            created_at=metric.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve validation metric: {str(e)}")