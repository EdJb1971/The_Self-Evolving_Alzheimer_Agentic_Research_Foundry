from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# Bayesian Uncertainty Schemas
class BayesianPredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the trained Bayesian model to use")
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    confidence_level: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Confidence level for uncertainty bounds")

class BayesianPredictionResponse(BaseModel):
    prediction_id: int
    prediction: Dict[str, Any]
    uncertainty_bounds: Dict[str, Any]
    confidence_level: float
    metadata: Dict[str, Any]

class BayesianModelTrainRequest(BaseModel):
    model_name: str
    model_config: Dict[str, Any] = Field(..., description="PyMC3 model configuration")
    training_data: Dict[str, Any]
    overwrite: bool = False

class BayesianModelTrainResponse(BaseModel):
    model_id: int
    model_name: str
    status: str
    estimated_completion_time: str

# Monte Carlo Uncertainty Schemas
class MonteCarloEnsembleRequest(BaseModel):
    model_configs: List[Dict[str, Any]] = Field(..., description="List of model configurations for ensemble")
    input_data: Dict[str, Any]
    n_samples: int = Field(1000, ge=100, le=10000, description="Number of Monte Carlo samples")
    confidence_level: Optional[float] = Field(0.95, ge=0.0, le=1.0)

class MonteCarloEnsembleResponse(BaseModel):
    prediction_id: int
    ensemble_prediction: Dict[str, Any]
    uncertainty_bounds: Dict[str, Any]
    individual_predictions: List[Dict[str, Any]]
    confidence_level: float
    metadata: Dict[str, Any]

# PINN Schemas
class PINNModelingRequest(BaseModel):
    disease_model: str = Field(..., description="Type of disease model: 'alzheimer_progression', 'biomarker_trajectory', 'drug_response'")
    input_conditions: Dict[str, Any] = Field(..., description="Initial conditions and parameters")
    time_horizon: float = Field(..., gt=0, description="Time horizon for prediction (years)")
    confidence_level: Optional[float] = Field(0.95, ge=0.0, le=1.0)

class PINNModelingResponse(BaseModel):
    prediction_id: int
    disease_trajectory: Dict[str, Any]
    uncertainty_bounds: Dict[str, Any]
    key_biomarkers: List[str]
    intervention_points: List[Dict[str, Any]]
    confidence_level: float
    metadata: Dict[str, Any]

class PINNTrainRequest(BaseModel):
    model_name: str
    disease_model: str
    neural_network_config: Dict[str, Any]
    physics_constraints: Dict[str, Any]
    training_data: Dict[str, Any]
    overwrite: bool = False

class PINNTrainResponse(BaseModel):
    model_id: int
    model_name: str
    disease_model: str
    status: str
    estimated_completion_time: str

class PINNEvolutionRequest(BaseModel):
    model_name: str
    new_constraints: Dict[str, Any] = Field(..., description="New or refined biological constraints")
    new_training_data: Dict[str, Any] = Field(..., description="Additional training data")
    feedback_data: Dict[str, Any] = Field(default_factory=dict, description="Prediction accuracy feedback and validation results")

# Risk Assessment Schemas
class RiskAssessmentRequest(BaseModel):
    assessment_type: str = Field(..., description="Type: 'clinical_significance', 'false_positive_rate', 'decision_confidence'")
    research_question: str
    input_parameters: Dict[str, Any]
    clinical_thresholds: Optional[Dict[str, Any]] = None

class RiskAssessmentResponse(BaseModel):
    assessment_id: int
    risk_metrics: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    recommendations: List[str]
    clinical_significance: str
    decision_confidence: float
    metadata: Dict[str, Any]

# Error Propagation Schemas
class ErrorPropagationRequest(BaseModel):
    pipeline_steps: List[Dict[str, Any]] = Field(..., description="Sequence of analysis steps with their uncertainties")
    input_uncertainties: Dict[str, Any]
    correlation_matrix: Optional[Dict[str, Any]] = None

class ErrorPropagationResponse(BaseModel):
    propagation_id: int
    final_uncertainty: Dict[str, Any]
    step_by_step_breakdown: List[Dict[str, Any]]
    sensitivity_analysis: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any]

# Uncertainty Calibration Schemas
class UncertaintyCalibrationRequest(BaseModel):
    method_name: str
    calibration_dataset: str
    model_outputs: List[Dict[str, Any]]
    true_values: List[Dict[str, Any]]

class UncertaintyCalibrationResponse(BaseModel):
    calibration_id: int
    calibration_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    is_reliable: bool
    metadata: Dict[str, Any]