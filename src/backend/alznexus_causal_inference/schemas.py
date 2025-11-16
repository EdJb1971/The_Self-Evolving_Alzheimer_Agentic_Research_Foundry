"""
API request and response schemas for the Causal Inference Service

Comprehensive Pydantic models for all API endpoints with validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, model_validator
import pandas as pd
import networkx as nx

# Common base schemas

class BaseResponse(BaseModel):
    """Base response schema"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """Error response schema"""
    success: bool = False
    error_code: str = ""
    error_details: Optional[Dict[str, Any]] = None

# Dataset schemas

class DatasetUploadRequest(BaseModel):
    """Dataset upload request"""
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, max_length=1000, description="Dataset description")
    data: Dict[str, List[Union[float, int, str]]] = Field(..., description="Dataset columns as key-value pairs")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('data')
    def validate_data_structure(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")

        if not isinstance(v, dict):
            raise ValueError("Data must be a dictionary")

        # Check that all values are lists
        for key, value in v.items():
            if not isinstance(value, list):
                raise ValueError(f"Column '{key}' must be a list")
            if len(value) == 0:
                raise ValueError(f"Column '{key}' cannot be empty")

        return v

    @validator('data')
    def validate_data_consistency(cls, v):
        """Validate that all columns have the same length"""
        if not v:
            return v

        lengths = [len(values) for values in v.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All columns must have the same length")

        return v

    @validator('data')
    def validate_column_names(cls, v):
        """Validate column names"""
        if not v:
            return v

        for column_name in v.keys():
            if not isinstance(column_name, str):
                raise ValueError("Column names must be strings")
            if not column_name.strip():
                raise ValueError("Column names cannot be empty")
            if len(column_name) > 255:
                raise ValueError("Column names must be less than 255 characters")

        return v

class DatasetInfo(BaseModel):
    """Dataset information response"""
    dataset_id: str
    name: str
    description: Optional[str]
    columns: List[str]
    shape: List[int]  # [n_rows, n_cols]
    metadata: Dict[str, Any]
    uploaded_at: datetime
    data_types: Dict[str, str]  # Column data types

class DatasetListResponse(BaseResponse):
    """Dataset list response"""
    datasets: List[DatasetInfo]
    total_count: int

# Causal discovery schemas

class CausalDiscoveryRequest(BaseModel):
    """Causal discovery request"""
    dataset_id: str = Field(..., description="ID of uploaded dataset")
    algorithm: str = Field("pc", description="Discovery algorithm", pattern="^(pc|fci|ges)$")
    target_variables: Optional[List[str]] = Field(None, description="Variables to focus on")
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level for independence tests")
    max_degree: int = Field(5, gt=0, le=20, description="Maximum degree for skeleton search")
    n_bootstrap: int = Field(100, gt=0, le=1000, description="Number of bootstrap samples")
    biological_constraints: Optional[Dict[str, Any]] = Field(None, description="Biological knowledge constraints")

    @validator('target_variables')
    def validate_target_variables(cls, v, values):
        if v is not None and not isinstance(v, list):
            raise ValueError("target_variables must be a list")
        return v

class CausalGraphResponse(BaseModel):
    """Causal graph response"""
    graph_id: str
    algorithm: str
    variables: List[str]
    edges: List[List[str]]  # [source, target] pairs
    confidence_scores: Dict[str, float]
    bootstrap_stability: Dict[str, float]
    mechanistic_scores: Dict[str, float]
    is_dag: bool
    dataset_size: int
    graph_properties: Dict[str, Any]  # Additional graph statistics

class CausalDiscoveryResponse(BaseResponse):
    """Causal discovery response"""
    task_id: str
    estimated_duration: str = "5-15 minutes"  # Depending on dataset size

# Causal effect estimation schemas

class CausalEffectRequest(BaseModel):
    """Causal effect estimation request"""
    dataset_id: str = Field(..., description="ID of uploaded dataset")
    treatment: str = Field(..., description="Treatment variable name")
    outcome: str = Field(..., description="Outcome variable name")
    confounders: List[str] = Field(..., description="Confounder variable names")
    method: str = Field("auto", description="Estimation method", pattern="^(auto|backdoor|meta_learner|doubly_robust)$")
    analyze_heterogeneity: bool = Field(False, description="Analyze treatment effect heterogeneity")
    analyze_mediation: bool = Field(False, description="Analyze mediation effects")
    mediators: Optional[List[str]] = Field(None, description="Mediator variables for mediation analysis")
    moderators: Optional[List[str]] = Field(None, description="Moderator variables for heterogeneity analysis")

    @model_validator(mode='after')
    def validate_variables(self):
        """Validate that treatment, outcome, and confounders are different"""
        if self.treatment and self.outcome and self.treatment == self.outcome:
            raise ValueError("Treatment and outcome variables must be different")

        if self.treatment and self.treatment in self.confounders:
            raise ValueError("Treatment variable cannot be in confounders")

        if self.outcome and self.outcome in self.confounders:
            raise ValueError("Outcome variable cannot be in confounders")

        return self

class CausalEffectResultResponse(BaseModel):
    """Causal effect result response"""
    effect_id: str
    treatment_variable: str
    outcome_variable: str
    confounder_variables: List[str]
    estimator_used: str
    identification_strategy: str
    effect_estimate: float
    confidence_interval: List[float]
    p_value: Optional[float]
    standard_error: Optional[float]
    robustness_score: float
    sample_size: int
    refutation_results: Dict[str, Any]
    heterogeneous_effects: Optional[Dict[str, Any]]
    mediation_analysis: Optional[Dict[str, Any]]
    summary: str

class CausalEffectEstimationResponse(BaseResponse):
    """Causal effect estimation response"""
    task_id: str
    estimated_duration: str = "2-10 minutes"

# Mechanistic modeling schemas

class MechanisticModelingRequest(BaseModel):
    """Mechanistic modeling request"""
    causal_graph_id: str = Field(..., description="ID of causal graph")
    disease_context: str = Field("Alzheimer", description="Disease context for pathway selection")
    include_pathways: Optional[List[str]] = Field(None, description="Specific pathways to include")
    exclude_pathways: Optional[List[str]] = Field(None, description="Pathways to exclude")

class MechanisticModelResponse(BaseModel):
    """Mechanistic model response"""
    model_id: str
    causal_graph_id: str
    disease_context: str
    biological_pathways: List[Dict[str, Any]]
    mechanistic_scores: Dict[str, float]
    pathway_coverage: Dict[str, float]
    validation_results: Dict[str, Any]
    integrated_graph_stats: Dict[str, Any]

class MechanisticModelingResponse(BaseResponse):
    """Mechanistic modeling response"""
    task_id: str
    estimated_duration: str = "3-8 minutes"

# Intervention simulation schemas

class InterventionSimulationRequest(BaseModel):
    """Intervention simulation request"""
    mechanistic_model_id: str = Field(..., description="ID of mechanistic model")
    intervention: Dict[str, Union[float, int]] = Field(..., description="Intervention parameters")
    time_horizon: float = Field(10.0, gt=0, le=1000, description="Simulation time horizon")
    initial_conditions: Optional[Dict[str, Union[float, int]]] = Field(None, description="Initial conditions")
    simulation_method: str = Field("mechanistic", description="Simulation method", pattern="^(mechanistic|pinn|hybrid)$")

    @validator('intervention')
    def validate_intervention(cls, v):
        if not v:
            raise ValueError("Intervention cannot be empty")

        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("Intervention keys must be strings")
            if not isinstance(value, (int, float)):
                raise ValueError("Intervention values must be numeric")

        return v

class SimulationResultResponse(BaseModel):
    """Simulation result response"""
    simulation_id: str
    mechanistic_model_id: str
    intervention: Dict[str, float]
    time_horizon: float
    initial_conditions: Dict[str, float]
    time_points: List[float]
    trajectories: Dict[str, List[float]]  # Variable trajectories over time
    key_metrics: Dict[str, Any]  # Summary statistics

class InterventionSimulationResponse(BaseResponse):
    """Intervention simulation response"""
    task_id: str
    estimated_duration: str = "1-5 minutes"

# Counterfactual analysis schemas

class CounterfactualAnalysisRequest(BaseModel):
    """Counterfactual analysis request"""
    mechanistic_model_id: str = Field(..., description="ID of mechanistic model")
    observed_data_id: str = Field(..., description="ID of observed data")
    hypothetical_intervention: Dict[str, Union[float, int]] = Field(..., description="Hypothetical intervention")
    comparison_metrics: Optional[List[str]] = Field(None, description="Metrics to compare")

class CounterfactualResultResponse(BaseModel):
    """Counterfactual result response"""
    analysis_id: str
    mechanistic_model_id: str
    observed_data_id: str
    hypothetical_intervention: Dict[str, float]
    factual_trajectory: Dict[str, List[float]]
    counterfactual_trajectory: Dict[str, List[float]]
    counterfactual_effects: Dict[str, Dict[str, Any]]
    key_insights: List[str]

class CounterfactualAnalysisResponse(BaseResponse):
    """Counterfactual analysis response"""
    task_id: str
    estimated_duration: str = "2-7 minutes"

# Task management schemas

class TaskStatus(BaseModel):
    """Task status response"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    estimated_completion: Optional[datetime] = None

class TaskListResponse(BaseResponse):
    """Task list response"""
    tasks: List[TaskStatus]
    total_count: int
    active_count: int

# Result retrieval schemas

class ResultMetadata(BaseModel):
    """Result metadata"""
    result_id: str
    result_type: str  # causal_graph, causal_effect, mechanistic_model, etc.
    created_at: datetime
    size_bytes: Optional[int] = None
    description: str = ""

class ResultListResponse(BaseResponse):
    """Result list response"""
    results: List[ResultMetadata]
    total_count: int

# Validation and utility schemas

class ValidationRequest(BaseModel):
    """Validation request"""
    data: Dict[str, Any] = Field(..., description="Data to validate")
    validation_type: str = Field(..., description="Type of validation", pattern="^(causal_graph|dataset|intervention)$")

class ValidationResponse(BaseResponse):
    """Validation response"""
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

# Batch processing schemas

class BatchRequest(BaseModel):
    """Batch processing request"""
    requests: List[Dict[str, Any]] = Field(..., description="List of individual requests")
    parallel_processing: bool = Field(True, description="Process requests in parallel")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent requests")

class BatchResponse(BaseResponse):
    """Batch processing response"""
    batch_id: str
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

# Export schemas

class ExportRequest(BaseModel):
    """Export request"""
    result_ids: List[str] = Field(..., description="IDs of results to export")
    export_format: str = Field("json", description="Export format", pattern="^(json|csv|pdf|png)$")
    include_metadata: bool = Field(True, description="Include metadata in export")

class ExportResponse(BaseResponse):
    """Export response"""
    export_id: str
    export_format: str
    download_url: str
    expires_at: datetime

# Health and monitoring schemas

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime_seconds: float
    timestamp: datetime
    services: Dict[str, str]  # Status of individual services
    metrics: Dict[str, Any] = Field(default_factory=dict)

class MetricsResponse(BaseModel):
    """Metrics response"""
    timestamp: datetime
    time_range: str  # e.g., "last_24h", "last_7d"
    metrics: Dict[str, Any]
    aggregations: Dict[str, Any] = Field(default_factory=dict)

# Configuration schemas

class ServiceConfig(BaseModel):
    """Service configuration"""
    max_dataset_size: int = Field(100000, description="Maximum dataset size (rows)")
    max_concurrent_tasks: int = Field(10, description="Maximum concurrent tasks")
    cache_ttl_seconds: int = Field(3600, description="Cache time-to-live in seconds")
    enable_biological_validation: bool = Field(True, description="Enable biological validation")
    default_bootstrap_samples: int = Field(100, description="Default bootstrap samples")

class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    config: ServiceConfig
    restart_required: bool = False

# Utility functions

def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        success=False,
        message=message,
        error_code=error_code,
        error_details=details
    )

def validate_dataset_compatibility(dataset_id: str, required_columns: List[str]) -> bool:
    """Validate that a dataset has required columns"""
    # This would check against stored dataset metadata
    # For now, return True as placeholder
    return True

def estimate_task_duration(request_type: str, dataset_size: int) -> str:
    """Estimate task duration based on request type and data size"""
    base_times = {
        "causal_discovery": 300,  # 5 minutes
        "causal_effect": 120,     # 2 minutes
        "mechanistic_model": 180, # 3 minutes
        "intervention_simulate": 60,  # 1 minute
        "counterfactual_analyze": 150  # 2.5 minutes
    }

    base_time = base_times.get(request_type, 300)
    # Scale by dataset size (rough heuristic)
    scale_factor = min(max(dataset_size / 1000, 0.5), 5.0)
    estimated_seconds = base_time * scale_factor

    if estimated_seconds < 60:
        return f"{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        return f"{int(estimated_seconds / 60)} minutes"
    else:
        return f"{int(estimated_seconds / 3600)} hours"