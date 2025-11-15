from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Statistical Analysis Schemas
class StatisticalAnalysisRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of statistical analysis")
    data: Dict[str, Any] = Field(..., description="Input data for analysis")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Analysis parameters")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    agent_id: Optional[str] = Field(default=None, description="Requesting agent ID")
    task_id: Optional[str] = Field(default=None, description="Associated task ID")

class StatisticalAnalysisResponse(BaseModel):
    analysis_id: int
    analysis_type: str
    results: Dict[str, Any]
    confidence_intervals: Optional[Dict[str, Any]] = None
    interpretation: Optional[str] = None
    warnings: Optional[List[str]] = None
    created_at: datetime

# Data Quality Schemas
class DataQualityRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Dataset to analyze")
    checks: List[str] = Field(..., description="Quality checks to perform")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    agent_id: Optional[str] = Field(default=None, description="Requesting agent ID")

class MissingDataAnalysis(BaseModel):
    total_missing: int
    missing_percentage: float
    missing_by_column: Dict[str, int]
    missing_patterns: Dict[str, Any]

class OutlierAnalysis(BaseModel):
    outlier_count: int
    outlier_percentage: float
    outlier_indices: List[int]
    outlier_methods_used: List[str]

class NormalityTest(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_normal: bool

class CorrelationAnalysis(BaseModel):
    correlation_matrix: Dict[str, Dict[str, float]]
    significant_correlations: List[Dict[str, Any]]
    multicollinearity_warnings: List[str]

class DataQualityResponse(BaseModel):
    report_id: int
    dataset_id: str
    overall_quality_score: float
    missing_data: Optional[MissingDataAnalysis] = None
    outliers: Optional[OutlierAnalysis] = None
    normality_tests: Optional[List[NormalityTest]] = None
    correlations: Optional[CorrelationAnalysis] = None
    recommendations: List[str]
    created_at: datetime

# Validation Metrics Schemas
class ValidationRequest(BaseModel):
    model_type: str = Field(..., description="Type of model/validation")
    data: Dict[str, Any] = Field(..., description="Dataset and predictions")
    method: str = Field(default="cross_validation", description="Validation method")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Validation parameters")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    agent_id: Optional[str] = Field(default=None, description="Requesting agent ID")
    task_id: Optional[str] = Field(default=None, description="Associated task ID")

class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: List[List[int]]
    class_report: Dict[str, Any]

class RegressionMetrics(BaseModel):
    mse: float
    rmse: float
    mae: float
    r_squared: float
    explained_variance: float

class BiomarkerMetrics(BaseModel):
    sensitivity: float
    specificity: float
    positive_predictive_value: float
    negative_predictive_value: float
    diagnostic_accuracy: float
    likelihood_ratios: Dict[str, float]

class CrossValidationResults(BaseModel):
    scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Optional[Dict[str, float]] = None

class ValidationResponse(BaseModel):
    validation_id: int
    model_type: str
    method: str
    classification_metrics: Optional[ClassificationMetrics] = None
    regression_metrics: Optional[RegressionMetrics] = None
    biomarker_metrics: Optional[BiomarkerMetrics] = None
    cross_validation: Optional[CrossValidationResults] = None
    confidence_intervals: Optional[Dict[str, Any]] = None
    interpretation: Optional[str] = None
    created_at: datetime

# Hypothesis Testing Schemas
class HypothesisTestRequest(BaseModel):
    test_type: str = Field(..., description="Type of hypothesis test")
    data: Dict[str, Any] = Field(..., description="Data for testing")
    null_hypothesis: str = Field(..., description="Null hypothesis statement")
    alternative_hypothesis: str = Field(..., description="Alternative hypothesis statement")
    alpha: float = Field(default=0.05, description="Significance level")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Test parameters")

class HypothesisTestResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    reject_null: bool
    interpretation: str

class HypothesisTestResponse(BaseModel):
    test_id: int
    test_type: str
    results: HypothesisTestResult
    multiple_testing_correction: Optional[str] = None
    corrected_p_value: Optional[float] = None
    created_at: datetime

# Power Analysis Schemas
class PowerAnalysisRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of power analysis")
    effect_size: float = Field(..., description="Expected effect size")
    alpha: float = Field(default=0.05, description="Significance level")
    power: float = Field(default=0.80, description="Desired statistical power")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Analysis parameters")

class PowerAnalysisResponse(BaseModel):
    analysis_id: int
    required_sample_size: int
    achieved_power: float
    effect_size: float
    alpha: float
    recommendations: List[str]
    created_at: datetime

# Effect Size Schemas
class EffectSizeRequest(BaseModel):
    effect_type: str = Field(..., description="Type of effect size")
    data: Dict[str, Any] = Field(..., description="Data for effect size calculation")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Calculation parameters")

class EffectSizeResponse(BaseModel):
    effect_size_id: int
    effect_type: str
    value: float
    confidence_interval: Optional[Dict[str, float]] = None
    interpretation: str
    magnitude: str  # 'small', 'medium', 'large'
    created_at: datetime

# Additional Schemas for Router Endpoints
class CorrelationAnalysisRequest(BaseModel):
    data: List[List[float]] = Field(..., description="Data matrix for correlation analysis")
    method: str = Field(default="pearson", description="Correlation method: 'pearson', 'spearman', 'kendall'")
    confidence_level: Optional[float] = Field(default=0.95, description="Confidence level for intervals")

class CorrelationAnalysisResponse(BaseModel):
    analysis_id: int
    correlation_matrix: Union[List[List[float]], Dict[str, Any]]
    p_values: Union[List[List[float]], Dict[str, Any]]
    method: str
    confidence_level: float
    sample_size: int

class StatisticalAnalysisCreate(BaseModel):
    analysis_type: str
    method: str
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    confidence_level: Optional[float] = 0.95

class StatisticalAnalysisResponse(BaseModel):
    id: int
    analysis_type: str
    method: str
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    confidence_level: float
    created_at: datetime

class DataQualityReportCreate(BaseModel):
    dataset_name: str
    report_data: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    issues_found: Optional[List[str]] = None

class DataQualityReportResponse(BaseModel):
    id: int
    dataset_name: str
    report_data: Dict[str, Any]
    quality_score: float
    issues_found: List[str]
    created_at: datetime

class ValidationMetricCreate(BaseModel):
    metric_name: str
    metric_value: float
    metric_type: str
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None

class ValidationMetricResponse(BaseModel):
    id: int
    metric_name: str
    metric_value: float
    metric_type: str
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    threshold: float
    created_at: datetime

class ModelValidationRequest(BaseModel):
    y_true: List[float]
    y_pred: List[float]
    task_type: str = Field(..., description="Task type: 'regression' or 'classification'")
    threshold: Optional[float] = Field(default=0.7, description="Performance threshold")

class ModelValidationResponse(BaseModel):
    validation_id: int
    metrics: Dict[str, Any]
    task_type: str
    sample_size: int
    meets_threshold: bool

class CrossValidationRequest(BaseModel):
    k_folds: int = Field(default=5, description="Number of cross-validation folds")
    scoring_metric: Optional[str] = Field(default="accuracy", description="Scoring metric")
    threshold: Optional[float] = Field(default=0.7, description="Performance threshold")

class CrossValidationResponse(BaseModel):
    validation_id: int
    mean_score: float
    std_score: float
    scores: List[float]
    k_folds: int
    meets_threshold: bool

class StatisticalValidationRequest(BaseModel):
    data: List[float]
    validation_type: str = Field(..., description="Validation type: 'normality', 'homoscedasticity', 'independence'")
    alpha: Optional[float] = Field(default=0.05, description="Significance level")
    groups: Optional[List[List[float]]] = Field(default=None, description="Groups for homoscedasticity test")
    contingency_table: Optional[List[List[int]]] = Field(default=None, description="Contingency table for independence test")

class StatisticalValidationResponse(BaseModel):
    validation_id: int
    validation_type: str
    results: Dict[str, Any]
    passes_validation: bool

class AnalysisSummaryRequest(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class AnalysisSummaryResponse(BaseModel):
    summary: Dict[str, Any]
    start_date: datetime
    end_date: datetime

class ReportGenerationRequest(BaseModel):
    report_title: Optional[str] = None
    start_date: datetime
    end_date: datetime

class ReportGenerationResponse(BaseModel):
    report_id: str
    report_data: Dict[str, Any]
    generated_at: datetime