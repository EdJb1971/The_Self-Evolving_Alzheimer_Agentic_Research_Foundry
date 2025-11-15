from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class StatisticalAnalysis(Base):
    """Model for storing statistical analysis results"""
    __tablename__ = "statistical_analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), nullable=False)  # 'correlation', 'regression', 'hypothesis_test', etc.
    dataset_id = Column(String(100), nullable=False)
    variables = Column(JSON, nullable=False)  # List of variable names analyzed
    results = Column(JSON, nullable=False)  # Statistical results (p-values, effect sizes, etc.)
    confidence_intervals = Column(JSON, nullable=True)  # Confidence intervals for estimates
    created_at = Column(DateTime, default=datetime.utcnow)
    agent_id = Column(String(100), nullable=True)  # Which agent requested the analysis
    task_id = Column(String(100), nullable=True)  # Associated task ID

class DataQualityReport(Base):
    """Model for storing data quality assessment results"""
    __tablename__ = "data_quality_reports"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(String(100), nullable=False)
    report_type = Column(String(50), nullable=False)  # 'missing_data', 'outliers', 'normality', etc.

    # Missing data analysis
    missing_percentage = Column(Float, nullable=True)
    missing_patterns = Column(JSON, nullable=True)

    # Outlier analysis
    outlier_count = Column(Integer, nullable=True)
    outlier_methods = Column(JSON, nullable=True)  # Methods used for outlier detection

    # Normality tests
    normality_tests = Column(JSON, nullable=True)  # Results of normality tests

    # Correlation analysis
    correlation_matrix = Column(JSON, nullable=True)
    multicollinearity_metrics = Column(JSON, nullable=True)  # VIF scores, etc.

    overall_quality_score = Column(Float, nullable=True)  # 0-100 quality score
    recommendations = Column(JSON, nullable=True)  # Suggested improvements

    created_at = Column(DateTime, default=datetime.utcnow)
    agent_id = Column(String(100), nullable=True)

class ValidationMetric(Base):
    """Model for storing validation metrics and performance results"""
    __tablename__ = "validation_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False)  # 'classification', 'regression', 'biomarker_detection'
    dataset_id = Column(String(100), nullable=False)
    validation_method = Column(String(50), nullable=False)  # 'cross_validation', 'bootstrap', 'holdout'

    # Classification metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_roc = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)

    # Regression metrics
    mse = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)

    # Biomarker detection metrics
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    positive_predictive_value = Column(Float, nullable=True)
    negative_predictive_value = Column(Float, nullable=True)

    # Confidence intervals
    confidence_intervals = Column(JSON, nullable=True)

    # Cross-validation results
    cv_scores = Column(JSON, nullable=True)
    cv_mean = Column(Float, nullable=True)
    cv_std = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    agent_id = Column(String(100), nullable=True)
    task_id = Column(String(100), nullable=True)

class AnalysisArtifact(Base):
    """Model for storing analysis artifacts and intermediate results"""
    __tablename__ = "analysis_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    artifact_type = Column(String(50), nullable=False)  # 'plot', 'model', 'dataset', 'report'
    analysis_id = Column(Integer, ForeignKey('statistical_analyses.id'), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    analysis = relationship("StatisticalAnalysis", back_populates="artifacts")

# Add relationship to StatisticalAnalysis
StatisticalAnalysis.artifacts = relationship("AnalysisArtifact", back_populates="analysis")