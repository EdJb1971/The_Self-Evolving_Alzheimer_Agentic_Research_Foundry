from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class UncertaintyAnalysis(Base):
    __tablename__ = "uncertainty_analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), nullable=False)  # 'bayesian', 'monte_carlo', 'pinn', 'risk'
    model_name = Column(String(100), nullable=False)
    input_data = Column(JSON, nullable=False)
    results = Column(JSON, nullable=False)
    uncertainty_bounds = Column(JSON, nullable=True)
    confidence_level = Column(Float, default=0.95)
    computation_time = Column(Float, nullable=True)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BayesianModel(Base):
    __tablename__ = "bayesian_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), unique=True, nullable=False)
    model_config = Column(JSON, nullable=False)  # PyMC3 model configuration
    trained_parameters = Column(JSON, nullable=True)
    training_data_hash = Column(String(64), nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PINNModel(Base):
    __tablename__ = "pinn_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), unique=True, nullable=False)
    disease_model = Column(String(100), nullable=False)  # 'alzheimer_progression', 'biomarker_trajectory', etc.
    physics_constraints = Column(JSON, nullable=False)  # PDEs and boundary conditions
    neural_network_config = Column(JSON, nullable=False)
    training_history = Column(JSON, nullable=True)
    convergence_metrics = Column(JSON, nullable=True)
    is_converged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    assessment_type = Column(String(50), nullable=False)  # 'clinical_significance', 'false_positive_rate', etc.
    research_question = Column(Text, nullable=False)
    input_parameters = Column(JSON, nullable=False)
    risk_metrics = Column(JSON, nullable=False)
    confidence_intervals = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    clinical_thresholds_used = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class UncertaintyCalibration(Base):
    __tablename__ = "uncertainty_calibrations"

    id = Column(Integer, primary_key=True, index=True)
    method_name = Column(String(50), nullable=False)
    calibration_dataset = Column(String(100), nullable=False)
    calibration_results = Column(JSON, nullable=False)
    performance_metrics = Column(JSON, nullable=False)
    is_current = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)