from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def create_uncertainty_analysis(db: Session, analysis: schemas.UncertaintyAnalysisCreate):
    db_analysis = models.UncertaintyAnalysis(**analysis.dict())
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_uncertainty_analysis(db: Session, analysis_id: int):
    return db.query(models.UncertaintyAnalysis).filter(models.UncertaintyAnalysis.id == analysis_id).first()

def get_uncertainty_analyses_by_type(db: Session, analysis_type: str, skip: int = 0, limit: int = 100):
    return db.query(models.UncertaintyAnalysis).filter(
        models.UncertaintyAnalysis.analysis_type == analysis_type
    ).offset(skip).limit(limit).all()

def create_bayesian_model(db: Session, model: schemas.BayesianModelCreate):
    db_model = models.BayesianModel(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_bayesian_model(db: Session, model_name: str):
    return db.query(models.BayesianModel).filter(models.BayesianModel.model_name == model_name).first()

def get_active_bayesian_models(db: Session):
    return db.query(models.BayesianModel).filter(models.BayesianModel.is_active == True).all()

def create_pinn_model(db: Session, model: schemas.PINNModelCreate):
    db_model = models.PINNModel(**model.dict())
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_pinn_model(db: Session, model_name: str):
    return db.query(models.PINNModel).filter(models.PINNModel.model_name == model_name).first()

def get_converged_pinn_models(db: Session, disease_model: str = None):
    query = db.query(models.PINNModel).filter(models.PINNModel.is_converged == True)
    if disease_model:
        query = query.filter(models.PINNModel.disease_model == disease_model)
    return query.all()

def create_risk_assessment(db: Session, assessment: schemas.RiskAssessmentCreate):
    db_assessment = models.RiskAssessment(**assessment.dict())
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    return db_assessment

def get_risk_assessment(db: Session, assessment_id: int):
    return db.query(models.RiskAssessment).filter(models.RiskAssessment.id == assessment_id).first()

def get_risk_assessments_by_type(db: Session, assessment_type: str, skip: int = 0, limit: int = 100):
    return db.query(models.RiskAssessment).filter(
        models.RiskAssessment.assessment_type == assessment_type
    ).offset(skip).limit(limit).all()