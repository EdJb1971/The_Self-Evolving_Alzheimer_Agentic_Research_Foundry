from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime
from typing import List, Optional
import json

# Statistical Analysis CRUD
def create_statistical_analysis(
    db: Session,
    analysis: schemas.StatisticalAnalysisCreate
) -> models.StatisticalAnalysis:
    """Create a new statistical analysis record"""
    db_analysis = models.StatisticalAnalysis(
        analysis_type=analysis.analysis_type,
        method=analysis.method,
        parameters=json.dumps(analysis.parameters) if analysis.parameters else None,
        results=json.dumps(analysis.results) if analysis.results else None,
        confidence_level=analysis.confidence_level
    )
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)
    return db_analysis

def get_statistical_analysis(db: Session, analysis_id: int) -> Optional[models.StatisticalAnalysis]:
    """Get a statistical analysis by ID"""
    return db.query(models.StatisticalAnalysis).filter(models.StatisticalAnalysis.id == analysis_id).first()

def get_statistical_analyses(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    analysis_type: Optional[str] = None
) -> List[models.StatisticalAnalysis]:
    """Get statistical analyses with optional filtering"""
    query = db.query(models.StatisticalAnalysis)
    if analysis_type:
        query = query.filter(models.StatisticalAnalysis.analysis_type == analysis_type)
    return query.offset(skip).limit(limit).all()

# Data Quality Report CRUD
def create_data_quality_report(
    db: Session,
    report: schemas.DataQualityReportCreate
) -> models.DataQualityReport:
    """Create a new data quality report"""
    db_report = models.DataQualityReport(
        dataset_name=report.dataset_name,
        report_data=json.dumps(report.report_data) if report.report_data else None,
        quality_score=report.quality_score,
        issues_found=report.issues_found or []
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_data_quality_report(db: Session, report_id: int) -> Optional[models.DataQualityReport]:
    """Get a data quality report by ID"""
    return db.query(models.DataQualityReport).filter(models.DataQualityReport.id == report_id).first()

def get_data_quality_reports(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> List[models.DataQualityReport]:
    """Get data quality reports"""
    return db.query(models.DataQualityReport).offset(skip).limit(limit).all()

# Validation Metrics CRUD
def create_validation_metric(
    db: Session,
    metric: schemas.ValidationMetricCreate
) -> models.ValidationMetric:
    """Create a new validation metric record"""
    db_metric = models.ValidationMetric(
        metric_name=metric.metric_name,
        metric_value=metric.metric_value,
        metric_type=metric.metric_type,
        parameters=json.dumps(metric.parameters) if metric.parameters else None,
        results=json.dumps(metric.results) if metric.results else None,
        threshold=metric.threshold
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric

def get_validation_metric(db: Session, metric_id: int) -> Optional[models.ValidationMetric]:
    """Get a validation metric by ID"""
    return db.query(models.ValidationMetric).filter(models.ValidationMetric.id == metric_id).first()

def get_validation_metrics(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> List[models.ValidationMetric]:
    """Get validation metrics"""
    return db.query(models.ValidationMetric).offset(skip).limit(limit).all()

# Analysis Artifacts CRUD
def create_analysis_artifact(
    db: Session,
    artifact_type: str,
    filename: str,
    file_path: str,
    analysis_id: Optional[int] = None,
    metadata: Optional[dict] = None
) -> models.AnalysisArtifact:
    """Create a new analysis artifact record"""
    db_artifact = models.AnalysisArtifact(
        artifact_type=artifact_type,
        analysis_id=analysis_id,
        filename=filename,
        file_path=file_path,
        metadata=json.dumps(metadata) if metadata else None
    )
    db.add(db_artifact)
    db.commit()
    db.refresh(db_artifact)
    return db_artifact

def get_analysis_artifacts(db: Session, analysis_id: int) -> List[models.AnalysisArtifact]:
    """Get all artifacts for a statistical analysis"""
    return db.query(models.AnalysisArtifact).filter(models.AnalysisArtifact.analysis_id == analysis_id).all()

def get_artifact_by_id(db: Session, artifact_id: int) -> Optional[models.AnalysisArtifact]:
    """Get an artifact by ID"""
    return db.query(models.AnalysisArtifact).filter(models.AnalysisArtifact.id == artifact_id).first()