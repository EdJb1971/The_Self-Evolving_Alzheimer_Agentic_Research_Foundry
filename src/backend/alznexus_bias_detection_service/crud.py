from sqlalchemy.orm import Session
from . import models, schemas

def create_bias_detection_report(db: Session, report: schemas.BiasDetectionReportCreate):
    db_report = models.BiasDetectionReport(
        entity_type=report.entity_type,
        entity_id=report.entity_id,
        data_snapshot=report.data_snapshot,
        detected_bias=report.detected_bias,
        bias_type=report.bias_type,
        severity=report.severity,
        analysis_summary=report.analysis_summary,
        proposed_corrections=report.proposed_corrections,
        metadata_json=report.metadata_json
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_bias_detection_report(db: Session, report_id: int):
    return db.query(models.BiasDetectionReport).filter(models.BiasDetectionReport.id == report_id).first()

def get_bias_reports_by_entity(db: Session, entity_type: str, entity_id: str):
    return db.query(models.BiasDetectionReport).filter(
        models.BiasDetectionReport.entity_type == entity_type,
        models.BiasDetectionReport.entity_id == entity_id
    ).order_by(models.BiasDetectionReport.timestamp.desc()).all()

def update_bias_detection_report(db: Session, report_id: int, update_data: schemas.BiasDetectionReportCreate):
    """CQ-BIAS-001: Updates an existing bias detection report."""
    db_report = db.query(models.BiasDetectionReport).filter(models.BiasDetectionReport.id == report_id).first()
    if not db_report:
        return None

    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(db_report, field, value)

    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report
