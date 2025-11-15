from sqlalchemy.orm import Session
from . import models, schemas

def create_audit_log_entry(db: Session, log_entry: schemas.AuditLogCreate):
    db_log = models.AuditLogEntry(
        entity_type=log_entry.entity_type,
        entity_id=log_entry.entity_id,
        event_type=log_entry.event_type,
        description=log_entry.description,
        metadata_json=log_entry.metadata
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_audit_history(db: Session, entity_type: str, entity_id: str):
    return db.query(models.AuditLogEntry).filter(
        models.AuditLogEntry.entity_type == entity_type,
        models.AuditLogEntry.entity_id == entity_id
    ).order_by(models.AuditLogEntry.timestamp).all()
