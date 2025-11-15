from sqlalchemy.orm import Session
from . import models, schemas

def create_llm_request_log(db: Session, log_entry: schemas.LLMRequestLogCreate):
    db_log = models.LLMRequestLog(
        model_name=log_entry.model_name,
        prompt=log_entry.prompt,
        response=log_entry.response,
        request_type=log_entry.request_type,
        detected_bias=log_entry.detected_bias,
        detected_injection=log_entry.detected_injection,
        ethical_flags=log_entry.ethical_flags,
        metadata_json=log_entry.metadata_json
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_llm_request_log(db: Session, log_id: int):
    return db.query(models.LLMRequestLog).filter(models.LLMRequestLog.id == log_id).first()

def get_all_llm_request_logs(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.LLMRequestLog).offset(skip).limit(limit).all()
