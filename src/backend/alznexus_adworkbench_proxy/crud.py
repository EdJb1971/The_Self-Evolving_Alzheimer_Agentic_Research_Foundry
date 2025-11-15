from sqlalchemy.orm import Session
from . import models, schemas

def create_adworkbench_query(db: Session, query: schemas.ADWorkbenchQueryCreate):
    db_query = models.ADWorkbenchQuery(query_text=query.query_text, status="PENDING")
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def get_adworkbench_query(db: Session, query_id: int):
    return db.query(models.ADWorkbenchQuery).filter(models.ADWorkbenchQuery.id == query_id).first()

def update_adworkbench_query_status(db: Session, query_id: int, status: str, result_data: str = None):
    db_query = db.query(models.ADWorkbenchQuery).filter(models.ADWorkbenchQuery.id == query_id).first()
    if db_query:
        db_query.status = status
        if result_data:
            db_query.result_data = result_data
        db.commit()
        db.refresh(db_query)
    return db_query

def create_adworkbench_insight(db: Session, insight: schemas.InsightPublishRequest):
    db_insight = models.ADWorkbenchInsight(
        insight_name=insight.insight_name,
        insight_description=insight.insight_description,
        data_source_ids=insight.data_source_ids,
        payload=insight.payload,
        tags=insight.tags,
        status="PUBLISHED"
    )
    db.add(db_insight)
    db.commit()
    db.refresh(db_insight)
    return db_insight

def get_adworkbench_insight(db: Session, insight_id: int):
    return db.query(models.ADWorkbenchInsight).filter(models.ADWorkbenchInsight.id == insight_id).first()

def get_adworkbench_insights(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ADWorkbenchInsight).offset(skip).limit(limit).all()
