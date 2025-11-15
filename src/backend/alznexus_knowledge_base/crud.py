from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from . import models, schemas

def upsert_knowledge_document(db: Session, document: schemas.KnowledgeDocumentUpsert) -> models.KnowledgeDocument:
    """Upsert a knowledge document - update if exists and version is newer, insert if not."""
    # Find existing document by unique combination (title + source_agent + source_task_id)
    existing_document = db.query(models.KnowledgeDocument).filter(
        and_(
            models.KnowledgeDocument.title == document.title,
            models.KnowledgeDocument.source_agent == document.source_agent,
            models.KnowledgeDocument.source_task_id == document.source_task_id
        )
    ).first()

    if existing_document:
        # Check version to prevent old data from overwriting new data
        if document.version <= existing_document.version:
            # Return existing document without changes - new data is not newer
            return existing_document

        # Update existing document with new data
        update_data = document.dict(exclude_unset=True)
        update_data.pop('version', None)  # Don't update version directly

        # Increment version
        update_data['version'] = existing_document.version + 1
        update_data['updated_at'] = datetime.utcnow()

        for field, value in update_data.items():
            if hasattr(existing_document, field):
                setattr(existing_document, field, value)

        db.commit()
        db.refresh(existing_document)

        # Log the update
        create_knowledge_update(
            db,
            schemas.KnowledgeUpdateCreate(
                operation_type="update",
                document_id=existing_document.id,
                change_description=f"Updated {document.document_type} document: {document.title} (version {existing_document.version})",
                performed_by=document.last_modified_by or document.source_agent,
                old_metadata={"version": existing_document.version - 1},
                new_metadata={"version": existing_document.version}
            )
        )

        return existing_document
    else:
        # Create new document
        db_document = models.KnowledgeDocument(**document.dict())
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        # Log the creation
        create_knowledge_update(
            db,
            schemas.KnowledgeUpdateCreate(
                operation_type="add",
                document_id=db_document.id,
                change_description=f"Created new {document.document_type} document: {document.title}",
                performed_by=document.last_modified_by or document.source_agent
            )
        )

        return db_document

def get_knowledge_document(db: Session, document_id: int) -> Optional[models.KnowledgeDocument]:
    """Get a knowledge document by ID."""
    return db.query(models.KnowledgeDocument).filter(models.KnowledgeDocument.id == document_id).first()

def get_knowledge_documents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    document_type: Optional[str] = None,
    source_agent: Optional[str] = None,
    is_validated: Optional[bool] = None
) -> List[models.KnowledgeDocument]:
    """Get knowledge documents with optional filtering."""
    query = db.query(models.KnowledgeDocument)

    if document_type:
        query = query.filter(models.KnowledgeDocument.document_type == document_type)
    if source_agent:
        query = query.filter(models.KnowledgeDocument.source_agent == source_agent)
    if is_validated is not None:
        query = query.filter(models.KnowledgeDocument.is_validated == is_validated)

    return query.offset(skip).limit(limit).all()

def update_knowledge_document(
    db: Session,
    document_id: int,
    updates: schemas.KnowledgeDocumentUpdate
) -> Optional[models.KnowledgeDocument]:
    """Update a knowledge document."""
    db_document = get_knowledge_document(db, document_id)
    if not db_document:
        return None

    update_data = updates.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_document, field, value)

    db.commit()
    db.refresh(db_document)
    return db_document

def delete_knowledge_document(db: Session, document_id: int) -> bool:
    """Delete a knowledge document."""
    db_document = get_knowledge_document(db, document_id)
    if not db_document:
        return False

    db.delete(db_document)
    db.commit()
    return True

def create_document_chunk(db: Session, chunk: schemas.DocumentChunkCreate) -> models.DocumentChunk:
    """Create a new document chunk."""
    db_chunk = models.DocumentChunk(**chunk.dict())
    db.add(db_chunk)
    db.commit()
    db.refresh(db_chunk)
    return db_chunk

def get_document_chunks_by_document(db: Session, document_id: int) -> List[models.DocumentChunk]:
    """Get all chunks for a document."""
    return db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document_id).order_by(models.DocumentChunk.chunk_index).all()

def create_knowledge_query(db: Session, query: schemas.KnowledgeQueryCreate) -> models.KnowledgeQuery:
    """Create a knowledge query record."""
    db_query = models.KnowledgeQuery(**query.dict())
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def get_recent_knowledge_queries(db: Session, days: int = 7) -> List[models.KnowledgeQuery]:
    """Get recent knowledge queries."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    return db.query(models.KnowledgeQuery).filter(models.KnowledgeQuery.created_at >= cutoff_date).all()

def create_knowledge_update(db: Session, update: schemas.KnowledgeUpdateCreate) -> models.KnowledgeUpdate:
    """Create a knowledge update record."""
    db_update = models.KnowledgeUpdate(**update.dict())
    db.add(db_update)
    db.commit()
    db.refresh(db_update)
    return db_update

def get_knowledge_updates(
    db: Session,
    operation_type: Optional[str] = None,
    performed_by: Optional[str] = None,
    days: int = 30
) -> List[models.KnowledgeUpdate]:
    """Get knowledge updates with optional filtering."""
    query = db.query(models.KnowledgeUpdate)
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    query = query.filter(models.KnowledgeUpdate.created_at >= cutoff_date)

    if operation_type:
        query = query.filter(models.KnowledgeUpdate.operation_type == operation_type)
    if performed_by:
        query = query.filter(models.KnowledgeUpdate.performed_by == performed_by)

    return query.order_by(models.KnowledgeUpdate.created_at.desc()).all()

def create_research_insight(db: Session, insight: schemas.ResearchInsightCreate) -> models.ResearchInsight:
    """Create a new research insight."""
    db_insight = models.ResearchInsight(**insight.dict())
    db.add(db_insight)
    db.commit()
    db.refresh(db_insight)
    return db_insight

def get_research_insight(db: Session, insight_id: int) -> Optional[models.ResearchInsight]:
    """Get a research insight by ID."""
    return db.query(models.ResearchInsight).filter(models.ResearchInsight.id == insight_id).first()

def get_research_insights(
    db: Session,
    insight_type: Optional[str] = None,
    validation_status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50
) -> List[models.ResearchInsight]:
    """Get research insights with optional filtering."""
    query = db.query(models.ResearchInsight)

    if insight_type:
        query = query.filter(models.ResearchInsight.insight_type == insight_type)
    if validation_status:
        query = query.filter(models.ResearchInsight.validation_status == validation_status)
    if min_confidence:
        query = query.filter(models.ResearchInsight.confidence_level >= min_confidence)

    return query.order_by(models.ResearchInsight.discovery_date.desc()).limit(limit).all()

def update_research_insight(
    db: Session,
    insight_id: int,
    updates: schemas.ResearchInsightUpdate
) -> Optional[models.ResearchInsight]:
    """Update a research insight."""
    db_insight = get_research_insight(db, insight_id)
    if not db_insight:
        return None

    update_data = updates.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_insight, field, value)

    if 'validation_status' in update_data and update_data['validation_status'] == 'validated':
        db_insight.last_validated = datetime.utcnow()

    db.commit()
    db.refresh(db_insight)
    return db_insight

def create_learning_pattern(db: Session, pattern: schemas.LearningPatternCreate) -> models.LearningPattern:
    """Create a new learning pattern."""
    db_pattern = models.LearningPattern(**pattern.dict())
    db.add(db_pattern)
    db.commit()
    db.refresh(db_pattern)
    return db_pattern

def get_learning_pattern(db: Session, pattern_id: int) -> Optional[models.LearningPattern]:
    """Get a learning pattern by ID."""
    return db.query(models.LearningPattern).filter(models.LearningPattern.id == pattern_id).first()

def get_learning_patterns(
    db: Session,
    pattern_type: Optional[str] = None,
    min_success_rate: Optional[float] = None,
    limit: int = 20
) -> List[models.LearningPattern]:
    """Get learning patterns with optional filtering."""
    query = db.query(models.LearningPattern)

    if pattern_type:
        query = query.filter(models.LearningPattern.pattern_type == pattern_type)
    if min_success_rate:
        query = query.filter(models.LearningPattern.success_rate >= min_success_rate)

    return query.order_by(models.LearningPattern.success_rate.desc()).limit(limit).all()

def update_learning_pattern_success(
    db: Session,
    pattern_id: int,
    success: bool
) -> Optional[models.LearningPattern]:
    """Update learning pattern success rate and application count."""
    db_pattern = get_learning_pattern(db, pattern_id)
    if not db_pattern:
        return None

    db_pattern.application_count += 1
    db_pattern.last_applied = datetime.utcnow()

    # Update success rate using exponential moving average
    alpha = 0.1  # Learning rate
    current_rate = 1.0 if success else 0.0
    db_pattern.success_rate = (1 - alpha) * db_pattern.success_rate + alpha * current_rate

    db.commit()
    db.refresh(db_pattern)
    return db_pattern

def get_knowledge_analytics(db: Session) -> Dict[str, Any]:
    """Get comprehensive knowledge base analytics."""
    # Total counts
    total_documents = db.query(func.count(models.KnowledgeDocument.id)).scalar()
    total_chunks = db.query(func.count(models.DocumentChunk.id)).scalar()

    # Document type distribution
    doc_types = db.query(
        models.KnowledgeDocument.document_type,
        func.count(models.KnowledgeDocument.id).label('count')
    ).group_by(models.KnowledgeDocument.document_type).all()
    document_types_distribution = {doc_type: count for doc_type, count in doc_types}

    # Validation status distribution
    validation_status = db.query(
        models.KnowledgeDocument.is_validated,
        func.count(models.KnowledgeDocument.id).label('count')
    ).group_by(models.KnowledgeDocument.is_validated).all()
    validation_status_distribution = {
        'validated' if validated else 'unvalidated': count
        for validated, count in validation_status
    }

    # Top contributing agents
    top_agents = db.query(
        models.KnowledgeDocument.source_agent,
        func.count(models.KnowledgeDocument.id).label('count')
    ).group_by(models.KnowledgeDocument.source_agent).order_by(func.count(models.KnowledgeDocument.id).desc()).limit(5).all()
    top_contributing_agents = [
        {'agent': agent, 'documents': count}
        for agent, count in top_agents
    ]

    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_updates = db.query(models.KnowledgeUpdate).filter(
        models.KnowledgeUpdate.created_at >= week_ago
    ).order_by(models.KnowledgeUpdate.created_at.desc()).limit(10).all()

    recent_activity = [
        {
            'operation': update.operation_type,
            'description': update.change_description,
            'performed_by': update.performed_by,
            'timestamp': update.created_at.isoformat()
        }
        for update in recent_updates
    ]

    # Knowledge growth trend (last 30 days, daily)
    growth_trend = []
    for i in range(30):
        date = datetime.utcnow() - timedelta(days=i)
        next_date = date + timedelta(days=1)
        daily_count = db.query(func.count(models.KnowledgeDocument.id)).filter(
            and_(
                models.KnowledgeDocument.created_at >= date.replace(hour=0, minute=0, second=0, microsecond=0),
                models.KnowledgeDocument.created_at < next_date.replace(hour=0, minute=0, second=0, microsecond=0)
            )
        ).scalar()
        growth_trend.append({
            'date': date.date().isoformat(),
            'documents_added': daily_count
        })

    return {
        'total_documents': total_documents,
        'total_chunks': total_chunks,
        'document_types_distribution': document_types_distribution,
        'validation_status_distribution': validation_status_distribution,
        'top_contributing_agents': top_contributing_agents,
        'recent_activity': recent_activity,
        'knowledge_growth_trend': list(reversed(growth_trend))  # Most recent first
    }