from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from . import models, schemas

# Agent Performance CRUD
def create_agent_performance(db: Session, performance: schemas.AgentPerformanceCreate) -> models.AgentPerformance:
    """Create a new agent performance record"""
    db_performance = models.AgentPerformance(**performance.dict())
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    return db_performance

def get_agent_performance(db: Session, performance_id: int) -> Optional[models.AgentPerformance]:
    """Get agent performance by ID"""
    return db.query(models.AgentPerformance).filter(models.AgentPerformance.id == performance_id).first()

def get_agent_performances(
    db: Session,
    agent_id: Optional[str] = None,
    task_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[models.AgentPerformance]:
    """Get agent performances with optional filtering"""
    query = db.query(models.AgentPerformance)
    if agent_id:
        query = query.filter(models.AgentPerformance.agent_id == agent_id)
    if task_type:
        query = query.filter(models.AgentPerformance.task_type == task_type)
    return query.order_by(desc(models.AgentPerformance.timestamp)).offset(skip).limit(limit).all()

def get_agent_performance_stats(db: Session, agent_id: str) -> Dict[str, Any]:
    """Get performance statistics for an agent"""
    performances = db.query(models.AgentPerformance).filter(
        models.AgentPerformance.agent_id == agent_id
    ).all()

    if not performances:
        return {"total_tasks": 0, "avg_success_rate": 0.0, "avg_execution_time": 0.0}

    total_tasks = len(performances)
    avg_success = sum(p.success_rate for p in performances) / total_tasks
    avg_time = sum(p.execution_time for p in performances) / total_tasks

    return {
        "total_tasks": total_tasks,
        "avg_success_rate": round(avg_success, 3),
        "avg_execution_time": round(avg_time, 2),
        "latest_task": performances[0].timestamp if performances else None
    }

# Learning Pattern CRUD
def create_learning_pattern(db: Session, pattern: schemas.LearningPatternCreate) -> models.LearningPattern:
    """Create a new learning pattern"""
    db_pattern = models.LearningPattern(**pattern.dict())
    db.add(db_pattern)
    db.commit()
    db.refresh(db_pattern)
    return db_pattern

def get_learning_pattern(db: Session, pattern_id: int) -> Optional[models.LearningPattern]:
    """Get learning pattern by ID"""
    return db.query(models.LearningPattern).filter(models.LearningPattern.id == pattern_id).first()

def get_learning_patterns(
    db: Session,
    pattern_type: Optional[str] = None,
    domain: Optional[str] = None,
    validation_status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    skip: int = 0,
    limit: int = 100
) -> List[models.LearningPattern]:
    """Get learning patterns with optional filtering"""
    query = db.query(models.LearningPattern)
    if pattern_type:
        query = query.filter(models.LearningPattern.pattern_type == pattern_type)
    if domain:
        query = query.filter(models.LearningPattern.domain == domain)
    if validation_status:
        query = query.filter(models.LearningPattern.validation_status == validation_status)
    if min_confidence:
        query = query.filter(models.LearningPattern.confidence >= min_confidence)
    return query.order_by(desc(models.LearningPattern.discovered_at)).offset(skip).limit(limit).all()

def update_pattern_validation(db: Session, pattern_id: int, validation_status: str) -> Optional[models.LearningPattern]:
    """Update pattern validation status"""
    pattern = db.query(models.LearningPattern).filter(models.LearningPattern.id == pattern_id).first()
    if pattern:
        pattern.validation_status = validation_status
        pattern.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(pattern)
    return pattern

def get_related_patterns(db: Session, pattern_id: int) -> List[models.LearningPattern]:
    """Get patterns related to the given pattern"""
    pattern = get_learning_pattern(db, pattern_id)
    if not pattern or not pattern.related_patterns:
        return []

    related_ids = pattern.related_patterns
    return db.query(models.LearningPattern).filter(models.LearningPattern.id.in_(related_ids)).all()

# Context Enrichment CRUD
def create_context_enrichment(db: Session, enrichment: schemas.ContextEnrichmentCreate) -> models.ContextEnrichment:
    """Create a new context enrichment record"""
    db_enrichment = models.ContextEnrichment(**enrichment.dict())
    db.add(db_enrichment)
    db.commit()
    db.refresh(db_enrichment)
    return db_enrichment

def get_context_enrichment(db: Session, enrichment_id: int) -> Optional[models.ContextEnrichment]:
    """Get context enrichment by ID"""
    return db.query(models.ContextEnrichment).filter(models.ContextEnrichment.id == enrichment_id).first()

def get_agent_context_history(
    db: Session,
    agent_id: str,
    skip: int = 0,
    limit: int = 50
) -> List[models.ContextEnrichment]:
    """Get context enrichment history for an agent"""
    return db.query(models.ContextEnrichment).filter(
        models.ContextEnrichment.agent_id == agent_id
    ).order_by(desc(models.ContextEnrichment.timestamp)).offset(skip).limit(limit).all()

# Feedback Loop CRUD
def create_feedback_loop(db: Session, loop: schemas.FeedbackLoopCreate) -> models.FeedbackLoop:
    """Create a new feedback loop"""
    db_loop = models.FeedbackLoop(**loop.dict())
    db.add(db_loop)
    db.commit()
    db.refresh(db_loop)
    return db_loop

def get_feedback_loop(db: Session, loop_id: str) -> Optional[models.FeedbackLoop]:
    """Get feedback loop by loop_id"""
    return db.query(models.FeedbackLoop).filter(models.FeedbackLoop.loop_id == loop_id).first()

def update_feedback_loop_status(
    db: Session,
    loop_id: str,
    status: str,
    end_time: Optional[datetime] = None,
    success_metric: Optional[float] = None
) -> Optional[models.FeedbackLoop]:
    """Update feedback loop status"""
    loop = get_feedback_loop(db, loop_id)
    if loop:
        loop.status = status
        if end_time:
            loop.end_time = end_time
        if success_metric is not None:
            loop.success_metric = success_metric
        db.commit()
        db.refresh(loop)
    return loop

def get_active_feedback_loops(db: Session) -> List[models.FeedbackLoop]:
    """Get all active feedback loops"""
    return db.query(models.FeedbackLoop).filter(models.FeedbackLoop.status == "active").all()

# Agent Memory CRUD
def create_agent_memory(db: Session, memory: schemas.AgentMemoryCreate) -> models.AgentMemory:
    """Create a new agent memory"""
    db_memory = models.AgentMemory(**memory.dict())
    if memory.ttl_seconds:
        db_memory.expires_at = datetime.utcnow() + timedelta(seconds=memory.ttl_seconds)
    db.add(db_memory)
    db.commit()
    db.refresh(db_memory)
    return db_memory

def get_agent_memory(db: Session, memory_id: int) -> Optional[models.AgentMemory]:
    """Get agent memory by ID"""
    return db.query(models.AgentMemory).filter(models.AgentMemory.id == memory_id).first()

def get_agent_memories(
    db: Session,
    agent_id: str,
    memory_type: Optional[str] = None,
    memory_key: Optional[str] = None
) -> List[models.AgentMemory]:
    """Get agent memories with optional filtering"""
    query = db.query(models.AgentMemory).filter(models.AgentMemory.agent_id == agent_id)
    if memory_type:
        query = query.filter(models.AgentMemory.memory_type == memory_type)
    if memory_key:
        query = query.filter(models.AgentMemory.memory_key == memory_key)

    # Filter out expired memories
    query = query.filter(
        and_(
            models.AgentMemory.expires_at.is_(None),
            models.AgentMemory.expires_at > datetime.utcnow()
        )
    )

    return query.order_by(desc(models.AgentMemory.last_accessed)).all()

def update_memory_access(db: Session, memory_id: int) -> Optional[models.AgentMemory]:
    """Update memory access count and timestamp"""
    memory = get_agent_memory(db, memory_id)
    if memory:
        memory.access_count += 1
        memory.last_accessed = datetime.utcnow()
        db.commit()
        db.refresh(memory)
    return memory

def delete_expired_memories(db: Session) -> int:
    """Delete expired memories, return count deleted"""
    deleted = db.query(models.AgentMemory).filter(
        and_(
            models.AgentMemory.expires_at.isnot(None),
            models.AgentMemory.expires_at <= datetime.utcnow()
        )
    ).delete()
    db.commit()
    return deleted

# Analytics and Insights
def get_learning_insights(db: Session) -> schemas.LearningInsights:
    """Get overall learning insights"""
    total_patterns = db.query(func.count(models.LearningPattern.id)).scalar()
    validated_patterns = db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.validation_status == "validated"
    ).scalar()

    avg_confidence_result = db.query(func.avg(models.LearningPattern.confidence)).scalar()
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0

    # Top domains
    domain_counts = db.query(
        models.LearningPattern.domain,
        func.count(models.LearningPattern.id).label('count')
    ).filter(
        models.LearningPattern.domain.isnot(None)
    ).group_by(models.LearningPattern.domain).order_by(desc('count')).limit(5).all()

    top_domains = [domain for domain, _ in domain_counts]

    # Recent learnings
    recent = db.query(models.LearningPattern).order_by(
        desc(models.LearningPattern.discovered_at)
    ).limit(10).all()

    recent_learnings = [
        {
            "id": p.id,
            "type": p.pattern_type,
            "confidence": p.confidence,
            "domain": p.domain,
            "discovered_at": p.discovered_at
        } for p in recent
    ]

    return schemas.LearningInsights(
        total_patterns=total_patterns,
        validated_patterns=validated_patterns,
        average_confidence=round(avg_confidence, 3),
        top_domains=top_domains,
        recent_learnings=recent_learnings
    )