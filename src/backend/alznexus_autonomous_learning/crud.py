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
    """Create a new learning pattern with supersession checking"""
    # Check for existing patterns that might be superseded by this new one
    existing_patterns = db.query(models.LearningPattern).filter(
        and_(
            models.LearningPattern.pattern_type == pattern.pattern_type,
            models.LearningPattern.domain == pattern.domain,
            models.LearningPattern.validation_status == "validated",
            models.LearningPattern.superseded_by.is_(None)  # Only check non-superseded patterns
        )
    ).all()

    # Check if this new pattern should supersede existing ones
    patterns_to_supersede = []
    for existing in existing_patterns:
        if _should_supersede(existing, pattern):
            patterns_to_supersede.append(existing)

    # Create the new pattern
    db_pattern = models.LearningPattern(**pattern.dict())
    db.add(db_pattern)
    db.commit()
    db.refresh(db_pattern)

    # Supersede old patterns
    for old_pattern in patterns_to_supersede:
        _supersede_pattern(db, old_pattern, db_pattern)

    return db_pattern

def _should_supersede(old_pattern: models.LearningPattern, new_pattern: schemas.LearningPatternCreate) -> bool:
    """Determine if new pattern should supersede old pattern"""
    # Supersede if new pattern has significantly higher confidence
    confidence_threshold = 0.15  # 15% improvement required
    if new_pattern.confidence > old_pattern.confidence + confidence_threshold:
        return True

    # Supersede if new pattern is much more recent and has similar confidence
    time_threshold_days = 30
    if (new_pattern.discovered_at - old_pattern.discovered_at).days > time_threshold_days:
        confidence_similarity = abs(new_pattern.confidence - old_pattern.confidence) < 0.1
        if confidence_similarity:
            return True

    # Supersede if new pattern has more validation evidence
    if hasattr(new_pattern, 'validation_count') and hasattr(old_pattern, 'validation_count'):
        if new_pattern.validation_count > old_pattern.validation_count * 1.5:
            return True

    return False

def _supersede_pattern(db: Session, old_pattern: models.LearningPattern, new_pattern: models.LearningPattern):
    """Mark old pattern as superseded by new pattern"""
    old_pattern.superseded_by = new_pattern.id
    old_pattern.superseded_at = datetime.utcnow()
    old_pattern.validation_status = "superseded"

    # Log the supersession
    db.commit()

    # Update any context enrichments that used the old pattern
    _update_context_enrichments_for_superseded_pattern(db, old_pattern, new_pattern)

def _update_context_enrichments_for_superseded_pattern(db: Session, old_pattern: models.LearningPattern, new_pattern: models.LearningPattern):
    """Update context enrichments to use new pattern instead of superseded one"""
    # Find enrichments that reference the old pattern
    enrichments_to_update = db.query(models.ContextEnrichment).filter(
        models.ContextEnrichment.source_patterns.contains([old_pattern.id])
    ).all()

    for enrichment in enrichments_to_update:
        # Replace old pattern ID with new pattern ID in source_patterns
        if enrichment.source_patterns:
            updated_patterns = [new_pattern.id if pid == old_pattern.id else pid
                              for pid in enrichment.source_patterns]
            enrichment.source_patterns = updated_patterns

            # Update enrichment metadata
            enrichment.enrichment_metadata = enrichment.enrichment_metadata or {}
            enrichment.enrichment_metadata["pattern_supersession"] = {
                "old_pattern_id": old_pattern.id,
                "new_pattern_id": new_pattern.id,
                "superseded_at": datetime.utcnow().isoformat()
            }

    db.commit()

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
    """Get learning patterns with optional filtering - excludes superseded patterns by default"""
    query = db.query(models.LearningPattern)

    # Always exclude superseded patterns unless explicitly requested
    if validation_status != "superseded":
        query = query.filter(models.LearningPattern.superseded_by.is_(None))

    if pattern_type:
        query = query.filter(models.LearningPattern.pattern_type == pattern_type)
    if domain:
        query = query.filter(models.LearningPattern.domain == domain)
    if validation_status and validation_status != "superseded":
        query = query.filter(models.LearningPattern.validation_status == validation_status)
    if min_confidence:
        query = query.filter(models.LearningPattern.confidence >= min_confidence)

    return query.order_by(desc(models.LearningPattern.discovered_at)).offset(skip).limit(limit).all()

def get_active_patterns_only(
    db: Session,
    pattern_type: Optional[str] = None,
    domain: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 100
) -> List[models.LearningPattern]:
    """Get only active (non-superseded) patterns for guaranteed forward progression"""
    return db.query(models.LearningPattern).filter(
        and_(
            models.LearningPattern.superseded_by.is_(None),
            models.LearningPattern.validation_status == "validated",
            models.LearningPattern.confidence >= (min_confidence or 0.0)
        )
    ).filter(
        pattern_type and models.LearningPattern.pattern_type == pattern_type or True
    ).filter(
        domain and models.LearningPattern.domain == domain or True
    ).order_by(
        desc(models.LearningPattern.discovered_at),
        desc(models.LearningPattern.confidence)
    ).limit(limit).all()

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

# Advanced Self-Evolution Metrics Functions
def get_total_patterns(db: Session) -> int:
    """Get total number of patterns extracted"""
    return db.query(func.count(models.LearningPattern.id)).scalar() or 0

def get_total_context_enrichments(db: Session) -> int:
    """Get total number of context enrichments performed"""
    return db.query(func.count(models.ContextEnrichment.id)).scalar() or 0

def get_average_confidence_score(db: Session) -> float:
    """Calculate average confidence score across all patterns"""
    result = db.query(func.avg(models.LearningPattern.confidence_level)).scalar()
    return round(result or 0.0, 3)

def get_knowledge_growth_rate(db: Session) -> float:
    """Calculate knowledge growth rate (patterns per day)"""
    # Get patterns from last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_patterns = db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.created_at >= thirty_days_ago
    ).scalar() or 0

    # Calculate daily rate
    return round(recent_patterns / 30.0, 2)

def get_active_learning_cycles(db: Session) -> int:
    """Get number of active learning cycles"""
    # Count patterns updated in last 24 hours as active cycles
    yesterday = datetime.utcnow() - timedelta(hours=24)
    return db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.updated_at >= yesterday
    ).scalar() or 0

def get_learning_effectiveness(db: Session) -> float:
    """Calculate learning effectiveness (patterns successfully applied / total patterns)"""
    total_patterns = get_total_patterns(db)
    if total_patterns == 0:
        return 0.0

    # Count patterns with high success rate (>80%)
    successful_patterns = db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.success_rate > 0.8
    ).scalar() or 0

    return round(successful_patterns / total_patterns, 3)

def get_adaptation_rate(db: Session) -> float:
    """Calculate adaptation rate (how quickly agents adapt to new patterns)"""
    # Get patterns from last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_patterns = db.query(models.LearningPattern).filter(
        models.LearningPattern.created_at >= week_ago
    ).all()

    if not recent_patterns:
        return 0.0

    # Calculate average time from creation to first application
    total_adaptation_time = 0
    count = 0

    for pattern in recent_patterns:
        if pattern.last_applied and pattern.created_at:
            adaptation_time = (pattern.last_applied - pattern.created_at).total_seconds() / 3600  # hours
            if adaptation_time > 0:
                total_adaptation_time += adaptation_time
                count += 1

    if count == 0:
        return 0.0

    avg_adaptation_hours = total_adaptation_time / count
    # Convert to adaptation rate (lower hours = higher rate)
    adaptation_rate = max(0, 1 - (avg_adaptation_hours / 168))  # 168 hours = 1 week

    return round(adaptation_rate, 3)

def get_knowledge_utilization(db: Session) -> float:
    """Calculate knowledge utilization (how much learned knowledge is being used)"""
    # Get total patterns
    total_patterns = get_total_patterns(db)
    if total_patterns == 0:
        return 0.0

    # Get patterns applied in last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    utilized_patterns = db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.last_applied >= thirty_days_ago
    ).scalar() or 0

    return round(utilized_patterns / total_patterns, 3)

def get_self_improvement_metrics(db: Session) -> Dict[str, float]:
    """Get self-improvement metrics"""
    # Pattern recognition accuracy (based on validation success)
    total_validated = db.query(func.count(models.LearningPattern.id)).filter(
        models.LearningPattern.is_validated == True
    ).scalar() or 0

    total_patterns = get_total_patterns(db)
    pattern_accuracy = total_validated / total_patterns if total_patterns > 0 else 0

    # Context enrichment quality (based on performance improvement)
    recent_performances = db.query(models.AgentPerformance).filter(
        models.AgentPerformance.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).all()

    if recent_performances:
        avg_improvement = sum(p.confidence_score for p in recent_performances if p.confidence_score) / len(recent_performances)
    else:
        avg_improvement = 0

    # Task success prediction (based on historical success rates)
    prediction_accuracy = get_learning_effectiveness(db)

    return {
        "pattern_recognition_accuracy": round(pattern_accuracy, 3),
        "context_enrichment_quality": round(avg_improvement, 3),
        "task_success_prediction": round(prediction_accuracy, 3)
    }

def determine_evolution_phase(db: Session) -> str:
    """Determine current evolution phase based on system maturity"""
    total_patterns = get_total_patterns(db)
    learning_effectiveness = get_learning_effectiveness(db)
    adaptation_rate = get_adaptation_rate(db)

    if total_patterns < 10:
        return "initialization"
    elif learning_effectiveness < 0.5:
        return "learning"
    elif adaptation_rate < 0.3:
        return "adaptation"
    elif learning_effectiveness > 0.8 and adaptation_rate > 0.7:
        return "optimization"
    else:
        return "maturation"

def get_total_knowledge_documents(db: Session) -> int:
    """Get total knowledge documents (would integrate with knowledge base)"""
    # This would normally query the knowledge base service
    # For now, return a placeholder based on patterns
    return get_total_patterns(db) * 3  # Estimate

def get_total_knowledge_chunks(db: Session) -> int:
    """Get total knowledge chunks"""
    return get_total_knowledge_documents(db) * 5  # Estimate

def get_vector_dimensions(db: Session) -> int:
    """Get vector dimensions for knowledge representation"""
    return 384  # Typical embedding dimension

def get_daily_growth_rate(db: Session) -> float:
    """Calculate daily growth rate of knowledge base"""
    return get_knowledge_growth_rate(db) * 3  # Estimate based on patterns

def get_quality_score_trend(db: Session, days: int = 30) -> List[float]:
    """Get quality score trend over time"""
    # Generate sample trend data (would be calculated from actual metrics)
    import random
    base_score = 0.7
    trend = []
    for i in range(days):
        # Simulate improving quality over time
        score = min(0.95, base_score + (i / days) * 0.2 + random.uniform(-0.05, 0.05))
        trend.append(round(score, 3))
    return trend

def get_quality_timestamps(db: Session, days: int = 30) -> List[str]:
    """Get timestamps for quality trend data"""
    timestamps = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=days-i-1)
        timestamps.append(date.isoformat())
    return timestamps

def get_agent_performance_history(db: Session, agent_id: str, limit: int = 50) -> List[models.AgentPerformance]:
    """Get historical performance data for an agent"""
    return db.query(models.AgentPerformance).filter(
        models.AgentPerformance.agent_id == agent_id
    ).order_by(desc(models.AgentPerformance.timestamp)).limit(limit).all()

def calculate_performance_trend(historical_data: List[models.AgentPerformance]) -> str:
    """Calculate performance trend from historical data"""
    if len(historical_data) < 2:
        return "insufficient_data"

    # Sort by timestamp
    sorted_data = sorted(historical_data, key=lambda x: x.timestamp)

    # Calculate trend using linear regression on success rates
    n = len(sorted_data)
    if n < 2:
        return "stable"

    x = list(range(n))
    y = [p.success_rate for p in sorted_data]

    # Simple linear regression
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        return "stable"

    slope = numerator / denominator

    if slope > 0.001:
        return "improving"
    elif slope < -0.001:
        return "declining"
    else:
        return "stable"

def predict_future_performance(historical_data: List[models.AgentPerformance]) -> Dict[str, float]:
    """Predict future performance using trend analysis"""
    if len(historical_data) < 3:
        return {"predicted_success_rate": 0.5, "confidence": 0.5}

    trend = calculate_performance_trend(historical_data)
    current_avg = sum(p.success_rate for p in historical_data[-5:]) / min(5, len(historical_data))

    if trend == "improving":
        predicted = min(0.95, current_avg + 0.05)
        confidence = 0.8
    elif trend == "declining":
        predicted = max(0.1, current_avg - 0.05)
        confidence = 0.7
    else:
        predicted = current_avg
        confidence = 0.9

    return {"predicted_success_rate": round(predicted, 3), "confidence": confidence}

def calculate_confidence_intervals(historical_data: List[models.AgentPerformance]) -> Dict[str, float]:
    """Calculate confidence intervals for performance predictions"""
    if len(historical_data) < 2:
        return {"lower_bound": 0.0, "upper_bound": 1.0}

    success_rates = [p.success_rate for p in historical_data]
    mean = sum(success_rates) / len(success_rates)
    variance = sum((x - mean) ** 2 for x in success_rates) / len(success_rates)
    std_dev = variance ** 0.5

    # 95% confidence interval
    margin = 1.96 * (std_dev / (len(success_rates) ** 0.5))

    return {
        "lower_bound": round(max(0.0, mean - margin), 3),
        "upper_bound": round(min(1.0, mean + margin), 3)
    }

def identify_performance_bottlenecks(historical_data: List[models.AgentPerformance]) -> List[str]:
    """Identify performance bottlenecks from historical data"""
    bottlenecks = []

    if len(historical_data) < 5:
        return ["insufficient_data"]

    # Check for high execution times
    avg_time = sum(p.execution_time for p in historical_data) / len(historical_data)
    high_time_count = sum(1 for p in historical_data if p.execution_time > avg_time * 1.5)
    if high_time_count > len(historical_data) * 0.3:
        bottlenecks.append("high_execution_times")

    # Check for low confidence scores
    low_confidence_count = sum(1 for p in historical_data if p.confidence_score and p.confidence_score < 0.5)
    if low_confidence_count > len(historical_data) * 0.4:
        bottlenecks.append("low_confidence_scores")

    # Check for declining success rate
    recent_success = sum(p.success_rate for p in historical_data[-3:]) / 3
    older_success = sum(p.success_rate for p in historical_data[:-3]) / max(1, len(historical_data) - 3)
    if recent_success < older_success * 0.9:
        bottlenecks.append("declining_success_rate")

    return bottlenecks if bottlenecks else ["no_bottlenecks_identified"]

def get_evolution_milestones(db: Session) -> List[Dict[str, Any]]:
    """Get evolution milestones"""
    # Create milestone data based on pattern creation dates
    patterns = db.query(models.LearningPattern).order_by(models.LearningPattern.created_at).all()

    milestones = []
    pattern_count = 0
    for pattern in patterns:
        pattern_count += 1
        if pattern_count in [1, 5, 10, 25, 50, 100]:  # Milestone pattern counts
            milestones.append({
                "milestone": f"{pattern_count}_patterns_learned",
                "timestamp": pattern.created_at.isoformat(),
                "description": f"System learned its {pattern_count}th pattern"
            })

    return milestones

def get_capability_progression(db: Session) -> List[Dict[str, Any]]:
    """Get capability progression over time"""
    # This would track how system capabilities improve over time
    # For now, return sample progression data
    return [
        {"capability": "pattern_recognition", "level": 0.8, "timestamp": datetime.utcnow().isoformat()},
        {"capability": "context_enrichment", "level": 0.7, "timestamp": datetime.utcnow().isoformat()},
        {"capability": "performance_prediction", "level": 0.6, "timestamp": datetime.utcnow().isoformat()},
        {"capability": "self_optimization", "level": 0.5, "timestamp": datetime.utcnow().isoformat()}
    ]

def get_learning_curve_data(db: Session) -> List[Dict[str, float]]:
    """Get learning curve data showing improvement over time"""
    # Get performance data over time
    performances = db.query(models.AgentPerformance).order_by(models.AgentPerformance.timestamp).all()

    if not performances:
        return []

    # Group by date and calculate average success rate
    daily_stats = {}
    for perf in performances:
        date_key = perf.timestamp.date().isoformat()
        if date_key not in daily_stats:
            daily_stats[date_key] = []
        daily_stats[date_key].append(perf.success_rate)

    learning_curve = []
    for date, rates in sorted(daily_stats.items()):
        avg_rate = sum(rates) / len(rates)
        learning_curve.append({
            "date": date,
            "average_success_rate": round(avg_rate, 3),
            "sample_size": len(rates)
        })

    return learning_curve

def calculate_evolution_velocity(db: Session) -> float:
    """Calculate evolution velocity (rate of improvement)"""
    learning_curve = get_learning_curve_data(db)

    if len(learning_curve) < 2:
        return 0.0

    # Calculate improvement rate
    first_rate = learning_curve[0]["average_success_rate"]
    last_rate = learning_curve[-1]["average_success_rate"]

    days_elapsed = len(learning_curve)
    improvement = last_rate - first_rate

    velocity = improvement / days_elapsed if days_elapsed > 0 else 0

    return round(velocity, 4)

def get_recent_performances_by_agent_and_task(db: Session, agent_id: str, task_type: str, limit: int = 10) -> List[models.AgentPerformance]:
    """Get recent performances for a specific agent and task type"""
    return db.query(models.AgentPerformance).filter(
        and_(
            models.AgentPerformance.agent_id == agent_id,
            models.AgentPerformance.task_type == task_type
        )
    ).order_by(desc(models.AgentPerformance.timestamp)).limit(limit).all()

def get_patterns_by_type_and_domain(db: Session, pattern_type: str, domain: str) -> List[models.LearningPattern]:
    """Get learning patterns by type and domain"""
    return db.query(models.LearningPattern).filter(
        and_(
            models.LearningPattern.pattern_type == pattern_type,
            models.LearningPattern.domain == domain,
            models.LearningPattern.superseded_by.is_(None)  # Only active patterns
        )
    ).order_by(desc(models.LearningPattern.confidence)).all()