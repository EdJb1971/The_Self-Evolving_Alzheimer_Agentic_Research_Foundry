from celery import Celery
import os

# Celery configuration
celery_app = Celery(
    "alznexus_autonomous_learning",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Import tasks
from .learning_engine import LearningEngine
from .context_enricher import ContextEnricher
from .feedback_processor import FeedbackProcessor

learning_engine = LearningEngine()
context_enricher = ContextEnricher()
feedback_processor = FeedbackProcessor()

@celery_app.task(name="analyze_performance")
def analyze_performance(performance_id: int):
    """Analyze agent performance for learning patterns"""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(learning_engine.analyze_performance(performance_id, db))
    finally:
        db.close()

@celery_app.task(name="run_comprehensive_analysis")
def run_comprehensive_analysis():
    """Run comprehensive learning analysis"""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(learning_engine.run_comprehensive_analysis(db))
    finally:
        db.close()

@celery_app.task(name="enrich_agent_context")
def enrich_agent_context(agent_id: str, context: dict):
    """Enrich agent context with learned data"""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(context_enricher.enrich_agent_context(agent_id, context, db))
    finally:
        db.close()

@celery_app.task(name="update_all_contexts")
def update_all_contexts():
    """Update contexts for all agents"""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(context_enricher.update_all_contexts(db))
    finally:
        db.close()

@celery_app.task(name="process_feedback_loop")
def process_feedback_loop(loop_id: str):
    """Process a feedback loop"""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(feedback_processor.process_feedback_loop(loop_id, db))
    finally:
        db.close()

@celery_app.task(name="cleanup_expired_memories")
def cleanup_expired_memories():
    """Clean up expired agent memories"""
    from .database import SessionLocal
    from . import crud

    db = SessionLocal()
    try:
        deleted_count = crud.delete_expired_memories(db)
        return f"Cleaned up {deleted_count} expired memories"
    finally:
        db.close()

@celery_app.task(name="sync_to_knowledge_base")
def sync_to_knowledge_base(min_confidence: float = 0.7, limit: int = 50):
    """Sync learned patterns and enrichments to knowledge base"""
    from .database import SessionLocal
    from .main import knowledge_base_client
    from . import crud, models

    db = SessionLocal()
    try:
        # Sync patterns
        patterns = crud.get_learning_patterns(
            db,
            min_confidence=min_confidence,
            validation_status="validated",
            limit=limit
        )

        pattern_sync_count = 0
        for pattern in patterns:
            if knowledge_base_client.push_learned_pattern(pattern):
                pattern_sync_count += 1

        # Sync enrichments
        enrichments = db.query(models.ContextEnrichment).order_by(
            models.ContextEnrichment.timestamp.desc()
        ).limit(limit).all()

        enrichment_sync_count = 0
        for enrichment in enrichments:
            if knowledge_base_client.push_context_enrichment(enrichment):
                enrichment_sync_count += 1

        return f"Synced {pattern_sync_count} patterns and {enrichment_sync_count} enrichments to knowledge base"

    finally:
        db.close()

# Periodic tasks
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic tasks"""
    sender.add_periodic_task(3600.0, run_comprehensive_analysis.s(), name='comprehensive-analysis')  # Every hour
    sender.add_periodic_task(7200.0, update_all_contexts.s(), name='update-contexts')  # Every 2 hours
    sender.add_periodic_task(86400.0, cleanup_expired_memories.s(), name='cleanup-memories')  # Daily
    sender.add_periodic_task(1800.0, sync_to_knowledge_base.s(), name='sync-knowledge-base')  # Every 30 minutes