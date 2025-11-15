import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import json
import requests
from datetime import datetime

from . import models, schemas, crud, database
from .learning_engine import LearningEngine
from .context_enricher import ContextEnricher
from .feedback_processor import FeedbackProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlzNexus Autonomous Learning Service",
    description="Self-evolving learning and feedback system for Alzheimer research agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Knowledge Base Integration
class KnowledgeBaseClient:
    """Client for integrating with the knowledge base service"""

    def __init__(self):
        self.knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://localhost:8006")
        self.api_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def push_learned_pattern(self, pattern: models.LearningPattern) -> bool:
        """Push a learned pattern to the knowledge base"""
        try:
            # Convert pattern to knowledge document format
            doc_data = {
                "title": f"Learned Pattern: {pattern.pattern_type}",
                "content": json.dumps({
                    "pattern_type": pattern.pattern_type,
                    "pattern_data": pattern.pattern_data,
                    "confidence": pattern.confidence,
                    "source_agent": pattern.source_agent,
                    "source_task": pattern.source_task,
                    "domain": pattern.domain,
                    "tags": pattern.tags,
                    "discovered_at": pattern.discovered_at.isoformat(),
                    "validation_status": pattern.validation_status
                }, indent=2),
                "document_type": "learned_pattern",
                "source_agent": "autonomous_learning_service",
                "source_task_id": f"pattern_{pattern.id}",
                "metadata_json": json.dumps({
                    "pattern_id": pattern.id,
                    "confidence": pattern.confidence,
                    "domain": pattern.domain,
                    "tags": pattern.tags
                }),
                "tags": ["learned_pattern", pattern.pattern_type, pattern.domain] + pattern.tags
            }

            response = requests.post(
                f"{self.knowledge_base_url}/documents/upsert/",
                headers=self.headers,
                json=doc_data,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Successfully pushed pattern {pattern.id} to knowledge base")
                return True
            else:
                logger.error(f"Failed to push pattern {pattern.id}: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error pushing pattern {pattern.id} to knowledge base: {str(e)}")
            return False

    def push_context_enrichment(self, enrichment: models.ContextEnrichment) -> bool:
        """Push context enrichment insights to knowledge base"""
        try:
            doc_data = {
                "title": f"Context Enrichment: {enrichment.agent_id}",
                "content": json.dumps({
                    "agent_id": enrichment.agent_id,
                    "context_type": enrichment.context_type,
                    "enrichment_metadata": enrichment.enrichment_metadata,
                    "source_patterns": enrichment.source_patterns,
                    "timestamp": enrichment.timestamp.isoformat(),
                    "enrichment_success": enrichment.enrichment_success
                }, indent=2),
                "document_type": "context_enrichment",
                "source_agent": "autonomous_learning_service",
                "source_task_id": f"enrichment_{enrichment.id}",
                "metadata_json": json.dumps({
                    "enrichment_id": enrichment.id,
                    "agent_id": enrichment.agent_id,
                    "patterns_used": len(enrichment.source_patterns) if enrichment.source_patterns else 0
                }),
                "tags": ["context_enrichment", enrichment.context_type, enrichment.agent_id]
            }

            response = requests.post(
                f"{self.knowledge_base_url}/documents/upsert/",
                headers=self.headers,
                json=doc_data,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Successfully pushed enrichment {enrichment.id} to knowledge base")
                return True
            else:
                logger.error(f"Failed to push enrichment {enrichment.id}: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error pushing enrichment {enrichment.id} to knowledge base: {str(e)}")
            return False

# Initialize components
learning_engine = LearningEngine()
context_enricher = ContextEnricher()
feedback_processor = FeedbackProcessor()
knowledge_base_client = KnowledgeBaseClient()

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    database.create_tables()
    logger.info("Autonomous Learning Service started and database initialized")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "alznexus_autonomous_learning"}

# Agent Performance Endpoints
@app.post("/performance/", response_model=schemas.AgentPerformanceResponse)
async def record_agent_performance(
    performance: schemas.AgentPerformanceCreate,
    db: Session = Depends(database.get_db)
):
    """Record agent performance data"""
    try:
        db_performance = crud.create_agent_performance(db, performance)

        # Trigger learning analysis in background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            learning_engine.analyze_performance,
            db_performance.id,
            db
        )

        logger.info(f"Recorded performance for agent {performance.agent_id}, task {performance.task_id}")
        return db_performance
    except Exception as e:
        logger.error(f"Error recording performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error recording performance: {str(e)}")

@app.get("/performance/{performance_id}", response_model=schemas.AgentPerformanceResponse)
async def get_performance(
    performance_id: int,
    db: Session = Depends(database.get_db)
):
    """Get agent performance by ID"""
    performance = crud.get_agent_performance(db, performance_id)
    if not performance:
        raise HTTPException(status_code=404, detail="Performance record not found")
    return performance

@app.get("/performance/", response_model=List[schemas.AgentPerformanceResponse])
async def get_performances(
    agent_id: Optional[str] = None,
    task_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db)
):
    """Get agent performances with optional filtering"""
    return crud.get_agent_performances(db, agent_id, task_type, skip, limit)

@app.get("/performance/stats/{agent_id}")
async def get_agent_stats(
    agent_id: str,
    db: Session = Depends(database.get_db)
):
    """Get performance statistics for an agent"""
    stats = crud.get_agent_performance_stats(db, agent_id)
    return stats

# Learning Pattern Endpoints
@app.post("/patterns/", response_model=schemas.LearningPatternResponse)
async def create_learning_pattern(
    pattern: schemas.LearningPatternCreate,
    db: Session = Depends(database.get_db)
):
    """Create a new learning pattern"""
    try:
        db_pattern = crud.create_learning_pattern(db, pattern)
        logger.info(f"Created learning pattern: {pattern.pattern_type} from {pattern.source_agent}")
        return db_pattern
    except Exception as e:
        logger.error(f"Error creating pattern: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating pattern: {str(e)}")

@app.get("/patterns/{pattern_id}", response_model=schemas.LearningPatternResponse)
async def get_pattern(
    pattern_id: int,
    db: Session = Depends(database.get_db)
):
    """Get learning pattern by ID"""
    pattern = crud.get_learning_pattern(db, pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern

@app.get("/patterns/", response_model=List[schemas.LearningPatternResponse])
async def get_patterns(
    pattern_type: Optional[str] = None,
    domain: Optional[str] = None,
    validation_status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db)
):
    """Get learning patterns with optional filtering"""
    return crud.get_learning_patterns(db, pattern_type, domain, validation_status, min_confidence, skip, limit)

@app.put("/patterns/{pattern_id}/validate")
async def validate_pattern(
    pattern_id: int,
    validation_status: str,
    db: Session = Depends(database.get_db)
):
    """Update pattern validation status"""
    if validation_status not in ["pending", "validated", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid validation status")

    pattern = crud.update_pattern_validation(db, pattern_id, validation_status)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")

    logger.info(f"Updated pattern {pattern_id} validation to {validation_status}")
    return {"message": f"Pattern validation updated to {validation_status}"}

@app.get("/patterns/{pattern_id}/related", response_model=List[schemas.LearningPatternResponse])
async def get_related_patterns(
    pattern_id: int,
    db: Session = Depends(database.get_db)
):
    """Get patterns related to the given pattern"""
    return crud.get_related_patterns(db, pattern_id)

# Context Enrichment Endpoints
@app.post("/context/enrich", response_model=schemas.ContextEnrichmentResponse)
async def enrich_context(
    enrichment: schemas.ContextEnrichmentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """Enrich agent context with learned data"""
    try:
        # Create enrichment record
        db_enrichment = crud.create_context_enrichment(db, enrichment)

        # Trigger context enrichment process
        background_tasks.add_task(
            context_enricher.enrich_agent_context,
            enrichment.agent_id,
            enrichment.enriched_context,
            db
        )

        logger.info(f"Context enrichment initiated for agent {enrichment.agent_id}")
        return db_enrichment
    except Exception as e:
        logger.error(f"Error enriching context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enriching context: {str(e)}")

@app.get("/context/history/{agent_id}", response_model=List[schemas.ContextEnrichmentResponse])
async def get_context_history(
    agent_id: str,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(database.get_db)
):
    """Get context enrichment history for an agent"""
    return crud.get_agent_context_history(db, agent_id, skip, limit)

# Feedback Loop Endpoints
@app.post("/feedback/loops/", response_model=schemas.FeedbackLoopResponse)
async def create_feedback_loop(
    loop: schemas.FeedbackLoopCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """Create a new feedback loop"""
    try:
        # Check if loop_id already exists
        existing = crud.get_feedback_loop(db, loop.loop_id)
        if existing:
            raise HTTPException(status_code=400, detail="Feedback loop with this ID already exists")

        db_loop = crud.create_feedback_loop(db, loop)

        # Start feedback processing
        background_tasks.add_task(
            feedback_processor.process_feedback_loop,
            loop.loop_id,
            db
        )

        logger.info(f"Created feedback loop: {loop.loop_id}")
        return db_loop
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating feedback loop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating feedback loop: {str(e)}")

@app.get("/feedback/loops/{loop_id}", response_model=schemas.FeedbackLoopResponse)
async def get_feedback_loop(
    loop_id: str,
    db: Session = Depends(database.get_db)
):
    """Get feedback loop by ID"""
    loop = crud.get_feedback_loop(db, loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")
    return loop

@app.put("/feedback/loops/{loop_id}/complete")
async def complete_feedback_loop(
    loop_id: str,
    success_metric: Optional[float] = None,
    db: Session = Depends(database.get_db)
):
    """Mark feedback loop as completed"""
    loop = crud.update_feedback_loop_status(
        db,
        loop_id,
        "completed",
        datetime.utcnow(),
        success_metric
    )
    if not loop:
        raise HTTPException(status_code=404, detail="Feedback loop not found")

    logger.info(f"Completed feedback loop: {loop_id}")
    return {"message": "Feedback loop completed"}

@app.get("/feedback/loops/active", response_model=List[schemas.FeedbackLoopResponse])
async def get_active_loops(db: Session = Depends(database.get_db)):
    """Get all active feedback loops"""
    return crud.get_active_feedback_loops(db)

# Agent Memory Endpoints
@app.post("/memory/", response_model=schemas.AgentMemoryResponse)
async def create_agent_memory(
    memory: schemas.AgentMemoryCreate,
    db: Session = Depends(database.get_db)
):
    """Create a new agent memory"""
    try:
        db_memory = crud.create_agent_memory(db, memory)
        logger.info(f"Created memory for agent {memory.agent_id}: {memory.memory_key}")
        return db_memory
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating memory: {str(e)}")

@app.get("/memory/{agent_id}", response_model=List[schemas.AgentMemoryResponse])
async def get_agent_memories(
    agent_id: str,
    memory_type: Optional[str] = None,
    memory_key: Optional[str] = None,
    db: Session = Depends(database.get_db)
):
    """Get agent memories with optional filtering"""
    return crud.get_agent_memories(db, agent_id, memory_type, memory_key)

@app.put("/memory/{memory_id}/access")
async def access_memory(
    memory_id: int,
    db: Session = Depends(database.get_db)
):
    """Update memory access count and timestamp"""
    memory = crud.update_memory_access(db, memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory accessed"}

@app.delete("/memory/expired")
async def cleanup_expired_memories(db: Session = Depends(database.get_db)):
    """Delete expired memories"""
    deleted_count = crud.delete_expired_memories(db)
    logger.info(f"Cleaned up {deleted_count} expired memories")
    return {"message": f"Deleted {deleted_count} expired memories"}

# Analytics Endpoints
@app.get("/analytics/insights", response_model=schemas.LearningInsights)
async def get_learning_insights(db: Session = Depends(database.get_db)):
    """Get overall learning insights"""
    return crud.get_learning_insights(db)

@app.get("/analytics/performance/{agent_id}")
async def get_agent_performance_analytics(
    agent_id: str,
    db: Session = Depends(database.get_db)
):
    """Get detailed performance analytics for an agent"""
    stats = crud.get_agent_performance_stats(db, agent_id)

    # Get recent performances for trend analysis
    recent_performances = crud.get_agent_performances(db, agent_id, limit=20)

    trend_data = [
        {
            "timestamp": p.timestamp.isoformat(),
            "success_rate": p.success_rate,
            "execution_time": p.execution_time,
            "accuracy_score": p.accuracy_score
        } for p in recent_performances
    ]

    return {
        "stats": stats,
        "trend": trend_data
    }

# Knowledge Base Integration Endpoints
@app.post("/knowledge/sync-patterns")
async def sync_patterns_to_knowledge_base(
    background_tasks: BackgroundTasks,
    min_confidence: float = 0.7,
    limit: int = 50,
    db: Session = Depends(database.get_db)
):
    """Sync learned patterns to knowledge base"""
    background_tasks.add_task(
        _sync_patterns_to_kb,
        min_confidence,
        limit,
        db
    )
    return {"message": "Pattern sync to knowledge base triggered"}

@app.post("/knowledge/sync-enrichments")
async def sync_enrichments_to_knowledge_base(
    background_tasks: BackgroundTasks,
    limit: int = 50,
    db: Session = Depends(database.get_db)
):
    """Sync context enrichments to knowledge base"""
    background_tasks.add_task(
        _sync_enrichments_to_kb,
        limit,
        db
    )
    return {"message": "Enrichment sync to knowledge base triggered"}

@app.post("/knowledge/sync-all")
async def sync_all_to_knowledge_base(
    background_tasks: BackgroundTasks,
    min_confidence: float = 0.7,
    limit: int = 50,
    db: Session = Depends(database.get_db)
):
    """Sync all learned data to knowledge base"""
    background_tasks.add_task(
        _sync_all_to_kb,
        min_confidence,
        limit,
        db
    )
    return {"message": "Full sync to knowledge base triggered"}

# Background sync functions
async def _sync_patterns_to_kb(min_confidence: float, limit: int, db: Session):
    """Background task to sync patterns to knowledge base"""
    try:
        # Get patterns that haven't been synced yet
        patterns = crud.get_learning_patterns(
            db,
            min_confidence=min_confidence,
            validation_status="validated",
            limit=limit
        )

        synced_count = 0
        for pattern in patterns:
            # Check if already synced (you could add a synced flag to the model)
            if knowledge_base_client.push_learned_pattern(pattern):
                synced_count += 1

        logger.info(f"Synced {synced_count}/{len(patterns)} patterns to knowledge base")

    except Exception as e:
        logger.error(f"Error syncing patterns to knowledge base: {str(e)}")

async def _sync_enrichments_to_kb(limit: int, db: Session):
    """Background task to sync enrichments to knowledge base"""
    try:
        # Get recent enrichments
        enrichments = db.query(models.ContextEnrichment).order_by(
            models.ContextEnrichment.timestamp.desc()
        ).limit(limit).all()

        synced_count = 0
        for enrichment in enrichments:
            if knowledge_base_client.push_context_enrichment(enrichment):
                synced_count += 1

        logger.info(f"Synced {synced_count}/{len(enrichments)} enrichments to knowledge base")

    except Exception as e:
        logger.error(f"Error syncing enrichments to knowledge base: {str(e)}")

async def _sync_all_to_kb(min_confidence: float, limit: int, db: Session):
    """Background task to sync all data to knowledge base"""
    try:
        await _sync_patterns_to_kb(min_confidence, limit, db)
        await _sync_enrichments_to_kb(limit, db)
        logger.info("Completed full sync to knowledge base")

    except Exception as e:
        logger.error(f"Error in full sync to knowledge base: {str(e)}")

# Learning Engine Control Endpoints
@app.post("/learning/trigger-analysis")
async def trigger_learning_analysis(
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """Trigger comprehensive learning analysis"""
    background_tasks.add_task(learning_engine.run_comprehensive_analysis, db)
    return {"message": "Learning analysis triggered"}

@app.post("/learning/update-contexts")
async def update_all_contexts(
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    """Update all agent contexts with latest learnings"""
    background_tasks.add_task(context_enricher.update_all_contexts, db)
    return {"message": "Context updates triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)