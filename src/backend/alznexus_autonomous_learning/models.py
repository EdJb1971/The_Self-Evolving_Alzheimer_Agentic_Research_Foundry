from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class AgentPerformance(Base):
    """Tracks individual agent performance metrics and outcomes"""
    __tablename__ = "agent_performance"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(100), nullable=False)
    task_id = Column(String(100), nullable=False, index=True)
    task_type = Column(String(100), nullable=False)

    # Performance metrics
    success_rate = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)  # seconds
    accuracy_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Outcome data
    outcome_data = Column(JSON, nullable=True)  # Store structured outcome data
    feedback_received = Column(JSON, nullable=True)  # Feedback from other agents/users

    # Learning context
    context_used = Column(JSON, nullable=True)  # Context data used for this task
    learned_patterns = Column(JSON, nullable=True)  # Patterns learned from this execution

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    version = Column(String(50), nullable=True)  # Agent version
    environment = Column(String(50), nullable=True)  # dev/staging/prod

class LearningPattern(Base):
    """Stores learned patterns and insights from agent executions"""
    __tablename__ = "learning_patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(100), nullable=False, index=True)  # biomarker, drug, hypothesis, etc.
    pattern_data = Column(JSON, nullable=False)  # The actual pattern/insight
    confidence = Column(Float, nullable=False)
    source_agent = Column(String(100), nullable=False, index=True)
    source_task = Column(String(100), nullable=False)

    # Pattern metadata
    domain = Column(String(100), nullable=True)  # alzheimer, neuroscience, etc.
    tags = Column(JSON, nullable=True)  # Categorization tags
    validation_status = Column(String(50), default="pending")  # pending, validated, rejected

    # Relationships
    related_patterns = Column(JSON, nullable=True)  # IDs of related patterns
    citations = Column(JSON, nullable=True)  # Supporting evidence/references

    # Temporal data
    discovered_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # When pattern becomes stale

class ContextEnrichment(Base):
    """Tracks how contexts are enriched with learned data"""
    __tablename__ = "context_enrichment"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    context_type = Column(String(100), nullable=False)  # initial, enriched, learned

    # Context data
    original_context = Column(JSON, nullable=True)
    enriched_context = Column(JSON, nullable=False)
    enrichment_metadata = Column(JSON, nullable=True)  # What was added/changed

    # Enrichment source
    source_patterns = Column(JSON, nullable=True)  # Pattern IDs used for enrichment
    source_performance = Column(JSON, nullable=True)  # Performance data used

    # Impact tracking
    performance_improvement = Column(Float, nullable=True)  # Measured improvement
    enrichment_success = Column(Boolean, default=True)

    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    task_id = Column(String(100), nullable=True)

class FeedbackLoop(Base):
    """Tracks the complete feedback loop from execution to learning"""
    __tablename__ = "feedback_loops"

    id = Column(Integer, primary_key=True, index=True)
    loop_id = Column(String(100), unique=True, nullable=False, index=True)

    # Loop stages
    execution_stage = Column(JSON, nullable=False)  # Agent execution data
    evaluation_stage = Column(JSON, nullable=False)  # Performance evaluation
    learning_stage = Column(JSON, nullable=False)  # Pattern extraction/learning
    enrichment_stage = Column(JSON, nullable=False)  # Context enrichment

    # Loop metadata
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(50), default="active")  # active, completed, failed
    success_metric = Column(Float, nullable=True)  # Overall loop success

    # Relationships
    agent_ids = Column(JSON, nullable=False)  # Agents involved in loop
    pattern_ids = Column(JSON, nullable=True)  # Patterns generated

class AgentMemory(Base):
    """Persistent memory for agents to store learned experiences"""
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(100), nullable=False, index=True)
    memory_type = Column(String(100), nullable=False)  # episodic, semantic, procedural

    # Memory content
    memory_key = Column(String(200), nullable=False)  # Lookup key
    memory_value = Column(JSON, nullable=False)  # Stored data
    memory_metadata = Column(JSON, nullable=True)  # Additional metadata

    # Memory management
    importance_score = Column(Float, default=0.5)  # 0-1 scale
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Expiration
    ttl_seconds = Column(Integer, nullable=True)  # Time to live
    expires_at = Column(DateTime, nullable=True)