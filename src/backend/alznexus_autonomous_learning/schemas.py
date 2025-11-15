from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

# Agent Performance Schemas
class AgentPerformanceCreate(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Type of agent (e.g., biomarker_hunter, drug_screener)")
    task_id: str = Field(..., description="Unique identifier for the task")
    task_type: str = Field(..., description="Type of task performed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate of the task")
    execution_time: float = Field(..., gt=0, description="Execution time in seconds")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Accuracy score if applicable")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score if applicable")
    outcome_data: Optional[Dict[str, Any]] = Field(None, description="Structured outcome data")
    feedback_received: Optional[Dict[str, Any]] = Field(None, description="Feedback from other agents/users")
    context_used: Optional[Dict[str, Any]] = Field(None, description="Context data used for this task")
    learned_patterns: Optional[Dict[str, Any]] = Field(None, description="Patterns learned from this execution")
    version: Optional[str] = Field(None, description="Agent version")
    environment: Optional[str] = Field("dev", description="Environment: dev/staging/prod")

class AgentPerformanceResponse(AgentPerformanceCreate):
    id: int
    timestamp: datetime

# Learning Pattern Schemas
class LearningPatternCreate(BaseModel):
    pattern_type: str = Field(..., description="Type of pattern (biomarker, drug, hypothesis, etc.)")
    pattern_data: Dict[str, Any] = Field(..., description="The actual pattern/insight data")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the pattern")
    source_agent: str = Field(..., description="Agent that discovered this pattern")
    source_task: str = Field(..., description="Task that led to this pattern")
    domain: Optional[str] = Field(None, description="Domain (alzheimer, neuroscience, etc.)")
    tags: Optional[List[str]] = Field(None, description="Categorization tags")
    validation_status: str = Field("pending", description="Validation status")
    related_patterns: Optional[List[int]] = Field(None, description="IDs of related patterns")
    citations: Optional[List[str]] = Field(None, description="Supporting evidence/references")
    expires_at: Optional[datetime] = Field(None, description="When pattern becomes stale")

class LearningPatternResponse(LearningPatternCreate):
    id: int
    discovered_at: datetime
    last_updated: datetime

# Context Enrichment Schemas
class ContextEnrichmentCreate(BaseModel):
    agent_id: str = Field(..., description="Agent receiving enriched context")
    context_type: str = Field(..., description="Type of context: initial, enriched, learned")
    original_context: Optional[Dict[str, Any]] = Field(None, description="Original context before enrichment")
    enriched_context: Dict[str, Any] = Field(..., description="Enriched context data")
    enrichment_metadata: Optional[Dict[str, Any]] = Field(None, description="What was added/changed")
    source_patterns: Optional[List[int]] = Field(None, description="Pattern IDs used for enrichment")
    source_performance: Optional[List[int]] = Field(None, description="Performance data IDs used")
    performance_improvement: Optional[float] = Field(None, description="Measured improvement")
    enrichment_success: bool = Field(True, description="Whether enrichment was successful")
    task_id: Optional[str] = Field(None, description="Associated task ID")

class ContextEnrichmentResponse(ContextEnrichmentCreate):
    id: int
    timestamp: datetime

# Feedback Loop Schemas
class FeedbackLoopCreate(BaseModel):
    loop_id: str = Field(..., description="Unique identifier for the feedback loop")
    execution_stage: Dict[str, Any] = Field(..., description="Agent execution data")
    evaluation_stage: Dict[str, Any] = Field(..., description="Performance evaluation data")
    learning_stage: Dict[str, Any] = Field(..., description="Pattern extraction/learning data")
    enrichment_stage: Dict[str, Any] = Field(..., description="Context enrichment data")
    agent_ids: List[str] = Field(..., description="Agents involved in the loop")
    pattern_ids: Optional[List[int]] = Field(None, description="Patterns generated in this loop")
    success_metric: Optional[float] = Field(None, description="Overall loop success metric")

class FeedbackLoopResponse(FeedbackLoopCreate):
    id: int
    start_time: datetime
    end_time: Optional[datetime]
    status: str

# Agent Memory Schemas
class AgentMemoryCreate(BaseModel):
    agent_id: str = Field(..., description="Agent that owns this memory")
    memory_type: str = Field(..., description="Type: episodic, semantic, procedural")
    memory_key: str = Field(..., description="Lookup key for the memory")
    memory_value: Dict[str, Any] = Field(..., description="Stored memory data")
    memory_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Importance score 0-1")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")

class AgentMemoryResponse(AgentMemoryCreate):
    id: int
    access_count: int
    last_accessed: datetime
    created_at: datetime
    expires_at: Optional[datetime]

# API Response Models
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(100, ge=1, le=1000)

class LearningInsights(BaseModel):
    total_patterns: int
    validated_patterns: int
    average_confidence: float
    top_domains: List[str]
    recent_learnings: List[Dict[str, Any]]