from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Knowledge Document Schemas
class KnowledgeDocumentBase(BaseModel):
    title: str = Field(..., max_length=500)
    content: str
    document_type: str = Field(..., description="Type: 'research_finding', 'hypothesis', 'validation_result', 'literature_summary'")
    source_agent: str
    source_task_id: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    is_validated: bool = False
    validation_score: Optional[float] = None
    tags: Optional[List[str]] = None

class KnowledgeDocumentCreate(KnowledgeDocumentBase):
    pass

class KnowledgeDocumentUpsert(KnowledgeDocumentBase):
    """Schema for upserting knowledge documents with versioning."""
    version: int = Field(default=1, description="Version number for optimistic locking")
    last_modified_by: Optional[str] = None

class KnowledgeDocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    document_type: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    is_validated: Optional[bool] = None
    validation_score: Optional[float] = None
    tags: Optional[List[str]] = None

class KnowledgeDocument(KnowledgeDocumentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    version: int
    last_modified_by: Optional[str] = None

    class Config:
        from_attributes = True

# Document Chunk Schemas
class DocumentChunkBase(BaseModel):
    document_id: int
    chunk_index: int
    content: str
    embedding_model: str
    chunk_metadata: Optional[Dict[str, Any]] = None

class DocumentChunkCreate(DocumentChunkBase):
    pass

class DocumentChunk(DocumentChunkBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Knowledge Query Schemas
class KnowledgeQueryBase(BaseModel):
    query_text: str
    query_type: str = Field(..., description="Type: 'semantic_search', 'rag_context', 'similarity'")
    requester_agent: Optional[str] = None
    retrieved_document_ids: Optional[List[int]] = None
    relevance_scores: Optional[Dict[str, float]] = None
    query_metadata: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None

class KnowledgeQueryCreate(KnowledgeQueryBase):
    pass

class KnowledgeQuery(KnowledgeQueryBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Knowledge Update Schemas
class KnowledgeUpdateBase(BaseModel):
    operation_type: str = Field(..., description="Type: 'add', 'update', 'delete', 'validate'")
    document_id: Optional[int] = None
    change_description: str
    performed_by: str
    old_metadata: Optional[Dict[str, Any]] = None
    new_metadata: Optional[Dict[str, Any]] = None

class KnowledgeUpdateCreate(KnowledgeUpdateBase):
    pass

class KnowledgeUpdate(KnowledgeUpdateBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Research Insight Schemas
class ResearchInsightBase(BaseModel):
    insight_text: str
    insight_type: str
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: Optional[Dict[str, Any]] = None
    related_insights: Optional[List[int]] = None
    discovered_by: str
    validation_status: str = "pending"
    impact_score: Optional[float] = None

class ResearchInsightCreate(ResearchInsightBase):
    pass

class ResearchInsightUpdate(BaseModel):
    insight_text: Optional[str] = None
    insight_type: Optional[str] = None
    confidence_level: Optional[float] = None
    supporting_evidence: Optional[Dict[str, Any]] = None
    related_insights: Optional[List[int]] = None
    validation_status: Optional[str] = None
    impact_score: Optional[float] = None

class ResearchInsight(ResearchInsightBase):
    id: int
    discovery_date: datetime
    last_validated: Optional[datetime] = None

    class Config:
        from_attributes = True

# Learning Pattern Schemas
class LearningPatternBase(BaseModel):
    pattern_type: str
    pattern_description: str
    success_rate: float = Field(..., ge=0.0, le=1.0)
    application_count: int = 0
    discovered_from: Optional[Dict[str, Any]] = None

class LearningPatternCreate(LearningPatternBase):
    pass

class LearningPatternUpdate(BaseModel):
    pattern_type: Optional[str] = None
    pattern_description: Optional[str] = None
    success_rate: Optional[float] = None
    application_count: Optional[int] = None
    discovered_from: Optional[Dict[str, Any]] = None

class LearningPattern(LearningPatternBase):
    id: int
    last_applied: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# RAG and Search Schemas
class RAGContextRequest(BaseModel):
    query: str
    max_chunks: int = 5
    similarity_threshold: float = 0.7
    context_window: int = 2  # Number of surrounding chunks to include
    requester_agent: Optional[str] = None

class IntelligentRAGRequest(BaseModel):
    query: str
    model_name: str = "gemini-1.5-flash"  # For token limit calculation
    max_tokens: Optional[int] = None  # If not provided, will be inferred from model
    token_reserve: int = 1000  # Tokens to reserve for response
    min_relevance_score: float = 0.6
    max_chunks: int = 10
    context_window: int = 1
    requester_agent: Optional[str] = None
    document_types: Optional[List[str]] = None
    prioritize_recent: bool = True  # Prioritize more recent findings
    include_metadata: bool = False  # Include source metadata in context

class RAGContextResponse(BaseModel):
    query: str
    relevant_chunks: List[Dict[str, Any]]
    total_chunks_found: int
    context_text: str
    metadata: Dict[str, Any]

class IntelligentRAGResponse(BaseModel):
    query: str
    selected_chunks: List[Dict[str, Any]]
    context_text: str
    total_tokens_used: int
    max_tokens_allowed: int
    relevance_stats: Dict[str, Any]
    metadata: Dict[str, Any]
    rate_limit_info: Dict[str, Any]

class RateLimitInfo(BaseModel):
    agent_requests_per_minute: int
    current_usage: int
    backoff_seconds: int
    priority_level: str  # 'high', 'medium', 'low'

class ContextChunk(BaseModel):
    chunk_id: str
    document_id: int
    content: str
    relevance_score: float
    token_count: int
    source_metadata: Dict[str, Any]
    created_at: str

class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 10
    document_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_validation_score: Optional[float] = None

class SemanticSearchResult(BaseModel):
    document: KnowledgeDocument
    relevance_score: float
    matched_chunks: List[DocumentChunk]

class SemanticSearchResponse(BaseModel):
    query: str
    results: List[SemanticSearchResult]
    total_found: int
    search_metadata: Dict[str, Any]

# Knowledge Ingestion Schemas
class KnowledgeIngestionRequest(BaseModel):
    title: str
    content: str
    document_type: str
    source_agent: str
    source_task_id: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    auto_chunk: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200

class KnowledgeIngestionResponse(BaseModel):
    document_id: int
    chunks_created: int
    embedding_model_used: str
    ingestion_metadata: Dict[str, Any]

# Analytics Schemas
class KnowledgeAnalytics(BaseModel):
    total_documents: int
    total_chunks: int
    document_types_distribution: Dict[str, int]
    validation_status_distribution: Dict[str, int]
    top_contributing_agents: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]
    knowledge_growth_trend: List[Dict[str, Any]]