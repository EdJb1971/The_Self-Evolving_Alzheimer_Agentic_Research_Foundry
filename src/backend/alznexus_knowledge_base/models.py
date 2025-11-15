from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class KnowledgeDocument(Base):
    """Represents a document or research finding stored in the knowledge base."""
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String(100), nullable=False)  # 'research_finding', 'hypothesis', 'validation_result', 'literature_summary'
    source_agent = Column(String(100), nullable=False)
    source_task_id = Column(String(100), nullable=True)
    metadata_json = Column(JSON, nullable=True)  # Additional metadata like confidence scores, validation status
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_validated = Column(Boolean, default=False)
    validation_score = Column(Float, nullable=True)  # Statistical validation score
    tags = Column(JSON, nullable=True)  # List of tags for categorization
    version = Column(Integer, default=1, nullable=False)  # Version number for optimistic locking
    last_modified_by = Column(String(100), nullable=True)  # Agent that last modified this document

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Represents a chunk of a document for vector storage and retrieval."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("knowledge_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    chunk_metadata = Column(JSON, nullable=True)  # Position info, semantic info
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("KnowledgeDocument", back_populates="chunks")

class KnowledgeQuery(Base):
    """Tracks knowledge base queries for analytics and improvement."""
    __tablename__ = "knowledge_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)  # 'semantic_search', 'rag_context', 'similarity'
    requester_agent = Column(String(100), nullable=True)
    retrieved_document_ids = Column(JSON, nullable=True)  # List of retrieved document IDs
    relevance_scores = Column(JSON, nullable=True)  # Relevance scores for retrieved documents
    query_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    response_time_ms = Column(Integer, nullable=True)

class KnowledgeUpdate(Base):
    """Tracks updates to the knowledge base for audit and versioning."""
    __tablename__ = "knowledge_updates"

    id = Column(Integer, primary_key=True, index=True)
    operation_type = Column(String(50), nullable=False)  # 'add', 'update', 'delete', 'validate'
    document_id = Column(Integer, nullable=True)
    change_description = Column(Text, nullable=False)
    performed_by = Column(String(100), nullable=False)  # Agent or service that performed the update
    old_metadata = Column(JSON, nullable=True)
    new_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResearchInsight(Base):
    """Stores distilled research insights and their relationships."""
    __tablename__ = "research_insights"

    id = Column(Integer, primary_key=True, index=True)
    insight_text = Column(Text, nullable=False)
    insight_type = Column(String(100), nullable=False)  # 'biomarker_discovery', 'pathway_connection', 'trial_optimization', etc.
    confidence_level = Column(Float, nullable=False)  # 0.0 to 1.0
    supporting_evidence = Column(JSON, nullable=True)  # References to supporting documents/chunks
    related_insights = Column(JSON, nullable=True)  # IDs of related insights
    discovered_by = Column(String(100), nullable=False)
    discovery_date = Column(DateTime, default=datetime.utcnow)
    last_validated = Column(DateTime, nullable=True)
    validation_status = Column(String(50), default="pending")  # 'pending', 'validated', 'rejected'
    impact_score = Column(Float, nullable=True)  # Estimated scientific impact

class LearningPattern(Base):
    """Tracks patterns in successful research approaches for meta-learning."""
    __tablename__ = "learning_patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(100), nullable=False)  # 'successful_methodology', 'failed_approach', 'effective_combination'
    pattern_description = Column(Text, nullable=False)
    success_rate = Column(Float, nullable=False)  # Historical success rate
    application_count = Column(Integer, default=0)
    last_applied = Column(DateTime, nullable=True)
    discovered_from = Column(JSON, nullable=True)  # Source tasks/insights that led to this pattern
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)