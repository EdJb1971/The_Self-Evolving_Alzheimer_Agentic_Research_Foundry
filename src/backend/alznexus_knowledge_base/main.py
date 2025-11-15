from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from .database import get_db, create_tables
from . import crud, models, schemas
from .vector_db import get_vector_db_manager, IntelligentContextRetriever, get_rate_limiter, IntelligentTextChunker
from .celery_app import celery_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AlzNexus Knowledge Base Service",
    description="Vector database and knowledge accumulation service for continuous learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key dependency
def get_api_key(api_key: str = None):
    expected_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Initialize database and vector DB on startup
@app.on_event("startup")
async def startup_event():
    create_tables()
    vector_db = get_vector_db_manager()
    logger.info("Knowledge Base Service started successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "alznexus_knowledge_base"}

# Knowledge Document Endpoints
@app.post("/documents/", response_model=schemas.KnowledgeDocument, dependencies=[Depends(get_api_key)])
async def create_document(
    document: schemas.KnowledgeDocumentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new knowledge document (legacy - use upsert for new implementations)."""
    db_document = crud.create_knowledge_document(db, document)

    # Log the creation
    crud.create_knowledge_update(
        db,
        schemas.KnowledgeUpdateCreate(
            operation_type="add",
            document_id=db_document.id,
            change_description=f"Created new {document.document_type} document: {document.title}",
            performed_by=document.source_agent
        )
    )

    # Trigger background chunking and embedding
    background_tasks.add_task(process_document_chunks, db_document.id)

    return db_document

@app.post("/documents/upsert/", response_model=schemas.KnowledgeDocument, dependencies=[Depends(get_api_key)])
async def upsert_document(
    document: schemas.KnowledgeDocumentUpsert,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Upsert a knowledge document - only updates if new data is actually newer."""
    db_document = crud.upsert_knowledge_document(db, document)

    # Check if this was an update or insert
    was_updated = db_document.version > 1

    # Trigger background chunking and embedding (only if content actually changed)
    if not was_updated or document.content != db_document.content:
        background_tasks.add_task(process_document_chunks, db_document.id)

    return db_document

@app.get("/documents/{document_id}", response_model=schemas.KnowledgeDocument, dependencies=[Depends(get_api_key)])
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get a knowledge document by ID."""
    db_document = crud.get_knowledge_document(db, document_id)
    if not db_document:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.get("/documents/", response_model=List[schemas.KnowledgeDocument], dependencies=[Depends(get_api_key)])
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    document_type: Optional[str] = None,
    source_agent: Optional[str] = None,
    is_validated: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get knowledge documents with optional filtering."""
    documents = crud.get_knowledge_documents(
        db, skip=skip, limit=limit,
        document_type=document_type,
        source_agent=source_agent,
        is_validated=is_validated
    )
    return documents

@app.put("/documents/{document_id}", response_model=schemas.KnowledgeDocument, dependencies=[Depends(get_api_key)])
async def update_document(
    document_id: int,
    updates: schemas.KnowledgeDocumentUpdate,
    db: Session = Depends(get_db)
):
    """Update a knowledge document."""
    db_document = crud.update_knowledge_document(db, document_id, updates)
    if not db_document:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.delete("/documents/{document_id}", dependencies=[Depends(get_api_key)])
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a knowledge document."""
    success = crud.delete_knowledge_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from vector database
    vector_db = get_vector_db_manager()
    vector_db.delete_document_chunks(document_id)

    return {"message": "Document deleted successfully"}

# Knowledge Ingestion Endpoint
@app.post("/ingest/", response_model=schemas.KnowledgeIngestionResponse, dependencies=[Depends(get_api_key)])
async def ingest_knowledge(
    request: schemas.KnowledgeIngestionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest new knowledge with automatic chunking and embedding."""
    # Create the document
    doc_create = schemas.KnowledgeDocumentCreate(
        title=request.title,
        content=request.content,
        document_type=request.document_type,
        source_agent=request.source_agent,
        source_task_id=request.source_task_id,
        metadata_json=request.metadata_json,
        tags=request.tags
    )

    db_document = crud.create_knowledge_document(db, doc_create)

    # Log the ingestion
    crud.create_knowledge_update(
        db,
        schemas.KnowledgeUpdateCreate(
            operation_type="add",
            document_id=db_document.id,
            change_description=f"Ingested {request.document_type}: {request.title}",
            performed_by=request.source_agent
        )
    )

    # Process chunks in background
    background_tasks.add_task(
        process_document_chunks,
        db_document.id,
        request.chunk_size,
        request.chunk_overlap
    )

    return schemas.KnowledgeIngestionResponse(
        document_id=db_document.id,
        chunks_created=0,  # Will be updated after processing
        embedding_model_used=get_vector_db_manager().embedding_model_name,
        ingestion_metadata={"status": "processing"}
    )

# RAG Context Endpoint
@app.post("/rag/context/", response_model=schemas.RAGContextResponse, dependencies=[Depends(get_api_key)])
async def get_rag_context(
    request: schemas.RAGContextRequest,
    db: Session = Depends(get_db)
):
    """Get intelligent context for RAG using semantic search with rich metadata."""
    start_time = datetime.utcnow()

    # Use intelligent context retriever
    vector_db = get_vector_db_manager()
    retriever = IntelligentContextRetriever(vector_db)

    retrieval_result = retriever.retrieve_context(
        query=request.query,
        model_name=request.model_name or "gemini-1.5-flash",
        max_tokens=request.max_tokens,
        min_relevance_score=request.similarity_threshold,
        max_chunks=request.max_chunks,
        context_window=request.context_window,
        requester_agent=request.requester_agent,
        document_types=request.document_types,
        prioritize_recent=request.prioritize_recent,
        include_metadata=request.include_metadata
    )

    # Handle rate limiting
    if 'error' in retrieval_result:
        if retrieval_result['error'] == 'rate_limit_exceeded':
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retrieval_result['backoff_seconds']} seconds."
            )
        else:
            raise HTTPException(status_code=400, detail=retrieval_result['error'])

    # Extract relevant chunks for logging
    relevant_chunks = [
        {
            'chunk_id': chunk['chunk_id'],
            'document_id': chunk['document_id'],
            'content': chunk['content'],
            'similarity_score': chunk['relevance_score'],
            'metadata': chunk['metadata']
        }
        for chunk in retrieval_result['selected_chunks']
    ]

    # Log the query
    response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    crud.create_knowledge_query(
        db,
        schemas.KnowledgeQueryCreate(
            query_text=request.query,
            query_type="intelligent_rag_context",
            requester_agent=request.requester_agent,
            retrieved_document_ids=[chunk['document_id'] for chunk in relevant_chunks],
            relevance_scores={str(chunk['document_id']): chunk['relevance_score'] for chunk in relevant_chunks},
            response_time_ms=response_time
        )
    )

    return schemas.RAGContextResponse(
        query=request.query,
        relevant_chunks=relevant_chunks,
        total_chunks_found=len(relevant_chunks),
        context_text=retrieval_result['context_text'],
        metadata={
            'total_tokens_used': retrieval_result['total_tokens_used'],
            'max_tokens_allowed': retrieval_result['max_tokens_allowed'],
            'relevance_stats': retrieval_result['relevance_stats'],
            'rate_limit_info': retrieval_result['rate_limit_info'],
            'model_name': request.model_name or "gemini-1.5-flash",
            'response_time_ms': response_time
        }
    )

# Intelligent RAG Context Endpoint with Token Awareness and Rate Limiting
@app.post("/rag/intelligent/", response_model=schemas.IntelligentRAGResponse, dependencies=[Depends(get_api_key)])
async def get_intelligent_rag_context(
    request: schemas.IntelligentRAGRequest,
    db: Session = Depends(get_db)
):
    """Get intelligent context for RAG with token limits, relevance ranking, and rate limiting."""
    start_time = datetime.utcnow()

    # Get intelligent retriever
    retriever = get_intelligent_retriever()

    # Perform intelligent context retrieval
    result = retriever.retrieve_context(
        query=request.query,
        model_name=request.model_name,
        max_tokens=request.max_tokens,
        token_reserve=request.token_reserve,
        min_relevance_score=request.min_relevance_score,
        max_chunks=request.max_chunks,
        context_window=request.context_window,
        requester_agent=request.requester_agent,
        document_types=request.document_types,
        prioritize_recent=request.prioritize_recent,
        include_metadata=request.include_metadata
    )

    # Handle rate limiting errors
    if result.get("error") == "rate_limit_exceeded":
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Backoff for {result['backoff_seconds']} seconds.",
            headers={"Retry-After": str(result["backoff_seconds"])}
        )

    if result.get("error") == "insufficient_token_budget":
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient token budget. Available: {result['available_tokens']}"
        )

    response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    # Log the intelligent query
    crud.create_knowledge_query(
        db,
        schemas.KnowledgeQueryCreate(
            query_text=request.query,
            query_type="intelligent_rag",
            requester_agent=request.requester_agent,
            retrieved_document_ids=[chunk['document_id'] for chunk in result.get('selected_chunks', [])],
            relevance_scores={
                str(chunk['document_id']): chunk['relevance_score']
                for chunk in result.get('selected_chunks', [])
            },
            response_time_ms=response_time
        )
    )

    return schemas.IntelligentRAGResponse(
        query=request.query,
        selected_chunks=result['selected_chunks'],
        context_text=result['context_text'],
        total_tokens_used=result['total_tokens_used'],
        max_tokens_allowed=result['max_tokens_allowed'],
        relevance_stats=result['relevance_stats'],
        metadata={
            **result['metadata'],
            "response_time_ms": response_time,
            "rate_limit_info": result.get('rate_limit_info')
        },
        rate_limit_info=result.get('rate_limit_info')
    )

# Rate Limiting Status Endpoint
@app.get("/rate-limit/status/{agent_id}", dependencies=[Depends(get_api_key)])
async def get_rate_limit_status(agent_id: str):
    """Get rate limiting status for a specific agent."""
    rate_limiter = get_rate_limiter()
    stats = rate_limiter.get_agent_stats(agent_id)

    return {
        "agent_id": agent_id,
        "current_usage": stats["current_usage"],
        "limit_per_minute": stats["limit_per_minute"],
        "remaining_requests": stats["remaining_requests"],
        "can_make_request": stats["remaining_requests"] > 0
    }

# Semantic Search Endpoint
@app.post("/search/semantic/", response_model=schemas.SemanticSearchResponse, dependencies=[Depends(get_api_key)])
async def semantic_search(
    request: schemas.SemanticSearchRequest,
    db: Session = Depends(get_db)
):
    """Perform semantic search across knowledge base."""
    # Build search filters
    where_clause = {}
    if request.document_types:
        # ChromaDB doesn't support complex where clauses easily, so we'll filter in Python
        pass

    # Perform vector search
    vector_db = get_vector_db_manager()
    search_results = vector_db.search_similar(
        query=request.query,
        n_results=request.limit * 2  # Get more to allow for filtering
    )

    # Filter and format results
    results = []
    for chunk_result in search_results['results'][:request.limit]:
        # Get the full document
        document = crud.get_knowledge_document(db, chunk_result['document_id'])
        if not document:
            continue

        # Apply filters
        if request.document_types and document.document_type not in request.document_types:
            continue
        if request.date_from and document.created_at < request.date_from:
            continue
        if request.date_to and document.created_at > request.date_to:
            continue
        if request.min_validation_score and document.validation_score and document.validation_score < request.min_validation_score:
            continue

        # Get matched chunks for this document
        matched_chunks = vector_db.get_chunks_by_document(document.id)

        results.append(schemas.SemanticSearchResult(
            document=document,
            relevance_score=chunk_result['similarity_score'],
            matched_chunks=[schemas.DocumentChunk.from_orm(chunk) for chunk in matched_chunks[:3]]  # Top 3 chunks
        ))

    # Log the search
    crud.create_knowledge_query(
        db,
        schemas.KnowledgeQueryCreate(
            query_text=request.query,
            query_type="semantic_search",
            requester_agent=None,  # Could be added to request schema
            retrieved_document_ids=[r.document.id for r in results],
            relevance_scores={str(r.document.id): r.relevance_score for r in results}
        )
    )

    return schemas.SemanticSearchResponse(
        query=request.query,
        results=results,
        total_found=len(results),
        search_metadata={
            "filters_applied": {
                "document_types": request.document_types,
                "date_from": request.date_from.isoformat() if request.date_from else None,
                "date_to": request.date_to.isoformat() if request.date_to else None,
                "min_validation_score": request.min_validation_score
            }
        }
    )

# Research Insights Endpoints
@app.post("/insights/", response_model=schemas.ResearchInsight, dependencies=[Depends(get_api_key)])
async def create_research_insight(
    insight: schemas.ResearchInsightCreate,
    db: Session = Depends(get_db)
):
    """Create a new research insight."""
    db_insight = crud.create_research_insight(db, insight)

    # Log the creation
    crud.create_knowledge_update(
        db,
        schemas.KnowledgeUpdateCreate(
            operation_type="add",
            change_description=f"Created research insight: {insight.insight_text[:100]}...",
            performed_by=insight.discovered_by
        )
    )

    return db_insight

@app.get("/insights/", response_model=List[schemas.ResearchInsight], dependencies=[Depends(get_api_key)])
async def get_research_insights(
    insight_type: Optional[str] = None,
    validation_status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get research insights with optional filtering."""
    insights = crud.get_research_insights(
        db,
        insight_type=insight_type,
        validation_status=validation_status,
        min_confidence=min_confidence,
        limit=limit
    )
    return insights

@app.put("/insights/{insight_id}", response_model=schemas.ResearchInsight, dependencies=[Depends(get_api_key)])
async def update_research_insight(
    insight_id: int,
    updates: schemas.ResearchInsightUpdate,
    db: Session = Depends(get_db)
):
    """Update a research insight."""
    db_insight = crud.update_research_insight(db, insight_id, updates)
    if not db_insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    return db_insight

# Learning Patterns Endpoints
@app.post("/patterns/", response_model=schemas.LearningPattern, dependencies=[Depends(get_api_key)])
async def create_learning_pattern(
    pattern: schemas.LearningPatternCreate,
    db: Session = Depends(get_db)
):
    """Create a new learning pattern."""
    return crud.create_learning_pattern(db, pattern)

@app.get("/patterns/", response_model=List[schemas.LearningPattern], dependencies=[Depends(get_api_key)])
async def get_learning_patterns(
    pattern_type: Optional[str] = None,
    min_success_rate: Optional[float] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get learning patterns with optional filtering."""
    patterns = crud.get_learning_patterns(
        db,
        pattern_type=pattern_type,
        min_success_rate=min_success_rate,
        limit=limit
    )
    return patterns

# Analytics Endpoint
@app.get("/analytics/", response_model=schemas.KnowledgeAnalytics, dependencies=[Depends(get_api_key)])
async def get_knowledge_analytics(db: Session = Depends(get_db)):
    """Get comprehensive knowledge base analytics."""
    return crud.get_knowledge_analytics(db)

# Background task functions
def process_document_chunks(document_id: int, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Process document chunks and add to vector database."""
    from .database import SessionLocal

    db = SessionLocal()
    try:
        # Get the document
        document = crud.get_knowledge_document(db, document_id)
        if not document:
            logger.error(f"Document {document_id} not found for chunking")
            return

        # Initialize chunker and vector DB
        # Determine domain context based on document type
        domain_context = "alzheimer" if "alzheimer" in document.document_type.lower() else "general"
        chunker = IntelligentTextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            domain_context=domain_context
        )
        vector_db = get_vector_db_manager()

        # Chunk the document
        chunks = chunker.chunk_text(
            document.content,
            metadata={
                "document_type": document.document_type,
                "source_agent": document.source_agent,
                "tags": document.tags or [],
                "document_id": document.id,
                "title": document.title
            }
        )

        # Add embedding model info to chunks
        for chunk in chunks:
            chunk['embedding_model'] = vector_db.embedding_model_name

        # Store chunks in database
        for chunk_data in chunks:
            chunk_create = schemas.DocumentChunkCreate(
                document_id=document_id,
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                embedding_model=chunk_data['embedding_model'],
                chunk_metadata=chunk_data['chunk_metadata']
            )
            crud.create_document_chunk(db, chunk_create)

        # Add to vector database
        vector_db.add_document_chunks(chunks, document_id)

        logger.info(f"Successfully processed {len(chunks)} chunks for document {document_id}")

    except Exception as e:
        logger.error(f"Failed to process chunks for document {document_id}: {str(e)}")
    finally:
        db.close()

def get_surrounding_chunks(db: Session, document_id: int, chunk_index: int, window: int) -> List[Dict[str, Any]]:
    """Get surrounding chunks for context."""
    chunks = crud.get_document_chunks_by_document(db, document_id)

    surrounding = []
    for chunk in chunks:
        if abs(chunk.chunk_index - chunk_index) <= window and chunk.chunk_index != chunk_index:
            surrounding.append({
                'chunk_index': chunk.chunk_index,
                'content': chunk.content,
                'metadata': chunk.chunk_metadata
            })

    return sorted(surrounding, key=lambda x: x['chunk_index'])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)