from .celery_app import celery_app
from .database import SessionLocal
from . import crud, schemas
from .vector_db import get_vector_db_manager, TextChunker
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="ingest_knowledge_task")
def ingest_knowledge_task(self, knowledge_data: dict):
    """Async task to ingest knowledge into the system."""
    db = SessionLocal()
    try:
        # Create document
        doc_create = schemas.KnowledgeDocumentCreate(**knowledge_data)
        db_document = crud.create_knowledge_document(db, doc_create)

        # Process chunks
        chunker = TextChunker()
        vector_db = get_vector_db_manager()

        chunks = chunker.chunk_text(db_document.content)
        for chunk in chunks:
            chunk['embedding_model'] = vector_db.embedding_model_name

        # Store chunks in database
        for chunk_data in chunks:
            chunk_create = schemas.DocumentChunkCreate(
                document_id=db_document.id,
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                embedding_model=chunk_data['embedding_model'],
                chunk_metadata=chunk_data['chunk_metadata']
            )
            crud.create_document_chunk(db, chunk_create)

        # Add to vector database
        vector_db.add_document_chunks(chunks, db_document.id)

        # Log the update
        crud.create_knowledge_update(
            db,
            schemas.KnowledgeUpdateCreate(
                operation_type="add",
                document_id=db_document.id,
                change_description=f"Async ingestion completed: {db_document.title}",
                performed_by=knowledge_data.get('source_agent', 'system')
            )
        )

        logger.info(f"Successfully ingested knowledge document {db_document.id}")
        return {"document_id": db_document.id, "chunks_created": len(chunks)}

    except Exception as e:
        logger.error(f"Failed to ingest knowledge: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="update_embeddings_task")
def update_embeddings_task(self, document_ids: list):
    """Update embeddings for specified documents."""
    db = SessionLocal()
    vector_db = get_vector_db_manager()

    try:
        updated_count = 0
        for doc_id in document_ids:
            # Remove old embeddings
            vector_db.delete_document_chunks(doc_id)

            # Get document and chunks
            document = crud.get_knowledge_document(db, doc_id)
            if not document:
                continue

            chunks = crud.get_document_chunks_by_document(db, doc_id)

            # Re-embed chunks
            chunk_dicts = []
            for chunk in chunks:
                chunk_dicts.append({
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'embedding_model': vector_db.embedding_model_name,
                    'chunk_metadata': chunk.chunk_metadata or {}
                })

            vector_db.add_document_chunks(chunk_dicts, doc_id)
            updated_count += 1

        logger.info(f"Updated embeddings for {updated_count} documents")
        return {"updated_documents": updated_count}

    except Exception as e:
        logger.error(f"Failed to update embeddings: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="validate_knowledge_task")
def validate_knowledge_task(self, document_ids: list, validation_method: str = "statistical"):
    """Validate knowledge documents using specified method."""
    db = SessionLocal()
    try:
        validated_count = 0
        for doc_id in document_ids:
            document = crud.get_knowledge_document(db, doc_id)
            if not document:
                continue

            # Perform validation based on method
            if validation_method == "statistical":
                # Placeholder for statistical validation integration
                validation_score = 0.85  # Mock score
            elif validation_method == "peer_review":
                validation_score = 0.90  # Mock score
            else:
                validation_score = 0.75  # Default

            # Update document
            updates = schemas.KnowledgeDocumentUpdate(
                is_validated=True,
                validation_score=validation_score
            )
            crud.update_knowledge_document(db, doc_id, updates)

            # Log validation
            crud.create_knowledge_update(
                db,
                schemas.KnowledgeUpdateCreate(
                    operation_type="validate",
                    document_id=doc_id,
                    change_description=f"Validated document using {validation_method} method",
                    performed_by="validation_system",
                    new_metadata={"validation_score": validation_score}
                )
            )

            validated_count += 1

        logger.info(f"Validated {validated_count} documents using {validation_method}")
        return {"validated_documents": validated_count, "method": validation_method}

    except Exception as e:
        logger.error(f"Failed to validate knowledge: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="extract_insights_task")
def extract_insights_task(self, document_ids: list, extraction_prompt: str = None):
    """Extract research insights from documents."""
    db = SessionLocal()
    try:
        insights_extracted = 0

        for doc_id in document_ids:
            document = crud.get_knowledge_document(db, doc_id)
            if not document:
                continue

            # Get document chunks for context
            chunks = crud.get_document_chunks_by_document(db, doc_id)
            full_content = " ".join([chunk.content for chunk in chunks])

            # Extract insights (placeholder - would integrate with LLM service)
            # For now, create mock insights based on document type
            if document.document_type == "research_finding":
                insight_text = f"Key finding from {document.source_agent}: {document.title}"
                confidence_level = 0.8
            elif document.document_type == "hypothesis":
                insight_text = f"Research hypothesis identified: {document.title}"
                confidence_level = 0.7
            else:
                insight_text = f"General insight: {document.title}"
                confidence_level = 0.6

            # Create insight
            insight_create = schemas.ResearchInsightCreate(
                insight_text=insight_text,
                insight_type=document.document_type,
                confidence_level=confidence_level,
                supporting_evidence={"document_id": doc_id, "source": document.source_agent},
                discovered_by=document.source_agent,
                validation_status="pending"
            )

            crud.create_research_insight(db, insight_create)
            insights_extracted += 1

        logger.info(f"Extracted {insights_extracted} insights from {len(document_ids)} documents")
        return {"insights_extracted": insights_extracted, "documents_processed": len(document_ids)}

    except Exception as e:
        logger.error(f"Failed to extract insights: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()

@celery_app.task(bind=True, name="consolidate_learning_patterns_task")
def consolidate_learning_patterns_task(self):
    """Analyze recent activity to identify and update learning patterns."""
    db = SessionLocal()
    try:
        # Get recent successful tasks and patterns
        recent_updates = crud.get_knowledge_updates(db, operation_type="add", days=30)

        # Analyze patterns (simplified version)
        pattern_updates = 0

        # Group by agent and operation type
        agent_patterns = {}
        for update in recent_updates:
            agent = update.performed_by
            if agent not in agent_patterns:
                agent_patterns[agent] = {"total_actions": 0, "successful_contributions": 0}

            agent_patterns[agent]["total_actions"] += 1
            if "research_finding" in update.change_description.lower():
                agent_patterns[agent]["successful_contributions"] += 1

        # Update or create learning patterns
        for agent, stats in agent_patterns.items():
            success_rate = stats["successful_contributions"] / stats["total_actions"] if stats["total_actions"] > 0 else 0

            # Check if pattern exists
            existing_patterns = crud.get_learning_patterns(db, pattern_type="successful_methodology")
            pattern_exists = any(p.pattern_description.startswith(f"Agent {agent}") for p in existing_patterns)

            if not pattern_exists and success_rate > 0.7:
                # Create new pattern
                pattern_create = schemas.LearningPatternCreate(
                    pattern_type="successful_methodology",
                    pattern_description=f"Agent {agent} demonstrates high success rate ({success_rate:.2%}) in knowledge contribution",
                    success_rate=success_rate,
                    discovered_from={"agent": agent, "analysis_period_days": 30}
                )
                crud.create_learning_pattern(db, pattern_create)
                pattern_updates += 1

        logger.info(f"Consolidated {pattern_updates} learning patterns")
        return {"patterns_updated": pattern_updates}

    except Exception as e:
        logger.error(f"Failed to consolidate learning patterns: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise
    finally:
        db.close()