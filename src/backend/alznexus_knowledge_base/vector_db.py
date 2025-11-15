import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Manages ChromaDB vector database operations for knowledge base."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "alznexus_knowledge"
    ):
        """Initialize the vector database manager."""
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)

        logger.info(f"Vector database initialized with collection: {collection_name}")

    def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add document chunks to the vector database."""
        if not chunks:
            return []

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"doc_{document_id}_chunk_{chunk['chunk_index']}"
            ids.append(chunk_id)

            documents.append(chunk['content'])

            # Prepare metadata
            chunk_metadata = {
                "document_id": str(document_id),
                "chunk_index": chunk['chunk_index'],
                "embedding_model": chunk.get('embedding_model', self.embedding_model_name),
                "created_at": datetime.utcnow().isoformat(),
                **(metadata or {}),
                **(chunk.get('chunk_metadata') or {})
            }
            metadatas.append(chunk_metadata)

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks for document {document_id}")
        return ids

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using semantic similarity."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, (doc_id, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'chunk_id': doc_id,
                    'document_id': int(metadata.get('document_id', 0)),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'content': results['documents'][0][i] if results['documents'] and results['documents'][0] else "",
                    'metadata': metadata,
                    'similarity_score': 1.0 - distance  # Convert distance to similarity
                })

        return {
            'query': query,
            'results': formatted_results,
            'total_found': len(formatted_results)
        }

    def get_chunks_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        results = self.collection.get(
            where={"document_id": str(document_id)},
            include=['documents', 'metadatas']
        )

        chunks = []
        if results['ids']:
            for i, chunk_id in enumerate(results['ids']):
                chunks.append({
                    'chunk_id': chunk_id,
                    'document_id': document_id,
                    'chunk_index': results['metadatas'][i].get('chunk_index', 0),
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })

        return sorted(chunks, key=lambda x: x['chunk_index'])

    def delete_document_chunks(self, document_id: int) -> bool:
        """Delete all chunks for a specific document."""
        try:
            self.collection.delete(where={"document_id": str(document_id)})
            logger.info(f"Deleted chunks for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
            return False

    def update_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a specific chunk."""
        try:
            # Get existing metadata
            existing = self.collection.get(ids=[chunk_id], include=['metadatas'])
            if not existing['metadatas']:
                return False

            current_metadata = existing['metadatas'][0]
            updated_metadata = {**current_metadata, **metadata}

            self.collection.update(
                ids=[chunk_id],
                metadatas=[updated_metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for chunk {chunk_id}: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {'error': str(e)}

class TextChunker:
    """Intelligent text chunking for knowledge documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator_patterns: Optional[List[str]] = None
    ):
        """Initialize the text chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_patterns = separator_patterns or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            " ",     # Word boundaries
        ]

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into semantically meaningful pieces."""
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Try to find a good breaking point
                chunk_text = text[start:end]
                best_break = self._find_best_break_point(chunk_text, text[end:])

                if best_break > 0:
                    chunk_text = chunk_text[:best_break]
                    end = start + best_break

            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    'chunk_index': chunk_index,
                    'content': chunk_text.strip(),
                    'chunk_metadata': {
                        'start_position': start,
                        'end_position': end,
                        'length': len(chunk_text),
                        **(metadata or {})
                    }
                })
                chunk_index += 1

            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _find_best_break_point(self, chunk_text: str, remaining_text: str) -> int:
        """Find the best point to break the chunk."""
        # Try separator patterns in order of preference
        for separator in self.separator_patterns:
            # Look for separator in the last part of chunk_text
            search_start = max(0, len(chunk_text) - 200)  # Look in last 200 chars
            last_separator = chunk_text.rfind(separator, search_start)

            if last_separator > len(chunk_text) * 0.5:  # Don't break too early
                return last_separator + len(separator)

        # If no good separator found, try to break at word boundary in remaining text
        for i in range(min(50, len(remaining_text))):
            if remaining_text[i] in [' ', '\n', '\t']:
                return len(chunk_text) + i

        # Fallback: break at chunk_size
        return len(chunk_text)

# Global instance for easy access
vector_db_manager = None

def get_vector_db_manager() -> VectorDatabaseManager:
    """Get or create the global vector database manager."""
    global vector_db_manager
    if vector_db_manager is None:
        persist_dir = os.getenv("KNOWLEDGE_VECTOR_DB_PATH", "./chroma_db")
        embedding_model = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        collection_name = os.getenv("KNOWLEDGE_COLLECTION_NAME", "alznexus_knowledge")

        vector_db_manager = VectorDatabaseManager(
            persist_directory=persist_dir,
            embedding_model_name=embedding_model,
            collection_name=collection_name
        )
    return vector_db_manager

# Intelligent Context Retrieval System
import tiktoken
import time
import threading
from collections import defaultdict

class TokenEstimator:
    """Estimates token counts for different LLM models."""

    # Token limits for common models (approximate)
    MODEL_TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "gemini-1.0-pro": 30720,
        "gemini-1.5-flash": 1048576,
        "gemini-1.5-pro": 2097152,
        "claude-3-haiku": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-opus": 200000,
    }

    def __init__(self):
        # Initialize tiktoken encoders for different models
        self.encoders = {}
        try:
            self.encoders["gpt"] = tiktoken.get_encoding("cl100k_base")  # GPT-4, GPT-3.5
        except:
            self.encoders["gpt"] = None

        try:
            self.encoders["claude"] = tiktoken.get_encoding("cl100k_base")  # Claude uses similar
        except:
            self.encoders["claude"] = None

    def estimate_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """Estimate token count for a given text and model."""
        if not text:
            return 0

        # Choose appropriate encoder
        if "gpt" in model_name.lower():
            encoder = self.encoders.get("gpt")
        elif "claude" in model_name.lower():
            encoder = self.encoders.get("claude")
        else:
            # Fallback: rough character-based estimation
            return len(text) // 4  # ~4 chars per token

        if encoder:
            try:
                return len(encoder.encode(text))
            except:
                return len(text) // 4
        else:
            return len(text) // 4

    def get_model_token_limit(self, model_name: str) -> int:
        """Get the token limit for a specific model."""
        # Extract base model name
        base_name = model_name.lower().split("-")[0] if "-" in model_name else model_name.lower()

        for model_key, limit in self.MODEL_TOKEN_LIMITS.items():
            if base_name in model_key or model_key in model_name.lower():
                return limit

        # Default fallback
        return 4096

class RateLimiter:
    """Rate limiter for agent requests to prevent 429 errors."""

    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.agent_requests = defaultdict(list)
        self.lock = threading.Lock()

    def check_rate_limit(self, agent_id: str) -> Tuple[bool, int]:
        """
        Check if agent can make a request.
        Returns: (allowed: bool, backoff_seconds: int)
        """
        with self.lock:
            current_time = time.time()
            # Clean old requests (older than 1 minute)
            cutoff_time = current_time - 60
            self.agent_requests[agent_id] = [
                req_time for req_time in self.agent_requests[agent_id]
                if req_time > cutoff_time
            ]

            current_requests = len(self.agent_requests[agent_id])

            if current_requests >= self.requests_per_minute:
                # Calculate backoff time until oldest request expires
                oldest_request = min(self.agent_requests[agent_id])
                backoff_seconds = int(60 - (current_time - oldest_request)) + 1
                return False, backoff_seconds

            return True, 0

    def record_request(self, agent_id: str):
        """Record a successful request."""
        with self.lock:
            self.agent_requests[agent_id].append(time.time())

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get rate limiting stats for an agent."""
        with self.lock:
            current_requests = len([
                req_time for req_time in self.agent_requests[agent_id]
                if time.time() - req_time < 60
            ])

            return {
                "current_usage": current_requests,
                "limit_per_minute": self.requests_per_minute,
                "remaining_requests": max(0, self.requests_per_minute - current_requests)
            }

class IntelligentContextRetriever:
    """Intelligent context retrieval with token awareness and rate limiting."""

    def __init__(self, vector_db_manager: VectorDatabaseManager):
        self.vector_db = vector_db_manager
        self.token_estimator = TokenEstimator()
        self.rate_limiter = RateLimiter(requests_per_minute=10)  # Configurable

    def retrieve_context(
        self,
        query: str,
        model_name: str = "gemini-1.5-flash",
        max_tokens: Optional[int] = None,
        token_reserve: int = 1000,
        min_relevance_score: float = 0.6,
        max_chunks: int = 10,
        context_window: int = 1,
        requester_agent: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        prioritize_recent: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Intelligently retrieve context with token limits and relevance ranking.
        """

        # Check rate limiting
        if requester_agent:
            allowed, backoff = self.rate_limiter.check_rate_limit(requester_agent)
            if not allowed:
                return {
                    "error": "rate_limit_exceeded",
                    "backoff_seconds": backoff,
                    "rate_limit_info": self.rate_limiter.get_agent_stats(requester_agent)
                }

        # Get model token limit
        if max_tokens is None:
            max_tokens = self.token_estimator.get_model_token_limit(model_name)

        available_tokens = max_tokens - token_reserve
        if available_tokens <= 0:
            return {"error": "insufficient_token_budget", "available_tokens": available_tokens}

        # Perform initial search with higher limit to allow for filtering
        search_limit = max_chunks * 3  # Get more candidates for better selection

        # Build search filters
        where_clause = {}
        if document_types:
            # Note: ChromaDB has limitations with complex filters, so we'll filter in Python
            pass

        search_results = self.vector_db.search_similar(
            query=query,
            n_results=search_limit,
            where=where_clause
        )

        # Filter and rank candidates
        candidates = []
        for chunk_result in search_results['results']:
            if chunk_result['similarity_score'] >= min_relevance_score:
                # Get document info
                document_id = chunk_result['document_id']
                # Note: In a real implementation, you'd fetch document metadata here

                # Estimate tokens for this chunk
                token_count = self.token_estimator.estimate_tokens(
                    chunk_result['content'], model_name
                )

                candidates.append({
                    'chunk_id': chunk_result['chunk_id'],
                    'document_id': document_id,
                    'content': chunk_result['content'],
                    'relevance_score': chunk_result['similarity_score'],
                    'token_count': token_count,
                    'metadata': chunk_result['metadata']
                })

        # Filter by document types if specified
        if document_types:
            candidates = [
                c for c in candidates
                if c['metadata'].get('document_type') in document_types
            ]

        # Sort by relevance score (highest first)
        candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        # If prioritizing recent, boost more recent documents
        if prioritize_recent:
            current_time = time.time()
            for candidate in candidates:
                # Simple recency boost based on some timestamp in metadata
                # In practice, you'd have proper timestamp fields
                age_days = candidate['metadata'].get('age_days', 30)  # Default assumption
                recency_boost = max(0, (30 - age_days) / 30) * 0.1  # Small boost for recent items
                candidate['relevance_score'] += recency_boost

            # Re-sort after boosting
            candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Select chunks within token limit using greedy algorithm
        selected_chunks = []
        total_tokens = 0

        for candidate in candidates:
            if len(selected_chunks) >= max_chunks:
                break

            if total_tokens + candidate['token_count'] <= available_tokens:
                selected_chunks.append(candidate)
                total_tokens += candidate['token_count']
            else:
                # Try to fit a smaller portion if possible
                remaining_tokens = available_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful chunk size
                    truncated_content = self._truncate_to_tokens(
                        candidate['content'], remaining_tokens, model_name
                    )
                    if truncated_content:
                        candidate_copy = candidate.copy()
                        candidate_copy['content'] = truncated_content
                        candidate_copy['token_count'] = remaining_tokens
                        candidate_copy['truncated'] = True
                        selected_chunks.append(candidate_copy)
                        total_tokens += remaining_tokens
                        break

        # Add context window chunks if requested
        if context_window > 0:
            selected_chunks = self._add_context_window(
                selected_chunks, context_window, available_tokens, total_tokens, model_name
            )

        # Build final context text
        context_parts = []
        final_tokens = 0

        for chunk in selected_chunks:
            context_parts.append(chunk['content'])
            final_tokens += chunk['token_count']

        context_text = "\n\n".join(context_parts)

        # Record the request for rate limiting
        if requester_agent:
            self.rate_limiter.record_request(requester_agent)

        # Calculate relevance statistics
        relevance_stats = {
            'total_candidates': len(candidates),
            'selected_chunks': len(selected_chunks),
            'avg_relevance': sum(c['relevance_score'] for c in selected_chunks) / len(selected_chunks) if selected_chunks else 0,
            'min_relevance': min((c['relevance_score'] for c in selected_chunks), default=0),
            'max_relevance': max((c['relevance_score'] for c in selected_chunks), default=0)
        }

        return {
            'query': query,
            'selected_chunks': selected_chunks,
            'context_text': context_text,
            'total_tokens_used': final_tokens,
            'max_tokens_allowed': max_tokens,
            'available_tokens': available_tokens,
            'relevance_stats': relevance_stats,
            'rate_limit_info': self.rate_limiter.get_agent_stats(requester_agent) if requester_agent else None,
            'metadata': {
                'model_name': model_name,
                'min_relevance_score': min_relevance_score,
                'context_window': context_window,
                'prioritize_recent': prioritize_recent,
                'include_metadata': include_metadata
            }
        }

    def _truncate_to_tokens(self, text: str, max_tokens: int, model_name: str) -> Optional[str]:
        """Truncate text to fit within token limit."""
        if self.token_estimator.estimate_tokens(text, model_name) <= max_tokens:
            return text

        # Binary search for truncation point
        left, right = 0, len(text)
        best_truncation = ""

        while left <= right:
            mid = (left + right) // 2
            truncated = text[:mid]
            token_count = self.token_estimator.estimate_tokens(truncated, model_name)

            if token_count <= max_tokens:
                best_truncation = truncated
                left = mid + 1
            else:
                right = mid - 1

        # Try to end at a sentence boundary
        if best_truncation:
            last_sentence_end = max(
                best_truncation.rfind('. '),
                best_truncation.rfind('! '),
                best_truncation.rfind('? ')
            )
            if last_sentence_end > len(best_truncation) * 0.8:  # Don't truncate too much
                best_truncation = best_truncation[:last_sentence_end + 1]

        return best_truncation if best_truncation else None

    def _add_context_window(
        self,
        selected_chunks: List[Dict[str, Any]],
        context_window: int,
        available_tokens: int,
        current_tokens: int,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Add surrounding context chunks within token limits."""
        if context_window <= 0:
            return selected_chunks

        enhanced_chunks = []
        used_document_chunks = set()

        for chunk in selected_chunks:
            enhanced_chunks.append(chunk)
            used_document_chunks.add((chunk['document_id'], chunk['metadata'].get('chunk_index', 0)))

            # Get surrounding chunks
            surrounding = self.vector_db.get_chunks_by_document(chunk['document_id'])

            # Find current chunk index
            current_index = chunk['metadata'].get('chunk_index', 0)

            # Add chunks within window
            for surround_chunk in surrounding:
                surround_index = surround_chunk['metadata'].get('chunk_index', 0)
                if abs(surround_index - current_index) <= context_window and surround_index != current_index:
                    # Check if we haven't already included this chunk
                    chunk_key = (chunk['document_id'], surround_index)
                    if chunk_key not in used_document_chunks:
                        # Check token limit
                        surround_tokens = self.token_estimator.estimate_tokens(
                            surround_chunk['content'], model_name
                        )

                        if current_tokens + surround_tokens <= available_tokens:
                            context_chunk = {
                                'chunk_id': surround_chunk['chunk_id'],
                                'document_id': chunk['document_id'],
                                'content': surround_chunk['content'],
                                'relevance_score': chunk['relevance_score'] * 0.8,  # Slightly lower relevance for context
                                'token_count': surround_tokens,
                                'metadata': {**surround_chunk['metadata'], 'is_context': True},
                                'is_context': True
                            }
                            enhanced_chunks.append(context_chunk)
                            used_document_chunks.add(chunk_key)
                            current_tokens += surround_tokens

        # Sort by document and chunk index for coherent reading
        enhanced_chunks.sort(key=lambda x: (x['document_id'], x['metadata'].get('chunk_index', 0)))

        return enhanced_chunks

# Global instances
intelligent_retriever = None
rate_limiter = None

def get_intelligent_retriever() -> IntelligentContextRetriever:
    """Get or create the global intelligent context retriever."""
    global intelligent_retriever
    if intelligent_retriever is None:
        vector_db = get_vector_db_manager()
        intelligent_retriever = IntelligentContextRetriever(vector_db)
    return intelligent_retriever

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    return rate_limiter