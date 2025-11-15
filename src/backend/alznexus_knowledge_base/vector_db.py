import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from datetime import datetime
import re
from collections import defaultdict

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

class IntelligentTextChunker:
    """Advanced text chunking that analyzes content semantics and adds rich metadata."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        domain_context: str = "general"
    ):
        """Initialize the intelligent text chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.domain_context = domain_context

        # Domain-specific patterns for Alzheimer's research
        self.domain_patterns = {
            "alzheimer": {
                "biomarkers": [
                    r"biomarker", r"amyloid", r"tau", r"p-tau", r"abeta", r"CSF",
                    r"plasma", r"blood", r"imaging", r"PET", r"MRI", r"FDG"
                ],
                "symptoms": [
                    r"cognition", r"memory", r"dementia", r"MCI", r"cognitive decline",
                    r"Alzheimer's disease", r"AD", r"neurodegenerative"
                ],
                "treatments": [
                    r"drug", r"therapy", r"treatment", r"clinical trial", r"phase",
                    r"efficacy", r"side effects", r"dosage", r"administration"
                ],
                "genetics": [
                    r"APOE", r"PSEN1", r"PSEN2", r"APP", r"mutation", r"genetic",
                    r"risk factor", r"heritability"
                ],
                "pathways": [
                    r"pathway", r"mechanism", r"cascade", r"amyloid cascade",
                    r"tau pathology", r"neuroinflammation", r"synaptic dysfunction"
                ]
            }
        }

        # Content type patterns
        self.content_patterns = {
            "methodology": [
                r"method", r"protocol", r"procedure", r"experimental design",
                r"statistical analysis", r"data collection", r"assay"
            ],
            "results": [
                r"result", r"finding", r"outcome", r"significant", r"p-value",
                r"correlation", r"association", r"effect size"
            ],
            "conclusion": [
                r"conclusion", r"summary", r"implication", r"future direction",
                r"clinical relevance", r"therapeutic potential"
            ],
            "references": [
                r"reference", r"citation", r"literature", r"previous study",
                r"published", r"review"
            ]
        }

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text with intelligent semantic analysis and rich metadata."""
        if not text:
            return []

        # Pre-analyze the entire text for structure
        text_analysis = self._analyze_text_structure(text)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Find optimal chunk boundaries based on semantic analysis
            chunk_boundaries = self._find_semantic_chunk_boundary(
                text, start, self.chunk_size, text_analysis
            )

            chunk_text = text[start:chunk_boundaries['end']]
            chunk_metadata = self._extract_chunk_metadata(
                chunk_text, start, chunk_boundaries['end'], text_analysis
            )

            if chunk_text.strip():
                chunks.append({
                    'chunk_index': chunk_index,
                    'content': chunk_text.strip(),
                    'chunk_metadata': {
                        'start_position': start,
                        'end_position': chunk_boundaries['end'],
                        'length': len(chunk_text),
                        'semantic_score': chunk_boundaries['semantic_score'],
                        'content_type': chunk_metadata['content_type'],
                        'key_entities': chunk_metadata['key_entities'],
                        'domain_relevance': chunk_metadata['domain_relevance'],
                        'recall_context': chunk_metadata['recall_context'],
                        'relationship_hints': chunk_metadata['relationship_hints'],
                        **(metadata or {})
                    }
                })
                chunk_index += 1

            # Move start position with intelligent overlap
            start = max(start + 1, chunk_boundaries['end'] - self.chunk_overlap)

        return chunks

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the overall structure and content of the text."""
        analysis = {
            'sentences': [],
            'paragraphs': [],
            'sections': [],
            'domain_entities': defaultdict(list),
            'content_types': defaultdict(list),
            'key_phrases': []
        }

        # Split into sentences and paragraphs
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        analysis['sentences'] = sentences
        analysis['paragraphs'] = paragraphs

        # Analyze domain-specific content
        for category, patterns in self.domain_patterns.get(self.domain_context, {}).items():
            for pattern in patterns:
                matches = [i for i, sent in enumerate(sentences) if pattern.lower() in sent.lower()]
                if matches:
                    analysis['domain_entities'][category].extend(matches)

        # Analyze content types
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                matches = [i for i, sent in enumerate(sentences) if pattern.lower() in sent.lower()]
                if matches:
                    analysis['content_types'][content_type].extend(matches)

        return analysis

    def _find_semantic_chunk_boundary(self, text: str, start: int, max_size: int,
                                    text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Find the optimal boundary for a chunk based on semantic analysis."""
        end = min(start + max_size, len(text))

        if end >= len(text):
            return {
                'end': end,
                'semantic_score': 1.0,
                'boundary_type': 'end_of_text'
            }

        # Look for semantic boundaries within the chunk window
        best_boundary = end
        best_score = 0.0
        boundary_type = 'size_limit'

        chunk_text = text[start:end]

        # Prefer paragraph boundaries
        paragraph_breaks = []
        pos = start
        for para in text_analysis['paragraphs']:
            para_start = text.find(para, pos)
            if para_start >= start and para_start < end:
                para_end = para_start + len(para)
                if para_end > start and para_end < end:
                    paragraph_breaks.append(para_end)

        if paragraph_breaks:
            best_boundary = max(paragraph_breaks)
            best_score = 0.9
            boundary_type = 'paragraph'

        # Look for sentence boundaries if no good paragraph break
        if best_score < 0.8:
            sentences = text_analysis['sentences']
            for i, sentence in enumerate(sentences):
                sent_pos = text.find(sentence, start)
                if sent_pos >= start and sent_pos + len(sentence) <= end:
                    sent_end = sent_pos + len(sentence)
                    if sent_end > start and abs(sent_end - end) < abs(best_boundary - end):
                        # Calculate semantic coherence score
                        semantic_score = self._calculate_semantic_coherence(
                            text[start:sent_end], text_analysis, i
                        )
                        if semantic_score > best_score:
                            best_boundary = sent_end
                            best_score = semantic_score
                            boundary_type = 'sentence'

        return {
            'end': best_boundary,
            'semantic_score': best_score,
            'boundary_type': boundary_type
        }

    def _calculate_semantic_coherence(self, chunk_text: str, text_analysis: Dict[str, Any],
                                    sentence_idx: int) -> float:
        """Calculate how semantically coherent a chunk is."""
        score = 0.5  # Base score

        # Check for domain entity concentration
        domain_entities = 0
        for category, positions in text_analysis['domain_entities'].items():
            if sentence_idx in positions:
                domain_entities += 1

        if domain_entities > 0:
            score += min(domain_entities * 0.2, 0.3)  # Up to 0.3 for domain relevance

        # Check for content type coherence
        content_types = 0
        for content_type, positions in text_analysis['content_types'].items():
            if sentence_idx in positions:
                content_types += 1

        if content_types > 0:
            score += min(content_types * 0.15, 0.2)  # Up to 0.2 for content coherence

        # Length appropriateness (prefer chunks that are substantial but not too long)
        length_ratio = len(chunk_text) / self.chunk_size
        if 0.7 <= length_ratio <= 1.0:
            score += 0.1

        return min(score, 1.0)

    def _extract_chunk_metadata(self, chunk_text: str, start: int, end: int,
                              text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rich metadata for the chunk to aid recall."""
        metadata = {
            'content_type': 'general',
            'key_entities': [],
            'domain_relevance': {},
            'recall_context': {},
            'relationship_hints': []
        }

        # Determine primary content type
        content_type_scores = defaultdict(float)
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if pattern.lower() in chunk_text.lower():
                    content_type_scores[content_type] += 1

        if content_type_scores:
            metadata['content_type'] = max(content_type_scores, key=content_type_scores.get)

        # Extract key entities and domain relevance
        for category, patterns in self.domain_patterns.get(self.domain_context, {}).items():
            entities_found = []
            relevance_score = 0

            for pattern in patterns:
                if pattern.lower() in chunk_text.lower():
                    entities_found.append(pattern)
                    relevance_score += 1

            if entities_found:
                metadata['key_entities'].extend(entities_found)
                metadata['domain_relevance'][category] = {
                    'score': relevance_score,
                    'entities': entities_found
                }

        # Generate recall context hints
        metadata['recall_context'] = {
            'temporal_indicators': self._extract_temporal_context(chunk_text),
            'methodological_context': self._extract_methodological_context(chunk_text),
            'relationship_context': self._extract_relationship_context(chunk_text),
            'importance_indicators': self._extract_importance_indicators(chunk_text)
        }

        # Generate relationship hints for connecting this chunk to others
        metadata['relationship_hints'] = self._generate_relationship_hints(
            chunk_text, text_analysis, start, end
        )

        return metadata

    def _extract_temporal_context(self, text: str) -> List[str]:
        """Extract temporal indicators for recall context."""
        temporal_keywords = [
            'recently', 'previously', 'currently', 'future', 'ongoing',
            'clinical trial', 'phase', 'study', 'research', 'published'
        ]
        return [word for word in temporal_keywords if word in text.lower()]

    def _extract_methodological_context(self, text: str) -> List[str]:
        """Extract methodological context for scientific recall."""
        method_keywords = [
            'statistical', 'analysis', 'correlation', 'regression', 'p-value',
            'significant', 'cohort', 'randomized', 'controlled', 'double-blind'
        ]
        return [word for word in method_keywords if word in text.lower()]

    def _extract_relationship_context(self, text: str) -> List[str]:
        """Extract relationship indicators between concepts."""
        relationship_keywords = [
            'associated with', 'correlated with', 'linked to', 'related to',
            'causes', 'leads to', 'results in', 'affects', 'influences'
        ]
        found_relationships = []
        for keyword in relationship_keywords:
            if keyword in text.lower():
                found_relationships.append(keyword)
        return found_relationships

    def _extract_importance_indicators(self, text: str) -> List[str]:
        """Extract indicators of importance or significance."""
        importance_keywords = [
            'important', 'significant', 'key', 'critical', 'major',
            'breakthrough', 'novel', 'innovative', 'promising', 'potential'
        ]
        return [word for word in importance_keywords if word in text.lower()]

    def _generate_relationship_hints(self, chunk_text: str, text_analysis: Dict[str, Any],
                                   start: int, end: int) -> List[str]:
        """Generate hints about how this chunk relates to other content."""
        hints = []

        # Check for continuation patterns
        if 'continued' in chunk_text.lower() or 'following' in chunk_text.lower():
            hints.append("continuation_of_previous")

        # Check for summary patterns
        if 'summary' in chunk_text.lower() or 'conclusion' in chunk_text.lower():
            hints.append("summarizes_previous_content")

        # Check for methodological connections
        if any(word in chunk_text.lower() for word in ['method', 'protocol', 'procedure']):
            hints.append("describes_methodology")

        # Check for result connections
        if any(word in chunk_text.lower() for word in ['result', 'finding', 'outcome']):
            hints.append("contains_results")

        return hints

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

        # Filter and rank candidates with semantic metadata enhancement
        candidates = []
        for chunk_result in search_results['results']:
            if chunk_result['similarity_score'] >= min_relevance_score:
                # Get document info
                document_id = chunk_result['document_id']

                # Estimate tokens for this chunk
                token_count = self.token_estimator.estimate_tokens(
                    chunk_result['content'], model_name
                )

                # Calculate enhanced relevance score using semantic metadata
                enhanced_score = self._calculate_enhanced_relevance_score(
                    chunk_result, query, min_relevance_score
                )

                candidates.append({
                    'chunk_id': chunk_result['chunk_id'],
                    'document_id': document_id,
                    'content': chunk_result['content'],
                    'relevance_score': enhanced_score,
                    'base_similarity': chunk_result['similarity_score'],
                    'token_count': token_count,
                    'metadata': chunk_result['metadata'],
                    'semantic_metadata': chunk_result['metadata'].get('chunk_metadata', {})
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

    def _calculate_enhanced_relevance_score(self, chunk_result: Dict[str, Any],
                                          query: str, base_threshold: float) -> float:
        """Calculate enhanced relevance score using semantic metadata."""
        base_score = chunk_result['similarity_score']
        enhancement = 0.0

        # Get semantic metadata
        semantic_meta = chunk_result['metadata'].get('chunk_metadata', {})

        # Boost for domain relevance
        domain_relevance = semantic_meta.get('domain_relevance', {})
        if domain_relevance:
            # Check if query contains domain-related terms
            query_lower = query.lower()
            domain_boost = 0.0

            for category, data in domain_relevance.items():
                if isinstance(data, dict) and 'entities' in data:
                    for entity in data['entities']:
                        if entity.lower() in query_lower:
                            domain_boost += 0.15 * data.get('score', 1)

            enhancement += min(domain_boost, 0.3)  # Cap at 0.3

        # Boost for content type relevance
        content_type = semantic_meta.get('content_type', '')
        query_intent = self._analyze_query_intent(query)

        if content_type and query_intent:
            if self._content_type_matches_intent(content_type, query_intent):
                enhancement += 0.2

        # Boost for key entities in query
        key_entities = semantic_meta.get('key_entities', [])
        if key_entities:
            entity_matches = sum(1 for entity in key_entities
                               if entity.lower() in query.lower())
            if entity_matches > 0:
                enhancement += min(entity_matches * 0.1, 0.25)

        # Boost for recall context relevance
        recall_context = semantic_meta.get('recall_context', {})
        if recall_context:
            context_boost = self._calculate_context_boost(query, recall_context)
            enhancement += context_boost

        # Boost for semantic coherence score
        semantic_score = semantic_meta.get('semantic_score', 0.5)
        if semantic_score > 0.7:
            enhancement += (semantic_score - 0.7) * 0.2

        # Ensure final score doesn't exceed reasonable bounds
        final_score = base_score + enhancement
        return min(final_score, 1.0)

    def _analyze_query_intent(self, query: str) -> str:
        """Analyze the intent of the query to match with content types."""
        query_lower = query.lower()

        # Check for methodological queries
        if any(word in query_lower for word in ['how', 'method', 'protocol', 'procedure']):
            return 'methodology'

        # Check for results/finding queries
        if any(word in query_lower for word in ['result', 'finding', 'outcome', 'what']):
            return 'results'

        # Check for conclusion/summary queries
        if any(word in query_lower for word in ['conclusion', 'summary', 'implication']):
            return 'conclusion'

        # Check for reference/literature queries
        if any(word in query_lower for word in ['reference', 'study', 'research', 'literature']):
            return 'references'

        return 'general'

    def _content_type_matches_intent(self, content_type: str, query_intent: str) -> bool:
        """Check if content type matches query intent."""
        intent_mapping = {
            'methodology': ['methodology'],
            'results': ['results'],
            'conclusion': ['conclusion'],
            'references': ['references']
        }

        return content_type in intent_mapping.get(query_intent, [])

    def _calculate_context_boost(self, query: str, recall_context: Dict[str, Any]) -> float:
        """Calculate boost based on recall context relevance."""
        boost = 0.0
        query_lower = query.lower()

        # Check temporal indicators
        temporal_indicators = recall_context.get('temporal_indicators', [])
        if temporal_indicators:
            if any(indicator in query_lower for indicator in temporal_indicators):
                boost += 0.1

        # Check methodological context
        methodological_context = recall_context.get('methodological_context', [])
        if methodological_context:
            if any(context in query_lower for context in methodological_context):
                boost += 0.1

        # Check relationship context
        relationship_context = recall_context.get('relationship_context', [])
        if relationship_context:
            if any(relationship in query_lower for relationship in relationship_context):
                boost += 0.1

        # Check importance indicators
        importance_indicators = recall_context.get('importance_indicators', [])
        if importance_indicators:
            if any(indicator in query_lower for indicator in importance_indicators):
                boost += 0.1

        return min(boost, 0.2)  # Cap at 0.2

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