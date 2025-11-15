import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import requests
import os

from . import crud, models, schemas

logger = logging.getLogger(__name__)

class ContextEnricher:
    """Handles context enrichment with learned data"""

    def __init__(self):
        self.orchestrator_url = os.getenv("ORCHESTRATOR_API_URL", "http://localhost:8001")
        self.knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://localhost:8006")
        self.knowledge_api_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")
        self.enrichment_threshold = 0.75  # Minimum confidence for enrichment

    async def enrich_agent_context(self, agent_id: str, base_context: Dict[str, Any], db: Session):
        """Enrich agent context with relevant learned patterns"""
        try:
            logger.info(f"Enriching context for agent {agent_id}")

            # Get relevant patterns for this agent
            relevant_patterns = self._get_relevant_patterns(agent_id, db)

            if not relevant_patterns:
                logger.info(f"No relevant patterns found for agent {agent_id}")
                return

            # Build enriched context
            enriched_context = self._build_enriched_context(base_context, relevant_patterns)

            # Record enrichment
            enrichment_record = schemas.ContextEnrichmentCreate(
                agent_id=agent_id,
                context_type="enriched",
                original_context=base_context,
                enriched_context=enriched_context,
                enrichment_metadata={
                    "patterns_used": len(relevant_patterns),
                    "enrichment_timestamp": datetime.utcnow().isoformat(),
                    "pattern_ids": [p.id for p in relevant_patterns]
                },
                source_patterns=[p.id for p in relevant_patterns]
            )

            crud.create_context_enrichment(db, enrichment_record)

            # Push enrichment insights to knowledge base for RAG
            await self._push_enrichment_to_knowledge_base(enrichment_record)

            # Send enriched context to agent
            await self._send_enriched_context_to_agent(agent_id, enriched_context)

            logger.info(f"Successfully enriched context for agent {agent_id} with {len(relevant_patterns)} patterns")

        except Exception as e:
            logger.error(f"Error enriching context for agent {agent_id}: {str(e)}")

    async def update_all_contexts(self, db: Session):
        """Update contexts for all active agents"""
        try:
            # Get all agent IDs from recent performances
            agent_ids = db.query(models.AgentPerformance.agent_id).distinct().limit(50).all()
            agent_ids = [aid[0] for aid in agent_ids]

            for agent_id in agent_ids:
                # Get latest context for agent (this would come from orchestrator)
                base_context = await self._get_agent_current_context(agent_id)
                if base_context:
                    await self.enrich_agent_context(agent_id, base_context, db)

            logger.info(f"Updated contexts for {len(agent_ids)} agents")

        except Exception as e:
            logger.error(f"Error updating all contexts: {str(e)}")

    def _get_relevant_patterns(self, agent_id: str, db: Session) -> List[models.LearningPattern]:
        """Get patterns relevant to the agent"""
        try:
            # Get agent's recent performance to understand its domain
            recent_perf = crud.get_agent_performances(db, agent_id, limit=10)

            if not recent_perf:
                return []

            # Determine agent's primary task types
            task_types = list(set(p.task_type for p in recent_perf))
            domains = [self._task_type_to_domain(tt) for tt in task_types]

            # Get patterns from relevant domains with high confidence - ONLY non-superseded
            relevant_patterns = []
            for domain in domains:
                patterns = crud.get_learning_patterns(
                    db,
                    domain=domain,
                    min_confidence=self.enrichment_threshold,
                    validation_status="validated",  # Only validated, non-superseded patterns
                    limit=20
                )
                relevant_patterns.extend(patterns)

            # Remove duplicates and sort by recency and confidence (newest first)
            seen_ids = set()
            unique_patterns = []
            for pattern in sorted(relevant_patterns,
                                key=lambda x: (x.discovered_at, x.confidence),
                                reverse=True):  # Newest and most confident first
                if pattern.id not in seen_ids and pattern.superseded_by is None:
                    unique_patterns.append(pattern)
                    seen_ids.add(pattern.id)

            return unique_patterns[:10]  # Limit to top 10 most recent and relevant

        except Exception as e:
            logger.error(f"Error getting relevant patterns for {agent_id}: {str(e)}")
            return []

    def _build_enriched_context(self, base_context: Dict[str, Any], patterns: List[models.LearningPattern]) -> Dict[str, Any]:
        """Build enriched context from base context and patterns"""
        enriched = base_context.copy()

        # Initialize enrichment sections
        enriched.setdefault("learned_patterns", [])
        enriched.setdefault("performance_insights", {})
        enriched.setdefault("domain_knowledge", {})
        enriched.setdefault("optimization_hints", [])

        # Add pattern-based enrichments
        for pattern in patterns:
            pattern_info = {
                "id": pattern.id,
                "type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "data": pattern.pattern_data,
                "tags": pattern.tags,
                "source_agent": pattern.source_agent,
                "discovered_at": pattern.discovered_at.isoformat()
            }

            enriched["learned_patterns"].append(pattern_info)

            # Add domain-specific knowledge
            if pattern.domain:
                enriched["domain_knowledge"].setdefault(pattern.domain, [])
                enriched["domain_knowledge"][pattern.domain].append(pattern_info)

            # Add optimization hints
            if pattern.pattern_type in ["efficiency_optimization", "success_factor"]:
                enriched["optimization_hints"].append({
                    "pattern_id": pattern.id,
                    "hint": pattern.pattern_data,
                    "confidence": pattern.confidence
                })

        # Add performance insights
        enriched["performance_insights"] = {
            "total_patterns": len(patterns),
            "avg_confidence": sum(p.confidence for p in patterns) / len(patterns) if patterns else 0,
            "enrichment_timestamp": datetime.utcnow().isoformat(),
            "pattern_types": list(set(p.pattern_type for p in patterns))
        }

        # Add metadata
        enriched["_enrichment_metadata"] = {
            "enriched_at": datetime.utcnow().isoformat(),
            "patterns_applied": len(patterns),
            "enrichment_version": "1.0"
        }

        return enriched

    def _task_type_to_domain(self, task_type: str) -> str:
        """Convert task type to domain"""
        domain_map = {
            "biomarker_hunting": "alzheimer_biomarkers",
            "drug_screening": "alzheimer_therapeutics",
            "hypothesis_validation": "alzheimer_research",
            "literature_bridging": "alzheimer_literature",
            "pathway_modeling": "alzheimer_pathways",
            "trial_optimization": "clinical_trials"
        }
        return domain_map.get(task_type, "alzheimer_research")

    async def _get_agent_current_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent's current context from orchestrator"""
        try:
            # This would make an API call to the orchestrator to get current context
            # For now, return a mock context
            return {
                "agent_id": agent_id,
                "current_task": None,
                "domain_knowledge": {},
                "performance_history": {},
                "active_since": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current context for {agent_id}: {str(e)}")
            return None

    async def _send_enriched_context_to_agent(self, agent_id: str, enriched_context: Dict[str, Any]):
        """Send enriched context to the agent"""
        try:
            # This would make an API call to update the agent's context
            # For now, just log the action
            logger.info(f"Would send enriched context to agent {agent_id}: {len(str(enriched_context))} chars")

            # In a real implementation, this would be:
            # response = requests.post(
            #     f"{self.orchestrator_url}/agents/{agent_id}/context",
            #     json={"context": enriched_context},
            #     headers={"Authorization": f"Bearer {self.api_key}"}
            # )

        except Exception as e:
            logger.error(f"Error sending enriched context to agent {agent_id}: {str(e)}")

    async def _push_enrichment_to_knowledge_base(self, enrichment_record: schemas.ContextEnrichmentCreate):
        """Push context enrichment insights to knowledge base for RAG"""
        try:
            doc_data = {
                "title": f"Context Enrichment: {enrichment_record.agent_id}",
                "content": json.dumps({
                    "agent_id": enrichment_record.agent_id,
                    "context_type": enrichment_record.context_type,
                    "enrichment_metadata": enrichment_record.enrichment_metadata,
                    "source_patterns": enrichment_record.source_patterns,
                    "timestamp": datetime.utcnow().isoformat(),
                    "enrichment_success": True
                }, indent=2),
                "document_type": "context_enrichment",
                "source_agent": "autonomous_learning_service",
                "source_task_id": f"enrichment_{datetime.utcnow().timestamp()}",
                "metadata_json": json.dumps({
                    "agent_id": enrichment_record.agent_id,
                    "patterns_used": len(enrichment_record.source_patterns) if enrichment_record.source_patterns else 0,
                    "enrichment_type": enrichment_record.context_type
                }),
                "tags": ["context_enrichment", enrichment_record.context_type, enrichment_record.agent_id]
            }

            headers = {"X-API-Key": self.knowledge_api_key, "Content-Type": "application/json"}
            response = requests.post(
                f"{self.knowledge_base_url}/documents/upsert/",
                headers=headers,
                json=doc_data,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Successfully pushed enrichment for agent {enrichment_record.agent_id} to knowledge base")
            else:
                logger.error(f"Failed to push enrichment to knowledge base: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error pushing enrichment to knowledge base: {str(e)}")

    def get_context_enrichment_stats(self, db: Session) -> Dict[str, Any]:
        """Get statistics about context enrichment"""
        try:
            total_enrichments = db.query(models.ContextEnrichment).count()

            recent_enrichments = db.query(models.ContextEnrichment).filter(
                models.ContextEnrichment.timestamp >= datetime.utcnow().replace(hour=0, minute=0, second=0)
            ).count()

            successful_enrichments = db.query(models.ContextEnrichment).filter(
                models.ContextEnrichment.enrichment_success == True
            ).count()

            return {
                "total_enrichments": total_enrichments,
                "recent_enrichments": recent_enrichments,
                "successful_enrichments": successful_enrichments,
                "success_rate": successful_enrichments / total_enrichments if total_enrichments > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error getting enrichment stats: {str(e)}")
            return {"error": str(e)}