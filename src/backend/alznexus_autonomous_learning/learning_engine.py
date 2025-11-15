import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import requests
import os

from . import crud, models, schemas

logger = logging.getLogger(__name__)

class LearningEngine:
    """Core learning engine for pattern extraction and analysis"""

    def __init__(self):
        self.min_pattern_confidence = 0.7
        self.max_pattern_age_days = 30
        self.knowledge_base_url = os.getenv("KNOWLEDGE_BASE_URL", "http://localhost:8006")
        self.knowledge_api_key = os.getenv("KNOWLEDGE_API_KEY", "test_knowledge_key_123")

    async def analyze_performance(self, performance_id: int, db: Session):
        """Analyze agent performance to extract learning patterns"""
        try:
            performance = crud.get_agent_performance(db, performance_id)
            if not performance:
                logger.warning(f"Performance record {performance_id} not found")
                return

            # Extract patterns from performance data
            patterns = self._extract_patterns_from_performance(performance)

            # Validate and store patterns
            for pattern_data in patterns:
                if pattern_data['confidence'] >= self.min_pattern_confidence:
                    pattern = schemas.LearningPatternCreate(
                        pattern_type=pattern_data['type'],
                        pattern_data=pattern_data['data'],
                        confidence=pattern_data['confidence'],
                        source_agent=performance.agent_id,
                        source_task=performance.task_id,
                        domain=self._infer_domain(performance.task_type),
                        tags=pattern_data.get('tags', []),
                        related_patterns=self._find_related_patterns(pattern_data, db)
                    )
                    crud.create_learning_pattern(db, pattern)
                    logger.info(f"Extracted pattern: {pattern_data['type']} with confidence {pattern_data['confidence']}")

                    # Push high-confidence patterns to knowledge base for RAG
                    if pattern_data['confidence'] >= 0.8:
                        await self._push_pattern_to_knowledge_base(pattern)

            # Update performance with learned patterns
            if patterns:
                performance.learned_patterns = {"patterns": patterns, "extracted_at": datetime.utcnow().isoformat()}
                db.commit()

        except Exception as e:
            logger.error(f"Error analyzing performance {performance_id}: {str(e)}")

    async def run_comprehensive_analysis(self, db: Session):
        """Run comprehensive learning analysis across all data"""
        try:
            logger.info("Starting comprehensive learning analysis")

            # Analyze recent performances
            recent_performances = crud.get_agent_performances(db, limit=1000)
            for perf in recent_performances:
                if not perf.learned_patterns:  # Only analyze unanalyzed performances
                    await self.analyze_performance(perf.id, db)

            # Cluster patterns for insights
            self._cluster_patterns(db)

            # Update pattern relationships
            self._update_pattern_relationships(db)

            # Clean up old patterns
            self._cleanup_old_patterns(db)

            logger.info("Comprehensive learning analysis completed")

        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")

    def _extract_patterns_from_performance(self, performance: models.AgentPerformance) -> List[Dict[str, Any]]:
        """Extract learning patterns from performance data"""
        patterns = []

        # Success pattern
        if performance.success_rate > 0.8:
            patterns.append({
                'type': 'success_factor',
                'data': {
                    'task_type': performance.task_type,
                    'success_rate': performance.success_rate,
                    'execution_time': performance.execution_time,
                    'context_factors': self._extract_context_factors(performance.context_used)
                },
                'confidence': min(performance.success_rate, 0.95),
                'tags': ['success', performance.task_type]
            })

        # Efficiency pattern
        if performance.execution_time < self._get_task_time_baseline(performance.task_type):
            patterns.append({
                'type': 'efficiency_optimization',
                'data': {
                    'task_type': performance.task_type,
                    'execution_time': performance.execution_time,
                    'optimization_factors': self._extract_optimization_factors(performance)
                },
                'confidence': 0.8,
                'tags': ['efficiency', 'optimization', performance.task_type]
            })

        # Accuracy pattern
        if performance.accuracy_score and performance.accuracy_score > 0.85:
            patterns.append({
                'type': 'accuracy_pattern',
                'data': {
                    'task_type': performance.task_type,
                    'accuracy_score': performance.accuracy_score,
                    'confidence_score': performance.confidence_score,
                    'accuracy_factors': self._extract_accuracy_factors(performance)
                },
                'confidence': performance.accuracy_score,
                'tags': ['accuracy', performance.task_type]
            })

        # Learning from feedback
        if performance.feedback_received:
            feedback_patterns = self._extract_feedback_patterns(performance.feedback_received)
            patterns.extend(feedback_patterns)

        return patterns

    def _extract_context_factors(self, context: Optional[Dict[str, Any]]) -> List[str]:
        """Extract key factors from context data"""
        if not context:
            return []

        factors = []
        if 'data_quality' in context:
            factors.append(f"data_quality_{context['data_quality']}")
        if 'complexity' in context:
            factors.append(f"complexity_{context['complexity']}")
        if 'domain_knowledge' in context:
            factors.append(f"domain_knowledge_{context['domain_knowledge']}")

        return factors

    def _extract_optimization_factors(self, performance: models.AgentPerformance) -> Dict[str, Any]:
        """Extract factors that contributed to optimization"""
        factors = {}

        if performance.context_used:
            # Look for optimization hints in context
            context = performance.context_used
            if 'optimization_hints' in context:
                factors.update(context['optimization_hints'])

        # Add performance-based factors
        factors['success_rate'] = performance.success_rate
        factors['accuracy_score'] = performance.accuracy_score

        return factors

    def _extract_accuracy_factors(self, performance: models.AgentPerformance) -> Dict[str, Any]:
        """Extract factors contributing to high accuracy"""
        factors = {
            'confidence_score': performance.confidence_score,
            'execution_time': performance.execution_time
        }

        if performance.context_used and 'validation_methods' in performance.context_used:
            factors['validation_methods'] = performance.context_used['validation_methods']

        return factors

    def _extract_feedback_patterns(self, feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from feedback data"""
        patterns = []

        if 'improvement_suggestions' in feedback:
            patterns.append({
                'type': 'improvement_suggestion',
                'data': {
                    'suggestions': feedback['improvement_suggestions'],
                    'feedback_source': feedback.get('source', 'unknown')
                },
                'confidence': 0.75,
                'tags': ['feedback', 'improvement']
            })

        if 'learned_lessons' in feedback:
            patterns.append({
                'type': 'learned_lesson',
                'data': {
                    'lessons': feedback['learned_lessons'],
                    'context': feedback.get('context', {})
                },
                'confidence': 0.8,
                'tags': ['lesson', 'feedback']
            })

        return patterns

    def _get_task_time_baseline(self, task_type: str) -> float:
        """Get baseline execution time for task type"""
        baselines = {
            'biomarker_hunting': 300.0,  # 5 minutes
            'drug_screening': 600.0,    # 10 minutes
            'hypothesis_validation': 180.0,  # 3 minutes
            'literature_bridging': 240.0,    # 4 minutes
            'pathway_modeling': 480.0,      # 8 minutes
            'trial_optimization': 360.0      # 6 minutes
        }
        return baselines.get(task_type, 300.0)

    def _infer_domain(self, task_type: str) -> str:
        """Infer domain from task type"""
        domain_map = {
            'biomarker_hunting': 'alzheimer_biomarkers',
            'drug_screening': 'alzheimer_therapeutics',
            'hypothesis_validation': 'alzheimer_research',
            'literature_bridging': 'alzheimer_literature',
            'pathway_modeling': 'alzheimer_pathways',
            'trial_optimization': 'clinical_trials'
        }
        return domain_map.get(task_type, 'alzheimer_research')

    def _find_related_patterns(self, pattern_data: Dict[str, Any], db: Session) -> Optional[List[int]]:
        """Find patterns related to the given pattern data"""
        try:
            # Simple relatedness based on tags and type
            related_patterns = []

            # Get recent patterns with similar tags
            if 'tags' in pattern_data:
                tags = pattern_data['tags']
                existing_patterns = crud.get_learning_patterns(db, limit=100)

                for existing in existing_patterns:
                    if existing.tags and any(tag in existing.tags for tag in tags):
                        related_patterns.append(existing.id)
                        if len(related_patterns) >= 5:  # Limit related patterns
                            break

            return related_patterns if related_patterns else None

        except Exception as e:
            logger.error(f"Error finding related patterns: {str(e)}")
            return None

    def _cluster_patterns(self, db: Session):
        """Cluster patterns to identify insights"""
        try:
            patterns = crud.get_learning_patterns(db, limit=500)

            if len(patterns) < 10:
                return  # Not enough data for clustering

            # Extract features for clustering
            features = []
            pattern_objects = []

            for pattern in patterns:
                # Simple feature extraction based on pattern data
                feature_vector = [
                    pattern.confidence,
                    len(pattern.pattern_data) if pattern.pattern_data else 0,
                    hash(pattern.pattern_type) % 1000 / 1000.0,  # Normalize hash
                    hash(str(pattern.tags)) % 1000 / 1000.0 if pattern.tags else 0
                ]
                features.append(feature_vector)
                pattern_objects.append(pattern)

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Perform clustering
            clustering = DBSCAN(eps=0.5, min_samples=3)
            clusters = clustering.fit_predict(features_scaled)

            # Update patterns with cluster information
            for i, pattern in enumerate(pattern_objects):
                if clusters[i] != -1:  # Not noise
                    cluster_id = int(clusters[i])
                    if not pattern.tags:
                        pattern.tags = []
                    if f"cluster_{cluster_id}" not in pattern.tags:
                        pattern.tags.append(f"cluster_{cluster_id}")
                        db.commit()

            logger.info(f"Clustered {len(patterns)} patterns into {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters")

        except Exception as e:
            logger.error(f"Error clustering patterns: {str(e)}")

    def _update_pattern_relationships(self, db: Session):
        """Update relationships between patterns"""
        try:
            patterns = crud.get_learning_patterns(db, limit=200)

            for pattern in patterns:
                if not pattern.related_patterns:
                    related = self._find_related_patterns({
                        'type': pattern.pattern_type,
                        'tags': pattern.tags,
                        'data': pattern.pattern_data
                    }, db)

                    if related:
                        pattern.related_patterns = related
                        db.commit()

        except Exception as e:
            logger.error(f"Error updating pattern relationships: {str(e)}")

    def _cleanup_old_patterns(self, db: Session):
        """Clean up old or low-confidence patterns"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.max_pattern_age_days)

            # Mark old patterns as expired
            old_patterns = db.query(models.LearningPattern).filter(
                models.LearningPattern.discovered_at < cutoff_date,
                models.LearningPattern.validation_status == "pending"
            ).all()

            for pattern in old_patterns:
                pattern.expires_at = datetime.utcnow() + timedelta(days=7)  # Grace period
                db.commit()

            logger.info(f"Marked {len(old_patterns)} old patterns for expiration")

        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {str(e)}")

    async def _push_pattern_to_knowledge_base(self, pattern: schemas.LearningPatternCreate):
        """Push high-confidence learned pattern to knowledge base for RAG"""
        try:
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
                    "discovered_at": datetime.utcnow().isoformat(),
                    "validation_status": "validated"
                }, indent=2),
                "document_type": "learned_pattern",
                "source_agent": "autonomous_learning_service",
                "source_task_id": f"pattern_{datetime.utcnow().timestamp()}",
                "metadata_json": json.dumps({
                    "confidence": pattern.confidence,
                    "domain": pattern.domain,
                    "tags": pattern.tags,
                    "pattern_type": pattern.pattern_type
                }),
                "tags": ["learned_pattern", pattern.pattern_type, pattern.domain] + pattern.tags
            }

            headers = {"X-API-Key": self.knowledge_api_key, "Content-Type": "application/json"}
            response = requests.post(
                f"{self.knowledge_base_url}/documents/upsert/",
                headers=headers,
                json=doc_data,
                timeout=30
            )

            if response.status_code == 200:
                logger.info(f"Successfully pushed pattern {pattern.pattern_type} to knowledge base")
            else:
                logger.error(f"Failed to push pattern to knowledge base: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error pushing pattern to knowledge base: {str(e)}")