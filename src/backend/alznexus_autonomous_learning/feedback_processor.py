import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import asyncio

from . import crud, models, schemas

logger = logging.getLogger(__name__)

class FeedbackProcessor:
    """Processes feedback loops from execution to learning"""

    def __init__(self):
        self.loop_timeout_seconds = 3600  # 1 hour timeout for feedback loops

    async def process_feedback_loop(self, loop_id: str, db: Session):
        """Process a complete feedback loop"""
        try:
            logger.info(f"Processing feedback loop: {loop_id}")

            loop = crud.get_feedback_loop(db, loop_id)
            if not loop:
                logger.error(f"Feedback loop {loop_id} not found")
                return

            # Execute the feedback loop stages
            await self._execute_feedback_stages(loop, db)

            # Mark loop as completed
            success_metric = self._calculate_loop_success(loop)
            crud.update_feedback_loop_status(
                db,
                loop_id,
                "completed",
                datetime.utcnow(),
                success_metric
            )

            logger.info(f"Completed feedback loop: {loop_id} with success metric {success_metric}")

        except Exception as e:
            logger.error(f"Error processing feedback loop {loop_id}: {str(e)}")
            # Mark loop as failed
            crud.update_feedback_loop_status(db, loop_id, "failed", datetime.utcnow())

    async def _execute_feedback_stages(self, loop: models.FeedbackLoop, db: Session):
        """Execute all stages of the feedback loop"""
        try:
            # Stage 1: Process execution data
            execution_insights = await self._process_execution_stage(loop.execution_stage, db)

            # Stage 2: Evaluate performance
            evaluation_results = await self._process_evaluation_stage(
                loop.evaluation_stage,
                execution_insights,
                db
            )

            # Stage 3: Extract learning patterns
            learning_patterns = await self._process_learning_stage(
                loop.learning_stage,
                evaluation_results,
                db
            )

            # Stage 4: Enrich contexts
            enrichment_results = await self._process_enrichment_stage(
                loop.enrichment_stage,
                learning_patterns,
                db
            )

            # Update loop with processed data
            loop.execution_stage = {**loop.execution_stage, "processed_insights": execution_insights}
            loop.evaluation_stage = {**loop.evaluation_stage, "evaluation_results": evaluation_results}
            loop.learning_stage = {**loop.learning_stage, "extracted_patterns": learning_patterns}
            loop.enrichment_stage = {**loop.enrichment_stage, "enrichment_results": enrichment_results}

            db.commit()

        except Exception as e:
            logger.error(f"Error executing feedback stages for loop {loop.loop_id}: {str(e)}")
            raise

    async def _process_execution_stage(self, execution_data: Dict[str, Any], db: Session) -> Dict[str, Any]:
        """Process execution stage data"""
        insights = {
            "total_agents": len(execution_data.get("agent_executions", [])),
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time": 0.0,
            "execution_patterns": []
        }

        executions = execution_data.get("agent_executions", [])
        if executions:
            execution_times = []
            for execution in executions:
                if execution.get("success", False):
                    insights["successful_executions"] += 1
                else:
                    insights["failed_executions"] += 1

                if "execution_time" in execution:
                    execution_times.append(execution["execution_time"])

            if execution_times:
                insights["avg_execution_time"] = sum(execution_times) / len(execution_times)

            # Extract execution patterns
            insights["execution_patterns"] = self._extract_execution_patterns(executions)

        return insights

    async def _process_evaluation_stage(
        self,
        evaluation_data: Dict[str, Any],
        execution_insights: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Process evaluation stage data"""
        results = {
            "performance_scores": {},
            "improvement_areas": [],
            "success_factors": [],
            "evaluation_timestamp": datetime.utcnow().isoformat()
        }

        # Evaluate individual agent performances
        agent_performances = evaluation_data.get("agent_performances", [])
        for perf_data in agent_performances:
            agent_id = perf_data.get("agent_id")
            if agent_id:
                score = self._calculate_performance_score(perf_data)
                results["performance_scores"][agent_id] = score

                # Record performance in database
                performance_record = schemas.AgentPerformanceCreate(
                    agent_id=agent_id,
                    agent_type=perf_data.get("agent_type", "unknown"),
                    task_id=perf_data.get("task_id", f"loop_{datetime.utcnow().timestamp()}"),
                    task_type=perf_data.get("task_type", "feedback_loop"),
                    success_rate=perf_data.get("success_rate", 0.0),
                    execution_time=perf_data.get("execution_time", 0.0),
                    accuracy_score=perf_data.get("accuracy_score"),
                    confidence_score=perf_data.get("confidence_score"),
                    outcome_data=perf_data.get("outcome_data"),
                    feedback_received={"loop_evaluation": True},
                    context_used=perf_data.get("context_used"),
                    learned_patterns=perf_data.get("learned_patterns")
                )
                crud.create_agent_performance(db, performance_record)

        # Identify improvement areas and success factors
        results["improvement_areas"] = self._identify_improvement_areas(agent_performances)
        results["success_factors"] = self._identify_success_factors(agent_performances, execution_insights)

        return results

    async def _process_learning_stage(
        self,
        learning_data: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        db: Session
    ) -> List[Dict[str, Any]]:
        """Process learning stage and extract patterns"""
        patterns = []

        # Extract patterns from evaluation results
        performance_patterns = self._extract_performance_patterns(evaluation_results)
        patterns.extend(performance_patterns)

        # Extract collaboration patterns
        collaboration_patterns = self._extract_collaboration_patterns(learning_data)
        patterns.extend(collaboration_patterns)

        # Store patterns in database
        stored_patterns = []
        for pattern_data in patterns:
            if pattern_data.get("confidence", 0) > 0.6:  # Minimum confidence threshold
                pattern = schemas.LearningPatternCreate(
                    pattern_type=pattern_data.get("type", "feedback_pattern"),
                    pattern_data=pattern_data.get("data", {}),
                    confidence=pattern_data.get("confidence", 0.5),
                    source_agent="feedback_processor",
                    source_task=f"loop_{datetime.utcnow().timestamp()}",
                    domain=pattern_data.get("domain", "feedback_loops"),
                    tags=pattern_data.get("tags", ["feedback", "learning"]),
                    validation_status="validated"  # Auto-validate feedback patterns
                )
                stored_pattern = crud.create_learning_pattern(db, pattern)
                stored_patterns.append({
                    "id": stored_pattern.id,
                    "type": stored_pattern.pattern_type,
                    "confidence": stored_pattern.confidence
                })

        return stored_patterns

    async def _process_enrichment_stage(
        self,
        enrichment_data: Dict[str, Any],
        learning_patterns: List[Dict[str, Any]],
        db: Session
    ) -> Dict[str, Any]:
        """Process context enrichment stage"""
        results = {
            "contexts_enriched": 0,
            "enrichment_success_rate": 0.0,
            "patterns_applied": len(learning_patterns),
            "enrichment_timestamp": datetime.utcnow().isoformat()
        }

        # Get agents involved in the loop
        agent_ids = enrichment_data.get("target_agents", [])

        for agent_id in agent_ids:
            try:
                # Get current context (would come from orchestrator)
                current_context = await self._get_agent_context_for_enrichment(agent_id)

                if current_context:
                    # Create enriched context
                    enriched_context = self._enrich_context_with_patterns(
                        current_context,
                        learning_patterns
                    )

                    # Record enrichment
                    enrichment_record = schemas.ContextEnrichmentCreate(
                        agent_id=agent_id,
                        context_type="feedback_enriched",
                        original_context=current_context,
                        enriched_context=enriched_context,
                        enrichment_metadata={
                            "source": "feedback_loop",
                            "patterns_applied": len(learning_patterns)
                        },
                        source_patterns=[p["id"] for p in learning_patterns if "id" in p]
                    )

                    crud.create_context_enrichment(db, enrichment_record)
                    results["contexts_enriched"] += 1

            except Exception as e:
                logger.error(f"Error enriching context for agent {agent_id}: {str(e)}")

        if agent_ids:
            results["enrichment_success_rate"] = results["contexts_enriched"] / len(agent_ids)

        return results

    def _extract_execution_patterns(self, executions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from execution data"""
        patterns = []

        # Success rate patterns
        success_rates = [e.get("success_rate", 0) for e in executions if "success_rate" in e]
        if success_rates:
            avg_success = sum(success_rates) / len(success_rates)
            patterns.append({
                "pattern": "execution_success_rate",
                "value": avg_success,
                "threshold": 0.8
            })

        # Execution time patterns
        exec_times = [e.get("execution_time", 0) for e in executions if "execution_time" in e]
        if exec_times:
            avg_time = sum(exec_times) / len(exec_times)
            patterns.append({
                "pattern": "execution_time",
                "value": avg_time,
                "unit": "seconds"
            })

        return patterns

    def _calculate_performance_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        weights = {
            "success_rate": 0.4,
            "accuracy_score": 0.3,
            "execution_efficiency": 0.2,
            "confidence_score": 0.1
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in performance_data and performance_data[metric] is not None:
                score += performance_data[metric] * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def _identify_improvement_areas(self, performances: List[Dict[str, Any]]) -> List[str]:
        """Identify areas needing improvement"""
        areas = []

        for perf in performances:
            if perf.get("success_rate", 1.0) < 0.7:
                areas.append(f"success_rate_{perf.get('agent_id')}")
            if perf.get("accuracy_score", 1.0) < 0.8:
                areas.append(f"accuracy_{perf.get('agent_id')}")
            if perf.get("execution_time", 0) > 600:  # 10 minutes
                areas.append(f"efficiency_{perf.get('agent_id')}")

        return list(set(areas))  # Remove duplicates

    def _identify_success_factors(self, performances: List[Dict[str, Any]], execution_insights: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to success"""
        factors = []

        # High performers
        high_performers = [
            p for p in performances
            if p.get("success_rate", 0) > 0.8 and p.get("accuracy_score", 0) > 0.8
        ]

        for perf in high_performers:
            factors.append(f"high_performance_{perf.get('agent_id')}")

        # Efficient executions
        if execution_insights.get("avg_execution_time", float('inf')) < 300:  # 5 minutes
            factors.append("efficient_execution")

        return factors

    def _extract_performance_patterns(self, evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract patterns from performance evaluation"""
        patterns = []

        performance_scores = evaluation_results.get("performance_scores", {})

        if performance_scores:
            avg_score = sum(performance_scores.values()) / len(performance_scores)
            patterns.append({
                "type": "performance_pattern",
                "data": {
                    "average_score": avg_score,
                    "individual_scores": performance_scores,
                    "best_performer": max(performance_scores.items(), key=lambda x: x[1])[0]
                },
                "confidence": 0.85,
                "domain": "agent_performance",
                "tags": ["performance", "evaluation"]
            })

        return patterns

    def _extract_collaboration_patterns(self, learning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract collaboration patterns"""
        patterns = []

        collaborations = learning_data.get("agent_collaborations", [])
        if collaborations:
            patterns.append({
                "type": "collaboration_pattern",
                "data": {
                    "total_collaborations": len(collaborations),
                    "collaboration_types": list(set(c.get("type") for c in collaborations)),
                    "success_rate": sum(1 for c in collaborations if c.get("success")) / len(collaborations)
                },
                "confidence": 0.75,
                "domain": "agent_collaboration",
                "tags": ["collaboration", "teamwork"]
            })

        return patterns

    def _calculate_loop_success(self, loop: models.FeedbackLoop) -> float:
        """Calculate overall success metric for the feedback loop"""
        try:
            # Simple success calculation based on completion and results
            if loop.status == "completed":
                # Check enrichment results
                enrichment = loop.enrichment_stage
                if isinstance(enrichment, dict) and "enrichment_results" in enrichment:
                    success_rate = enrichment["enrichment_results"].get("enrichment_success_rate", 0.5)
                    return success_rate

            return 0.5  # Default neutral score

        except Exception as e:
            logger.error(f"Error calculating loop success: {str(e)}")
            return 0.0

    async def _get_agent_context_for_enrichment(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent context for enrichment (mock implementation)"""
        # In real implementation, this would call the orchestrator
        return {
            "agent_id": agent_id,
            "current_capabilities": [],
            "learned_patterns": [],
            "performance_history": {}
        }

    def _enrich_context_with_patterns(
        self,
        context: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enrich context with learning patterns"""
        enriched = context.copy()

        enriched.setdefault("feedback_learnings", [])
        enriched["feedback_learnings"].extend(patterns)

        # Add metadata
        enriched["_feedback_enrichment"] = {
            "enriched_at": datetime.utcnow().isoformat(),
            "patterns_added": len(patterns)
        }

        return enriched