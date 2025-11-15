from celery import Celery
import os
import json
import hashlib
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any

try:
    # Try relative imports first (when run as package)
    from .database import engine
    from .crud import update_validation_status, get_validation_by_id
    from .models import AnalysisSnapshot
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database import engine
    from crud import update_validation_status, get_validation_by_id
    from models import AnalysisSnapshot

# Create Celery app
celery_app = Celery(
    "reproducibility_service",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@celery_app.task(bind=True)
def run_reproducibility_test(
    self,
    validation_id: int,
    original_snapshot_id: int,
    reproduction_snapshot_id: int,
    validation_type: str
):
    """
    Background task to run reproducibility validation
    """
    db = SessionLocal()
    try:
        # Update status to running
        update_validation_status(db, validation_id, "running")

        # Get snapshots
        original = db.query(AnalysisSnapshot).filter(
            AnalysisSnapshot.id == original_snapshot_id
        ).first()
        reproduction = db.query(AnalysisSnapshot).filter(
            AnalysisSnapshot.id == reproduction_snapshot_id
        ).first()

        if not original or not reproduction:
            update_validation_status(db, validation_id, "failed")
            return {"error": "One or both snapshots not found"}

        # Perform comprehensive validation
        validation_results = perform_comprehensive_validation(
            original, reproduction, validation_type
        )

        # Update validation record with results
        update_validation_status(
            db,
            validation_id,
            "completed",
            results=validation_results
        )

        return validation_results

    except Exception as e:
        db.rollback()
        update_validation_status(db, validation_id, "failed")
        raise e
    finally:
        db.close()

def perform_comprehensive_validation(
    original: AnalysisSnapshot,
    reproduction: AnalysisSnapshot,
    validation_type: str
) -> Dict[str, Any]:
    """
    Perform comprehensive reproducibility validation
    """
    results = {
        "validation_type": validation_type,
        "timestamp": datetime.utcnow().isoformat(),
        "comparisons": {}
    }

    # Environment comparison
    env_comparison = compare_environments(original, reproduction)
    results["comparisons"]["environment"] = env_comparison

    # Code comparison
    code_comparison = compare_code_versions(original, reproduction)
    results["comparisons"]["code"] = code_comparison

    # Data comparison
    data_comparison = compare_data_versions(original, reproduction)
    results["comparisons"]["data"] = data_comparison

    # Random seed comparison
    seed_comparison = compare_random_seeds(original, reproduction)
    results["comparisons"]["random_seed"] = seed_comparison

    # Statistical comparison (placeholder - would need actual results)
    stat_comparison = compare_statistical_results(original, reproduction)
    results["comparisons"]["statistical"] = stat_comparison

    # Calculate overall score
    scores = [
        env_comparison.get("match_score", 0),
        code_comparison.get("match_score", 0),
        data_comparison.get("match_score", 0),
        seed_comparison.get("match_score", 0),
        stat_comparison.get("match_score", 0)
    ]
    results["overall_score"] = sum(scores) / len(scores)

    # Identify issues
    results["issues_found"] = []
    for comparison_name, comparison in results["comparisons"].items():
        if not comparison.get("matches", False):
            results["issues_found"].extend(comparison.get("issues", []))

    # Generate recommendations
    results["recommendations"] = generate_recommendations(results["issues_found"])

    return results

def compare_environments(snapshot1: AnalysisSnapshot, snapshot2: AnalysisSnapshot) -> Dict[str, Any]:
    """
    Compare execution environments
    """
    matches = snapshot1.environment_hash == snapshot2.environment_hash

    result = {
        "matches": matches,
        "match_score": 1.0 if matches else 0.0,
        "snapshot1_hash": snapshot1.environment_hash,
        "snapshot2_hash": snapshot2.environment_hash,
        "issues": [] if matches else ["Environment configurations differ"]
    }

    return result

def compare_code_versions(snapshot1: AnalysisSnapshot, snapshot2: AnalysisSnapshot) -> Dict[str, Any]:
    """
    Compare code versions
    """
    matches = snapshot1.code_version == snapshot2.code_version

    result = {
        "matches": matches,
        "match_score": 1.0 if matches else 0.0,
        "snapshot1_version": snapshot1.code_version,
        "snapshot2_version": snapshot2.code_version,
        "issues": [] if matches else ["Code versions differ"]
    }

    return result

def compare_data_versions(snapshot1: AnalysisSnapshot, snapshot2: AnalysisSnapshot) -> Dict[str, Any]:
    """
    Compare data versions
    """
    matches = snapshot1.data_version == snapshot2.data_version

    result = {
        "matches": matches,
        "match_score": 1.0 if matches else 0.0,
        "snapshot1_version": snapshot1.data_version,
        "snapshot2_version": snapshot2.data_version,
        "issues": [] if matches else ["Data versions differ"]
    }

    return result

def compare_random_seeds(snapshot1: AnalysisSnapshot, snapshot2: AnalysisSnapshot) -> Dict[str, Any]:
    """
    Compare random seeds
    """
    matches = snapshot1.random_seed == snapshot2.random_seed

    result = {
        "matches": matches,
        "match_score": 1.0 if matches else 0.0,
        "snapshot1_seed": snapshot1.random_seed,
        "snapshot2_seed": snapshot2.random_seed,
        "issues": [] if matches else ["Random seeds differ"]
    }

    return result

def compare_statistical_results(snapshot1: AnalysisSnapshot, snapshot2: AnalysisSnapshot) -> Dict[str, Any]:
    """
    Compare statistical analysis results (placeholder)
    """
    # In a real implementation, this would compare actual statistical outputs
    # For now, assume they match if parameters are identical
    params_match = snapshot1.parameters == snapshot2.parameters

    result = {
        "matches": params_match,
        "match_score": 1.0 if params_match else 0.5,  # Partial credit for parameter match
        "parameters_match": params_match,
        "snapshot1_params": snapshot1.parameters,
        "snapshot2_params": snapshot2.parameters,
        "issues": [] if params_match else ["Analysis parameters differ"]
    }

    return result

def generate_recommendations(issues: list) -> list:
    """
    Generate recommendations based on identified issues
    """
    recommendations = []

    issue_map = {
        "Environment configurations differ": "Ensure identical Python environment and package versions",
        "Code versions differ": "Use the same code commit/version for reproduction",
        "Data versions differ": "Use identical dataset versions and preprocessing steps",
        "Random seeds differ": "Use the same random seed for reproducible analysis",
        "Analysis parameters differ": "Ensure identical analysis parameters and configurations"
    }

    for issue in issues:
        if issue in issue_map:
            recommendations.append(issue_map[issue])

    if not recommendations:
        recommendations.append("Analysis appears reproducible")

    return recommendations

@celery_app.task
def cleanup_expired_seeds():
    """
    Periodic task to clean up expired random seeds
    """
    db = SessionLocal()
    try:
        from .models import RandomSeed
        from datetime import datetime

        # Delete expired seeds older than 30 days
        cutoff_date = datetime.utcnow()
        expired_seeds = db.query(RandomSeed).filter(
            RandomSeed.expires_at < cutoff_date
        ).delete()

        db.commit()
        return f"Cleaned up {expired_seeds} expired seeds"

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

@celery_app.task
def validate_provenance_chains():
    """
    Periodic task to validate data provenance chains
    """
    db = SessionLocal()
    try:
        from .models import DataProvenance

        # Get all provenance records
        provenances = db.query(DataProvenance).all()
        validated_count = 0

        for provenance in provenances:
            # Validate chain integrity (simplified)
            if validate_chain_integrity(provenance):
                if not provenance.validated_at:
                    provenance.validated_at = datetime.utcnow()
                    validated_count += 1

        db.commit()
        return f"Validated {validated_count} provenance chains"

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def validate_chain_integrity(provenance) -> bool:
    """
    Validate the integrity of a provenance chain
    """
    # Simplified validation - in practice would check hashes, signatures, etc.
    return bool(provenance.data_hash and provenance.parent_provenance_ids is not None)