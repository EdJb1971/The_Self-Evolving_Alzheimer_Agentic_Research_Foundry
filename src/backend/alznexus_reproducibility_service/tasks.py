from celery import Celery
import os
import json
import hashlib
import logging
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any, List, Union

try:
    # Try relative imports first (when run as package)
    from .database import engine
    from .crud import update_validation_status, get_validation_by_id
    from .models import AnalysisSnapshot, DataProvenance
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database import engine
    from crud import update_validation_status, get_validation_by_id
    from models import AnalysisSnapshot, DataProvenance

logger = logging.getLogger(__name__)

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
    Compare statistical analysis results between two snapshots
    """
    results1 = snapshot1.final_results or {}
    results2 = snapshot2.final_results or {}

    # If no results to compare, fall back to parameter comparison
    if not results1 and not results2:
        params_match = snapshot1.input_parameters == snapshot2.input_parameters
        return {
            "matches": params_match,
            "match_score": 1.0 if params_match else 0.5,
            "parameters_match": params_match,
            "snapshot1_params": snapshot1.input_parameters,
            "snapshot2_params": snapshot2.input_parameters,
            "issues": [] if params_match else ["Analysis parameters differ"],
            "comparison_type": "parameters_only"
        }

    # Extract statistical metrics for comparison
    metrics1 = extract_statistical_metrics(results1)
    metrics2 = extract_statistical_metrics(results2)

    # Compare key statistical measures
    comparisons = {}

    # Compare p-values if available
    if metrics1.get('p_values') and metrics2.get('p_values'):
        comparisons['p_values'] = compare_p_values(metrics1['p_values'], metrics2['p_values'])

    # Compare effect sizes if available
    if metrics1.get('effect_sizes') and metrics2.get('effect_sizes'):
        comparisons['effect_sizes'] = compare_effect_sizes(metrics1['effect_sizes'], metrics2['effect_sizes'])

    # Compare correlation matrices if available
    if metrics1.get('correlation_matrix') and metrics2.get('correlation_matrix'):
        comparisons['correlations'] = compare_correlation_matrices(
            metrics1['correlation_matrix'], metrics2['correlation_matrix']
        )

    # Compare regression metrics if available
    if metrics1.get('regression_metrics') and metrics2.get('regression_metrics'):
        comparisons['regression'] = compare_regression_metrics(
            metrics1['regression_metrics'], metrics2['regression_metrics']
        )

    # Calculate overall match score
    match_scores = [comp.get('match_score', 0) for comp in comparisons.values()]
    overall_score = sum(match_scores) / len(match_scores) if match_scores else 0.0

    # Determine if results are statistically equivalent
    matches = overall_score >= 0.95  # Very high threshold for statistical reproducibility

    # Collect issues
    issues = []
    for comp_name, comp in comparisons.items():
        if not comp.get('matches', True):
            issues.extend(comp.get('issues', []))

    # Add parameter comparison as additional check
    params_match = snapshot1.input_parameters == snapshot2.input_parameters
    if not params_match:
        issues.append("Analysis parameters differ")
        overall_score *= 0.8  # Reduce score if parameters don't match

    return {
        "matches": matches,
        "match_score": overall_score,
        "comparison_type": "statistical_results",
        "comparisons": comparisons,
        "parameters_match": params_match,
        "snapshot1_params": snapshot1.input_parameters,
        "snapshot2_params": snapshot2.input_parameters,
        "issues": issues
    }

def extract_statistical_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract statistical metrics from analysis results
    """
    metrics = {}

    # Extract p-values
    if 'p_value' in results:
        metrics['p_values'] = [results['p_value']]
    elif 'p_values' in results:
        metrics['p_values'] = results['p_values']
    elif 'statistical_analysis' in results and 'p_value' in results['statistical_analysis']:
        metrics['p_values'] = [results['statistical_analysis']['p_value']]

    # Extract effect sizes
    if 'effect_size' in results:
        metrics['effect_sizes'] = [results['effect_size']]
    elif 'effect_sizes' in results:
        metrics['effect_sizes'] = results['effect_sizes']
    elif 'statistical_analysis' in results and 'effect_size' in results['statistical_analysis']:
        metrics['effect_sizes'] = [results['statistical_analysis']['effect_size']]

    # Extract correlation matrix
    if 'correlation_matrix' in results:
        metrics['correlation_matrix'] = results['correlation_matrix']
    elif 'correlations' in results and 'correlation_matrix' in results['correlations']:
        metrics['correlation_matrix'] = results['correlations']['correlation_matrix']

    # Extract regression metrics
    if 'regression_metrics' in results:
        metrics['regression_metrics'] = results['regression_metrics']
    elif 'regression' in results and 'metrics' in results['regression']:
        metrics['regression_metrics'] = results['regression']['metrics']

    return metrics

def compare_p_values(p_values1: List[float], p_values2: List[float]) -> Dict[str, Any]:
    """
    Compare p-values between two analyses
    """
    if len(p_values1) != len(p_values2):
        return {
            "matches": False,
            "match_score": 0.0,
            "issues": ["Different number of p-values compared"]
        }

    # Check if p-values are statistically equivalent (within reasonable tolerance)
    differences = []
    max_diff = 0.0

    for p1, p2 in zip(p_values1, p_values2):
        diff = abs(p1 - p2)
        differences.append(diff)
        max_diff = max(max_diff, diff)

    # P-values should be very close for reproducibility
    # Allow small tolerance for numerical precision
    tolerance = 1e-6
    matches = max_diff <= tolerance

    # Calculate match score based on maximum difference
    # Score decreases as difference increases
    if max_diff <= tolerance:
        match_score = 1.0
    elif max_diff <= 0.01:  # Small difference
        match_score = 0.8
    elif max_diff <= 0.05:  # Moderate difference
        match_score = 0.5
    else:  # Large difference
        match_score = 0.0

    issues = []
    if not matches:
        issues.append(".6f")

    return {
        "matches": matches,
        "match_score": match_score,
        "max_difference": max_diff,
        "tolerance": tolerance,
        "issues": issues
    }

def compare_effect_sizes(effect_sizes1: List[float], effect_sizes2: List[float]) -> Dict[str, Any]:
    """
    Compare effect sizes between two analyses
    """
    if len(effect_sizes1) != len(effect_sizes2):
        return {
            "matches": False,
            "match_score": 0.0,
            "issues": ["Different number of effect sizes compared"]
        }

    # Effect sizes should be very close for reproducibility
    differences = []
    max_diff = 0.0

    for es1, es2 in zip(effect_sizes1, effect_sizes2):
        diff = abs(es1 - es2)
        differences.append(diff)
        max_diff = max(max_diff, diff)

    # Allow small tolerance for effect sizes (0.01 is quite strict)
    tolerance = 0.01
    matches = max_diff <= tolerance

    # Calculate match score
    if max_diff <= tolerance:
        match_score = 1.0
    elif max_diff <= 0.05:
        match_score = 0.8
    elif max_diff <= 0.1:
        match_score = 0.5
    else:
        match_score = 0.0

    issues = []
    if not matches:
        issues.append(".4f")

    return {
        "matches": matches,
        "match_score": match_score,
        "max_difference": max_diff,
        "tolerance": tolerance,
        "issues": issues
    }

def compare_correlation_matrices(matrix1: Union[List[List[float]], Dict], matrix2: Union[List[List[float]], Dict]) -> Dict[str, Any]:
    """
    Compare correlation matrices between two analyses
    """
    # Convert dict format to list if needed
    if isinstance(matrix1, dict):
        matrix1 = matrix1.get('values', [])
    if isinstance(matrix2, dict):
        matrix2 = matrix2.get('values', [])

    if len(matrix1) != len(matrix2) or (matrix1 and len(matrix1[0]) != len(matrix2[0])):
        return {
            "matches": False,
            "match_score": 0.0,
            "issues": ["Correlation matrices have different dimensions"]
        }

    # Flatten matrices and compare element-wise
    flat1 = [val for row in matrix1 for val in row]
    flat2 = [val for row in matrix2 for val in row]

    differences = [abs(a - b) for a, b in zip(flat1, flat2)]
    max_diff = max(differences) if differences else 0.0

    # Correlation coefficients should be very close
    tolerance = 0.01
    matches = max_diff <= tolerance

    # Calculate match score
    if max_diff <= tolerance:
        match_score = 1.0
    elif max_diff <= 0.05:
        match_score = 0.8
    elif max_diff <= 0.1:
        match_score = 0.5
    else:
        match_score = 0.0

    issues = []
    if not matches:
        issues.append(".4f")

    return {
        "matches": matches,
        "match_score": match_score,
        "max_difference": max_diff,
        "tolerance": tolerance,
        "matrix_size": f"{len(matrix1)}x{len(matrix1[0]) if matrix1 else 0}",
        "issues": issues
    }

def compare_regression_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare regression metrics between two analyses
    """
    issues = []

    # Compare R-squared values
    r2_match = True
    if 'r_squared' in metrics1 and 'r_squared' in metrics2:
        r2_diff = abs(metrics1['r_squared'] - metrics2['r_squared'])
        if r2_diff > 0.01:  # Allow 1% difference
            r2_match = False
            issues.append(".4f")

    # Compare coefficients if available
    coeff_match = True
    if 'coefficients' in metrics1 and 'coefficients' in metrics2:
        coeff1 = metrics1['coefficients']
        coeff2 = metrics2['coefficients']

        if isinstance(coeff1, list) and isinstance(coeff2, list):
            if len(coeff1) == len(coeff2):
                max_coeff_diff = max(abs(a - b) for a, b in zip(coeff1, coeff2))
                if max_coeff_diff > 0.01:
                    coeff_match = False
                    issues.append(".4f")
            else:
                coeff_match = False
                issues.append("Different number of regression coefficients")

    # Overall match
    matches = r2_match and coeff_match

    # Calculate match score
    match_score = 1.0 if matches else 0.5

    return {
        "matches": matches,
        "match_score": match_score,
        "r_squared_match": r2_match,
        "coefficients_match": coeff_match,
        "issues": issues
    }

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
            # Validate chain integrity using cryptographic methods
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
    Validate the integrity of a provenance chain using cryptographic methods
    """
    try:
        # Check data integrity using hash verification
        if not provenance.data_hash:
            logger.warning(f"Provenance {provenance.id} missing data hash")
            return False

        # Verify parent chain consistency
        if provenance.parent_provenance_ids:
            for parent_id in provenance.parent_provenance_ids:
                # Create session for parent lookup
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                temp_db = SessionLocal()
                try:
                    parent = temp_db.query(DataProvenance).filter(DataProvenance.id == parent_id).first()
                    if not parent:
                        logger.warning(f"Provenance {provenance.id} references non-existent parent {parent_id}")
                        return False
                    # Check temporal consistency (child should be after parent)
                    if parent.created_at >= provenance.created_at:
                        logger.warning(f"Provenance {provenance.id} has invalid temporal ordering with parent {parent_id}")
                        return False
                finally:
                    temp_db.close()

        # Verify metadata integrity
        if provenance.metadata:
            # Check for required metadata fields
            required_fields = ['agent_id', 'operation_type']
            for field in required_fields:
                if field not in provenance.metadata:
                    logger.warning(f"Provenance {provenance.id} missing required metadata field: {field}")
                    return False

        # Check for data tampering (if signature is available)
        if provenance.digital_signature:
            # In production, would verify cryptographic signature
            # For now, check signature format consistency
            import hashlib
            expected_signature = hashlib.sha256(
                f"{provenance.data_hash}{provenance.created_at}{provenance.metadata}".encode()
            ).hexdigest()
            if provenance.digital_signature != expected_signature:
                logger.warning(f"Provenance {provenance.id} has invalid digital signature")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating provenance chain {provenance.id}: {str(e)}")
        return False