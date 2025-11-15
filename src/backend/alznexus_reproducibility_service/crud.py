from sqlalchemy.orm import Session
try:
    # Try relative imports first (when run as package)
    from . import models, schemas
except ImportError:
    # Fall back to absolute imports (when run directly)
    import models, schemas
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import hashlib
import platform
import sys
import subprocess
import os
import uuid

# Try to import git, handle gracefully if not available
try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False

# Try to import pkg_resources for package info
try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

# Seed Management CRUD
def create_random_seed(db: Session, seed_request: schemas.SeedRequest) -> models.RandomSeed:
    """Create a new random seed for reproducible analysis"""
    # Generate a unique seed value (simple approach - in production use cryptographically secure)
    import random
    seed_value = random.randint(0, 2**31 - 1)

    db_seed = models.RandomSeed(
        seed_value=seed_value,
        purpose=seed_request.purpose,
        agent_id=seed_request.agent_id,
        task_id=seed_request.task_id,
        analysis_type=seed_request.analysis_type,
        parameters=seed_request.parameters,
        expires_at=datetime.utcnow() + timedelta(hours=24)  # Default 24 hour expiration
    )
    db.add(db_seed)
    db.commit()
    db.refresh(db_seed)
    return db_seed

def get_seed_by_value(db: Session, seed_value: int) -> Optional[models.RandomSeed]:
    """Get seed by its value"""
    return db.query(models.RandomSeed).filter(models.RandomSeed.seed_value == seed_value).first()

def get_seeds_by_agent(db: Session, agent_id: str) -> List[models.RandomSeed]:
    """Get all seeds used by a specific agent"""
    return db.query(models.RandomSeed).filter(models.RandomSeed.agent_id == agent_id).all()

def get_active_seeds(db: Session) -> List[models.RandomSeed]:
    """Get all currently active (non-expired) seeds"""
    return db.query(models.RandomSeed).filter(
        (models.RandomSeed.expires_at.is_(None)) |
        (models.RandomSeed.expires_at > datetime.utcnow())
    ).all()

# Data Provenance CRUD
def create_data_provenance(db: Session, provenance: schemas.DataProvenanceCreate) -> models.DataProvenance:
    """Create a new data provenance record"""
    db_provenance = models.DataProvenance(
        data_source_id=provenance.data_source_id,
        data_hash=provenance.data_hash,
        data_size=provenance.data_size,
        data_format=provenance.data_format,
        schema_version=provenance.schema_version,
        parent_provenance_id=provenance.parent_provenance_id,
        transformation_type=provenance.transformation_type,
        transformation_params=provenance.transformation_params,
        created_by_agent=provenance.created_by_agent,
        quality_metrics=provenance.quality_metrics,
        privacy_level=provenance.privacy_level
    )
    db.add(db_provenance)
    db.commit()
    db.refresh(db_provenance)
    return db_provenance

def get_data_provenance(db: Session, provenance_id: int) -> Optional[models.DataProvenance]:
    """Get data provenance by ID"""
    return db.query(models.DataProvenance).filter(models.DataProvenance.id == provenance_id).first()

def get_provenance_chain(db: Session, provenance_id: int) -> List[models.DataProvenance]:
    """Get the complete provenance chain from source to current"""
    chain = []
    current = get_data_provenance(db, provenance_id)

    while current:
        chain.insert(0, current)  # Insert at beginning to maintain chronological order
        if current.parent_provenance_id:
            current = get_data_provenance(db, current.parent_provenance_id)
        else:
            current = None

    return chain

def get_provenance_by_hash(db: Session, data_hash: str) -> Optional[models.DataProvenance]:
    """Get provenance by data hash"""
    return db.query(models.DataProvenance).filter(models.DataProvenance.data_hash == data_hash).first()

# Analysis Snapshot CRUD
def create_analysis_snapshot(db: Session, snapshot: schemas.AnalysisSnapshotCreate) -> models.AnalysisSnapshot:
    """Create a new analysis snapshot for reproducibility"""
    # Capture environment information
    env_info = capture_environment()

    # Generate unique snapshot ID
    snapshot_id = str(uuid.uuid4())

    db_snapshot = models.AnalysisSnapshot(
        snapshot_id=snapshot_id,
        analysis_type=snapshot.analysis_type,
        agent_id=snapshot.agent_id,
        task_id=snapshot.task_id,
        seed_id=snapshot.seed_id,
        data_provenance_id=snapshot.data_provenance_id,
        python_version=env_info["python_version"],
        package_versions=env_info["package_versions"],
        system_info=env_info["system_info"],
        git_commit_hash=env_info.get("git_commit_hash"),
        environment_variables=env_info.get("environment_variables", {}),
        input_parameters=snapshot.input_parameters,
        intermediate_results=snapshot.intermediate_results,
        final_results=snapshot.final_results,
        execution_time=snapshot.execution_time,
        expires_at=datetime.utcnow() + timedelta(days=90)  # Default 90 day retention
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot

def get_analysis_snapshot(db: Session, snapshot_id: int) -> Optional[models.AnalysisSnapshot]:
    """Get analysis snapshot by ID"""
    return db.query(models.AnalysisSnapshot).filter(models.AnalysisSnapshot.id == snapshot_id).first()

def get_snapshot_by_uuid(db: Session, snapshot_uuid: str) -> Optional[models.AnalysisSnapshot]:
    """Get snapshot by UUID"""
    return db.query(models.AnalysisSnapshot).filter(models.AnalysisSnapshot.snapshot_id == snapshot_uuid).first()

def get_snapshots_by_agent(db: Session, agent_id: str, skip: int = 0, limit: int = 100) -> List[models.AnalysisSnapshot]:
    """Get all snapshots for a specific agent"""
    return db.query(models.AnalysisSnapshot).filter(
        models.AnalysisSnapshot.agent_id == agent_id
    ).offset(skip).limit(limit).all()

def get_snapshots_by_task(db: Session, task_id: str) -> List[models.AnalysisSnapshot]:
    """Get all snapshots for a specific task"""
    return db.query(models.AnalysisSnapshot).filter(
        models.AnalysisSnapshot.task_id == task_id
    ).all()

# Analysis Artifact CRUD
def create_analysis_artifact(db: Session, artifact: schemas.AnalysisArtifactCreate) -> models.AnalysisArtifact:
    """Create a new analysis artifact record"""
    db_artifact = models.AnalysisArtifact(
        snapshot_id=artifact.snapshot_id,
        artifact_type=artifact.artifact_type,
        filename=artifact.filename,
        file_path=artifact.file_path,
        file_hash=artifact.file_hash,
        file_size=artifact.file_size,
        mime_type=artifact.mime_type,
        description=artifact.description,
        parameters=artifact.parameters
    )
    db.add(db_artifact)
    db.commit()
    db.refresh(db_artifact)
    return db_artifact

def get_analysis_artifacts(db: Session, snapshot_id: int) -> List[models.AnalysisArtifact]:
    """Get all artifacts for a snapshot"""
    return db.query(models.AnalysisArtifact).filter(
        models.AnalysisArtifact.snapshot_id == snapshot_id
    ).all()

# Reproducibility Validation CRUD
def create_reproducibility_validation(
    db: Session,
    validation: schemas.ReproducibilityValidationRequest,
    is_successful: bool,
    reproducibility_score: Optional[float] = None,
    execution_time: Optional[float] = None,
    error_message: Optional[str] = None,
    original_results: Optional[Dict[str, Any]] = None,
    reproduced_results: Optional[Dict[str, Any]] = None,
    differences: Optional[Dict[str, Any]] = None,
    environment_match: bool = True,
    package_differences: Optional[Dict[str, Any]] = None
) -> models.ReproducibilityValidation:
    """Create a reproducibility validation record"""

    db_validation = models.ReproducibilityValidation(
        snapshot_id=validation.snapshot_id,
        validation_type=validation.validation_type,
        is_successful=is_successful,
        reproducibility_score=reproducibility_score,
        execution_time=execution_time,
        error_message=error_message,
        original_results=original_results,
        reproduced_results=reproduced_results,
        differences=differences,
        environment_match=environment_match,
        package_differences=package_differences
    )
    db.add(db_validation)
    db.commit()
    db.refresh(db_validation)

    # Update the snapshot's reproducibility status
    snapshot = get_analysis_snapshot(db, validation.snapshot_id)
    if snapshot:
        snapshot.is_reproducible = is_successful
        snapshot.reproducibility_score = reproducibility_score
        snapshot.validation_attempts += 1
        snapshot.last_validated_at = datetime.utcnow()
        db.commit()

    return db_validation

def get_reproducibility_validations(db: Session, snapshot_id: int) -> List[models.ReproducibilityValidation]:
    """Get all validation attempts for a snapshot"""
    return db.query(models.ReproducibilityValidation).filter(
        models.ReproducibilityValidation.snapshot_id == snapshot_id
    ).order_by(models.ReproducibilityValidation.attempted_at.desc()).all()

# Seed Rotation Policy CRUD
def create_seed_rotation_policy(db: Session, policy: schemas.SeedRotationPolicyCreate) -> models.SeedRotationPolicy:
    """Create a new seed rotation policy"""
    db_policy = models.SeedRotationPolicy(
        policy_name=policy.policy_name,
        agent_type=policy.agent_type,
        analysis_type=policy.analysis_type,
        rotation_interval_hours=policy.rotation_interval_hours,
        max_uses_per_seed=policy.max_uses_per_seed,
        seed_range_start=policy.seed_range_start,
        seed_range_end=policy.seed_range_end
    )
    db.add(db_policy)
    db.commit()
    db.refresh(db_policy)
    return db_policy

def get_seed_rotation_policy(db: Session, policy_name: str) -> Optional[models.SeedRotationPolicy]:
    """Get seed rotation policy by name"""
    return db.query(models.SeedRotationPolicy).filter(
        models.SeedRotationPolicy.policy_name == policy_name
    ).first()

def get_active_policies(db: Session) -> List[models.SeedRotationPolicy]:
    """Get all active seed rotation policies"""
    return db.query(models.SeedRotationPolicy).filter(
        models.SeedRotationPolicy.is_active == True
    ).all()

# Utility Functions
def capture_environment() -> Dict[str, Any]:
    """Capture current environment information for reproducibility"""
    env_info = {
        "python_version": sys.version,
        "system_info": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "machine": platform.machine(),
            "system": platform.system(),
            "version": platform.version()
        }
    }

    # Capture package versions
    try:
        packages = {}
        for pkg in pkg_resources.working_set:
            packages[pkg.project_name] = pkg.version
        env_info["package_versions"] = packages
    except Exception:
        env_info["package_versions"] = {}

    # Capture git commit hash if in a git repository
    try:
        repo = git.Repo(search_parent_directories=True)
        env_info["git_commit_hash"] = repo.head.object.hexsha
    except Exception:
        env_info["git_commit_hash"] = None

    # Capture relevant environment variables (excluding sensitive ones)
    relevant_env_vars = {}
    sensitive_patterns = ['key', 'secret', 'password', 'token', 'auth']
    for key, value in os.environ.items():
        if not any(pattern in key.lower() for pattern in sensitive_patterns):
            relevant_env_vars[key] = value
    env_info["environment_variables"] = relevant_env_vars

    return env_info

def calculate_data_hash(data: Any) -> str:
    """Calculate SHA-256 hash of data for integrity checking"""
    if isinstance(data, str):
        data_str = data
    elif isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)

    return hashlib.sha256(data_str.encode()).hexdigest()

def validate_reproducibility(snapshot: models.AnalysisSnapshot, new_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate reproducibility by comparing original and new results"""
    validation = {
        "is_reproducible": False,
        "score": 0.0,
        "differences": {},
        "execution_time_ratio": None
    }

    if not snapshot.final_results:
        return validation

    original_results = snapshot.final_results
    differences = {}

    # Compare key metrics (this would be customized per analysis type)
    for key in original_results.keys():
        if key in new_results:
            orig_val = original_results[key]
            new_val = new_results[key]

            if isinstance(orig_val, (int, float)) and isinstance(new_val, (int, float)):
                diff = abs(orig_val - new_val)
                rel_diff = diff / abs(orig_val) if orig_val != 0 else 0
                differences[key] = {
                    "original": orig_val,
                    "reproduced": new_val,
                    "absolute_difference": diff,
                    "relative_difference": rel_diff
                }
            else:
                differences[key] = {
                    "original": orig_val,
                    "reproduced": new_val,
                    "match": orig_val == new_val
                }

    # Calculate reproducibility score (simple approach)
    if differences:
        matches = sum(1 for diff in differences.values() if diff.get("match", False))
        total_comparisons = len(differences)
        validation["score"] = matches / total_comparisons if total_comparisons > 0 else 0.0
        validation["is_reproducible"] = validation["score"] >= 0.95  # 95% threshold

    validation["differences"] = differences
    return validation