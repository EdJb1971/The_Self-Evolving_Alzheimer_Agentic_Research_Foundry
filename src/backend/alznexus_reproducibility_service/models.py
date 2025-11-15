from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class RandomSeed(Base):
    """Tracks random seeds used across all analyses for reproducibility"""
    __tablename__ = "random_seeds"

    id = Column(Integer, primary_key=True, index=True)
    seed_value = Column(Integer, nullable=False, unique=True)
    purpose = Column(String(255), nullable=False)  # e.g., "biomarker_analysis", "hypothesis_test"
    agent_id = Column(String(100), nullable=False)
    task_id = Column(String(100), nullable=False)
    analysis_type = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=True)  # Parameters used with this seed
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # For seed rotation

    # Relationships
    snapshots = relationship("AnalysisSnapshot", back_populates="seed")

class DataProvenance(Base):
    """Tracks data lineage from source to final insight"""
    __tablename__ = "data_provenance"

    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(String(255), nullable=False)  # Original data source identifier
    data_hash = Column(String(128), nullable=False)  # SHA-256 hash of data
    data_size = Column(Integer, nullable=False)  # Size in bytes
    data_format = Column(String(50), nullable=False)  # csv, json, parquet, etc.
    schema_version = Column(String(50), nullable=False)

    # Provenance chain
    parent_provenance_id = Column(Integer, ForeignKey("data_provenance.id"), nullable=True)
    transformation_type = Column(String(100), nullable=True)  # harmonize, filter, aggregate, etc.
    transformation_params = Column(JSON, nullable=True)

    # Metadata
    created_by_agent = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    quality_metrics = Column(JSON, nullable=True)  # Data quality scores
    privacy_level = Column(String(50), default="public")  # public, restricted, confidential

    # Relationships
    parent = relationship("DataProvenance", remote_side=[id])
    snapshots = relationship("AnalysisSnapshot", back_populates="data_provenance")

class AnalysisSnapshot(Base):
    """Complete snapshot of analysis state for reproducibility"""
    __tablename__ = "analysis_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(String(100), unique=True, nullable=False)  # UUID-based identifier
    analysis_type = Column(String(100), nullable=False)
    agent_id = Column(String(100), nullable=False)
    task_id = Column(String(100), nullable=False)

    # Reproducibility data
    seed_id = Column(Integer, ForeignKey("random_seeds.id"), nullable=False)
    data_provenance_id = Column(Integer, ForeignKey("data_provenance.id"), nullable=False)

    # Environment capture
    python_version = Column(String(20), nullable=False)
    package_versions = Column(JSON, nullable=False)  # All installed packages
    system_info = Column(JSON, nullable=False)  # OS, CPU, memory, etc.
    git_commit_hash = Column(String(40), nullable=True)
    environment_variables = Column(JSON, nullable=True)  # Relevant env vars

    # Analysis state
    input_parameters = Column(JSON, nullable=False)
    intermediate_results = Column(JSON, nullable=True)
    final_results = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)  # In seconds

    # Validation
    is_reproducible = Column(Boolean, default=None)  # Null = not tested
    reproducibility_score = Column(Float, nullable=True)  # 0-1 scale
    validation_attempts = Column(Integer, default=0)
    last_validated_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # For cleanup

    # Relationships
    seed = relationship("RandomSeed", back_populates="snapshots")
    data_provenance = relationship("DataProvenance", back_populates="snapshots")
    artifacts = relationship("AnalysisArtifact", back_populates="snapshot")

class AnalysisArtifact(Base):
    """Stores analysis artifacts like models, plots, reports"""
    __tablename__ = "analysis_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("analysis_snapshots.id"), nullable=False)
    artifact_type = Column(String(50), nullable=False)  # model, plot, report, data
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # Relative to storage root
    file_hash = Column(String(128), nullable=False)  # SHA-256
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)

    # Metadata
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)  # Generation parameters
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    snapshot = relationship("AnalysisSnapshot", back_populates="artifacts")

class ReproducibilityValidation(Base):
    """Records reproducibility validation attempts"""
    __tablename__ = "reproducibility_validations"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("analysis_snapshots.id"), nullable=False)
    validation_type = Column(String(50), nullable=False)  # full, partial, quick
    attempted_at = Column(DateTime, default=datetime.utcnow)

    # Validation results
    is_successful = Column(Boolean, nullable=False)
    reproducibility_score = Column(Float, nullable=True)
    execution_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Comparison data
    original_results = Column(JSON, nullable=True)
    reproduced_results = Column(JSON, nullable=True)
    differences = Column(JSON, nullable=True)  # Statistical differences

    # Environment comparison
    environment_match = Column(Boolean, default=True)
    package_differences = Column(JSON, nullable=True)

class SeedRotationPolicy(Base):
    """Manages automatic seed rotation policies"""
    __tablename__ = "seed_rotation_policies"

    id = Column(Integer, primary_key=True, index=True)
    policy_name = Column(String(100), unique=True, nullable=False)
    agent_type = Column(String(100), nullable=False)
    analysis_type = Column(String(100), nullable=False)

    # Rotation rules
    rotation_interval_hours = Column(Integer, nullable=False)  # How often to rotate
    max_uses_per_seed = Column(Integer, nullable=False)  # Max analyses per seed
    seed_range_start = Column(Integer, nullable=False)
    seed_range_end = Column(Integer, nullable=False)

    # Tracking
    last_rotation = Column(DateTime, default=datetime.utcnow)
    current_seed_id = Column(Integer, ForeignKey("random_seeds.id"), nullable=True)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)