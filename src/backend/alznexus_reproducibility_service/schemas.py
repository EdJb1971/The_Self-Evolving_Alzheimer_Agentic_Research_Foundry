from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid

# Seed Management Schemas
class SeedRequest(BaseModel):
    purpose: str = Field(..., description="Purpose of the seed (e.g., 'biomarker_analysis')")
    agent_id: str = Field(..., description="ID of the requesting agent")
    task_id: str = Field(..., description="ID of the current task")
    analysis_type: str = Field(..., description="Type of analysis being performed")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Analysis parameters")

class SeedResponse(BaseModel):
    seed_id: int
    seed_value: int
    purpose: str
    agent_id: str
    task_id: str
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime
    expires_at: Optional[datetime] = None

class SeedRotationPolicyCreate(BaseModel):
    policy_name: str
    agent_type: str
    analysis_type: str
    rotation_interval_hours: int = Field(..., gt=0)
    max_uses_per_seed: int = Field(..., gt=0)
    seed_range_start: int = Field(..., ge=0)
    seed_range_end: int = Field(..., gt=0)

class SeedRotationPolicyResponse(BaseModel):
    id: int
    policy_name: str
    agent_type: str
    analysis_type: str
    rotation_interval_hours: int
    max_uses_per_seed: int
    seed_range_start: int
    seed_range_end: int
    last_rotation: datetime
    current_seed_id: Optional[int] = None
    is_active: bool
    created_at: datetime

# Data Provenance Schemas
class DataProvenanceCreate(BaseModel):
    data_source_id: str
    data_hash: str
    data_size: int
    data_format: str
    schema_version: str
    parent_provenance_id: Optional[int] = None
    transformation_type: Optional[str] = None
    transformation_params: Optional[Dict[str, Any]] = None
    created_by_agent: str
    quality_metrics: Optional[Dict[str, Any]] = None
    privacy_level: str = "public"

class DataProvenanceResponse(BaseModel):
    id: int
    data_source_id: str
    data_hash: str
    data_size: int
    data_format: str
    schema_version: str
    parent_provenance_id: Optional[int] = None
    transformation_type: Optional[str] = None
    transformation_params: Optional[Dict[str, Any]] = None
    created_by_agent: str
    created_at: datetime
    quality_metrics: Optional[Dict[str, Any]] = None
    privacy_level: str

class DataLineageResponse(BaseModel):
    provenance_chain: List[DataProvenanceResponse]
    total_transformations: int
    data_quality_trend: Dict[str, Any]

# Analysis Snapshot Schemas
class AnalysisSnapshotCreate(BaseModel):
    analysis_type: str
    agent_id: str
    task_id: str
    seed_id: int
    data_provenance_id: int
    input_parameters: Dict[str, Any]
    intermediate_results: Optional[Dict[str, Any]] = None
    final_results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None

class AnalysisSnapshotResponse(BaseModel):
    id: int
    snapshot_id: str
    analysis_type: str
    agent_id: str
    task_id: str
    seed_id: int
    data_provenance_id: int
    python_version: str
    package_versions: Dict[str, str]
    system_info: Dict[str, Any]
    git_commit_hash: Optional[str] = None
    input_parameters: Dict[str, Any]
    intermediate_results: Optional[Dict[str, Any]] = None
    final_results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    is_reproducible: Optional[bool] = None
    reproducibility_score: Optional[float] = None
    validation_attempts: int
    last_validated_at: Optional[datetime] = None
    created_at: datetime
    expires_at: Optional[datetime] = None

# Analysis Artifact Schemas
class AnalysisArtifactCreate(BaseModel):
    snapshot_id: int
    artifact_type: str
    filename: str
    file_path: str
    file_hash: str
    file_size: int
    mime_type: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class AnalysisArtifactResponse(BaseModel):
    id: int
    snapshot_id: int
    artifact_type: str
    filename: str
    file_path: str
    file_hash: str
    file_size: int
    mime_type: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime

# Reproducibility Validation Schemas
class ReproducibilityValidationRequest(BaseModel):
    snapshot_id: int
    validation_type: str = Field(default="full", description="Validation type: 'full', 'partial', 'quick'")

class ReproducibilityValidationResponse(BaseModel):
    id: int
    snapshot_id: int
    validation_type: str
    attempted_at: datetime
    is_successful: bool
    reproducibility_score: Optional[float] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    original_results: Optional[Dict[str, Any]] = None
    reproduced_results: Optional[Dict[str, Any]] = None
    differences: Optional[Dict[str, Any]] = None
    environment_match: bool
    package_differences: Optional[Dict[str, Any]] = None

class ReproducibilityReport(BaseModel):
    snapshot_id: str
    overall_score: float
    validations: List[ReproducibilityValidationResponse]
    recommendations: List[str]
    generated_at: datetime

# Environment Capture Schemas
class EnvironmentInfo(BaseModel):
    python_version: str
    package_versions: Dict[str, str]
    system_info: Dict[str, Any]
    git_commit_hash: Optional[str] = None
    environment_variables: Dict[str, str]

# Bulk Operations Schemas
class BulkValidationRequest(BaseModel):
    snapshot_ids: List[int]
    validation_type: str = "quick"

class BulkValidationResponse(BaseModel):
    total_requested: int
    successful_validations: int
    failed_validations: int
    results: List[ReproducibilityValidationResponse]

# Query and Filter Schemas
class SnapshotQuery(BaseModel):
    agent_id: Optional[str] = None
    analysis_type: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_reproducible: Optional[bool] = None
    min_reproducibility_score: Optional[float] = None

class ProvenanceQuery(BaseModel):
    data_source_id: Optional[str] = None
    created_by_agent: Optional[str] = None
    data_format: Optional[str] = None
    privacy_level: Optional[str] = None

# Statistics and Analytics Schemas
class ReproducibilityStats(BaseModel):
    total_snapshots: int
    reproducible_snapshots: int
    average_reproducibility_score: float
    validation_success_rate: float
    agent_breakdown: Dict[str, Dict[str, Any]]
    analysis_type_breakdown: Dict[str, Dict[str, Any]]
    time_based_trends: Dict[str, List[Dict[str, Any]]]

class DataQualityTrends(BaseModel):
    provenance_count: int
    average_quality_score: float
    quality_distribution: Dict[str, int]
    transformation_impact: Dict[str, Dict[str, Any]]
    agent_contributions: Dict[str, Dict[str, Any]]