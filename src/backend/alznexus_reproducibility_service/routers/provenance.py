from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

try:
    # Try relative imports first (when run as package)
    from ..database import get_db
    from ..models import DataProvenance, AnalysisSnapshot, AnalysisArtifact
    from ..schemas import (
        DataProvenanceCreate, DataProvenanceResponse,
        DataLineageResponse, AnalysisSnapshotCreate, AnalysisSnapshotResponse,
        AnalysisArtifactCreate, AnalysisArtifactResponse
    )
    from ..crud import (
        create_data_provenance, get_data_provenance, get_provenance_chain,
        create_analysis_snapshot, get_analysis_snapshot, get_snapshots_by_analysis,
        create_analysis_artifact, get_artifacts_by_snapshot
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from database import get_db
    from models import DataProvenance, AnalysisSnapshot, AnalysisArtifact
    from schemas import (
        DataProvenanceCreate, DataProvenanceResponse,
        DataLineageResponse, AnalysisSnapshotCreate, AnalysisSnapshotResponse,
        AnalysisArtifactCreate, AnalysisArtifactResponse
    )
    from crud import (
        create_data_provenance, get_data_provenance, get_provenance_chain,
        create_analysis_snapshot, get_analysis_snapshot, get_snapshots_by_task,
        create_analysis_artifact, get_analysis_artifacts
    )

router = APIRouter()

@router.post("/provenance", response_model=DataProvenanceResponse)
async def create_provenance_record(
    provenance: DataProvenanceCreate,
    db: Session = Depends(get_db)
) -> DataProvenanceResponse:
    """
    Create a new data provenance record
    """
    try:
        db_provenance = create_data_provenance(db, provenance)
        return DataProvenanceResponse(
            id=db_provenance.id,
            data_source_id=db_provenance.data_source_id,
            data_hash=db_provenance.data_hash,
            data_size=db_provenance.data_size,
            data_format=db_provenance.data_format,
            schema_version=db_provenance.schema_version,
            parent_provenance_id=db_provenance.parent_provenance_id,
            transformation_type=db_provenance.transformation_type,
            transformation_params=db_provenance.transformation_params,
            created_by_agent=db_provenance.created_by_agent,
            created_at=db_provenance.created_at,
            quality_metrics=db_provenance.quality_metrics,
            privacy_level=db_provenance.privacy_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create provenance record: {str(e)}")

@router.get("/provenance/{provenance_id}", response_model=DataProvenanceResponse)
async def get_provenance_record(
    provenance_id: int,
    db: Session = Depends(get_db)
) -> DataProvenanceResponse:
    """
    Get a specific provenance record by ID
    """
    try:
        provenance = get_data_provenance(db, provenance_id)
        if not provenance:
            raise HTTPException(status_code=404, detail="Provenance record not found")
        return DataProvenanceResponse(
            id=provenance.id,
            data_source_id=provenance.data_source_id,
            data_hash=provenance.data_hash,
            data_size=provenance.data_size,
            data_format=provenance.data_format,
            schema_version=provenance.schema_version,
            parent_provenance_id=provenance.parent_provenance_id,
            transformation_type=provenance.transformation_type,
            transformation_params=provenance.transformation_params,
            created_by_agent=provenance.created_by_agent,
            created_at=provenance.created_at,
            quality_metrics=provenance.quality_metrics,
            privacy_level=provenance.privacy_level
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve provenance record: {str(e)}")

@router.post("/snapshots", response_model=AnalysisSnapshotResponse)
async def create_analysis_snapshot(
    snapshot: AnalysisSnapshotCreate,
    db: Session = Depends(get_db)
) -> AnalysisSnapshotResponse:
    """
    Create a snapshot of analysis state for reproducibility
    """
    try:
        db_snapshot = create_analysis_snapshot(db, snapshot)
        return AnalysisSnapshotResponse(
            id=db_snapshot.id,
            snapshot_id=db_snapshot.snapshot_id,
            analysis_type=db_snapshot.analysis_type,
            agent_id=db_snapshot.agent_id,
            task_id=db_snapshot.task_id,
            seed_id=db_snapshot.seed_id,
            data_provenance_id=db_snapshot.data_provenance_id,
            python_version=db_snapshot.python_version,
            package_versions=db_snapshot.package_versions,
            system_info=db_snapshot.system_info,
            git_commit_hash=db_snapshot.git_commit_hash,
            input_parameters=db_snapshot.input_parameters,
            intermediate_results=db_snapshot.intermediate_results,
            final_results=db_snapshot.final_results,
            execution_time=db_snapshot.execution_time,
            is_reproducible=db_snapshot.is_reproducible,
            reproducibility_score=db_snapshot.reproducibility_score,
            validation_attempts=db_snapshot.validation_attempts,
            last_validated_at=db_snapshot.last_validated_at,
            created_at=db_snapshot.created_at,
            expires_at=db_snapshot.expires_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create analysis snapshot: {str(e)}")

@router.get("/snapshots/{snapshot_id}", response_model=AnalysisSnapshotResponse)
async def get_snapshot(
    snapshot_id: int,
    db: Session = Depends(get_db)
) -> AnalysisSnapshotResponse:
    """
    Get a specific analysis snapshot
    """
    try:
        snapshot = get_analysis_snapshot(db, snapshot_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Analysis snapshot not found")
        return AnalysisSnapshotResponse(
            id=snapshot.id,
            snapshot_id=snapshot.snapshot_id,
            analysis_type=snapshot.analysis_type,
            agent_id=snapshot.agent_id,
            task_id=snapshot.task_id,
            seed_id=snapshot.seed_id,
            data_provenance_id=snapshot.data_provenance_id,
            python_version=snapshot.python_version,
            package_versions=snapshot.package_versions,
            system_info=snapshot.system_info,
            git_commit_hash=snapshot.git_commit_hash,
            input_parameters=snapshot.input_parameters,
            intermediate_results=snapshot.intermediate_results,
            final_results=snapshot.final_results,
            execution_time=snapshot.execution_time,
            is_reproducible=snapshot.is_reproducible,
            reproducibility_score=snapshot.reproducibility_score,
            validation_attempts=snapshot.validation_attempts,
            last_validated_at=snapshot.last_validated_at,
            created_at=snapshot.created_at,
            expires_at=snapshot.expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis snapshot: {str(e)}")