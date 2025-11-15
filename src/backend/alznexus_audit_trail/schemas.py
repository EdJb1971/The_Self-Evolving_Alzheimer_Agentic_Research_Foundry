from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Optional, Dict, Any, TypeAlias
import json

MAX_METADATA_SIZE_BYTES = 10 * 1024 # 10KB limit for metadata JSON

# Custom type for validated metadata to prevent excessively large or arbitrary data
ValidatedMetadata: TypeAlias = Dict[str, Any]

class MetadataValidator(BaseModel):
    metadata: ValidatedMetadata

    @model_validator(mode='after')
    def validate_metadata_size(self) -> 'MetadataValidator':
        if self.metadata is not None:
            serialized_metadata = json.dumps(self.metadata)
            if len(serialized_metadata.encode('utf-8')) > MAX_METADATA_SIZE_BYTES:
                raise ValueError(f"Metadata JSON size exceeds {MAX_METADATA_SIZE_BYTES} bytes.")
        return self

class AuditLogCreate(BaseModel):
    entity_type: str = Field(..., description="Type of entity, e.g., ORCHESTRATOR, AGENT, AD_PROXY.")
    entity_id: str = Field(..., description="ID of the specific entity (e.g., goal_id, agent_id-task_id, query_id).")
    event_type: str = Field(..., description="Type of event, e.g., GOAL_SET, DAILY_SCAN_INITIATED, TASK_EXECUTED, REFLECTION_COMPLETED, SELF_CORRECTION_COMPLETED.")
    description: str = Field(..., description="A brief description of the event.")
    metadata: Optional[ValidatedMetadata] = Field(None, description="Additional context, detailed reasoning steps, or intermediate thoughts related to the event.")

    @model_validator(mode='after')
    def validate_all_metadata(self) -> 'AuditLogCreate':
        if self.metadata is not None:
            MetadataValidator(metadata=self.metadata) # Re-use validator logic
        return self

class AuditLogEntry(BaseModel):
    id: int
    entity_type: str
    entity_id: str
    event_type: str
    description: str
    timestamp: datetime
    metadata_json: Optional[ValidatedMetadata] = Field(None, description="Additional context, detailed reasoning steps, or intermediate thoughts related to the event.")

    model_config = {'from_attributes': True}

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'AuditLogEntry':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json)
        return self

class AuditHistoryResponse(BaseModel):
    entity_type: str
    entity_id: str
    history: list[AuditLogEntry]
