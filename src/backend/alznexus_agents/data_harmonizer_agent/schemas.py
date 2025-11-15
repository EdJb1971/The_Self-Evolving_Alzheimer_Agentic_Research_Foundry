from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Optional, Dict, Any, TypeAlias, List
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

class AgentTaskBase(BaseModel):
    agent_id: str = Field(..., description="The ID of the agent executing the task.")
    orchestrator_task_id: Optional[int] = Field(None, description="Optional ID of the orchestrator task that initiated this agent task.")
    task_description: str = Field(..., description="Description of the task to be executed.")
    metadata_json: Optional[ValidatedMetadata] = Field(None, description="Additional task-specific metadata.")

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'AgentTaskBase':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json) # Re-use validator logic
        return self

class AgentTaskCreate(AgentTaskBase):
    pass

class AgentTask(AgentTaskBase):
    id: int
    status: str
    result_data: Optional[ValidatedMetadata] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {'from_attributes': True}

    @model_validator(mode='after')
    def validate_all_result_data(self) -> 'AgentTask':
        if self.result_data is not None:
            MetadataValidator(metadata=self.result_data)
        return self

class AgentStatusResponse(BaseModel):
    agent_id: str
    status: str
    current_task_id: Optional[int] = None
    current_goal: Optional[str] = None
    message: str

class AgentStateUpdate(BaseModel):
    current_goal: Optional[str] = None
    current_task_id: Optional[int] = None
    last_reflection_at: Optional[datetime] = None
    metadata_json: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'AgentStateUpdate':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json)
        return self

# STORY-304 / STORY-106: Schema for publishing insights to AD Workbench API Proxy
class InsightPublishRequest(BaseModel):
    insight_name: str = Field(..., description="Name of the insight.")
    insight_description: str = Field(..., description="Detailed description of the insight.")
    data_source_ids: List[str] = Field(..., description="List of data source IDs related to this insight.")
    payload: Dict[str, Any] = Field(..., description="The actual insight data payload.")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the insight.")
