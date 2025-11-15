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

class ResearchGoalBase(BaseModel):
    goal_text: str = Field(..., description="The high-level research goal for the agent swarm.")

class ResearchGoalCreate(ResearchGoalBase):
    pass

class ResearchGoal(ResearchGoalBase):
    id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {'from_attributes': True}

class OrchestratorTaskBase(BaseModel):
    goal_id: int
    task_type: str = Field(..., description="Type of task, e.g., DAILY_SCAN, COORDINATE_SUB_AGENT, RESOLVE_DEBATE.")
    description: str
    assigned_agent_id: Optional[str] = None
    metadata_json: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'OrchestratorTaskBase':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json) # Re-use validator logic
        return self

class OrchestratorTaskCreate(OrchestratorTaskBase):
    pass

class OrchestratorTask(OrchestratorTaskBase):
    id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {'from_attributes': True}

class DailyScanInitiateResponse(BaseModel):
    orchestrator_task_id: int
    status: str
    message: str

class OrchestratorTaskStatusResponse(BaseModel):
    id: int
    status: str
    message: str
    result_data: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_result_data(self) -> 'OrchestratorTaskStatusResponse':
        if self.result_data is not None:
            MetadataValidator(metadata=self.result_data)
        return self

# New schemas for multi-agent communication
class SubAgentTaskRequest(BaseModel):
    agent_id: str = Field(..., description="The ID of the sub-agent to assign the task to.")
    task_description: str = Field(..., description="Description of the specific task for this sub-agent.")
    task_metadata: Optional[ValidatedMetadata] = Field(None, description="Additional metadata for the sub-agent's task.")

    @model_validator(mode='after')
    def validate_all_task_metadata(self) -> 'SubAgentTaskRequest':
        if self.task_metadata is not None:
            MetadataValidator(metadata=self.task_metadata)
        return self

# CQ-001 FIX: Define AgentTaskCreate schema
class AgentTaskCreate(BaseModel):
    agent_id: str = Field(..., description="The ID of the agent to which the task is being assigned.")
    orchestrator_task_id: int = Field(..., description="The ID of the orchestrator task initiating this sub-agent task.")
    task_description: str = Field(..., description="A detailed description of the task for the sub-agent.")
    metadata_json: Optional[ValidatedMetadata] = Field(None, description="Additional task-specific metadata for the sub-agent.")

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'AgentTaskCreate':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json)
        return self

class TaskCoordinationRequest(BaseModel):
    goal_id: int = Field(..., description="The ID of the research goal this coordination effort contributes to.")
    overall_description: str = Field(..., description="Overall description of the multi-agent coordination task.")
    sub_agent_tasks: List[SubAgentTaskRequest] = Field(..., min_length=1, description="List of tasks for individual sub-agents.")
    coordination_metadata: Optional[ValidatedMetadata] = Field(None, description="Overall metadata for the coordination task.")

    @model_validator(mode='after')
    def validate_all_coordination_metadata(self) -> 'TaskCoordinationRequest':
        if self.coordination_metadata is not None:
            MetadataValidator(metadata=self.coordination_metadata)
        return self

class DebateInitiateRequest(BaseModel):
    goal_id: int = Field(..., description="The ID of the research goal related to this debate.")
    description: str = Field(..., description="A brief description of the debate topic or conflict.")
    conflicting_agents: List[str] = Field(..., min_length=2, description="List of agent IDs involved in the debate.")
    points_of_contention: List[str] = Field(..., description="Key points or findings that are in conflict.")
    evidence_summary: Optional[str] = Field(None, description="Summary of evidence supporting the conflicting points.")
    debate_metadata: Optional[ValidatedMetadata] = Field(None, description="Additional metadata for the debate.")

    @model_validator(mode='after')
    def validate_all_debate_metadata(self) -> 'DebateInitiateRequest':
        if self.debate_metadata is not None:
            MetadataValidator(metadata=self.debate_metadata)
        return self
