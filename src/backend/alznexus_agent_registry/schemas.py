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

class AgentRegister(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the sub-agent.")
    capabilities: Optional[ValidatedMetadata] = Field(None, description="JSON object describing the agent's capabilities, tools, and domain expertise.")
    api_endpoint: str = Field(..., description="The base URL where the agent's API can be reached.")

    @model_validator(mode='after')
    def validate_all_capabilities(self) -> 'AgentRegister':
        if self.capabilities is not None:
            MetadataValidator(metadata=self.capabilities)
        return self

class AgentDetails(AgentRegister):
    id: int
    registered_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {'from_attributes': True}

class AgentListResponse(BaseModel):
    agents: List[AgentDetails]
