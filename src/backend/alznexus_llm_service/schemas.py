from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Optional, Dict, Any, TypeAlias, List
import json

MAX_METADATA_SIZE_BYTES = 10 * 1024 # 10KB limit for metadata JSON

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

class LLMChatRequest(BaseModel):
    model_name: str = Field(..., description="The name of the LLM to use (e.g., 'grok-1', 'llama-3').")
    prompt: str = Field(..., description="The user's input prompt for chat completion.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for generation.")
    max_tokens: int = Field(500, ge=1, description="Maximum number of tokens to generate.")
    metadata: Optional[ValidatedMetadata] = Field(None, description="Additional request metadata.")

    @model_validator(mode='after')
    def validate_all_metadata(self) -> 'LLMChatRequest':
        if self.metadata is not None:
            MetadataValidator(metadata=self.metadata)
        return self

class LLMToolUseRequest(BaseModel):
    model_name: str = Field(..., description="The name of the LLM to use (e.g., 'grok-1', 'llama-3').")
    prompt: str = Field(..., description="The user's input prompt for tool use.")
    tools: List[Dict[str, Any]] = Field(..., description="List of available tools for the LLM to use.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for generation.")
    max_tokens: int = Field(500, ge=1, description="Maximum number of tokens to generate.")
    metadata: Optional[ValidatedMetadata] = Field(None, description="Additional request metadata.")

    @model_validator(mode='after')
    def validate_all_metadata(self) -> 'LLMToolUseRequest':
        if self.metadata is not None:
            MetadataValidator(metadata=self.metadata)
        return self

class LLMResponse(BaseModel):
    model_name: str
    response_text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    detected_bias: bool = False
    detected_injection: bool = False
    ethical_flags: Optional[ValidatedMetadata] = None
    metadata: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_metadata(self) -> 'LLMResponse':
        if self.metadata is not None:
            MetadataValidator(metadata=self.metadata)
        if self.ethical_flags is not None:
            MetadataValidator(metadata=self.ethical_flags)
        return self

class LLMRequestLogCreate(BaseModel):
    model_name: str
    prompt: str
    response: Optional[str] = None
    request_type: str
    detected_bias: bool = False
    detected_injection: bool = False
    ethical_flags: Optional[ValidatedMetadata] = None
    metadata_json: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'LLMRequestLogCreate':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json)
        if self.ethical_flags is not None:
            MetadataValidator(metadata=self.ethical_flags)
        return self

class LLMRequestLogEntry(LLMRequestLogCreate):
    id: int
    timestamp: datetime

    model_config = {'from_attributes': True}
