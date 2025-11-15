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

class BiasDetectionRequest(BaseModel):
    entity_type: str = Field(..., description="Type of entity being analyzed (e.g., 'AGENT_OUTPUT', 'LLM_RESPONSE', 'DATA_INPUT').")
    entity_id: Optional[str] = Field(None, description="ID of the specific entity being analyzed, if applicable.")
    data_to_analyze: str = Field(..., description="The data or text content to be analyzed for bias.")
    analysis_context: Optional[ValidatedMetadata] = Field(None, description="Additional context for bias analysis.")

    @model_validator(mode='after')
    def validate_all_analysis_context(self) -> 'BiasDetectionRequest':
        if self.analysis_context is not None:
            MetadataValidator(metadata=self.analysis_context)
        return self

class BiasDetectionReportCreate(BaseModel):
    entity_type: str
    entity_id: Optional[str] = None
    data_snapshot: str
    detected_bias: bool
    bias_type: Optional[str] = None
    severity: Optional[str] = None
    analysis_summary: Optional[str] = None
    proposed_corrections: Optional[ValidatedMetadata] = None
    metadata_json: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_metadata_json(self) -> 'BiasDetectionReportCreate':
        if self.metadata_json is not None:
            MetadataValidator(metadata=self.metadata_json)
        if self.proposed_corrections is not None:
            MetadataValidator(metadata=self.proposed_corrections)
        return self

class BiasDetectionReport(BiasDetectionReportCreate):
    id: int
    timestamp: datetime

    model_config = {'from_attributes': True}

class BiasDetectionResponse(BaseModel):
    report_id: int
    status: str
    message: str
    detected_bias: bool
    bias_type: Optional[str] = None
    severity: Optional[str] = None
    analysis_summary: Optional[str] = None
    proposed_corrections: Optional[ValidatedMetadata] = None

    @model_validator(mode='after')
    def validate_all_proposed_corrections(self) -> 'BiasDetectionResponse':
        if self.proposed_corrections is not None:
            MetadataValidator(metadata=self.proposed_corrections)
        return self
