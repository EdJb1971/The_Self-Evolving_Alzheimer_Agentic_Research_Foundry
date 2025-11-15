from pydantic import BaseModel, Field
from datetime import datetime

class ADWorkbenchQueryBase(BaseModel):
    query_text: str

class ADWorkbenchQueryCreate(ADWorkbenchQueryBase):
    pass

class ADWorkbenchQuery(ADWorkbenchQueryBase):
    id: int
    status: str
    result_data: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    # Pydantic v2 compatibility for ORM mode
    model_config = {'from_attributes': True}

class QueryStatusResponse(BaseModel):
    id: int
    status: str
    message: str
    result_data: str | None = None

class InsightPublishRequest(BaseModel):
    insight_name: str = Field(..., description="Name or title of the insight.")
    insight_description: str | None = Field(None, description="Detailed description of the insight.")
    data_source_ids: list[str] = Field(..., description="List of data source IDs related to the insight.")
    payload: dict = Field(..., description="The actual insight data payload.")
    tags: list[str] = Field(default_factory=list, description="Optional tags for the insight.")

class InsightPublishResponse(BaseModel):
    insight_id: int = Field(..., description="Internal ID of the published insight.")
    message: str = Field(..., description="Status message.")
    workbench_insight_id: str | None = Field(None, description="ID from AD Workbench if successfully synced.")
