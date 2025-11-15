from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from .database import Base

class LLMRequestLog(Base):
    __tablename__ = "llm_request_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    request_type = Column(String, nullable=False) # e.g., 'chat', 'tool_use'
    detected_bias = Column(Boolean, default=False)
    detected_injection = Column(Boolean, default=False)
    ethical_flags = Column(JSON, nullable=True) # e.g., {'harmful_content': True, 'pii_leak': False}
    metadata_json = Column(JSON, nullable=True) # Additional request/response metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<LLMRequestLog(id={self.id}, model='{self.model_name}', type='{self.request_type}', bias={self.detected_bias})>"
