from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from .database import Base

class BiasDetectionReport(Base):
    __tablename__ = "bias_detection_reports"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String, nullable=False) # e.g., 'AGENT_OUTPUT', 'LLM_RESPONSE', 'DATA_INPUT'
    entity_id = Column(String, nullable=True) # ID of the entity being analyzed, if applicable
    data_snapshot = Column(Text, nullable=False) # Snapshot of the data/text analyzed
    detected_bias = Column(Boolean, nullable=False)
    bias_type = Column(String, nullable=True) # e.g., 'demographic', 'selection', 'confirmation'
    severity = Column(String, nullable=True) # e.g., 'low', 'medium', 'high'
    analysis_summary = Column(Text, nullable=True)
    proposed_corrections = Column(JSON, nullable=True) # JSON list of suggested actions
    metadata_json = Column(JSON, nullable=True) # Additional detection metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<BiasDetectionReport(id={self.id}, entity='{self.entity_type}:{self.entity_id}', bias={self.detected_bias})>"
