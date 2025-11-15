from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class AuditLogEntry(Base):
    __tablename__ = "audit_log_entries"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String, nullable=False) # e.g., ORCHESTRATOR, AGENT, AD_PROXY
    entity_id = Column(String, nullable=False) # ID of the specific entity (goal_id, agent_id-task_id, query_id)
    event_type = Column(String, nullable=False) # e.g., GOAL_SET, DAILY_SCAN_INITIATED, TASK_EXECUTED
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata_json = Column(JSON, nullable=True) # Additional context in JSON format

    def __repr__(self):
        return f"<AuditLogEntry(id={self.id}, type='{self.event_type}', entity='{self.entity_type}:{self.entity_id}')>"
