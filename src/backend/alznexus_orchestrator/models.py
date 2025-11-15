from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class ResearchGoal(Base):
    __tablename__ = "research_goals"

    id = Column(Integer, primary_key=True, index=True)
    goal_text = Column(Text, nullable=False)
    status = Column(String, default="ACTIVE", nullable=False) # ACTIVE, COMPLETED, ARCHIVED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    orchestrator_tasks = relationship("OrchestratorTask", back_populates="research_goal")

    def __repr__(self):
        return f"<ResearchGoal(id={self.id}, status='{self.status}')>"

class OrchestratorTask(Base):
    __tablename__ = "orchestrator_tasks"

    id = Column(Integer, primary_key=True, index=True)
    goal_id = Column(Integer, ForeignKey("research_goals.id"), nullable=False)
    task_type = Column(String, nullable=False) # e.g., "DAILY_SCAN", "COORDINATE_SUB_AGENT", "RESOLVE_DEBATE", "SELF_CORRECTION"
    description = Column(Text, nullable=False)
    status = Column(String, default="PENDING", nullable=False) # PENDING, IN_PROGRESS, COMPLETED, FAILED
    assigned_agent_id = Column(String, nullable=True) # ID of the sub-agent if assigned
    metadata_json = Column(JSON, nullable=True) # Additional task-specific metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    research_goal = relationship("ResearchGoal", back_populates="orchestrator_tasks")

    def __repr__(self):
        return f"<OrchestratorTask(id={self.id}, type='{self.task_type}', status='{self.status}')>"
