from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class AgentTask(Base):
    __tablename__ = "collaboration_matchmaker_agent_tasks"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, nullable=False)
    orchestrator_task_id = Column(Integer, nullable=True)
    task_description = Column(Text, nullable=False)
    status = Column(String, default="PENDING", nullable=False)
    result_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<AgentTask(id={self.id}, agent='{self.agent_id}', status='{self.status}')>"

class AgentState(Base):
    __tablename__ = "collaboration_matchmaker_agent_states"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, nullable=False)
    current_goal = Column(Text, nullable=True)
    current_task_id = Column(Integer, nullable=True)
    last_reflection_at = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<AgentState(agent_id='{self.agent_id}', current_task_id={self.current_task_id})>"
