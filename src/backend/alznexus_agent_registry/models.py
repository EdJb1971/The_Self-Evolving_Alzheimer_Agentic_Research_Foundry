from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class RegisteredAgent(Base):
    __tablename__ = "registered_agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, nullable=False, index=True)
    capabilities = Column(JSON, nullable=True) # JSON describing agent's functions, tools, etc.
    api_endpoint = Column(String, nullable=False) # Base URL for the agent's API
    registered_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<RegisteredAgent(agent_id='{self.agent_id}', api_endpoint='{self.api_endpoint}')>"
