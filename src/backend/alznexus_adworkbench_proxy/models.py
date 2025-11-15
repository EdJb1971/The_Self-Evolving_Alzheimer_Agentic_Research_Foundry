from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base

class ADWorkbenchQuery(Base):
    __tablename__ = "adworkbench_queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    status = Column(String, default="PENDING", nullable=False) # PENDING, PROCESSING, COMPLETED, FAILED
    result_data = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ADWorkbenchQuery(id={self.id}, status='{self.status}')>"

class ADWorkbenchInsight(Base):
    __tablename__ = "adworkbench_insights"

    id = Column(Integer, primary_key=True, index=True)
    insight_name = Column(String, nullable=False)
    insight_description = Column(Text, nullable=True)
    data_source_ids = Column(JSON, nullable=False)  # List of data source IDs
    payload = Column(JSON, nullable=False)  # The insight data
    tags = Column(JSON, default=list)  # List of tags
    status = Column(String, default="PUBLISHED", nullable=False)  # PUBLISHED, ARCHIVED
    workbench_insight_id = Column(String, nullable=True)  # ID from AD Workbench if synced
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ADWorkbenchInsight(id={self.id}, name='{self.insight_name}')>"
