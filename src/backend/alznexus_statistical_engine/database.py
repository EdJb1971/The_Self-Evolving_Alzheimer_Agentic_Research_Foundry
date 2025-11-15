from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)

# Database URL from environment
DATABASE_URL = os.getenv("STATISTICAL_DATABASE_URL")
if not DATABASE_URL:
    if os.getenv("ENV") != "production":
        logger.warning("STATISTICAL_DATABASE_URL not set. Using SQLite for development.")
        DATABASE_URL = "sqlite:///./test_statistical.db"
    else:
        raise ValueError("STATISTICAL_DATABASE_URL must be set in production")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
    echo=False  # Set to True for SQL query logging in development
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()