from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
REPRODUCIBILITY_DATABASE_URL = os.getenv(
    "REPRODUCIBILITY_DATABASE_URL",
    "sqlite:///./test_reproducibility.db"
)

# Create engine
engine = create_engine(
    REPRODUCIBILITY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in REPRODUCIBILITY_DATABASE_URL else {}
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def reset_database():
    """Drop and recreate all tables (for testing)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)