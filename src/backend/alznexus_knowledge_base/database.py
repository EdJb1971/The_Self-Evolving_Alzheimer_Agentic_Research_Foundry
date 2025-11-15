from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from typing import Generator

DATABASE_URL = os.getenv("KNOWLEDGE_DATABASE_URL", "sqlite:///./test_knowledge.db")

# Configure engine with proper isolation level for concurrent updates
if "sqlite" in DATABASE_URL:
    # SQLite: Use SERIALIZABLE isolation to prevent concurrent update issues
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "isolation_level": "SERIALIZABLE"  # Highest isolation level for SQLite
        }
    )
else:
    # PostgreSQL/MySQL: Use REPEATABLE READ for good concurrency with consistency
    engine = create_engine(
        DATABASE_URL,
        isolation_level="REPEATABLE_READ"
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)