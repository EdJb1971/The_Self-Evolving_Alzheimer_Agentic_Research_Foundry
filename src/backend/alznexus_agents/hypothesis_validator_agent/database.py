import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv("HYPOTHESIS_VALIDATOR_DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("HYPOTHESIS_VALIDATOR_DATABASE_URL environment variable not set.")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
