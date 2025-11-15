import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os

from src.backend.alznexus_audit_trail.main import app, get_db, get_api_key
from src.backend.alznexus_audit_trail.database import Base
from src.backend.alznexus_audit_trail import models, schemas, crud

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(name="db_session")
def db_session_fixture():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(name="client")
def client_fixture(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"

    # Mock environment variables
    with patch.dict(os.environ, {
        "AUDIT_API_KEY": "test_api_key",
        "AUDIT_REDIS_URL": "redis://localhost:6379/1"
    }):
        with TestClient(app) as c:
            yield c

def test_log_event_success(client, db_session):
    request_data = {
        "entity_type": "AGENT",
        "entity_id": "biomarker_hunter_001",
        "event_type": "TASK_COMPLETED",
        "description": "Biomarker discovery task finished successfully",
        "metadata": {
            "task_id": "task_123",
            "biomarkers_found": 5
        }
    }

    response = client.post(
        "/audit/log",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["entity_type"] == "AGENT"
    assert data["event_type"] == "TASK_COMPLETED"
    assert "biomarkers_found" in data["metadata"]

    # Verify database entry
    db_log = db_session.query(models.AuditLogEntry).first()
    assert db_log is not None
    assert db_log.entity_type == "AGENT"
    assert db_log.event_type == "TASK_COMPLETED"

def test_get_audit_history_success(client, db_session):
    # Create some test audit entries
    entries = [
        schemas.AuditLogCreate(
            entity_type="AGENT",
            entity_id="test_agent",
            event_type="TASK_STARTED",
            description="Task started",
            metadata={"task_id": "task_1"}
        ),
        schemas.AuditLogCreate(
            entity_type="AGENT",
            entity_id="test_agent",
            event_type="TASK_COMPLETED",
            description="Task completed",
            metadata={"task_id": "task_1", "result": "success"}
        )
    ]

    for entry in entries:
        crud.create_audit_log_entry(db_session, entry)

    response = client.get(
        "/audit/history/AGENT/test_agent",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["entity_type"] == "AGENT"
    assert data["entity_id"] == "test_agent"
    assert len(data["history"]) == 2
    assert data["history"][0]["event_type"] == "TASK_STARTED"
    assert data["history"][1]["event_type"] == "TASK_COMPLETED"

def test_get_audit_history_not_found(client):
    response = client.get(
        "/audit/history/AGENT/nonexistent",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 404
    data = response.json()
    assert "no audit history found" in data["detail"].lower()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alznexus_audit_trail"

def test_log_event_validation_error(client):
    # Missing required fields
    request_data = {
        "entity_type": "AGENT",
        # missing entity_id, event_type, description
    }

    response = client.post(
        "/audit/log",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 422  # Validation error