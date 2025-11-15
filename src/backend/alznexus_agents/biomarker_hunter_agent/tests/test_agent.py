import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os

from src.backend.alznexus_agents.biomarker_hunter_agent.main import app, get_db, get_api_key
from src.backend.alznexus_agents.biomarker_hunter_agent.database import Base
from src.backend.alznexus_agents.biomarker_hunter_agent import models, schemas, crud

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
        "AGENT_API_KEY": "test_api_key",
        "AGENT_REDIS_URL": "redis://localhost:6379/1",
        "ADWORKBENCH_PROXY_URL": "http://mock-adworkbench",
        "ADWORKBENCH_API_KEY": "mock-ad-key",
        "AGENT_REGISTRY_URL": "http://mock-registry",
        "AGENT_REGISTRY_API_KEY": "mock-reg-key",
        "AGENT_EXTERNAL_API_ENDPOINT": "http://test-agent:8000",
        "AUDIT_TRAIL_URL": "http://mock-audit",
        "AUDIT_API_KEY": "mock-audit-key"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_agents.biomarker_hunter_agent.tasks.execute_agent_task.delay")
@patch("src.backend.alznexus_agents.biomarker_hunter_agent.tasks.log_audit_event")
def test_execute_task_success(mock_log_audit_event, mock_delay, client, db_session):
    request_data = {
        "task_description": "Identify novel biomarkers for Alzheimer's disease",
        "parameters": {
            "data_sources": ["adni", "rosmap"],
            "analysis_type": "differential_expression"
        }
    }

    response = client.post(
        "/agent/execute-task",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"
    assert "Biomarker discovery task initiated" in data["message"]

    mock_delay.assert_called_once()

    # Verify database entry
    db_task = db_session.query(models.AgentTask).first()
    assert db_task is not None
    assert db_task.agent_id == "biomarker_hunter_agent_001"
    assert "novel biomarkers" in db_task.task_description.lower()

@patch("src.backend.alznexus_agents.biomarker_hunter_agent.tasks.perform_reflection_task.delay")
@patch("src.backend.alznexus_agents.biomarker_hunter_agent.tasks.log_audit_event")
def test_perform_reflection_success(mock_log_audit_event, mock_delay, client, db_session):
    request_data = {
        "reason": "weekly_performance_review",
        "context": {
            "time_period": "last_7_days"
        }
    }

    response = client.post(
        "/agent/perform-reflection",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "reflection initiated" in data["message"].lower()

    mock_delay.assert_called_once()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "biomarker_hunter_agent"

@patch("src.backend.alznexus_agents.biomarker_hunter_agent.tasks.execute_agent_task.delay")
def test_execute_task_invalid_agent_id(mock_delay, client, db_session):
    request_data = {
        "task_description": "Some task",
        "agent_id": "wrong_agent_id"  # This should be rejected
    }

    response = client.post(
        "/agent/execute-task",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 400
    data = response.json()
    assert "invalid agent_id" in data["detail"].lower()

    mock_delay.assert_not_called()