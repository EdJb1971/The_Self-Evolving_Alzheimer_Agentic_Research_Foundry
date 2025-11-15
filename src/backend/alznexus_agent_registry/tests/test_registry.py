import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os

from src.backend.alznexus_agent_registry.main import app, get_db, get_api_key
from src.backend.alznexus_agent_registry.database import Base
from src.backend.alznexus_agent_registry import models, schemas, crud

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
        "REGISTRY_API_KEY": "test_api_key",
        "REGISTRY_REDIS_URL": "redis://localhost:6379/1"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_agent_registry.tasks.log_audit_event")
def test_register_agent_success(mock_log_audit_event, client, db_session):
    request_data = {
        "agent_id": "test_agent_001",
        "capabilities": {
            "domain": "biomarkers",
            "functions": ["discovery", "validation"]
        },
        "api_endpoint": "http://test-agent:8000",
        "status": "active"
    }

    response = client.post(
        "/registry/register",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["agent_id"] == "test_agent_001"
    assert data["capabilities"]["domain"] == "biomarkers"
    assert data["status"] == "active"

    # Verify database entry
    db_agent = db_session.query(models.Agent).filter(models.Agent.agent_id == "test_agent_001").first()
    assert db_agent is not None
    assert db_agent.api_endpoint == "http://test-agent:8000"

    mock_log_audit_event.assert_called()

def test_get_agent_details_success(client, db_session):
    # First register an agent
    agent_data = schemas.AgentRegister(
        agent_id="test_agent_002",
        capabilities={"domain": "drugs"},
        api_endpoint="http://test-agent-2:8000",
        status="active"
    )
    crud.register_agent(db_session, agent_data)

    response = client.get(
        "/registry/agents/test_agent_002",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == "test_agent_002"
    assert data["capabilities"]["domain"] == "drugs"

def test_get_agent_details_not_found(client):
    response = client.get(
        "/registry/agents/nonexistent_agent",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alznexus_agent_registry"

@patch("src.backend.alznexus_agent_registry.tasks.log_audit_event")
def test_register_agent_update_existing(mock_log_audit_event, client, db_session):
    # Register agent first time
    request_data = {
        "agent_id": "update_agent_001",
        "capabilities": {"domain": "initial"},
        "api_endpoint": "http://update-agent:8000",
        "status": "active"
    }

    client.post(
        "/registry/register",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    # Update the same agent
    update_data = {
        "agent_id": "update_agent_001",
        "capabilities": {"domain": "updated"},
        "api_endpoint": "http://update-agent:8000",
        "status": "active"
    }

    response = client.post(
        "/registry/register",
        json=update_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["capabilities"]["domain"] == "updated"

    # Verify only one record exists
    agents = db_session.query(models.Agent).filter(models.Agent.agent_id == "update_agent_001").all()
    assert len(agents) == 1