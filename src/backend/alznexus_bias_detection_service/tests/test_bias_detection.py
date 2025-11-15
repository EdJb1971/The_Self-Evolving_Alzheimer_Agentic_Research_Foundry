import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os

from main import app, get_db, get_api_key
from database import Base
import models, schemas, crud

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
        "BIAS_DETECTION_API_KEY": "test_api_key",
        "BIAS_DETECTION_REDIS_URL": "redis://localhost:6379/1",
        "LLM_SERVICE_URL": "http://mock-llm-service",
        "LLM_API_KEY": "mock-llm-key"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_bias_detection_service.tasks.detect_bias_task.delay")
@patch("src.backend.alznexus_bias_detection_service.tasks.log_audit_event")
def test_detect_bias_success(mock_log_audit_event, mock_delay, client, db_session):
    request_data = {
        "content_type": "agent_reasoning",
        "content": "Analysis shows clear biomarker patterns in the data.",
        "context": {
            "agent_id": "biomarker_hunter_agent_001",
            "task_id": "task_123"
        }
    }

    response = client.post(
        "/bias/detect",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "PENDING"

    mock_delay.assert_called_once()

    # Verify database entry
    db_request = db_session.query(models.BiasDetectionRequest).first()
    assert db_request is not None
    assert db_request.content_type == "agent_reasoning"
    assert db_request.content == "Analysis shows clear biomarker patterns in the data."

@patch("src.backend.alznexus_bias_detection_service.tasks.log_audit_event")
def test_get_bias_report(mock_log_audit_event, client, db_session):
    # Create a test report
    report_data = schemas.BiasDetectionReportCreate(
        request_id=1,
        detected_biases=[
            {
                "bias_type": "confirmation_bias",
                "severity": "medium",
                "description": "Analysis may be influenced by prior assumptions",
                "suggested_corrections": ["Consider alternative hypotheses"]
            }
        ],
        overall_risk_level="low",
        recommendations=["Diversify data sources"]
    )

    db_report = crud.create_bias_detection_report(db_session, report_data)

    response = client.get(
        f"/bias/report/{db_report.id}",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == db_report.id
    assert data["overall_risk_level"] == "low"
    assert len(data["detected_biases"]) == 1
    assert data["detected_biases"][0]["bias_type"] == "confirmation_bias"

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alznexus_bias_detection_service"

@patch("src.backend.alznexus_bias_detection_service.tasks.detect_bias_task.delay")
def test_detect_bias_invalid_content_type(mock_delay, client, db_session):
    request_data = {
        "content_type": "invalid_type",
        "content": "Some content",
        "context": {}
    }

    response = client.post(
        "/bias/detect",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 422  # Validation error

    mock_delay.assert_not_called()