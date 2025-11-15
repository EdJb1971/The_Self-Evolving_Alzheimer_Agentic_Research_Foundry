import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os
import json

from src.backend.alznexus_llm_service.main import app, get_db, get_api_key
from src.backend.alznexus_llm_service.database import Base
from src.backend.alznexus_llm_service import models, schemas, crud

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
        "LLM_API_KEY": "test_api_key",
        "LLM_REDIS_URL": "redis://localhost:6379/1",
        "OPENAI_API_KEY": "mock-openai-key",
        "GEMINI_API_KEY": "mock-gemini-key"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_llm_service.tasks.call_llm_api")
@patch("src.backend.alznexus_llm_service.tasks.log_audit_event")
def test_llm_completion_success(mock_log_audit_event, mock_call_llm_api, client, db_session):
    # Mock LLM API response
    mock_response = MagicMock()
    mock_response.model_name = "gpt-4"
    mock_response.response_text = "This is a test response about Alzheimer's research."
    mock_response.usage_tokens = 150
    mock_response.confidence_score = 0.95
    mock_response.ethical_flags = []
    mock_response.injection_detected = False
    mock_call_llm_api.return_value = mock_response

    request_data = {
        "prompt": "Explain the role of beta-amyloid in Alzheimer's disease",
        "model": "gpt-4",
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = client.post(
        "/llm/completion",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "gpt-4"
    assert data["response_text"] == "This is a test response about Alzheimer's research."
    assert data["usage_tokens"] == 150
    assert data["confidence_score"] == 0.95

    # Verify database logging
    db_log = db_session.query(models.LLMRequestLog).first()
    assert db_log is not None
    assert db_log.model_name == "gpt-4"
    assert db_log.usage_tokens == 150

    mock_log_audit_event.assert_called()

@patch("src.backend.alznexus_llm_service.tasks.call_llm_api")
def test_llm_completion_ethical_filtering(mock_call_llm_api, client, db_session):
    # Mock LLM API response with ethical flags
    mock_response = MagicMock()
    mock_response.model_name = "gpt-4"
    mock_response.response_text = "Filtered response due to ethical concerns."
    mock_response.usage_tokens = 50
    mock_response.confidence_score = 0.8
    mock_response.ethical_flags = ["bias_detected"]
    mock_response.injection_detected = False
    mock_call_llm_api.return_value = mock_response

    request_data = {
        "prompt": "Biased prompt that might cause issues",
        "model": "gpt-4",
        "max_tokens": 200
    }

    response = client.post(
        "/llm/completion",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "bias_detected" in data["ethical_flags"]

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alznexus_llm_service"

@patch("src.backend.alznexus_llm_service.tasks.call_llm_api")
def test_llm_tool_use_completion(mock_call_llm_api, client, db_session):
    # Mock tool-use response
    mock_response = MagicMock()
    mock_response.model_name = "gpt-4"
    mock_response.response_text = "I'll analyze this data using the biomarker tool."
    mock_response.usage_tokens = 200
    mock_response.confidence_score = 0.9
    mock_response.ethical_flags = []
    mock_response.injection_detected = False
    mock_response.tool_calls = [
        {
            "tool_name": "analyze_biomarkers",
            "parameters": {"data": "sample_data"}
        }
    ]
    mock_call_llm_api.return_value = mock_response

    request_data = {
        "prompt": "Analyze these biomarker data",
        "model": "gpt-4",
        "tools": ["analyze_biomarkers"]
    }

    response = client.post(
        "/llm/tool-use-completion",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "tool_calls" in data
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool_name"] == "analyze_biomarkers"