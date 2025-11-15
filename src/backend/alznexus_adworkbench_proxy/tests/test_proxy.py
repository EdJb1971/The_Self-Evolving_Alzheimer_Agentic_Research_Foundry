import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock
import os

from src.backend.alznexus_adworkbench_proxy.main import app, get_db, get_api_key
from src.backend.alznexus_adworkbench_proxy.database import Base
from src.backend.alznexus_adworkbench_proxy import models, schemas, crud

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
        "ADWORKBENCH_API_KEY": "test_api_key"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_adworkbench_proxy.tasks.execute_federated_query.delay")
def test_query_adworkbench_success(mock_delay, client, db_session):
    request_data = {
        "query": "SELECT * FROM alzheimer_biomarkers WHERE p_value < 0.05",
        "datasets": ["adni", "rosmap"],
        "privacy_level": "federated"
    }

    response = client.post(
        "/adworkbench/query",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["status"] == "PENDING"
    assert "Query submitted" in data["message"]

    mock_delay.assert_called_once_with(data["id"])

    # Verify database entry
    db_query = db_session.query(models.ADWorkbenchQuery).first()
    assert db_query is not None
    assert db_query.query == "SELECT * FROM alzheimer_biomarkers WHERE p_value < 0.05"
    assert db_query.status == "PENDING"

def test_scan_adworkbench_data(client):
    response = client.get(
        "/adworkbench/data/scan",
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "datasets_found" in data
    assert isinstance(data["datasets_found"], list)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "alznexus_adworkbench_proxy"

@patch("src.backend.alznexus_adworkbench_proxy.tasks.execute_federated_query.delay")
def test_query_adworkbench_validation_error(mock_delay, client, db_session):
    # Missing required fields
    request_data = {
        "query": "SELECT * FROM table"
        # missing datasets and privacy_level
    }

    response = client.post(
        "/adworkbench/query",
        json=request_data,
        headers={"X-API-Key": "test_api_key"}
    )

    assert response.status_code == 422  # Validation error
    mock_delay.assert_not_called()