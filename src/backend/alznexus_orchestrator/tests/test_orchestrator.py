import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.alznexus_orchestrator.main import app, get_db, get_api_key
from src.backend.alznexus_orchestrator.database import Base
from src.backend.alznexus_orchestrator import models, schemas, crud
from unittest.mock import patch, MagicMock
import os

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

    # Mock environment variables for API keys and Redis URL during tests
    with patch.dict(os.environ, {
        "ORCHESTRATOR_API_KEY": "test_api_key",
        "ADWORKBENCH_PROXY_URL": "http://mock-adworkbench",
        "ADWORKBENCH_API_KEY": "mock-ad-key",
        "AUDIT_TRAIL_URL": "http://mock-audit-trail",
        "AUDIT_API_KEY": "mock-audit-key",
        "ORCHESTRATOR_REDIS_URL": "redis://localhost:6379/1", # Use a different DB for tests
        "AGENT_SERVICE_BASE_URL": "http://mock-agent-service",
        "AGENT_API_KEY": "mock-agent-key"
    }):
        with TestClient(app) as c:
            yield c

@patch("src.backend.alznexus_orchestrator.tasks.log_audit_event")
def test_set_research_goal(mock_log_audit_event, client, db_session):
    goal_text = "Develop new Alzheimer's biomarkers"
    response = client.post(
        "/orchestrator/set-goal",
        json={"goal_text": goal_text},
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["goal_text"] == goal_text
    assert data["status"] == "ACTIVE"
    assert "id" in data

    db_goal = db_session.query(models.ResearchGoal).filter(models.ResearchGoal.id == data["id"]).first()
    assert db_goal is not None
    assert db_goal.goal_text == goal_text

    mock_log_audit_event.assert_called_once()
    assert mock_log_audit_event.call_args[1]["event_type"] == "GOAL_SET"

@patch("src.backend.alznexus_orchestrator.tasks.initiate_daily_scan_task.delay")
@patch("src.backend.alznexus_orchestrator.tasks.log_audit_event")
def test_initiate_daily_scan(mock_log_audit_event, mock_delay, client, db_session):
    # First, create an active research goal
    goal = schemas.ResearchGoalCreate(goal_text="Find new drug targets")
    crud.create_research_goal(db_session, goal)

    response = client.post(
        "/orchestrator/initiate-daily-scan",
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "PENDING"
    assert "orchestrator_task_id" in data

    mock_delay.assert_called_once_with(data["orchestrator_task_id"])

    db_task = db_session.query(models.OrchestratorTask).filter(models.OrchestratorTask.id == data["orchestrator_task_id"]).first()
    assert db_task is not None
    assert db_task.task_type == "DAILY_SCAN"
    assert db_task.status == "PENDING"
    assert db_task.goal_id == 1 # Assuming the first created goal gets ID 1 in sqlite

    # Ensure audit log was NOT called directly by the endpoint, but by the task (which is mocked)
    mock_log_audit_event.assert_not_called()

@patch("src.backend.alznexus_orchestrator.tasks.log_audit_event")
def test_initiate_daily_scan_no_active_goal(mock_log_audit_event, client, db_session):
    # No active goal created in this test
    response = client.post(
        "/orchestrator/initiate-daily-scan",
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "No active research goal found to initiate scan."} 
    mock_log_audit_event.assert_not_called()

@patch("src.backend.alznexus_orchestrator.tasks.coordinate_sub_agents_task.delay")
@patch("src.backend.alznexus_orchestrator.tasks.log_audit_event")
def test_coordinate_task(mock_log_audit_event, mock_delay, client, db_session):
    # First, create an active research goal
    goal = schemas.ResearchGoalCreate(goal_text="Optimize drug trial design")
    crud.create_research_goal(db_session, goal)

    task_payload = {
        "goal_id": 1, # Assuming goal ID 1
        "overall_description": "Coordinate biomarker and trial agents for drug optimization.",
        "sub_agent_tasks": [
            {
                "agent_id": "biomarker_hunter_agent_001",
                "task_description": "Identify key biomarkers for trial inclusion.",
                "task_metadata": {"priority": "high"}
            },
            {
                "agent_id": "trial_optimizer_agent_001",
                "task_description": "Suggest optimal trial parameters based on biomarkers.",
                "task_metadata": {"region": "EU"}
            }
        ],
        "coordination_metadata": {"workflow_id": "wf-001"}
    }

    response = client.post(
        "/orchestrator/coordinate-task",
        json=task_payload,
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["message"] == "Task coordination initiated."
    assert "orchestrator_task_id" in data

    db_task = db_session.query(models.OrchestratorTask).filter(models.OrchestratorTask.id == data["orchestrator_task_id"]).first()
    assert db_task is not None
    assert db_task.task_type == "COORDINATE_SUB_AGENT"
    assert db_task.description == task_payload["overall_description"]
    assert db_task.status == "PENDING"
    assert db_task.metadata_json == task_payload["coordination_metadata"]

    mock_delay.assert_called_once()
    # Verify arguments passed to delay, converting pydantic models to dicts
    called_args, called_kwargs = mock_delay.call_args
    assert called_args[0] == data["orchestrator_task_id"]
    assert len(called_args[1]) == 2
    assert called_args[1][0]["agent_id"] == "biomarker_hunter_agent_001"
    assert called_args[1][1]["agent_id"] == "trial_optimizer_agent_001"

    mock_log_audit_event.assert_called_once()
    assert mock_log_audit_event.call_args[1]["event_type"] == "TASK_COORDINATION_INITIATED"

@patch("src.backend.alznexus_orchestrator.tasks.resolve_debate_task.delay")
@patch("src.backend.alznexus_orchestrator.tasks.log_audit_event")
def test_resolve_debate(mock_log_audit_event, mock_delay, client, db_session):
    # First, create an active research goal
    goal = schemas.ResearchGoalCreate(goal_text="Resolve conflicting biomarker findings")
    crud.create_research_goal(db_session, goal)

    debate_payload = {
        "goal_id": 1, # Assuming goal ID 1
        "description": "Conflict on APOE4 interpretation between Biomarker Hunter and Pathway Modeler.",
        "conflicting_agents": ["biomarker_hunter_agent_001", "pathway_modeler_agent_001"],
        "points_of_contention": ["APOE4 impact on early AD", "Interaction with other genetic factors"],
        "evidence_summary": "Agent A highlights direct correlation, Agent B emphasizes indirect effects.",
        "debate_metadata": {"urgency": "high"}
    }

    response = client.post(
        "/orchestrator/resolve-debate",
        json=debate_payload,
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["message"] == "Debate resolution initiated."
    assert "orchestrator_task_id" in data

    db_task = db_session.query(models.OrchestratorTask).filter(models.OrchestratorTask.id == data["orchestrator_task_id"]).first()
    assert db_task is not None
    assert db_task.task_type == "RESOLVE_DEBATE"
    assert db_task.description == debate_payload["description"]
    assert db_task.status == "PENDING"
    assert db_task.metadata_json == debate_payload["debate_metadata"]

    mock_delay.assert_called_once()
    # Verify arguments passed to delay, converting pydantic models to dicts
    called_args, called_kwargs = mock_delay.call_args
    assert called_args[0] == data["orchestrator_task_id"]
    assert called_args[1] == debate_payload

    mock_log_audit_event.assert_called_once()
    assert mock_log_audit_event.call_args[1]["event_type"] == "DEBATE_RESOLUTION_INITIATED"
