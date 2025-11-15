import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json
import os

# Mock environment variables before importing tasks
os.environ["AUDIT_TRAIL_URL"] = "http://mock-audit-trail"
os.environ["AUDIT_API_KEY"] = "mock-audit-key"
os.environ["ADWORKBENCH_PROXY_URL"] = "http://mock-adworkbench"
os.environ["ADWORKBENCH_API_KEY"] = "mock-adworkbench-key"
os.environ["AGENT_REGISTRY_URL"] = "http://mock-agent-registry"
os.environ["AGENT_REGISTRY_API_KEY"] = "mock-registry-key"

# Import tasks after setting environment variables
from src.backend.alznexus_agents.collaboration_matchmaker_agent.tasks import match_collaboration_task, perform_reflection_task
from src.backend.alznexus_agents.collaboration_matchmaker_agent import models

# Mock the Celery app's bind context for tasks
class MockTask:
    def update_state(self, state, meta):
        pass

@pytest.fixture
def mock_db_session():
    with patch('src.backend.alznexus_agents.collaboration_matchmaker_agent.database.SessionLocal') as mock_session_local:
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        yield mock_db

@pytest.fixture
def mock_requests():
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        yield mock_post, mock_get

@pytest.fixture
def mock_crud():
    with patch('src.backend.alznexus_agents.collaboration_matchmaker_agent.crud') as mock_crud_module:
        yield mock_crud_module

def test_match_collaboration_task_success(mock_db_session, mock_requests, mock_crud):
    mock_post, mock_get = mock_requests
    mock_task_instance = MockTask()

    # Mock crud.get_agent_task
    mock_agent_task = MagicMock(spec=models.AgentTask)
    mock_agent_task.id = 1
    mock_agent_task.agent_id = "test_matchmaker_agent"
    mock_agent_task.task_description = "Complex problem in neurodegeneration"
    mock_agent_task.model_dump.return_value = {"id": 1, "agent_id": "test_matchmaker_agent", "task_description": "Complex problem in neurodegeneration"}
    mock_crud.get_agent_task.return_value = mock_agent_task

    # Mock AD Workbench query initiation
    mock_post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"id": "query_789", "status": "PENDING"}), # AD Workbench query
        MagicMock(status_code=200, json=lambda: {"insight_id": "insight_987"}) # Publish insight
    ]

    # Mock AD Workbench query status/result and Agent Registry query
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: { # AD Workbench query status
            "id": "query_789",
            "status": "COMPLETED",
            "result_data": json.dumps({"data": [{"problem_description": "Protein aggregation mechanisms"}]})
        }),
        MagicMock(status_code=200, json=lambda: { # Agent Registry query
            "agents": [
                {"agent_id": "biomarker_hunter_agent_001", "capabilities": {"domain": "biomarkers"}},
                {"agent_id": "pathway_modeler_agent_001", "capabilities": {"domain": "pathways"}}
            ]
        })
    ]

    # Execute the task
    result = match_collaboration_task(mock_task_instance, 1)

    # Assertions
    mock_crud.update_agent_task_status.assert_any_call(mock_db_session, 1, "IN_PROGRESS")
    mock_crud.update_agent_task_status.assert_any_call(mock_db_session, 1, "COMPLETED", {
        "status": "success",
        "agent_output": "Collaboration matchmaking completed for task 1.",
        "collaboration_suggestions": {
            "research_problem": "Protein aggregation mechanisms",
            "suggested_teams": [
                {"team_id": "TEAM-001", "agents": [{"agent_id": "biomarker_hunter_agent_001", "role": "Biomarker Identification"}, {"agent_id": "pathway_modeler_agent_001", "role": "Disease Pathway Modeling"}], "rationale": "Combines biomarker discovery with pathway analysis for comprehensive understanding."},
                {"team_id": "TEAM-002", "agents": [{"agent_id": "data_harmonizer_agent_001", "role": "Data Integration"}, {"agent_id": "literature_bridger_agent_001", "role": "Contextual Literature Review"}], "rationale": "Ensures data consistency and enriches findings with relevant scientific context."}
            ],
            "external_experts": [
                {"name": "Dr. Jane Doe", "expertise": "Neurogenetics", "affiliation": "University X"}
            ],
            "matchmaking_summary": "Identified optimal teams based on problem complexity and 2 available agents."
        },
        "published_insight_id": "insight_987"
    })
    mock_crud.update_agent_state.assert_any_call(mock_db_session, "test_matchmaker_agent", current_task_id=1)
    mock_crud.update_agent_state.assert_any_call(mock_db_session, "test_matchmaker_agent", current_task_id=None)

    # Check audit logs (simplified check for calls, not full content)
    assert mock_post.call_count >= 2 # AD Workbench query + Publish insight
    assert mock_get.call_count >= 2 # AD Workbench query status + Agent Registry query

    assert result["status"] == "COMPLETED"
    assert result["agent_task_id"] == 1

def test_perform_reflection_task_success(mock_db_session, mock_requests, mock_crud):
    mock_post, mock_get = mock_requests
    mock_task_instance = MockTask()

    agent_id = "test_matchmaker_agent"
    reflection_metadata = {"reason": "daily_review"}

    # Mock crud.get_recent_agent_tasks
    mock_task_completed = MagicMock(spec=models.AgentTask, status="COMPLETED")
    mock_task_failed = MagicMock(spec=models.AgentTask, status="FAILED")
    mock_crud.get_recent_agent_tasks.return_value = [mock_task_completed, mock_task_failed]

    # Mock datetime.utcnow()
    fixed_now = datetime.utcnow()
    with patch('src.backend.alznexus_agents.collaboration_matchmaker_agent.tasks.datetime') as mock_dt:
        mock_dt.utcnow.return_value = fixed_now
        mock_dt.fromisoformat = datetime.fromisoformat # Keep original for parsing audit history
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw) # Allow other datetime calls

        # Mock Audit Trail history
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {
            "history": [
                {"timestamp": (fixed_now - timedelta(days=1)).isoformat() + "Z", "event_type": "TASK_RECEIVED"},
                {"timestamp": (fixed_now - timedelta(days=2)).isoformat() + "Z", "event_type": "COLLABORATION_MATCHMAKING_COMPLETED"}
            ]
        })

        # Execute the task
        result = perform_reflection_task(mock_task_instance, agent_id, reflection_metadata)

        # Assertions
        mock_crud.update_agent_state.assert_any_call(
            mock_db_session, agent_id, last_reflection_at=fixed_now,
            metadata_json={"reflection_status": "IN_PROGRESS"}
        )
        mock_crud.update_agent_state.assert_any_call(
            mock_db_session, agent_id, metadata_json={
                "reflection_status": "COMPLETED",
                "last_reflection_result": {
                    "analysis_summary": "Agent test_matchmaker_agent reviewed 2 tasks in the last 7 days. Completed: 1, Failed: 1. Identified potential for improving agent discovery and collaboration rationale generation.",
                    "proposed_adjustments": [
                        "Enhance querying of Agent Registry for more granular capabilities.",
                        "Develop more sophisticated algorithms for matching agents to complex problems.",
                        "Improve rationale generation for suggested collaborations."
                    ],
                    "task_performance_summary": {"total_tasks": 2, "completed": 1, "failed": 1, "pending": 0},
                    "recent_audit_event_count": 2,
                    "original_reflection_metadata": {"reason": "daily_review"}
                }
            }
        )
        mock_get.assert_called_once_with(
            f"{os.getenv('AUDIT_TRAIL_URL')}/audit/history/AGENT/{agent_id}",
            headers={"X-API-Key": os.getenv('AUDIT_API_KEY')}
        )
        assert result["status"] == "COMPLETED"
        assert result["agent_id"] == agent_id
