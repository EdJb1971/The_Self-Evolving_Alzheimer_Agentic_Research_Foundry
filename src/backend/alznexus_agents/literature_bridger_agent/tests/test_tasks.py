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

# Import tasks after setting environment variables
from src.backend.alznexus_agents.literature_bridger_agent.tasks import bridge_literature_task, perform_reflection_task
from src.backend.alznexus_agents.literature_bridger_agent import models

# Mock the Celery app's bind context for tasks
class MockTask:
    def update_state(self, state, meta):
        pass

@pytest.fixture
def mock_db_session():
    with patch('src.backend.alznexus_agents.literature_bridger_agent.database.SessionLocal') as mock_session_local:
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
    with patch('src.backend.alznexus_agents.literature_bridger_agent.crud') as mock_crud_module:
        yield mock_crud_module

def test_bridge_literature_task_success(mock_db_session, mock_requests, mock_crud):
    mock_post, mock_get = mock_requests
    mock_task_instance = MockTask()

    # Mock crud.get_agent_task
    mock_agent_task = MagicMock(spec=models.AgentTask)
    mock_agent_task.id = 1
    mock_agent_task.agent_id = "test_agent"
    mock_agent_task.task_description = "Alzheimer's disease and gut microbiome"
    mock_agent_task.model_dump.return_value = {"id": 1, "agent_id": "test_agent", "task_description": "Alzheimer's disease and gut microbiome"}
    mock_crud.get_agent_task.return_value = mock_agent_task

    # Mock AD Workbench query initiation
    mock_post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"id": "query_123", "status": "PENDING"}), # AD Workbench query
        MagicMock(status_code=200, json=lambda: {"insight_id": "insight_456"}) # Publish insight
    ]

    # Mock AD Workbench query status/result
    mock_get.return_value = MagicMock(status_code=200, json=lambda: {
        "id": "query_123",
        "status": "COMPLETED",
        "result_data": json.dumps({"data": [{"title": "Paper A", "pmid": "12345"}, {"title": "Paper B", "pmid": "67890"}]})
    })

    # Execute the task
    result = bridge_literature_task(mock_task_instance, 1)

    # Assertions
    mock_crud.update_agent_task_status.assert_any_call(mock_db_session, 1, "IN_PROGRESS")
    mock_crud.update_agent_task_status.assert_any_call(mock_db_session, 1, "COMPLETED", {
        "status": "success",
        "agent_output": "Literature bridging completed for task 1.",
        "literature_connections": {
            "topic": "Alzheimer's disease and gut microbiome",
            "bridged_areas": [
                {"area_a": "Neuroinflammation", "area_b": "Gut Microbiome", "connection": "Emerging evidence links gut dysbiosis to neuroinflammatory pathways in AD.", "references": ["PMID:12345", "PMID:67890"]},
                {"area_a": "Amyloid Beta", "area_b": "Sleep Disorders", "connection": "Poor sleep quality accelerates amyloid-beta accumulation and impairs clearance.", "references": ["PMID:11223", "PMID:44556"]}
            ],
            "summary": "Synthesized connections between 2 literature entries."
        },
        "published_insight_id": "insight_456"
    })
    mock_crud.update_agent_state.assert_any_call(mock_db_session, "test_agent", current_task_id=1)
    mock_crud.update_agent_state.assert_any_call(mock_db_session, "test_agent", current_task_id=None)

    # Check audit logs (simplified check for calls, not full content)
    assert mock_post.call_count >= 2 # AD Workbench query + Publish insight
    assert mock_get.call_count >= 1 # AD Workbench query status

    assert result["status"] == "COMPLETED"
    assert result["agent_task_id"] == 1

def test_perform_reflection_task_success(mock_db_session, mock_requests, mock_crud):
    mock_post, mock_get = mock_requests
    mock_task_instance = MockTask()

    agent_id = "test_agent"
    reflection_metadata = {"reason": "periodic_check"}

    # Mock crud.get_recent_agent_tasks
    mock_task_completed = MagicMock(spec=models.AgentTask, status="COMPLETED")
    mock_task_failed = MagicMock(spec=models.AgentTask, status="FAILED")
    mock_crud.get_recent_agent_tasks.return_value = [mock_task_completed, mock_task_failed]

    # Mock datetime.utcnow()
    fixed_now = datetime.utcnow()
    with patch('src.backend.alznexus_agents.literature_bridger_agent.tasks.datetime') as mock_dt:
        mock_dt.utcnow.return_value = fixed_now
        mock_dt.fromisoformat = datetime.fromisoformat # Keep original for parsing audit history
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw) # Allow other datetime calls

        # Mock Audit Trail history
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {
            "history": [
                {"timestamp": (fixed_now - timedelta(days=1)).isoformat() + "Z", "event_type": "TASK_RECEIVED"},
                {"timestamp": (fixed_now - timedelta(days=2)).isoformat() + "Z", "event_type": "LITERATURE_BRIDGING_COMPLETED"}
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
                    "analysis_summary": "Agent test_agent reviewed 2 tasks in the last 7 days. Completed: 1, Failed: 1. Identified potential for improving literature search strategies and connection synthesis.",
                    "proposed_adjustments": [
                        "Refine keywords for literature search to improve relevance.",
                        "Explore new NLP models for extracting nuanced connections.",
                        "Increase cross-referencing with external knowledge graphs."
                    ],
                    "task_performance_summary": {"total_tasks": 2, "completed": 1, "failed": 1, "pending": 0},
                    "recent_audit_event_count": 2,
                    "original_reflection_metadata": {"reason": "periodic_check"}
                }
            }
        )
        mock_get.assert_called_once_with(
            f"{os.getenv('AUDIT_TRAIL_URL')}/audit/history/AGENT/{agent_id}",
            headers={"X-API-Key": os.getenv('AUDIT_API_KEY')}
        )
        assert result["status"] == "COMPLETED"
        assert result["agent_id"] == agent_id
