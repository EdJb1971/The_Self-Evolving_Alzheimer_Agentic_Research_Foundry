import React, { useState, useEffect } from 'react';
import {
  getOrchestratorStatus,
  getActiveTasks,
  getTaskDetails,
  resolveDebate,
  cancelTask,
  OrchestratorStatus,
  Task,
  DebateResolution
} from '../api/alznexusApi';
import { AxiosError } from 'axios';

function AdvancedOrchestratorControls() {
  const [status, setStatus] = useState<OrchestratorStatus | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debateResolution, setDebateResolution] = useState('');
  const [showDebateModal, setShowDebateModal] = useState(false);
  const [debateTaskId, setDebateTaskId] = useState<string>('');

  useEffect(() => {
    loadOrchestratorData();
    const interval = setInterval(loadOrchestratorData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadOrchestratorData = async () => {
    try {
      const [statusResponse, tasksResponse] = await Promise.all([
        getOrchestratorStatus(),
        getActiveTasks()
      ]);
      setStatus(statusResponse);
      setTasks(tasksResponse);
    } catch (err) {
      console.error('Failed to load orchestrator data:', err);
      setError('Failed to load orchestrator status');
    }
  };

  const viewTaskDetails = async (taskId: string) => {
    try {
      const taskDetails = await getTaskDetails(taskId);
      setSelectedTask(taskDetails);
      setShowTaskModal(true);
    } catch (err) {
      setError('Failed to load task details');
    }
  };

  const handleCancelTask = async (taskId: string) => {
    if (!confirm('Are you sure you want to cancel this task?')) return;

    try {
      await cancelTask(taskId);
      await loadOrchestratorData(); // Refresh data
    } catch (err) {
      setError('Failed to cancel task');
    }
  };

  const handleResolveDebate = async (taskId: string) => {
    setDebateTaskId(taskId);
    setShowDebateModal(true);
  };

  const submitDebateResolution = async () => {
    if (!debateResolution.trim()) return;

    setLoading(true);
    try {
      const resolution: DebateResolution = {
        task_id: debateTaskId,
        resolution: debateResolution,
        resolved_by: 'user'
      };

      await resolveDebate(resolution);
      setShowDebateModal(false);
      setDebateResolution('');
      setDebateTaskId('');
      await loadOrchestratorData(); // Refresh data
    } catch (err) {
      setError('Failed to resolve debate');
    } finally {
      setLoading(false);
    }
  };

  const getTaskStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'pending': return 'text-yellow-600 bg-yellow-100';
      case 'debate': return 'text-purple-600 bg-purple-100';
      case 'cancelled': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getAgentTypeColor = (agentType: string) => {
    const colors: { [key: string]: string } = {
      'biomarker_hunter': 'bg-blue-100 text-blue-800',
      'collaboration_matchmaker': 'bg-green-100 text-green-800',
      'data_harmonizer': 'bg-purple-100 text-purple-800',
      'drug_screener': 'bg-red-100 text-red-800',
      'hypothesis_validator': 'bg-yellow-100 text-yellow-800',
      'literature_bridger': 'bg-indigo-100 text-indigo-800',
      'pathway_modeler': 'bg-pink-100 text-pink-800',
      'trial_optimizer': 'bg-orange-100 text-orange-800'
    };
    return colors[agentType] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Advanced Orchestrator Controls</h2>
      <p className="text-gray-600 mb-6">
        Monitor and control the multi-agent orchestration system, resolve debates, and manage active tasks.
      </p>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <p className="text-red-800">Error: {error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-2 text-sm text-red-600 hover:text-red-800"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Orchestrator Status */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">System Status</h3>

          {status ? (
            <div className="bg-gray-50 p-4 rounded-lg space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  status.is_active ? 'text-green-600 bg-green-100' : 'text-red-600 bg-red-100'
                }`}>
                  {status.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Active Tasks</span>
                <span className="text-sm">{status.active_tasks_count}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Registered Agents</span>
                <span className="text-sm">{status.registered_agents_count}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Pending Debates</span>
                <span className="text-sm">{status.pending_debates_count}</span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Uptime</span>
                <span className="text-sm">{Math.floor(status.uptime_seconds / 3600)}h {Math.floor((status.uptime_seconds % 3600) / 60)}m</span>
              </div>

              <div className="pt-2 border-t border-gray-200">
                <div className="text-xs text-gray-500 mb-2">Agent Health</div>
                <div className="space-y-1">
                  {Object.entries(status.agent_health).map(([agent, health]) => (
                    <div key={agent} className="flex items-center justify-between text-xs">
                      <span className="capitalize">{agent.replace('_', ' ')}</span>
                      <span className={`px-1 py-0.5 rounded text-xs ${
                        health === 'healthy' ? 'text-green-600 bg-green-100' :
                        health === 'degraded' ? 'text-yellow-600 bg-yellow-100' :
                        'text-red-600 bg-red-100'
                      }`}>
                        {health}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-500">Loading orchestrator status...</p>
            </div>
          )}
        </div>

        {/* Active Tasks */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Active Tasks</h3>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {tasks.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                No active tasks.
              </p>
            ) : (
              tasks.map((task) => (
                <div
                  key={task.id}
                  className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">
                      Task {task.id.slice(-8)}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTaskStatusColor(task.status)}`}>
                        {task.status}
                      </span>
                      {task.status === 'debate' && (
                        <button
                          onClick={() => handleResolveDebate(task.id)}
                          className="text-xs bg-purple-600 text-white px-2 py-1 rounded hover:bg-purple-700"
                        >
                          Resolve
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="mb-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAgentTypeColor(task.agent_type)}`}>
                      {task.agent_type.replace('_', ' ')}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                    {task.description}
                  </p>

                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Started: {new Date(task.created_at).toLocaleTimeString()}</span>
                    <div className="space-x-2">
                      <button
                        onClick={() => viewTaskDetails(task.id)}
                        className="text-blue-600 hover:text-blue-800"
                      >
                        Details
                      </button>
                      {task.status === 'running' && (
                        <button
                          onClick={() => handleCancelTask(task.id)}
                          className="text-red-600 hover:text-red-800"
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Task Details Modal */}
      {showTaskModal && selectedTask && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Task Details</h3>
                <button
                  onClick={() => setShowTaskModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Task Header */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-500">Task ID</span>
                    <p className="text-sm font-mono">{selectedTask.id}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Status</span>
                    <p className={`text-sm font-medium ${getTaskStatusColor(selectedTask.status)}`}>
                      {selectedTask.status}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Agent Type</span>
                    <p className={`text-sm font-medium ${getAgentTypeColor(selectedTask.agent_type)}`}>
                      {selectedTask.agent_type.replace('_', ' ')}
                    </p>
                  </div>
                </div>

                {/* Description */}
                <div>
                  <h4 className="font-medium mb-2">Description</h4>
                  <p className="text-sm">{selectedTask.description}</p>
                </div>

                {/* Progress */}
                {selectedTask.progress !== undefined && (
                  <div>
                    <h4 className="font-medium mb-2">Progress</h4>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${selectedTask.progress}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{selectedTask.progress}% complete</p>
                  </div>
                )}

                {/* Logs */}
                {selectedTask.logs && selectedTask.logs.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Task Logs</h4>
                    <div className="bg-gray-50 p-4 rounded-lg max-h-64 overflow-y-auto">
                      <div className="space-y-1 text-sm font-mono">
                        {selectedTask.logs.map((log, index) => (
                          <div key={index} className="text-gray-700">
                            <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span> {log.message}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Results */}
                {selectedTask.result && (
                  <div>
                    <h4 className="font-medium mb-2">Results</h4>
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(selectedTask.result, null, 2)}</pre>
                    </div>
                  </div>
                )}

                {/* Error */}
                {selectedTask.error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h4 className="font-medium text-red-800 mb-2">Error</h4>
                    <p className="text-red-700 text-sm">{selectedTask.error}</p>
                  </div>
                )}

                {/* Timestamps */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-500">Created:</span> {new Date(selectedTask.created_at).toLocaleString()}
                  </div>
                  {selectedTask.updated_at && (
                    <div>
                      <span className="font-medium text-gray-500">Updated:</span> {new Date(selectedTask.updated_at).toLocaleString()}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Debate Resolution Modal */}
      {showDebateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Resolve Agent Debate</h3>
                <button
                  onClick={() => setShowDebateModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                <p className="text-gray-600">
                  Multiple agents have reached conflicting conclusions. Please provide a resolution or guidance to help the system proceed.
                </p>

                <textarea
                  value={debateResolution}
                  onChange={(e) => setDebateResolution(e.target.value)}
                  placeholder="Enter your resolution or guidance for the agents..."
                  rows={6}
                  className="input-field w-full"
                />

                <div className="flex space-x-3">
                  <button
                    onClick={submitDebateResolution}
                    className="btn-primary flex-1"
                    disabled={loading || !debateResolution.trim()}
                  >
                    {loading ? 'Submitting...' : 'Submit Resolution'}
                  </button>
                  <button
                    onClick={() => setShowDebateModal(false)}
                    className="btn-secondary"
                    disabled={loading}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdvancedOrchestratorControls;