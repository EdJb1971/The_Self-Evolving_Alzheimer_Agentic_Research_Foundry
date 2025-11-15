import React, { useState, useEffect, useCallback } from 'react';
import { getOrchestratorTaskStatus, OrchestratorTaskStatus } from '../api/alznexusApi';
import { useSearchParams } from 'react-router-dom';
import { AxiosError } from 'axios';

function TaskStatusDashboard() {
  const [searchParams] = useSearchParams();
  const urlTaskId = searchParams.get('taskId');
  const [taskId, setTaskId] = useState<string>(urlTaskId || '');
  const [taskStatus, setTaskStatus] = useState<OrchestratorTaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!taskId) {
      setTaskStatus(null);
      setError(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const status = await getOrchestratorTaskStatus(Number(taskId));
      setTaskStatus(status);
    } catch (err: unknown) {
      const axiosError = err as AxiosError;
      setError((axiosError.response?.data as any)?.detail || 'Failed to fetch task status.');
      setTaskStatus(null);
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    // Fetch status immediately if taskId is present (either from URL or user input)
    if (taskId) {
      fetchStatus();
    }
    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (taskId) { // Only poll if a taskId is set
        fetchStatus();
      }
    }, 5000); // Poll every 5 seconds
    return () => clearInterval(interval); // Clean up on unmount or taskId change
  }, [fetchStatus, taskId]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTaskId(e.target.value);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchStatus();
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Active Agent Tasks Dashboard (STORY-203)</h2>
      <form onSubmit={handleSearch} className="flex space-x-2 mb-4">
        <input
          type="number"
          className="input-field flex-grow"
          placeholder="Enter Orchestrator Task ID"
          value={taskId}
          onChange={handleInputChange}
          min="1"
        />
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Loading...' : 'Get Status'}
        </button>
      </form>
      {error && <p className="text-red-600 mt-4">Error: {error}</p>}
      {taskStatus && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <h3 className="text-lg font-medium text-blue-800">Task Details: ID {taskStatus.id}</h3>
          <p className="text-blue-700">
            <strong>Status:</strong> {taskStatus.status}
          </p>
          <p className="text-blue-700">
            <strong>Message:</strong> {taskStatus.message}
          </p>
          {taskStatus.result_data && (
            <div className="mt-4">
              <h4 className="text-md font-medium text-blue-800">Result Data:</h4>
              <pre className="bg-blue-100 p-2 rounded-md text-sm overflow-auto max-h-60">
                {JSON.stringify(taskStatus.result_data, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
      {!taskStatus && !error && !loading && taskId && (
        <p className="text-gray-600 mt-4">Enter a task ID to view its status.</p>
      )}
    </div>
  );
}

export default TaskStatusDashboard;