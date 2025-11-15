import React, { useState } from 'react';
import { submitResearchGoal, ResearchGoal } from '../api/alznexusApi';
import { AxiosError } from 'axios';

function QuerySubmission() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<ResearchGoal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Research query cannot be empty.');
      return;
    }
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const result = await submitResearchGoal(query);
      setResponse(result);
      setQuery('');
    } catch (err: unknown) {
      const axiosError = err as AxiosError;
      setError((axiosError.response?.data as any)?.detail || 'Failed to submit query.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Submit New Research Query (STORY-201)</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="research-query" className="block text-sm font-medium text-gray-700 mb-1">
            Your Research Query:
          </label>
          <textarea
            id="research-query"
            className="input-field h-32"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Identify novel biomarkers for early Alzheimer's disease progression."
            required
          />
        </div>
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Submitting...' : 'Submit Query'}
        </button>
      </form>
      {error && <p className="text-red-600 mt-4">Error: {error}</p>}
      {response && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-md">
          <h3 className="text-lg font-medium text-green-800">Query Submitted Successfully!</h3>
          <p className="text-green-700">
            <strong>Goal ID:</strong> {response.id}
          </p>
          <p className="text-green-700">
            <strong>Status:</strong> {response.status}
          </p>
          <p className="text-green-700">
            <strong>Query:</strong> {response.goal_text}
          </p>
          <p className="text-sm text-green-600 mt-2">
            You can monitor its progress using the Task Status Dashboard.
          </p>
        </div>
      )}
    </div>
  );
}

export default QuerySubmission;