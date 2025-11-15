import React, { useState } from 'react';
import { getAuditHistory, AuditLogEntry } from '../api/alznexusApi';
import DOMPurify from 'dompurify';
import { AxiosError } from 'axios';

function AuditTrailViewer() {
  const [entityType, setEntityType] = useState<string>('');
  const [entityId, setEntityId] = useState<string>('');
  const [auditHistory, setAuditHistory] = useState<AuditLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!entityType.trim() || !entityId.trim()) {
      setError('Entity Type and Entity ID cannot be empty.');
      return;
    }
    setLoading(true);
    setError(null);
    setAuditHistory([]);
    try {
      const result = await getAuditHistory(entityType, entityId);
      setAuditHistory(result.history);
    } catch (err: unknown) {
      const axiosError = err as AxiosError;
      setError((axiosError.response?.data as any)?.detail || 'Failed to fetch audit history.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Audit Trail Viewer (STORY-204)</h2>
      <form onSubmit={handleSearch} className="space-y-4 mb-6">
        <div>
          <label htmlFor="entity-type" className="block text-sm font-medium text-gray-700 mb-1">
            Entity Type (e.g., ORCHESTRATOR, AGENT):
          </label>
          <input
            id="entity-type"
            type="text"
            className="input-field"
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
            placeholder="e.g., ORCHESTRATOR"
            required
          />
        </div>
        <div>
          <label htmlFor="entity-id" className="block text-sm font-medium text-gray-700 mb-1">
            Entity ID (e.g., 1, biomarker_hunter_agent_001-123):
          </label>
          <input
            id="entity-id"
            type="text"
            className="input-field"
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            placeholder="e.g., 1 or biomarker_hunter_agent_001-123"
            required
          />
        </div>
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Searching...' : 'Search Audit History'}
        </button>
      </form>
      {error && <p className="text-red-600 mt-4">Error: {error}</p>}
      {auditHistory.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-medium mb-3">History for {entityType}:{entityId}</h3>
          <div className="space-y-4">
            {auditHistory.map((entry) => (
              <div key={entry.id} className="p-4 border border-gray-200 rounded-md bg-gray-50">
                <p className="text-sm text-gray-500">
                  <strong>Timestamp:</strong> {new Date(entry.timestamp).toLocaleString()}
                </p>
                <p className="text-lg font-medium text-gray-800">
                  <strong>Event:</strong> <span dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(entry.event_type) }} />
                </p>
                <p className="text-gray-700">
                  <strong>Description:</strong> <span dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(entry.description) }} />
                </p>
                {entry.metadata_json && Object.keys(entry.metadata_json).length > 0 && (
                  <div className="mt-2">
                    <h4 className="text-md font-medium text-gray-700">Metadata:</h4>
                    <pre className="bg-gray-100 p-2 rounded-md text-sm overflow-auto max-h-40">
                      {JSON.stringify(entry.metadata_json, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      {!auditHistory.length && !error && !loading && entityType && entityId && (
        <p className="text-gray-600 mt-4">No audit history found for {entityType}:{entityId}.</p>
      )}
      {!entityType && !entityId && !error && !loading && (
        <p className="text-gray-600 mt-4">Enter an Entity Type and Entity ID to view audit history.</p>
      )}
    </div>
  );
}

export default AuditTrailViewer;