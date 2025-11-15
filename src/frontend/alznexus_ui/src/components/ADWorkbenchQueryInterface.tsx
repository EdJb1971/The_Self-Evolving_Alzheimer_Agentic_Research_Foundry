import React, { useState, useEffect } from 'react';
import {
  submitADWorkbenchQuery,
  getADWorkbenchQueries,
  getADWorkbenchQuery,
  ADWorkbenchQueryRequest,
  ADWorkbenchQuery,
  ADWorkbenchResult
} from '../api/alznexusApi';
import { AxiosError } from 'axios';

function ADWorkbenchQueryInterface() {
  const [query, setQuery] = useState('');
  const [queryType, setQueryType] = useState<'federated' | 'direct' | 'meta_analysis'>('federated');
  const [dataSources, setDataSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queries, setQueries] = useState<ADWorkbenchQuery[]>([]);
  const [selectedQuery, setSelectedQuery] = useState<ADWorkbenchQuery | null>(null);
  const [showResultsModal, setShowResultsModal] = useState(false);

  const availableDataSources = [
    'ADNI', 'NACC', 'ROSMAP', 'Mayo Clinic', 'WashU', 'Emory', 'Mount Sinai'
  ];

  useEffect(() => {
    loadQueries();
  }, []);

  const loadQueries = async () => {
    try {
      const response = await getADWorkbenchQueries();
      setQueries(response);
    } catch (err) {
      console.error('Failed to load queries:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const request: ADWorkbenchQueryRequest = {
        query,
        query_type: queryType,
        data_sources: dataSources.length > 0 ? dataSources : undefined,
        include_metadata: true
      };

      const response: ADWorkbenchResult = await submitADWorkbenchQuery(request);

      // Add the new query to the list
      setQueries(prev => [response.query, ...prev]);

      // Clear form
      setQuery('');
      setDataSources([]);
    } catch (err) {
      setError((err as AxiosError).message || 'Failed to submit AD Workbench query');
    } finally {
      setLoading(false);
    }
  };

  const viewResults = async (queryId: string) => {
    try {
      const queryDetails = await getADWorkbenchQuery(queryId);
      setSelectedQuery(queryDetails);
      setShowResultsModal(true);
    } catch (err) {
      setError('Failed to load query results');
    }
  };

  const toggleDataSource = (source: string) => {
    setDataSources(prev =>
      prev.includes(source)
        ? prev.filter(s => s !== source)
        : [...prev, source]
    );
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'running': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'pending': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">AD Workbench Query Interface</h2>
      <p className="text-gray-600 mb-6">
        Submit federated queries across Alzheimer's disease research databases and datasets.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Query Form */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Submit New Query</h3>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Query Type
              </label>
              <select
                value={queryType}
                onChange={(e) => setQueryType(e.target.value as 'federated' | 'direct' | 'meta_analysis')}
                className="input-field"
              >
                <option value="federated">Federated Query</option>
                <option value="direct">Direct Query</option>
                <option value="meta_analysis">Meta-Analysis</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Data Sources (Optional)
              </label>
              <div className="grid grid-cols-2 gap-2">
                {availableDataSources.map((source) => (
                  <label key={source} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={dataSources.includes(source)}
                      onChange={() => toggleDataSource(source)}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm">{source}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your AD research query (SQL, SPARQL, or natural language)..."
                rows={6}
                className="input-field"
                required
              />
            </div>

            <button
              type="submit"
              className="btn-primary w-full"
              disabled={loading || !query.trim()}
            >
              {loading ? 'Submitting Query...' : 'Submit Query'}
            </button>
          </form>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Recent Queries */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Recent Queries</h3>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {queries.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                No queries submitted yet.
              </p>
            ) : (
              queries.map((q) => (
                <div
                  key={q.id}
                  className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                  onClick={() => viewResults(q.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">
                      {q.query_type.charAt(0).toUpperCase() + q.query_type.slice(1)} Query
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(q.status)}`}>
                      {q.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                    {q.query_text}
                  </p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{q.data_sources?.join(', ') || 'All sources'}</span>
                    <span>{new Date(q.submitted_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Results Modal */}
      {showResultsModal && selectedQuery && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Query Results</h3>
                <button
                  onClick={() => setShowResultsModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Query Details */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-500">Query Type</span>
                    <p className="text-sm">{selectedQuery.query_type}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Status</span>
                    <p className={`text-sm font-medium ${getStatusColor(selectedQuery.status)}`}>
                      {selectedQuery.status}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Submitted</span>
                    <p className="text-sm">{new Date(selectedQuery.submitted_at).toLocaleString()}</p>
                  </div>
                </div>

                {/* Original Query */}
                <div>
                  <h4 className="font-medium mb-2">Query</h4>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm font-mono">{selectedQuery.query_text}</p>
                  </div>
                </div>

                {/* Data Sources */}
                {selectedQuery.data_sources && selectedQuery.data_sources.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Data Sources</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedQuery.data_sources.map((source, index) => (
                        <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                          {source}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Results */}
                {selectedQuery.results && selectedQuery.results.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Results ({selectedQuery.results.length} records)</h4>
                    <div className="bg-gray-50 p-4 rounded-lg max-h-96 overflow-y-auto">
                      <div className="overflow-x-auto">
                        <table className="min-w-full text-sm">
                          <thead>
                            <tr className="border-b border-gray-200">
                              {Object.keys(selectedQuery.results[0]).map((key) => (
                                <th key={key} className="text-left py-2 px-3 font-medium text-gray-700">
                                  {key}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {selectedQuery.results.slice(0, 100).map((row, index) => (
                              <tr key={index} className="border-b border-gray-100">
                                {Object.values(row).map((value, cellIndex) => (
                                  <td key={cellIndex} className="py-2 px-3">
                                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {selectedQuery.results.length > 100 && (
                          <p className="text-xs text-gray-500 mt-2">
                            Showing first 100 records of {selectedQuery.results.length} total
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Metadata */}
                {selectedQuery.metadata && (
                  <div>
                    <h4 className="font-medium mb-2">Query Metadata</h4>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Execution Time:</span> {selectedQuery.metadata.execution_time_ms}ms
                        </div>
                        <div>
                          <span className="font-medium">Records Processed:</span> {selectedQuery.metadata.records_processed}
                        </div>
                        <div>
                          <span className="font-medium">Data Sources Queried:</span> {selectedQuery.metadata.data_sources_queried}
                        </div>
                        <div>
                          <span className="font-medium">Federation Protocol:</span> {selectedQuery.metadata.federation_protocol}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Error Message */}
                {selectedQuery.error_message && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <h4 className="font-medium text-red-800 mb-2">Error</h4>
                    <p className="text-red-700 text-sm">{selectedQuery.error_message}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ADWorkbenchQueryInterface;