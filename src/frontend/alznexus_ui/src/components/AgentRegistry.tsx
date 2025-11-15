import { useState, useEffect } from 'react';
import { getRegisteredAgents, getAgentDetails, AgentDetails } from '../api/alznexusApi';
import { AxiosError } from 'axios';

function AgentRegistry() {
  const [agents, setAgents] = useState<AgentDetails[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAgents();
  }, []);

  const loadAgents = async () => {
    try {
      setLoading(true);
      const response = await getRegisteredAgents();
      setAgents(response.agents || []);
    } catch (err) {
      setError((err as AxiosError).message || 'Failed to load agents');
    } finally {
      setLoading(false);
    }
  };

  const viewAgentDetails = async (agentId: string) => {
    try {
      const agent = await getAgentDetails(agentId);
      setSelectedAgent(agent);
    } catch (err) {
      setError((err as AxiosError).message || 'Failed to load agent details');
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Agent Registry Management</h2>
        <p className="text-gray-600 mb-6">
          View and manage registered sub-agents in the AlzNexus system.
        </p>

        {agents.length === 0 ? (
          <p className="text-gray-600">No agents registered yet.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent) => (
              <div key={agent.agent_id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium text-lg">{agent.agent_id.replace(/_/g, ' ')}</h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    agent.status === 'active'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {agent.status}
                  </span>
                </div>

                <div className="mb-3">
                  <h4 className="font-medium text-sm text-gray-700 mb-1">Capabilities:</h4>
                  <div className="flex flex-wrap gap-1">
                    {Object.keys(agent.capabilities || {}).map((cap) => (
                      <span key={cap} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="text-xs text-gray-500 mb-3">
                  API: {agent.api_endpoint}
                </div>

                <button
                  onClick={() => viewAgentDetails(agent.agent_id)}
                  className="btn-secondary text-sm w-full"
                >
                  View Details
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Agent Details Modal */}
      {selectedAgent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">{selectedAgent.agent_id.replace(/_/g, ' ')}</h3>
                <button
                  onClick={() => setSelectedAgent(null)}
                  className="text-gray-400 hover:text-gray-600"
                  aria-label="Close agent details"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <span className="font-medium">Status:</span>
                  <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                    selectedAgent.status === 'active'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {selectedAgent.status}
                  </span>
                </div>

                <div>
                  <span className="font-medium">API Endpoint:</span>
                  <code className="ml-2 bg-gray-100 px-2 py-1 rounded text-sm">
                    {selectedAgent.api_endpoint}
                  </code>
                </div>

                <div>
                  <span className="font-medium">Registered:</span>
                  <span className="ml-2 text-gray-600">
                    {new Date(selectedAgent.registered_at).toLocaleString()}
                  </span>
                </div>

                <div>
                  <span className="font-medium block mb-2">Capabilities:</span>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <pre className="text-sm text-gray-700 whitespace-pre-wrap">
                      {JSON.stringify(selectedAgent.capabilities, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">Error: {error}</p>
        </div>
      )}
    </div>
  );
}

export default AgentRegistry;