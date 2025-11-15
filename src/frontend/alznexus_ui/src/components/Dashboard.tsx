import React, { useState, useEffect } from 'react';
import { getSystemHealth, HealthStatus, getRegisteredAgents, AgentListResponse } from '../api/alznexusApi';
import { AxiosError } from 'axios';

function Dashboard() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus[]>([]);
  const [agents, setAgents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);
        const [healthData, agentsData] = await Promise.all([
          getSystemHealth(),
          getRegisteredAgents()
        ]);
        setHealthStatus(healthData);
        setAgents(agentsData.agents || []);
      } catch (err) {
        setError((err as AxiosError).message || 'Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
    // Refresh every 30 seconds
    const interval = setInterval(loadDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const healthyServices = healthStatus.filter(s => s.status === 'healthy').length;
  const totalServices = healthStatus.length;

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
      {/* System Overview */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">AlzNexus System Overview</h2>

        {/* Health Status Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-600">{healthyServices}</div>
            <div className="text-sm text-green-800">Healthy Services</div>
          </div>
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <div className="text-2xl font-bold text-blue-600">{agents.length}</div>
            <div className="text-sm text-blue-800">Active Agents</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <div className="text-2xl font-bold text-purple-600">{totalServices}</div>
            <div className="text-sm text-purple-800">Total Services</div>
          </div>
        </div>

        {/* Service Health Details */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-3">Service Health Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {healthStatus.map((service) => (
              <div
                key={service.service}
                className={`p-3 rounded-lg border ${
                  service.status === 'healthy'
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium capitalize">{service.service.replace('_', ' ')}</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    service.status === 'healthy'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {service.status}
                  </span>
                </div>
                {service.error && (
                  <p className="text-xs text-red-600 mt-1">{service.error}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Registered Agents */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Registered Agents</h2>
        {agents.length === 0 ? (
          <p className="text-gray-600">No agents registered yet.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent) => (
              <div key={agent.agent_id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-medium text-lg">{agent.agent_id.replace('_', ' ')}</h3>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    agent.status === 'active'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {agent.status}
                  </span>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Capabilities:</strong> {Object.keys(agent.capabilities || {}).join(', ')}
                </div>
                <div className="text-xs text-gray-500">
                  Registered: {new Date(agent.registered_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-2xl font-semibold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <a href="/query-submission" className="btn-primary text-center">
            Submit Research Query
          </a>
          <a href="/task-status" className="btn-secondary text-center">
            Check Task Status
          </a>
          <a href="/audit-trail" className="btn-secondary text-center">
            View Audit Trail
          </a>
          <a href="/agents" className="btn-secondary text-center">
            Manage Agents
          </a>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">Error loading dashboard: {error}</p>
        </div>
      )}
    </div>
  );
}

export default Dashboard;