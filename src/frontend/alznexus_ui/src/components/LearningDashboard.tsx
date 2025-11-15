import { useState, useEffect } from 'react';
import {
  getLearningMetrics,
  getAgentPerformance,
  getLearningPatterns,
  getKnowledgeGrowth,
  getSelfEvolutionStatus,
  getPredictivePerformance,
  getEvolutionTrajectory,
  LearningMetrics,
  AgentPerformance,
  LearningPattern,
  KnowledgeGrowth,
  SelfEvolutionStatus,
  PredictivePerformance,
  EvolutionTrajectory
} from '../api/alznexusApi';
import { AxiosError } from 'axios';

function LearningDashboard() {
  const [learningMetrics, setLearningMetrics] = useState<LearningMetrics | null>(null);
  const [agentPerformance, setAgentPerformance] = useState<AgentPerformance[]>([]);
  const [learningPatterns, setLearningPatterns] = useState<LearningPattern[]>([]);
  const [knowledgeGrowth, setKnowledgeGrowth] = useState<KnowledgeGrowth | null>(null);
  const [evolutionStatus, setEvolutionStatus] = useState<SelfEvolutionStatus | null>(null);
  const [predictivePerformance, setPredictivePerformance] = useState<PredictivePerformance[]>([]);
  const [evolutionTrajectory, setEvolutionTrajectory] = useState<EvolutionTrajectory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadLearningData = async () => {
      try {
        setLoading(true);
        const [
          metricsData,
          performanceData,
          patternsData,
          growthData,
          evolutionData,
          trajectoryData
        ] = await Promise.all([
          getLearningMetrics(),
          getAgentPerformance(),
          getLearningPatterns(),
          getKnowledgeGrowth(),
          getSelfEvolutionStatus(),
          getEvolutionTrajectory()
        ]);

        setLearningMetrics(metricsData);
        setAgentPerformance(Array.isArray(performanceData) ? performanceData : []);
        setLearningPatterns(Array.isArray(patternsData) ? patternsData : []);
        setKnowledgeGrowth(growthData);
        setEvolutionStatus(evolutionData);
        setEvolutionTrajectory(trajectoryData);

        // Load predictive performance for each agent
        if (Array.isArray(performanceData) && performanceData.length > 0) {
          const predictiveData = await Promise.all(
            performanceData.slice(0, 3).map(agent => getPredictivePerformance(agent.agent_id))
          );
          setPredictivePerformance(predictiveData.filter(data => data !== undefined));
        }

      } catch (err) {
        setError((err as AxiosError).message || 'Failed to load learning data');
      } finally {
        setLoading(false);
      }
    };

    loadLearningData();
    // Refresh every 60 seconds for learning metrics
    const interval = setInterval(loadLearningData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Learning Data Error</h3>
              <div className="mt-2 text-sm text-red-700">{error}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Self-Evolution Status */}
      {evolutionStatus && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Self-Evolution Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-800">
                {evolutionStatus.learning_effectiveness.toFixed(2)}%
              </div>
              <div className="text-sm text-green-600">Learning Effectiveness</div>
            </div>
            <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-800">
                {evolutionStatus.adaptation_rate.toFixed(2)}%
              </div>
              <div className="text-sm text-blue-600">Adaptation Rate</div>
            </div>
            <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-800">
                {evolutionStatus.knowledge_utilization.toFixed(2)}%
              </div>
              <div className="text-sm text-purple-600">Knowledge Utilization</div>
            </div>
          </div>
          <div className="mt-4">
            <div className="text-sm text-gray-600">
              Current Phase: <span className="font-medium">{evolutionStatus.evolution_phase}</span>
            </div>
            <div className="text-sm text-gray-600">
              Last Evolution Cycle: {new Date(evolutionStatus.last_evolution_cycle).toLocaleString()}
            </div>
          </div>
        </div>
      )}

      {/* Learning Metrics */}
      {learningMetrics && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Learning Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-indigo-600">
                {learningMetrics.total_patterns_extracted}
              </div>
              <div className="text-sm text-gray-600">Patterns Extracted</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {learningMetrics.total_context_enrichments}
              </div>
              <div className="text-sm text-gray-600">Context Enrichments</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {learningMetrics.average_confidence_score.toFixed(2)}%
              </div>
              <div className="text-sm text-gray-600">Avg Confidence</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {learningMetrics.knowledge_growth_rate.toFixed(2)}%
              </div>
              <div className="text-sm text-gray-600">Growth Rate</div>
            </div>
          </div>
        </div>
      )}

      {/* Agent Performance */}
      {agentPerformance.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Performance</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Agent ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Success Rate
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Execution Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Learning Trend
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {agentPerformance.map((agent) => (
                  <tr key={agent.agent_id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {agent.agent_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {(agent.success_rate * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {agent.average_execution_time.toFixed(2)}s
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        agent.learning_trend === 'improving'
                          ? 'bg-green-100 text-green-800'
                          : agent.learning_trend === 'stable'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {agent.learning_trend}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Knowledge Growth */}
      {knowledgeGrowth && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Knowledge Base Growth</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-indigo-600">
                {knowledgeGrowth.total_documents}
              </div>
              <div className="text-sm text-gray-600">Total Documents</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {knowledgeGrowth.total_chunks}
              </div>
              <div className="text-sm text-gray-600">Total Chunks</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {knowledgeGrowth.growth_rate_per_day.toFixed(2)}/day
              </div>
              <div className="text-sm text-gray-600">Growth Rate</div>
            </div>
          </div>
        </div>
      )}

      {/* Learning Patterns */}
      {learningPatterns.length > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Discovered Learning Patterns</h3>
          <div className="space-y-4">
            {learningPatterns.slice(0, 5).map((pattern) => (
              <div key={pattern.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">{pattern.pattern_type}</h4>
                    <p className="text-sm text-gray-600 mt-1">{pattern.pattern_description}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-green-600">
                      {(pattern.success_rate * 100).toFixed(1)}% success
                    </div>
                    <div className="text-xs text-gray-500">
                      Used {pattern.application_count} times
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Predictive Performance and Evolution Trajectory */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Predictive Performance */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Predictive Performance</h3>
          {predictivePerformance.length > 0 ? (
            <div className="space-y-4">
              {predictivePerformance.map((prediction, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{prediction.agent_name}</span>
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      prediction.confidence > 0.8 ? 'bg-green-100 text-green-800' :
                      prediction.confidence > 0.6 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {Math.round(prediction.confidence * 100)}% confidence
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Predicted Success:</span>
                      <span className="ml-2 font-medium">{Math.round(prediction.predicted_success_rate * 100)}%</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Improvement:</span>
                      <span className="ml-2 font-medium text-green-600">
                        +{Math.round(prediction.expected_improvement * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No predictive data available</p>
          )}
        </div>

        {/* Evolution Trajectory */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Evolution Trajectory</h3>
          {evolutionTrajectory ? (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Current Phase:</span>
                <span className="font-medium">{evolutionTrajectory.current_phase}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Progress:</span>
                <span className="font-medium">{Math.round(evolutionTrajectory.progress_percentage)}%</span>
              </div>
              <div className={`w-full bg-gray-200 rounded-full h-2`}>
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${evolutionTrajectory.progress_percentage}%` }}
                ></div>
              </div>
              <div className="text-sm text-gray-600">
                <p>Next milestone: {evolutionTrajectory.next_milestone}</p>
                <p>Estimated completion: {new Date(evolutionTrajectory.estimated_completion).toLocaleDateString()}</p>
              </div>
            </div>
          ) : (
            <p className="text-gray-500">No trajectory data available</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default LearningDashboard;