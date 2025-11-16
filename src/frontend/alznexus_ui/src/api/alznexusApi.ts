import axios, { AxiosError } from 'axios';

// Environment variables for service URLs and API keys
const ORCHESTRATOR_BASE_URL = import.meta.env.VITE_ORCHESTRATOR_BASE_URL || 'http://localhost:8001';
const AUDIT_TRAIL_BASE_URL = import.meta.env.VITE_AUDIT_TRAIL_BASE_URL || 'http://localhost:8003';
const AGENT_REGISTRY_BASE_URL = import.meta.env.VITE_AGENT_REGISTRY_BASE_URL || 'http://localhost:8004';
const LLM_SERVICE_BASE_URL = import.meta.env.VITE_LLM_SERVICE_BASE_URL || 'http://localhost:8005';
const BIAS_DETECTION_BASE_URL = import.meta.env.VITE_BIAS_DETECTION_BASE_URL || 'http://localhost:8006';
const ADWORKBENCH_PROXY_BASE_URL = import.meta.env.VITE_ADWORKBENCH_PROXY_BASE_URL || 'http://localhost:8007';
const AUTONOMOUS_LEARNING_BASE_URL = import.meta.env.VITE_AUTONOMOUS_LEARNING_BASE_URL || 'http://localhost:8008';

const ORCHESTRATOR_API_KEY = import.meta.env.VITE_ORCHESTRATOR_API_KEY;
const AUDIT_TRAIL_API_KEY = import.meta.env.VITE_AUDIT_TRAIL_API_KEY;
const AGENT_REGISTRY_API_KEY = import.meta.env.VITE_AGENT_REGISTRY_API_KEY;
const LLM_API_KEY = import.meta.env.VITE_LLM_API_KEY;
const BIAS_DETECTION_API_KEY = import.meta.env.VITE_BIAS_DETECTION_API_KEY;
const ADWORKBENCH_API_KEY = import.meta.env.VITE_ADWORKBENCH_API_KEY;
const AUTONOMOUS_LEARNING_API_KEY = import.meta.env.VITE_AUTONOMOUS_LEARNING_API_KEY;

// Validation for required environment variables
const requiredEnvVars = [
  { key: ORCHESTRATOR_API_KEY, name: 'VITE_ORCHESTRATOR_API_KEY' },
  { key: AUDIT_TRAIL_API_KEY, name: 'VITE_AUDIT_TRAIL_API_KEY' },
  { key: AGENT_REGISTRY_API_KEY, name: 'VITE_AGENT_REGISTRY_API_KEY' },
  { key: LLM_API_KEY, name: 'VITE_LLM_API_KEY' },
  { key: BIAS_DETECTION_API_KEY, name: 'VITE_BIAS_DETECTION_API_KEY' },
  { key: ADWORKBENCH_API_KEY, name: 'VITE_ADWORKBENCH_API_KEY' },
  { key: AUTONOMOUS_LEARNING_API_KEY, name: 'VITE_AUTONOMOUS_LEARNING_API_KEY' }
];

requiredEnvVars.forEach(({ key, name }) => {
  if (!key) {
    if (import.meta.env.PROD) {
      throw new Error(`${name} is not set in production environment.`);
    } else {
      console.warn(`${name} is not set. Using potentially insecure defaults or will fail.`);
    }
  }
});

// Axios instances for each service
const orchestratorApi = axios.create({
  baseURL: ORCHESTRATOR_BASE_URL,
  headers: {
    'X-API-Key': ORCHESTRATOR_API_KEY,
    'Content-Type': 'application/json',
  },
});

const auditTrailApi = axios.create({
  baseURL: AUDIT_TRAIL_BASE_URL,
  headers: {
    'X-API-Key': AUDIT_TRAIL_API_KEY,
    'Content-Type': 'application/json',
  },
});

const agentRegistryApi = axios.create({
  baseURL: AGENT_REGISTRY_BASE_URL,
  headers: {
    'X-API-Key': AGENT_REGISTRY_API_KEY,
    'Content-Type': 'application/json',
  },
});

const llmServiceApi = axios.create({
  baseURL: LLM_SERVICE_BASE_URL,
  headers: {
    'X-API-Key': LLM_API_KEY,
    'Content-Type': 'application/json',
  },
});

const biasDetectionApi = axios.create({
  baseURL: BIAS_DETECTION_BASE_URL,
  headers: {
    'X-API-Key': BIAS_DETECTION_API_KEY,
    'Content-Type': 'application/json',
  },
});

const adworkbenchProxyApi = axios.create({
  baseURL: ADWORKBENCH_PROXY_BASE_URL,
  headers: {
    'X-API-Key': ADWORKBENCH_API_KEY,
    'Content-Type': 'application/json',
  },
});

const autonomousLearningApi = axios.create({
  baseURL: AUTONOMOUS_LEARNING_BASE_URL,
  headers: {
    'X-API-Key': AUTONOMOUS_LEARNING_API_KEY,
    'Content-Type': 'application/json',
  },
});

// Error handling utility
const handleApiError = (error: AxiosError, operation: string) => {
  console.error(`API Error in ${operation}:`, error);
  if (error.response) {
    throw new Error(`API Error: ${error.response.status} - ${error.response.statusText}`);
  } else if (error.request) {
    throw new Error('Network Error: No response received from server');
  } else {
    throw new Error(`Request Error: ${error.message}`);
  }
};

// Orchestrator API functions
export const getOrchestratorStatus = async () => {
  try {
    const response = await orchestratorApi.get('/orchestrator/status');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getOrchestratorStatus');
  }
};

export const getActiveTasks = async () => {
  try {
    const response = await orchestratorApi.get('/orchestrator/tasks/active');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getActiveTasks');
  }
};

export const getTaskDetails = async (taskId: number) => {
  try {
    const response = await orchestratorApi.get(`/orchestrator/tasks/${taskId}`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getTaskDetails');
  }
};

export const resolveDebate = async (debateId: number, resolution: any) => {
  try {
    const response = await orchestratorApi.post(`/orchestrator/debates/${debateId}/resolve`, resolution);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'resolveDebate');
  }
};

export const cancelTask = async (taskId: number) => {
  try {
    const response = await orchestratorApi.post(`/orchestrator/tasks/${taskId}/cancel`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'cancelTask');
  }
};

export const getOrchestratorTaskStatus = async (taskId: number) => {
  try {
    const response = await orchestratorApi.get(`/orchestrator/task/${taskId}/status`);
    return response.data;
  } catch (error: unknown) {
    console.error(`Error fetching orchestrator task status for ID ${taskId}:`, error);
    throw error;
  }
};

export const submitResearchGoal = async (goalText: string) => {
  try {
    const response = await orchestratorApi.post('/orchestrator/goals', { goal_text: goalText });
    return response.data;
  } catch (error: unknown) {
    console.error('Error submitting research goal:', error);
    throw error;
  }
};

// Audit Trail API functions
export const getAuditHistory = async (entityType: string, entityId: string) => {
  try {
    const response = await auditTrailApi.get(`/audit/history/${entityType}/${entityId}`);
    return response.data;
  } catch (error: unknown) {
    console.error(`Error fetching audit history for ${entityType}:${entityId}:`, error);
    throw error;
  }
};

// Agent Registry API functions
export const getRegisteredAgents = async () => {
  try {
    const response = await agentRegistryApi.get('/registry/agents');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getRegisteredAgents');
  }
};

export const getAgentDetails = async (agentId: string) => {
  try {
    const response = await agentRegistryApi.get(`/registry/agents/${agentId}`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getAgentDetails');
  }
};

// LLM Service API functions
export const chatWithLLM = async (request: { prompt: string; model: string; max_tokens: number; temperature: number; }) => {
  try {
    const response = await llmServiceApi.post('/llm/chat', request);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'chatWithLLM');
  }
};

// Bias Detection API functions
export const submitBiasAnalysis = async (request: BiasAnalysisRequest) => {
  try {
    const response = await biasDetectionApi.post('/bias/analyze', request);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'submitBiasAnalysis');
  }
};

export const getBiasReports = async () => {
  try {
    const response = await biasDetectionApi.get('/bias/reports');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getBiasReports');
  }
};

export const getBiasReport = async (reportId: string) => {
  try {
    const response = await biasDetectionApi.get(`/bias/reports/${reportId}`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getBiasReport');
  }
};

// AD Workbench Proxy API functions
export const submitADWorkbenchQuery = async (request: ADWorkbenchQueryRequest): Promise<{query: ADWorkbenchQuery} | undefined> => {
  try {
    const response = await adworkbenchProxyApi.post('/adworkbench/query', request);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'submitADWorkbenchQuery');
  }
};

export const getADWorkbenchQueries = async () => {
  try {
    const response = await adworkbenchProxyApi.get('/adworkbench/queries');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getADWorkbenchQueries');
  }
};

export const getADWorkbenchQuery = async (queryId: string) => {
  try {
    const response = await adworkbenchProxyApi.get(`/adworkbench/queries/${queryId}`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getADWorkbenchQuery');
  }
};

// System Health API function
export const getSystemHealth = async () => {
  try {
    const response = await orchestratorApi.get('/health');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getSystemHealth');
  }
};

// Autonomous Learning API functions
export const getLearningMetrics = async () => {
  try {
    const response = await autonomousLearningApi.get('/metrics/');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getLearningMetrics');
  }
};

export const getAgentPerformance = async (agentId?: string) => {
  try {
    const url = agentId ? `/performance/${agentId}` : '/performance/';
    const response = await autonomousLearningApi.get(url);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getAgentPerformance');
  }
};

export const getLearningPatterns = async () => {
  try {
    const response = await autonomousLearningApi.get('/patterns/');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getLearningPatterns');
  }
};

export const getKnowledgeGrowth = async () => {
  try {
    const response = await autonomousLearningApi.get('/knowledge/growth');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getKnowledgeGrowth');
  }
};

export const getSelfEvolutionStatus = async () => {
  try {
    const response = await autonomousLearningApi.get('/evolution/status');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getSelfEvolutionStatus');
  }
};

export const getPredictivePerformance = async (agentId: string) => {
  try {
    const response = await autonomousLearningApi.get(`/predictive/performance/${agentId}`);
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getPredictivePerformance');
  }
};

export const getEvolutionTrajectory = async () => {
  try {
    const response = await autonomousLearningApi.get('/evolution/trajectory');
    return response.data;
  } catch (error) {
    handleApiError(error as AxiosError, 'getEvolutionTrajectory');
  }
};

// TypeScript interfaces

// Orchestrator types
export interface OrchestratorStatus {
  status: string;
  active_tasks: number;
  total_tasks: number;
  uptime: string;
  is_active: boolean;
  active_tasks_count: number;
  registered_agents_count: number;
  pending_debates_count: number;
  uptime_seconds: number;
  agent_health: Record<string, string>;
}

export interface Task {
  id: number;
  type: string;
  status: string;
  created_at: string;
  updated_at: string;
  agent_id?: string;
  parameters?: any;
  agent_type?: string;
  description?: string;
  progress?: number;
  logs?: string[];
  result?: any;
  error?: string;
}

export interface DebateResolution {
  resolution: string;
  reasoning: string;
  confidence_score: number;
  task_id?: string;
  resolved_by?: string;
}

export interface OrchestratorTaskStatus {
  id: number;
  status: string;
  message: string;
  result_data?: any;
}

// Research Goal types
export interface ResearchGoal {
  id: number;
  goal_text: string;
  status: string;
  created_at: string;
  updated_at: string | null;
}

// Audit Trail types
export interface AuditLogEntry {
  id: number;
  entity_type: string;
  entity_id: string;
  event_type: string;
  description: string;
  timestamp: string;
  metadata_json?: any;
}

export interface AuditHistoryResponse {
  entity_type: string;
  entity_id: string;
  history: AuditLogEntry[];
}

// Agent Registry types
export interface AgentDetails {
  id: string;
  agent_id: string;
  name: string;
  type: string;
  capabilities: string[];
  status: string;
  registered_at: string;
  last_active: string;
  api_endpoint: string;
}

export interface AgentListResponse {
  agents: AgentDetails[];
  total_count: number;
}

// LLM Service types
export interface LLMResponse {
  response_text: string;
  model_name: string;
  usage_tokens: number;
  confidence_score: number;
  ethical_flags: string[];
}

// Bias Detection types
export interface BiasAnalysisRequest {
  content: string;
  analysis_type: 'text' | 'dataset' | 'model';
  parameters?: any;
  include_recommendations?: boolean;
}

export interface BiasReport {
  id: string;
  analysis_type: string;
  content_summary: string;
  bias_score: number;
  severity: string;
  description: string;
  suggested_corrections: string[];
  overall_risk_level: string;
  recommendations: string[];
  created_at: string;
  bias_level?: string;
  content_preview?: string;
  bias_categories?: Array<{
    name: string;
    description: string;
    confidence: number;
  }>;
  mitigation_strategies?: string[];
}

export interface BiasAnalysisResponse {
  report: BiasReport;
  processing_time: number;
}

// AD Workbench types
export interface ADWorkbenchQueryRequest {
  query: string;
  query_type: 'federated' | 'direct' | 'meta_analysis';
  data_sources?: string[];
  parameters?: any;
  include_metadata?: boolean;
}

export interface ADWorkbenchQuery {
  id: string;
  query: string;
  query_type: string;
  status: string;
  submitted_at: string;
  completed_at?: string;
  results?: any[];
  data_sources?: string[];
  metadata?: any;
  error_message?: string;
}

export interface ADWorkbenchResult {
  data: any[];
  metadata: any;
  statistics?: any;
  query?: ADWorkbenchQuery;
}

// System Health types
export interface HealthStatus {
  service: string;
  status: 'healthy' | 'unhealthy';
  details?: any;
  error?: string;
}

// Autonomous Learning types
export interface LearningMetrics {
  total_patterns_extracted: number;
  total_context_enrichments: number;
  average_confidence_score: number;
  knowledge_growth_rate: number;
  last_updated: string;
  active_learning_cycles: number;
}

export interface AgentPerformance {
  agent_id: string;
  total_tasks: number;
  success_rate: number;
  average_execution_time: number;
  average_accuracy: number;
  average_confidence: number;
  last_performance_update: string;
  learning_trend: 'improving' | 'stable' | 'declining';
}

export interface LearningPattern {
  id: number;
  pattern_type: string;
  pattern_description: string;
  success_rate: number;
  application_count: number;
  last_applied: string;
  discovered_from: any[];
  created_at: string;
}

export interface KnowledgeGrowth {
  total_documents: number;
  total_chunks: number;
  vector_dimensions: number;
  growth_rate_per_day: number;
  quality_score_trend: number[];
  timestamps: string[];
}

export interface SelfEvolutionStatus {
  evolution_phase: string;
  learning_effectiveness: number;
  adaptation_rate: number;
  knowledge_utilization: number;
  self_improvement_metrics: {
    pattern_recognition_accuracy: number;
    context_enrichment_quality: number;
    task_success_prediction: number;
  };
  last_evolution_cycle: string;
}

export interface PredictivePerformance {
  agent_id: string;
  agent_name: string;
  predicted_success_rate: number;
  confidence: number;
  expected_improvement: number;
  prediction_timestamp: string;
}

export interface EvolutionTrajectory {
  current_phase: string;
  progress_percentage: number;
  next_milestone: string;
  estimated_completion: string;
  trajectory_timestamp: string;
}

// Research-Grade Component APIs

// Evolution Dashboard APIs
export const getEvolutionMetrics = async () => {
  try {
    const response = await autonomousLearningApi.get('/evolution/metrics');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch evolution metrics:', error);
    throw error;
  }
};

// Research Canvas APIs
export const getAgentOrchestration = async () => {
  try {
    const response = await orchestratorApi.get('/orchestration/agents');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch agent orchestration:', error);
    throw error;
  }
};

export const getTaskFlows = async () => {
  try {
    const response = await orchestratorApi.get('/orchestration/task-flows');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch task flows:', error);
    throw error;
  }
};

export const getAgentDebates = async () => {
  try {
    const response = await orchestratorApi.get('/orchestration/debates');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch agent debates:', error);
    throw error;
  }
};

// Causal Inference Explorer APIs
export const getCausalGraphs = async () => {
  try {
    const response = await orchestratorApi.get('/causal/graphs');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch causal graphs:', error);
    throw error;
  }
};

export const simulateIntervention = async (intervention: any) => {
  try {
    const response = await orchestratorApi.post('/causal/intervene', intervention);
    return response.data;
  } catch (error) {
    console.error('Failed to simulate intervention:', error);
    throw error;
  }
};

export const getMechanisticPathways = async () => {
  try {
    const response = await orchestratorApi.get('/causal/pathways');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch mechanistic pathways:', error);
    throw error;
  }
};

// Uncertainty Quantification Center APIs
export const getUncertaintyMetrics = async () => {
  try {
    const response = await orchestratorApi.get('/uncertainty/metrics');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch uncertainty metrics:', error);
    throw error;
  }
};

export const getMonteCarloResults = async () => {
  try {
    const response = await orchestratorApi.get('/uncertainty/monte-carlo');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch Monte Carlo results:', error);
    throw error;
  }
};

export const getPINNConvergence = async () => {
  try {
    const response = await orchestratorApi.get('/uncertainty/pinn-convergence');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch PINN convergence:', error);
    throw error;
  }
};

// Knowledge Base Navigator APIs
export const semanticSearch = async (query: string) => {
  try {
    const response = await orchestratorApi.post('/knowledge/search', { query });
    return response.data;
  } catch (error) {
    console.error('Failed to perform semantic search:', error);
    throw error;
  }
};

export const getKnowledgeEvolution = async () => {
  try {
    const response = await orchestratorApi.get('/knowledge/evolution');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch knowledge evolution:', error);
    throw error;
  }
};

export const getPatternRecognition = async () => {
  try {
    const response = await orchestratorApi.get('/knowledge/patterns');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch pattern recognition:', error);
    throw error;
  }
};

// Research Output Studio APIs
export const generateResearchReport = async (findings: any) => {
  try {
    const response = await orchestratorApi.post('/reports/generate', findings);
    return response.data;
  } catch (error) {
    console.error('Failed to generate research report:', error);
    throw error;
  }
};

export const validateStatisticalOutputs = async (outputs: any) => {
  try {
    const response = await orchestratorApi.post('/reports/validate', outputs);
    return response.data;
  } catch (error) {
    console.error('Failed to validate statistical outputs:', error);
    throw error;
  }
};

export const exportPublication = async (reportId: string, format: string) => {
  try {
    const response = await orchestratorApi.post('/reports/export', { reportId, format });
    return response.data;
  } catch (error) {
    console.error('Failed to export publication:', error);
    throw error;
  }
};

// Performance Analytics Suite APIs
export const getSystemHealthDashboard = async () => {
  try {
    const response = await orchestratorApi.get('/health/dashboard');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch system health dashboard:', error);
    throw error;
  }
};

export const getAgentPerformanceMatrix = async () => {
  try {
    const response = await orchestratorApi.get('/performance/matrix');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch agent performance matrix:', error);
    throw error;
  }
};

export const getResourceUtilization = async () => {
  try {
    const response = await orchestratorApi.get('/performance/resources');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch resource utilization:', error);
    throw error;
  }
};

export const getErrorPatterns = async () => {
  try {
    const response = await orchestratorApi.get('/performance/errors');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch error patterns:', error);
    throw error;
  }
};

// Research Ethics & Bias Monitor APIs
export const getBiasDetectionResults = async () => {
  try {
    const response = await orchestratorApi.get('/ethics/bias-detection');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch bias detection results:', error);
    throw error;
  }
};

export const getEthicalCompliance = async () => {
  try {
    const response = await orchestratorApi.get('/ethics/compliance');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch ethical compliance:', error);
    throw error;
  }
};

export const getContentModerationLogs = async () => {
  try {
    const response = await orchestratorApi.get('/ethics/moderation-logs');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch content moderation logs:', error);
    throw error;
  }
};

export const getDataPrivacyMetrics = async () => {
  try {
    const response = await orchestratorApi.get('/ethics/privacy-metrics');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch data privacy metrics:', error);
    throw error;
  }
};

export const getAuditTrailExplorer = async () => {
  try {
    const response = await orchestratorApi.get('/ethics/audit-trail');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch audit trail explorer:', error);
    throw error;
  }
};