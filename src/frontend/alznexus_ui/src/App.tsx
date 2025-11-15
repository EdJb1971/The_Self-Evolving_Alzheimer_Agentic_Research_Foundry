import { Routes, Route } from 'react-router-dom';
import QuerySubmission from './components/QuerySubmission';
import TaskStatusDashboard from './components/TaskStatusDashboard';
import AuditTrailViewer from './components/AuditTrailViewer';
import Dashboard from './components/Dashboard';
import AgentRegistry from './components/AgentRegistry';
import LLMChat from './components/LLMChat';
import BiasDetectionPortal from './components/BiasDetectionPortal';
import ADWorkbenchQueryInterface from './components/ADWorkbenchQueryInterface';
import AdvancedOrchestratorControls from './components/AdvancedOrchestratorControls';
import LearningDashboard from './components/LearningDashboard';
import Settings from './components/Settings';
import Navbar from './components/Navbar';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />
      <div className="container mt-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/query-submission" element={<QuerySubmission />} />
          <Route path="/task-status" element={<TaskStatusDashboard />} />
          <Route path="/audit-trail" element={<AuditTrailViewer />} />
          <Route path="/agent-registry" element={<AgentRegistry />} />
          <Route path="/llm-chat" element={<LLMChat />} />
          <Route path="/bias-detection" element={<BiasDetectionPortal />} />
          <Route path="/ad-workbench" element={<ADWorkbenchQueryInterface />} />
          <Route path="/orchestrator" element={<AdvancedOrchestratorControls />} />
          <Route path="/learning-dashboard" element={<LearningDashboard />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;