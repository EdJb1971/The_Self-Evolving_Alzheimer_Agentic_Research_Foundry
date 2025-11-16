import { Link, useLocation } from 'react-router-dom';

function Navbar() {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="bg-blue-800 p-4 shadow-lg">
      <div className="container">
        <div className="flex justify-between items-center">
          <Link to="/" className="text-white text-2xl font-bold">
            AlzNexus Platform
          </Link>

          <div className="flex space-x-6">
            <Link
              to="/dashboard"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/dashboard') || isActive('/') ? 'text-blue-200' : ''
              }`}
            >
              Dashboard
            </Link>

            <Link
              to="/query-submission"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/query-submission') ? 'text-blue-200' : ''
              }`}
            >
              Submit Query
            </Link>

            <Link
              to="/task-status"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/task-status') ? 'text-blue-200' : ''
              }`}
            >
              Task Status
            </Link>

            <Link
              to="/agent-registry"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/agent-registry') ? 'text-blue-200' : ''
              }`}
            >
              Agents
            </Link>

            <Link
              to="/llm-chat"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/llm-chat') ? 'text-blue-200' : ''
              }`}
            >
              LLM Chat
            </Link>

            <Link
              to="/bias-detection"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/bias-detection') ? 'text-blue-200' : ''
              }`}
            >
              Bias Detection
            </Link>

            <Link
              to="/ad-workbench"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/ad-workbench') ? 'text-blue-200' : ''
              }`}
            >
              AD Workbench
            </Link>

            <Link
              to="/orchestrator"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/orchestrator') ? 'text-blue-200' : ''
              }`}
            >
              Orchestrator
            </Link>

            <Link
              to="/learning-dashboard"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/learning-dashboard') ? 'text-blue-200' : ''
              }`}
            >
              Learning Dashboard
            </Link>

            <Link
              to="/evolution-dashboard"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/evolution-dashboard') ? 'text-blue-200' : ''
              }`}
            >
              Evolution Dashboard
            </Link>

            <Link
              to="/research-canvas"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/research-canvas') ? 'text-blue-200' : ''
              }`}
            >
              Research Canvas
            </Link>

            <Link
              to="/causal-inference"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/causal-inference') ? 'text-blue-200' : ''
              }`}
            >
              Causal Inference
            </Link>

            <Link
              to="/uncertainty-center"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/uncertainty-center') ? 'text-blue-200' : ''
              }`}
            >
              Uncertainty Center
            </Link>

            <Link
              to="/knowledge-base"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/knowledge-base') ? 'text-blue-200' : ''
              }`}
            >
              Knowledge Base
            </Link>

            <Link
              to="/research-output"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/research-output') ? 'text-blue-200' : ''
              }`}
            >
              Research Output
            </Link>

            <Link
              to="/performance-analytics"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/performance-analytics') ? 'text-blue-200' : ''
              }`}
            >
              Performance Analytics
            </Link>

            <Link
              to="/ethics-monitor"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/ethics-monitor') ? 'text-blue-200' : ''
              }`}
            >
              Ethics Monitor
            </Link>

            <Link
              to="/audit-trail"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/audit-trail') ? 'text-blue-200' : ''
              }`}
            >
              Audit Trail
            </Link>

            <Link
              to="/settings"
              className={`text-white hover:text-blue-200 transition-colors ${
                isActive('/settings') ? 'text-blue-200' : ''
              }`}
            >
              Settings
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;