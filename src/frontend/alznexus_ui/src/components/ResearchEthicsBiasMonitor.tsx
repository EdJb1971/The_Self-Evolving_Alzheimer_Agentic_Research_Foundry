import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import Plotly from 'plotly.js';

interface BiasDetectionResult {
  category: string;
  biasScore: number;
  confidence: number;
  mitigationStatus: 'none' | 'partial' | 'complete';
  lastChecked: string;
}

interface EthicalCompliance {
  regulation: string;
  complianceScore: number;
  status: 'compliant' | 'warning' | 'non-compliant';
  lastAudit: string;
  nextReview: string;
}

interface ContentModerationLog {
  id: string;
  timestamp: string;
  contentType: string;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  action: 'approved' | 'flagged' | 'blocked';
  reason?: string;
}

interface PrivacyMetric {
  dataType: string;
  accessCount: number;
  anonymizationLevel: number;
  retentionDays: number;
  complianceStatus: 'compliant' | 'review' | 'breach';
}

const ResearchEthicsBiasMonitor: React.FC = () => {
  const biasRef = useRef<HTMLDivElement>(null);
  const complianceRef = useRef<HTMLDivElement>(null);
  const moderationRef = useRef<HTMLDivElement>(null);
  const privacyRef = useRef<HTMLDivElement>(null);
  const auditRef = useRef<HTMLDivElement>(null);

  const [biasResults] = useState<BiasDetectionResult[]>([
    { category: 'Demographic Bias', biasScore: 0.12, confidence: 0.89, mitigationStatus: 'complete', lastChecked: '2025-11-15T10:30:00Z' },
    { category: 'Selection Bias', biasScore: 0.08, confidence: 0.94, mitigationStatus: 'complete', lastChecked: '2025-11-15T10:25:00Z' },
    { category: 'Confirmation Bias', biasScore: 0.15, confidence: 0.76, mitigationStatus: 'partial', lastChecked: '2025-11-15T10:20:00Z' },
    { category: 'Publication Bias', biasScore: 0.05, confidence: 0.98, mitigationStatus: 'complete', lastChecked: '2025-11-15T10:15:00Z' },
    { category: 'Algorithmic Bias', biasScore: 0.09, confidence: 0.87, mitigationStatus: 'complete', lastChecked: '2025-11-15T10:10:00Z' }
  ]);

  const [ethicalCompliance] = useState<EthicalCompliance[]>([
    { regulation: 'HIPAA', complianceScore: 98, status: 'compliant', lastAudit: '2025-11-01', nextReview: '2026-05-01' },
    { regulation: 'GDPR', complianceScore: 96, status: 'compliant', lastAudit: '2025-10-15', nextReview: '2026-04-15' },
    { regulation: '21 CFR Part 11', complianceScore: 95, status: 'compliant', lastAudit: '2025-09-30', nextReview: '2026-03-30' },
    { regulation: 'IRB Guidelines', complianceScore: 97, status: 'compliant', lastAudit: '2025-11-10', nextReview: '2026-05-10' },
    { regulation: 'Data Privacy Act', complianceScore: 92, status: 'warning', lastAudit: '2025-10-01', nextReview: '2026-04-01' }
  ]);

  const [moderationLogs] = useState<ContentModerationLog[]>([
    { id: 'log1', timestamp: '2025-11-15T14:30:00Z', contentType: 'Research Query', riskLevel: 'low', action: 'approved' },
    { id: 'log2', timestamp: '2025-11-15T14:25:00Z', contentType: 'Agent Response', riskLevel: 'medium', action: 'flagged', reason: 'Potential PII exposure' },
    { id: 'log3', timestamp: '2025-11-15T14:20:00Z', contentType: 'Literature Summary', riskLevel: 'low', action: 'approved' },
    { id: 'log4', timestamp: '2025-11-15T14:15:00Z', contentType: 'Statistical Output', riskLevel: 'high', action: 'flagged', reason: 'Unusual statistical patterns' },
    { id: 'log5', timestamp: '2025-11-15T14:10:00Z', contentType: 'Clinical Recommendation', riskLevel: 'medium', action: 'approved' }
  ]);

  const [privacyMetrics] = useState<PrivacyMetric[]>([
    { dataType: 'Patient Records', accessCount: 1247, anonymizationLevel: 95, retentionDays: 2555, complianceStatus: 'compliant' },
    { dataType: 'Biomarker Data', accessCount: 2156, anonymizationLevel: 98, retentionDays: 1825, complianceStatus: 'compliant' },
    { dataType: 'Research Queries', accessCount: 3456, anonymizationLevel: 85, retentionDays: 365, complianceStatus: 'review' },
    { dataType: 'AI Model Outputs', accessCount: 5678, anonymizationLevel: 92, retentionDays: 1095, complianceStatus: 'compliant' },
    { dataType: 'Audit Logs', accessCount: 890, anonymizationLevel: 100, retentionDays: 2555, complianceStatus: 'compliant' }
  ]);

  // Initialize visualizations
  useEffect(() => {
    if (biasRef.current) createBiasDetectionChart();
    if (complianceRef.current) createComplianceDashboard();
    if (moderationRef.current) createModerationTimeline();
    if (privacyRef.current) createPrivacyVisualization();
    if (auditRef.current) createAuditTrail();
  }, []);

  const createBiasDetectionChart = () => {
    if (!biasRef.current) return;

    const data = [{
      x: biasResults.map(b => b.category),
      y: biasResults.map(b => b.biasScore * 100),
      type: 'bar' as const,
      name: 'Bias Score (%)',
      marker: {
        color: biasResults.map(b =>
          b.mitigationStatus === 'complete' ? '#4CAF50' :
          b.mitigationStatus === 'partial' ? '#FF9800' : '#F44336'
        )
      }
    }];

    const layout = {
      title: 'Bias Detection Results',
      xaxis: {
        title: 'Bias Category',
        tickangle: -45
      },
      yaxis: {
        title: 'Bias Score (%)',
        range: [0, 20]
      },
      showlegend: false,
      width: 600,
      height: 400,
      margin: { t: 50, r: 50, l: 60, b: 100 }
    };

    Plotly.newPlot(biasRef.current, data as any, layout as any);
  };

  const createComplianceDashboard = () => {
    if (!complianceRef.current) return;

    const data = [{
      x: ethicalCompliance.map(c => c.regulation),
      y: ethicalCompliance.map(c => c.complianceScore),
      type: 'bar' as const,
      name: 'Compliance Score (%)',
      marker: {
        color: ethicalCompliance.map(c =>
          c.status === 'compliant' ? '#4CAF50' :
          c.status === 'warning' ? '#FF9800' : '#F44336'
        )
      }
    }];

    const layout = {
      title: 'Ethical Compliance Dashboard',
      xaxis: {
        title: 'Regulation',
        tickangle: -45
      },
      yaxis: {
        title: 'Compliance Score (%)',
        range: [85, 100]
      },
      showlegend: false,
      width: 600,
      height: 400,
      margin: { t: 50, r: 50, l: 60, b: 100 }
    };

    Plotly.newPlot(complianceRef.current, data as any, layout as any);
  };

  const createModerationTimeline = () => {
    if (!moderationRef.current) return;

    const svg = d3.select(moderationRef.current);
    svg.selectAll('*').remove();

    const width = 600;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 60, left: 60 };

    svg.attr('width', width).attr('height', height);

    // Create timeline of moderation actions
    const timeScale = d3.scaleTime()
      .domain(d3.extent(moderationLogs, d => new Date(d.timestamp)) as [Date, Date])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleBand()
      .domain(moderationLogs.map((_, i) => i.toString()))
      .range([margin.top, height - margin.bottom])
      .padding(0.1);

    // Add timeline line
    svg.append('line')
      .attr('x1', margin.left)
      .attr('x2', width - margin.right)
      .attr('y1', height / 2)
      .attr('y2', height / 2)
      .attr('stroke', '#ddd')
      .attr('stroke-width', 2);

    // Add moderation events
    svg.selectAll('.moderation-event')
      .data(moderationLogs)
      .enter()
      .append('circle')
      .attr('class', 'moderation-event')
      .attr('cx', d => timeScale(new Date(d.timestamp)))
      .attr('cy', height / 2)
      .attr('r', 8)
      .attr('fill', d => getRiskColor(d.riskLevel))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .append('title')
      .text(d => `${d.contentType}: ${d.action}${d.reason ? ` - ${d.reason}` : ''}`);
  };

  const createPrivacyVisualization = () => {
    if (!privacyRef.current) return;

    const data = [{
      x: privacyMetrics.map(p => p.dataType),
      y: privacyMetrics.map(p => p.anonymizationLevel),
      type: 'bar' as const,
      name: 'Anonymization Level (%)',
      marker: { color: '#2196F3' }
    }, {
      x: privacyMetrics.map(p => p.dataType),
      y: privacyMetrics.map(p => p.accessCount / 100), // Scale down for visibility
      type: 'scatter' as const,
      mode: 'markers',
      name: 'Access Count (hundreds)',
      yaxis: 'y2',
      marker: { color: '#FF9800', size: 10 }
    }];

    const layout = {
      title: 'Data Privacy & Access Metrics',
      xaxis: {
        title: 'Data Type',
        tickangle: -45
      },
      yaxis: {
        title: 'Anonymization Level (%)',
        range: [80, 105]
      },
      yaxis2: {
        title: 'Access Count',
        overlaying: 'y',
        side: 'right'
      },
      showlegend: true,
      width: 600,
      height: 400,
      margin: { t: 50, r: 80, l: 60, b: 100 }
    };

    Plotly.newPlot(privacyRef.current, data as any, layout as any);
  };

  const createAuditTrail = () => {
    if (!auditRef.current) return;

    const svg = d3.select(auditRef.current);
    svg.selectAll('*').remove();

    const width = 600;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 60, left: 60 };

    svg.attr('width', width).attr('height', height);

    // Create audit trail visualization
    const auditEvents = [
      { time: '2025-11-15T08:00:00Z', event: 'System Startup', type: 'system' },
      { time: '2025-11-15T09:30:00Z', event: 'Bias Check Completed', type: 'compliance' },
      { time: '2025-11-15T10:15:00Z', event: 'Research Query Processed', type: 'research' },
      { time: '2025-11-15T11:00:00Z', event: 'Content Moderation', type: 'moderation' },
      { time: '2025-11-15T12:30:00Z', event: 'Privacy Audit', type: 'compliance' },
      { time: '2025-11-15T14:00:00Z', event: 'Agent Response Generated', type: 'research' },
      { time: '2025-11-15T15:45:00Z', event: 'Compliance Review', type: 'compliance' }
    ];

    const timeScale = d3.scaleTime()
      .domain(d3.extent(auditEvents, d => new Date(d.time)) as [Date, Date])
      .range([margin.left, width - margin.right]);

    // Add timeline
    svg.append('line')
      .attr('x1', margin.left)
      .attr('x2', width - margin.right)
      .attr('y1', height / 2)
      .attr('y2', height / 2)
      .attr('stroke', '#666')
      .attr('stroke-width', 2);

    // Add audit events
    svg.selectAll('.audit-event')
      .data(auditEvents)
      .enter()
      .append('circle')
      .attr('class', 'audit-event')
      .attr('cx', d => timeScale(new Date(d.time)))
      .attr('cy', height / 2)
      .attr('r', 6)
      .attr('fill', d => getEventColor(d.type))
      .append('title')
      .text(d => `${d.event} - ${new Date(d.time).toLocaleTimeString()}`);
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return '#F44336';
      case 'high': return '#FF9800';
      case 'medium': return '#FFEB3B';
      case 'low': return '#4CAF50';
      default: return '#9E9E9E';
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'system': return '#2196F3';
      case 'compliance': return '#4CAF50';
      case 'research': return '#9C27B0';
      case 'moderation': return '#FF9800';
      default: return '#9E9E9E';
    }
  };

  const getComplianceColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'non-compliant': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getPrivacyStatusColor = (status: string) => {
    switch (status) {
      case 'compliant': return 'text-green-600';
      case 'review': return 'text-yellow-600';
      case 'breach': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Research Ethics & Bias Monitor</h1>
      <p className="text-gray-600 mb-8">
        Comprehensive monitoring of ethical compliance, bias detection, and responsible AI practices
        in Alzheimer\'s research. Ensuring research integrity and patient privacy protection.
      </p>

      {/* Key Ethics Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
          <h3 className="text-lg font-semibold text-green-800">Overall Compliance</h3>
          <p className="text-2xl font-bold text-green-600">96.2%</p>
          <p className="text-sm text-green-600">Target: &gt;95%</p>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
          <h3 className="text-lg font-semibold text-blue-800">Bias Mitigation</h3>
          <p className="text-2xl font-bold text-blue-600">92.1%</p>
          <p className="text-sm text-blue-600">Complete</p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg border-l-4 border-purple-500">
          <h3 className="text-lg font-semibold text-purple-800">Content Moderated</h3>
          <p className="text-2xl font-bold text-purple-600">99.7%</p>
          <p className="text-sm text-purple-600">Auto-approved</p>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg border-l-4 border-orange-500">
          <h3 className="text-lg font-semibold text-orange-800">Privacy Compliant</h3>
          <p className="text-2xl font-bold text-orange-600">97.8%</p>
          <p className="text-sm text-orange-600">Data accesses</p>
        </div>
      </div>

      {/* Bias Detection & Compliance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Bias Detection Results</h2>
          <div ref={biasRef} className="w-full h-80"></div>
          <div className="mt-4 space-y-2">
            {biasResults.map(bias => (
              <div key={bias.category} className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-xs ${
                    bias.mitigationStatus === 'complete' ? 'bg-green-100 text-green-800' :
                    bias.mitigationStatus === 'partial' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {bias.mitigationStatus}
                  </span>
                  <span>{bias.category}</span>
                </div>
                <span className="font-mono">{(bias.biasScore * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Ethical Compliance Dashboard</h2>
          <div ref={complianceRef} className="w-full h-80"></div>
          <div className="mt-4 space-y-2">
            {ethicalCompliance.map(compliance => (
              <div key={compliance.regulation} className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-xs ${getComplianceColor(compliance.status)}`}>
                    {compliance.status}
                  </span>
                  <span>{compliance.regulation}</span>
                </div>
                <div className="text-right">
                  <div className="font-mono">{compliance.complianceScore}%</div>
                  <div className="text-xs text-gray-500">Next: {compliance.nextReview}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Content Moderation & Privacy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Content Moderation Timeline</h2>
          <div ref={moderationRef} className="w-full h-60"></div>
          <div className="mt-4 space-y-2 max-h-40 overflow-y-auto">
            {moderationLogs.slice(0, 5).map(log => (
              <div key={log.id} className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-xs ${getRiskColor(log.riskLevel)}`}>
                    {log.riskLevel}
                  </span>
                  <span>{log.contentType}</span>
                </div>
                <div className="text-right">
                  <div className={`font-medium ${
                    log.action === 'approved' ? 'text-green-600' :
                    log.action === 'flagged' ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {log.action}
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Data Privacy Visualizer</h2>
          <div ref={privacyRef} className="w-full h-80"></div>
          <div className="mt-4 space-y-2">
            {privacyMetrics.map(metric => (
              <div key={metric.dataType} className="flex items-center justify-between text-sm">
                <span>{metric.dataType}</span>
                <div className="text-right">
                  <div className={`font-medium ${getPrivacyStatusColor(metric.complianceStatus)}`}>
                    {metric.complianceStatus}
                  </div>
                  <div className="text-xs text-gray-500">
                    {metric.retentionDays} days retention
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Audit Trail Explorer */}
      <div className="bg-white p-4 rounded-lg border mb-8">
        <h2 className="text-xl font-semibold mb-4">Audit Trail Explorer</h2>
        <div ref={auditRef} className="w-full h-60"></div>
        <p className="text-sm text-gray-600 mt-2">
          Complete audit trail of all system activities, compliance checks, and research operations.
          Every action is logged with timestamps and user attribution for full accountability.
        </p>
      </div>

      {/* Ethics Summary */}
      <div className="bg-gray-50 p-6 rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Responsible AI Research Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Ethical Safeguards</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Continuous bias detection and mitigation</li>
              <li>• HIPAA/GDPR compliance monitoring</li>
              <li>• Automated content moderation</li>
              <li>• Data anonymization and privacy protection</li>
              <li>• Regular ethical audits and reviews</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Research Integrity</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Transparent methodology documentation</li>
              <li>• Reproducible research practices</li>
              <li>• Peer review simulation for outputs</li>
              <li>• Statistical validation requirements</li>
              <li>• Complete audit trails for accountability</li>
            </ul>
          </div>
        </div>
        <div className="mt-6 p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
          <p className="text-green-800 text-sm">
            <strong>Healthcare Research Ethics:</strong> AlzNexus maintains the highest standards of
            ethical AI research, ensuring patient privacy, eliminating bias, and providing complete
            transparency in all Alzheimer\'s disease research activities.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResearchEthicsBiasMonitor;