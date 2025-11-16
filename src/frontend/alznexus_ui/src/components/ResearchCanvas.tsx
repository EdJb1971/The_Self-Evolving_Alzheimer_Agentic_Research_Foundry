import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import * as d3 from 'd3';

cytoscape.use(dagre);

interface Agent {
  id: string;
  name: string;
  specialty: string;
  status: 'idle' | 'active' | 'collaborating' | 'learning';
  performance: number;
  connections: string[];
}

interface Task {
  id: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  assignedAgents: string[];
  progress: number;
}

interface ResearchSession {
  id: string;
  query: string;
  agents: Agent[];
  tasks: Task[];
  debateHistory: Array<{ agent: string; message: string; timestamp: string }>;
}

const ResearchCanvas: React.FC = () => {
  const networkRef = useRef<HTMLDivElement>(null);
  const flowRef = useRef<HTMLDivElement>(null);
  const debateRef = useRef<HTMLDivElement>(null);
  const [cy, setCy] = useState<cytoscape.Core | null>(null);
  const [currentSession, setCurrentSession] = useState<ResearchSession>({
    id: 'session-1',
    query: 'Investigate novel biomarkers for early Alzheimer\'s detection',
    agents: [
      { id: 'biomarker-hunter', name: 'Biomarker Hunter', specialty: 'Biomarker Discovery', status: 'active', performance: 89, connections: ['literature-bridger', 'hypothesis-validator'] },
      { id: 'literature-bridger', name: 'Literature Bridger', specialty: 'Literature Analysis', status: 'collaborating', performance: 92, connections: ['biomarker-hunter', 'drug-screener'] },
      { id: 'hypothesis-validator', name: 'Hypothesis Validator', specialty: 'Statistical Validation', status: 'active', performance: 87, connections: ['biomarker-hunter', 'pathway-modeler'] },
      { id: 'drug-screener', name: 'Drug Screener', specialty: 'Drug Discovery', status: 'idle', performance: 85, connections: ['literature-bridger', 'trial-optimizer'] },
      { id: 'pathway-modeler', name: 'Pathway Modeler', specialty: 'Disease Modeling', status: 'learning', performance: 91, connections: ['hypothesis-validator', 'data-harmonizer'] },
      { id: 'data-harmonizer', name: 'Data Harmonizer', specialty: 'Data Integration', status: 'active', performance: 88, connections: ['pathway-modeler', 'collaboration-matchmaker'] },
      { id: 'collaboration-matchmaker', name: 'Collaboration Matchmaker', specialty: 'Team Formation', status: 'idle', performance: 90, connections: ['data-harmonizer', 'trial-optimizer'] },
      { id: 'trial-optimizer', name: 'Trial Optimizer', specialty: 'Clinical Trials', status: 'pending', performance: 86, connections: ['drug-screener', 'collaboration-matchmaker'] }
    ],
    tasks: [
      { id: 'task-1', description: 'Identify candidate biomarkers from literature', status: 'completed', assignedAgents: ['biomarker-hunter', 'literature-bridger'], progress: 100 },
      { id: 'task-2', description: 'Validate biomarker hypotheses statistically', status: 'in_progress', assignedAgents: ['hypothesis-validator'], progress: 75 },
      { id: 'task-3', description: 'Model disease pathways with PINNs', status: 'pending', assignedAgents: ['pathway-modeler'], progress: 0 },
      { id: 'task-4', description: 'Screen potential therapeutic compounds', status: 'idle', assignedAgents: ['drug-screener'], progress: 0 }
    ],
    debateHistory: [
      { agent: 'biomarker-hunter', message: 'Found 15 promising biomarkers in recent GWAS studies', timestamp: '10:30:15' },
      { agent: 'literature-bridger', message: 'Cross-referencing with existing Alzheimer\'s literature...', timestamp: '10:30:22' },
      { agent: 'hypothesis-validator', message: 'Statistical validation shows p < 0.001 for 8 biomarkers', timestamp: '10:31:05' },
      { agent: 'pathway-modeler', message: 'Initiating PINN modeling for mechanistic understanding', timestamp: '10:31:18' }
    ]
  });

  // Initialize Cytoscape network visualization
  useEffect(() => {
    if (networkRef.current && !cy) {
      const cytoscapeInstance = cytoscape({
        container: networkRef.current,
        elements: generateNetworkElements(),
        style: [
          {
            selector: 'node',
            style: {
              'background-color': (ele: any) => getAgentColor(ele.data('status')),
              'label': 'data(label)',
              'width': 60,
              'height': 60,
              'font-size': '12px',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#fff',
              'text-outline-width': 2,
              'text-outline-color': '#000'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 3,
              'line-color': '#666',
              'target-arrow-color': '#666',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier'
            }
          }
        ],
        layout: {
          name: 'dagre',
          rankDir: 'TB',
          nodeSep: 100,
          edgeSep: 50,
          rankSep: 100
        }
      });

      setCy(cytoscapeInstance);

      // Add click handlers
      cytoscapeInstance.on('tap', 'node', (evt) => {
        const node = evt.target;
        console.log('Tapped node:', node.data());
      });
    }

    return () => {
      if (cy) {
        cy.destroy();
        setCy(null);
      }
    };
  }, [currentSession]);

  // Update network when session changes
  useEffect(() => {
    if (cy) {
      cy.elements().remove();
      cy.add(generateNetworkElements());
      cy.layout({ name: 'dagre', rankDir: 'TB' }).run();
    }
  }, [cy, currentSession]);

  // Initialize task flow diagram
  useEffect(() => {
    if (flowRef.current) {
      createTaskFlowDiagram();
    }
  }, [currentSession.tasks]);

  // Initialize debate visualization
  useEffect(() => {
    if (debateRef.current) {
      createDebateTimeline();
    }
  }, [currentSession.debateHistory]);

  const generateNetworkElements = () => {
    const elements: cytoscape.ElementDefinition[] = [];

    // Add nodes
    currentSession.agents.forEach(agent => {
      elements.push({
        data: {
          id: agent.id,
          label: agent.name.split(' ')[0], // Short name for display
          status: agent.status,
          performance: agent.performance,
          specialty: agent.specialty
        }
      });
    });

    // Add edges
    currentSession.agents.forEach(agent => {
      agent.connections.forEach(targetId => {
        elements.push({
          data: {
            id: `${agent.id}-${targetId}`,
            source: agent.id,
            target: targetId
          }
        });
      });
    });

    return elements;
  };

  const getAgentColor = (status: string) => {
    switch (status) {
      case 'active': return '#4CAF50';
      case 'collaborating': return '#2196F3';
      case 'learning': return '#FF9800';
      case 'idle': return '#9E9E9E';
      default: return '#666';
    }
  };

  const createTaskFlowDiagram = () => {
    if (!flowRef.current) return;

    const svg = d3.select(flowRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 60, left: 60 };

    svg.attr('width', width).attr('height', height);

    const xScale = d3.scaleBand()
      .domain(currentSession.tasks.map(t => t.id))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([height - margin.bottom, margin.top]);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `Task ${d.split('-')[1]}`));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}%`));

    // Add progress bars
    svg.selectAll('.bar')
      .data(currentSession.tasks)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.id)!)
      .attr('y', d => yScale(d.progress))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - margin.bottom - yScale(d.progress))
      .attr('fill', d => getTaskColor(d.status));

    // Add progress labels
    svg.selectAll('.label')
      .data(currentSession.tasks)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => xScale(d.id)! + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.progress) - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#333')
      .text(d => `${d.progress}%`);
  };

  const getTaskColor = (status: string) => {
    switch (status) {
      case 'completed': return '#4CAF50';
      case 'in_progress': return '#2196F3';
      case 'pending': return '#FF9800';
      case 'failed': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const createDebateTimeline = () => {
    if (!debateRef.current) return;

    const container = d3.select(debateRef.current);
    container.selectAll('*').remove();

    const messages = container.selectAll('.message')
      .data(currentSession.debateHistory)
      .enter()
      .append('div')
      .attr('class', 'message flex items-start space-x-3 mb-4 p-3 bg-gray-50 rounded-lg');

    messages.append('div')
      .attr('class', 'avatar w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold')
      .style('background-color', d => getAgentColor(currentSession.agents.find(a => a.id === d.agent)?.status || 'idle'))
      .text(d => d.agent.split('-').map(word => word[0]).join('').toUpperCase());

    const content = messages.append('div')
      .attr('class', 'content flex-1');

    content.append('div')
      .attr('class', 'agent-name text-sm font-semibold text-gray-700')
      .text(d => currentSession.agents.find(a => a.id === d.agent)?.name || d.agent);

    content.append('div')
      .attr('class', 'message-text text-gray-800')
      .text(d => d.message);

    content.append('div')
      .attr('class', 'timestamp text-xs text-gray-500')
      .text(d => d.timestamp);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Research Canvas</h1>
      <p className="text-gray-600 mb-8">
        Interactive visualization of multi-agent research orchestration. Watch as specialized AI agents
        collaborate autonomously to advance Alzheimer's research through swarm intelligence.
      </p>

      {/* Current Research Query */}
      <div className="bg-blue-50 p-4 rounded-lg mb-8 border-l-4 border-blue-500">
        <h2 className="text-lg font-semibold text-blue-800 mb-2">Active Research Session</h2>
        <p className="text-blue-700 font-medium">{currentSession.query}</p>
        <p className="text-sm text-blue-600 mt-1">
          Session ID: {currentSession.id} | {currentSession.agents.filter(a => a.status === 'active').length} agents active
        </p>
      </div>

      {/* Agent Network Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Agent Collaboration Network</h2>
          <div ref={networkRef} className="w-full h-96 border rounded"></div>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>Active</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded"></div>
              <span>Collaborating</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-orange-500 rounded"></div>
              <span>Learning</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-500 rounded"></div>
              <span>Idle</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Task Flow Progress</h2>
          <div ref={flowRef} className="w-full h-96"></div>
        </div>
      </div>

      {/* Debate Arena */}
      <div className="bg-white p-4 rounded-lg border mb-8">
        <h2 className="text-xl font-semibold mb-4">Agent Debate Arena</h2>
        <div ref={debateRef} className="max-h-96 overflow-y-auto">
          {/* Messages will be populated by D3 */}
        </div>
      </div>

      {/* Agent Performance Summary */}
      <div className="bg-gray-50 p-6 rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Agent Performance Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {currentSession.agents.map(agent => (
            <div key={agent.id} className="bg-white p-4 rounded-lg border">
              <h3 className="font-semibold text-gray-800">{agent.name}</h3>
              <p className="text-sm text-gray-600 mb-2">{agent.specialty}</p>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Performance:</span>
                <span className="font-bold text-green-600">{agent.performance}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getAgentColor(agent.status) }}
                ></div>
                <span className="text-sm capitalize">{agent.status}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResearchCanvas;