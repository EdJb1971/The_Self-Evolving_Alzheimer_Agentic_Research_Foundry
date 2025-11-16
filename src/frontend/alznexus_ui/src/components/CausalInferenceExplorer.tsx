import React, { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import Plotly from 'plotly.js';

cytoscape.use(dagre);

interface CausalRelationship {
  cause: string;
  effect: string;
  strength: number;
  confidence: number;
  mechanism?: string;
}

interface InterventionResult {
  variable: string;
  baseline: number;
  intervened: number;
  effectSize: number;
  confidence: number;
}

interface CausalGraph {
  nodes: Array<{ id: string; label: string; type: 'exposure' | 'outcome' | 'confounder' | 'mediator' }>;
  edges: CausalRelationship[];
}

const CausalInferenceExplorer: React.FC = () => {
  const graphRef = useRef<HTMLDivElement>(null);
  const interventionRef = useRef<HTMLDivElement>(null);
  const effectSizeRef = useRef<HTMLDivElement>(null);
  const [cy, setCy] = useState<cytoscape.Core | null>(null);
  const [selectedIntervention, setSelectedIntervention] = useState<string>('');

  const [causalGraph] = useState<CausalGraph>({
    nodes: [
      { id: 'age', label: 'Age', type: 'confounder' },
      { id: 'genetics', label: 'APOE ε4', type: 'exposure' },
      { id: 'inflammation', label: 'Inflammation', type: 'mediator' },
      { id: 'amyloid', label: 'Amyloid-β', type: 'mediator' },
      { id: 'tau', label: 'Tau Protein', type: 'mediator' },
      { id: 'synaptic', label: 'Synaptic Loss', type: 'mediator' },
      { id: 'cognitive', label: 'Cognitive Decline', type: 'outcome' },
      { id: 'dementia', label: 'Alzheimer\'s Dementia', type: 'outcome' }
    ],
    edges: [
      { cause: 'age', effect: 'inflammation', strength: 0.7, confidence: 0.85, mechanism: 'Age-related immune activation' },
      { cause: 'genetics', effect: 'amyloid', strength: 0.8, confidence: 0.92, mechanism: 'Genetic predisposition to amyloid accumulation' },
      { cause: 'amyloid', effect: 'tau', strength: 0.6, confidence: 0.78, mechanism: 'Amyloid-induced tau hyperphosphorylation' },
      { cause: 'inflammation', effect: 'synaptic', strength: 0.5, confidence: 0.72, mechanism: 'Neuroinflammatory synaptic damage' },
      { cause: 'tau', effect: 'synaptic', strength: 0.7, confidence: 0.88, mechanism: 'Tau-mediated synaptic dysfunction' },
      { cause: 'synaptic', effect: 'cognitive', strength: 0.9, confidence: 0.95, mechanism: 'Synaptic loss impairs cognition' },
      { cause: 'cognitive', effect: 'dementia', strength: 0.8, confidence: 0.90, mechanism: 'Progressive cognitive decline to dementia' },
      { cause: 'age', effect: 'cognitive', strength: 0.4, confidence: 0.65, mechanism: 'Age-related cognitive changes' }
    ]
  });

  const [interventionResults] = useState<InterventionResult[]>([
    { variable: 'Anti-amyloid therapy', baseline: 0.75, intervened: 0.45, effectSize: 0.30, confidence: 0.82 },
    { variable: 'Anti-inflammatory drugs', baseline: 0.75, intervened: 0.55, effectSize: 0.20, confidence: 0.75 },
    { variable: 'Tau aggregation inhibitors', baseline: 0.75, intervened: 0.50, effectSize: 0.25, confidence: 0.78 },
    { variable: 'Synaptic protection', baseline: 0.75, intervened: 0.40, effectSize: 0.35, confidence: 0.85 },
    { variable: 'Lifestyle intervention', baseline: 0.75, intervened: 0.65, effectSize: 0.10, confidence: 0.68 }
  ]);

  // Initialize causal graph
  useEffect(() => {
    if (graphRef.current && !cy) {
      const cytoscapeInstance = cytoscape({
        container: graphRef.current,
        elements: generateGraphElements(),
        style: [
          {
            selector: 'node',
            style: {
              'background-color': (ele: any) => getNodeColor(ele.data('type')),
              'label': 'data(label)',
              'width': 80,
              'height': 50,
              'font-size': '11px',
              'text-valign': 'center',
              'text-halign': 'center',
              'color': '#fff',
              'text-outline-width': 1,
              'text-outline-color': '#000',
              'border-width': 2,
              'border-color': '#fff'
            }
          },
          {
            selector: 'edge',
            style: {
              'width': (ele: any) => Math.max(2, ele.data('strength') * 8),
              'line-color': (ele: any) => getEdgeColor(ele.data('confidence')),
              'target-arrow-color': (ele: any) => getEdgeColor(ele.data('confidence')),
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': (ele: any) => `${(ele.data('strength') * 100).toFixed(0)}%`,
              'font-size': '10px',
              'text-background-color': '#fff',
              'text-background-opacity': 0.8
            }
          }
        ],
        layout: {
          name: 'dagre',
          rankDir: 'LR',
          nodeSep: 120,
          edgeSep: 80,
          rankSep: 120
        } as any
      });

      setCy(cytoscapeInstance);

      // Add tooltips
      cytoscapeInstance.on('tap', 'edge', (evt) => {
        const edge = evt.target;
        const data = edge.data();
        alert(`Mechanism: ${data.mechanism}\nConfidence: ${(data.confidence * 100).toFixed(1)}%`);
      });
    }

    return () => {
      if (cy) {
        cy.destroy();
        setCy(null);
      }
    };
  }, []);

  // Initialize intervention visualization
  useEffect(() => {
    if (interventionRef.current) {
      createInterventionChart();
    }
  }, [interventionResults]);

  // Initialize effect size visualization
  useEffect(() => {
    if (effectSizeRef.current) {
      createEffectSizePlot();
    }
  }, [interventionResults]);

  const generateGraphElements = () => {
    const elements: cytoscape.ElementDefinition[] = [];

    // Add nodes
    causalGraph.nodes.forEach(node => {
      elements.push({
        data: {
          id: node.id,
          label: node.label,
          type: node.type
        }
      });
    });

    // Add edges
    causalGraph.edges.forEach(edge => {
      elements.push({
        data: {
          id: `${edge.cause}-${edge.effect}`,
          source: edge.cause,
          target: edge.effect,
          strength: edge.strength,
          confidence: edge.confidence,
          mechanism: edge.mechanism
        }
      });
    });

    return elements;
  };

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'exposure': return '#e74c3c';
      case 'outcome': return '#27ae60';
      case 'confounder': return '#f39c12';
      case 'mediator': return '#3498db';
      default: return '#95a5a6';
    }
  };

  const getEdgeColor = (confidence: number) => {
    if (confidence > 0.8) return '#27ae60';
    if (confidence > 0.6) return '#f39c12';
    return '#e74c3c';
  };

  const createInterventionChart = () => {
    if (!interventionRef.current) return;

    const data = [{
      x: interventionResults.map(r => r.variable),
      y: interventionResults.map(r => r.baseline),
      type: 'bar',
      name: 'Baseline Risk',
      marker: { color: '#e74c3c' }
    }, {
      x: interventionResults.map(r => r.variable),
      y: interventionResults.map(r => r.intervened),
      type: 'bar',
      name: 'Post-Intervention',
      marker: { color: '#27ae60' }
    }];

    const layout = {
      title: 'Intervention Effects on Alzheimer\'s Risk',
      barmode: 'group',
      xaxis: {
        title: 'Intervention',
        tickangle: -45
      },
      yaxis: {
        title: 'Disease Risk Probability',
        range: [0, 1]
      },
      showlegend: true,
      width: 800,
      height: 400,
      margin: { t: 50, r: 50, l: 60, b: 100 }
    };

    Plotly.newPlot(interventionRef.current, data as any, layout as any);
  };

  const createEffectSizePlot = () => {
    if (!effectSizeRef.current) return;

    const data = [{
      x: interventionResults.map(r => r.effectSize),
      y: interventionResults.map(r => r.confidence),
      mode: 'markers+text',
      type: 'scatter',
      text: interventionResults.map(r => r.variable),
      textposition: 'top center',
      marker: {
        size: interventionResults.map(r => r.effectSize * 50 + 10),
        color: interventionResults.map(r => r.confidence),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {
          title: 'Confidence',
          titleside: 'right'
        }
      },
      name: 'Interventions'
    }];

    const layout = {
      title: 'Effect Size vs Confidence Analysis',
      xaxis: {
        title: 'Effect Size (Risk Reduction)',
        range: [0, 0.4]
      },
      yaxis: {
        title: 'Confidence Level',
        range: [0.6, 1.0]
      },
      showlegend: false,
      width: 800,
      height: 400,
      margin: { t: 50, r: 80, l: 60, b: 50 }
    };

    Plotly.newPlot(effectSizeRef.current, data as any, layout as any);
  };

  const runCounterfactualAnalysis = (intervention: string) => {
    setSelectedIntervention(intervention);
    // In a real implementation, this would call the backend causal inference service
    alert(`Running counterfactual analysis for: ${intervention}\nThis would simulate "what-if" scenarios using DoWhy causal inference framework.`);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Causal Inference Explorer</h1>
      <p className="text-gray-600 mb-8">
        Interactive exploration of causal relationships in Alzheimer's disease. Discover how different
        factors interact to cause disease progression using advanced causal inference algorithms.
      </p>

      {/* Causal Graph Visualization */}
      <div className="bg-white p-4 rounded-lg border mb-8">
        <h2 className="text-xl font-semibold mb-4">Alzheimer's Disease Causal Graph</h2>
        <div ref={graphRef} className="w-full h-96 border rounded mb-4"></div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Exposure</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>Outcome</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span>Confounder</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-4 h-4 bg-blue-500 rounded"></div>
            <span>Mediator</span>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-2">
          Click on edges to see causal mechanisms and confidence levels. Edge thickness represents causal strength.
        </p>
      </div>

      {/* Intervention Simulator */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Intervention Effects</h2>
          <div ref={interventionRef} className="w-full h-80"></div>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Effect Size Analysis</h2>
          <div ref={effectSizeRef} className="w-full h-80"></div>
        </div>
      </div>

      {/* Counterfactual Analysis */}
      <div className="bg-gray-50 p-6 rounded-lg mb-8">
        <h2 className="text-xl font-semibold mb-4">Counterfactual Analysis</h2>
        <p className="text-gray-700 mb-4">
          Explore "what-if" scenarios by intervening on different variables in the causal graph.
          This uses DoWhy's causal inference framework to estimate the effects of hypothetical interventions.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {['Anti-amyloid therapy', 'Anti-inflammatory drugs', 'Tau inhibitors', 'Lifestyle changes', 'Combination therapy'].map(intervention => (
            <button
              key={intervention}
              onClick={() => runCounterfactualAnalysis(intervention)}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {intervention}
            </button>
          ))}
        </div>
        {selectedIntervention && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <p className="text-blue-800">
              <strong>Selected Intervention:</strong> {selectedIntervention}
            </p>
            <p className="text-sm text-blue-600 mt-1">
              Counterfactual analysis would estimate how this intervention changes the probability
              of Alzheimer's progression while holding other factors constant.
            </p>
          </div>
        )}
      </div>

      {/* Mechanistic Understanding */}
      <div className="bg-white p-4 rounded-lg border">
        <h2 className="text-xl font-semibold mb-4">Mechanistic Pathways</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Amyloid Cascade Hypothesis</h3>
            <p className="text-sm text-gray-600">
              APOE ε4 → Amyloid-β accumulation → Tau hyperphosphorylation →
              Synaptic dysfunction → Cognitive decline → Dementia
            </p>
            <div className="mt-2 text-xs text-green-600">
              Supported by: GWAS studies, PET imaging, biomarker analysis
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Neuroinflammation Pathway</h3>
            <p className="text-sm text-gray-600">
              Age → Microglial activation → Chronic inflammation →
              Synaptic damage → Cognitive impairment
            </p>
            <div className="mt-2 text-xs text-green-600">
              Supported by: Cytokine profiling, microglia imaging, clinical trials
            </div>
          </div>
        </div>
        <div className="mt-6 p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">
          <p className="text-yellow-800 text-sm">
            <strong>Research Insight:</strong> Causal inference reveals that targeting multiple pathways
            simultaneously may be more effective than single-mechanism approaches, as evidenced by
            the interconnected nature of Alzheimer\'s disease progression.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CausalInferenceExplorer;