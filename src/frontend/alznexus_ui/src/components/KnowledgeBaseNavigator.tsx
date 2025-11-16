import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import Plotly from 'plotly.js';

interface KnowledgeItem {
  id: string;
  content: string;
  timestamp: string;
  confidence: number;
  source: string;
  category: string;
  supersededBy?: string;
}

interface SearchResult {
  item: KnowledgeItem;
  relevance: number;
  context: string[];
}

interface KnowledgePattern {
  pattern: string;
  frequency: number;
  trend: 'increasing' | 'stable' | 'decreasing';
  confidence: number;
}

const KnowledgeBaseNavigator: React.FC = () => {
  const timelineRef = useRef<HTMLDivElement>(null);
  const patternRef = useRef<HTMLDivElement>(null);
  const enrichmentRef = useRef<HTMLDivElement>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedItem, setSelectedItem] = useState<KnowledgeItem | null>(null);
  const [knowledgeItems] = useState<KnowledgeItem[]>([
    {
      id: 'k1',
      content: 'Amyloid-β plaques correlate with cognitive decline',
      timestamp: '2025-10-01T10:00:00Z',
      confidence: 0.85,
      source: 'Literature Analysis Agent',
      category: 'Biomarker Research'
    },
    {
      id: 'k2',
      content: 'Tau protein hyperphosphorylation precedes amyloid deposition',
      timestamp: '2025-10-15T14:30:00Z',
      confidence: 0.92,
      source: 'Hypothesis Validator Agent',
      category: 'Disease Mechanism'
    },
    {
      id: 'k3',
      content: 'Combined amyloid and tau targeting shows 40% efficacy improvement',
      timestamp: '2025-11-01T09:15:00Z',
      confidence: 0.88,
      source: 'Drug Screener Agent',
      category: 'Therapeutic Strategy',
      supersededBy: 'k4'
    },
    {
      id: 'k4',
      content: 'Multi-target therapy with amyloid, tau, and inflammation modulation achieves 55% efficacy',
      timestamp: '2025-11-10T16:45:00Z',
      confidence: 0.94,
      source: 'Trial Optimizer Agent',
      category: 'Therapeutic Strategy'
    }
  ]);

  const [patterns] = useState<KnowledgePattern[]>([
    { pattern: 'Amyloid-Tau Interaction', frequency: 45, trend: 'increasing', confidence: 0.89 },
    { pattern: 'Inflammation Pathways', frequency: 32, trend: 'stable', confidence: 0.76 },
    { pattern: 'Biomarker Combinations', frequency: 28, trend: 'increasing', confidence: 0.82 },
    { pattern: 'Synaptic Protection', frequency: 19, trend: 'increasing', confidence: 0.71 }
  ]);

  // Initialize visualizations
  useEffect(() => {
    if (timelineRef.current) createKnowledgeTimeline();
    if (patternRef.current) createPatternDashboard();
    if (enrichmentRef.current) createEnrichmentVisualization();
  }, [knowledgeItems, patterns]);

  const performSemanticSearch = (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    // Simulate semantic search with RAG
    const results: SearchResult[] = knowledgeItems
      .filter(item =>
        item.content.toLowerCase().includes(query.toLowerCase()) ||
        item.category.toLowerCase().includes(query.toLowerCase())
      )
      .map(item => ({
        item,
        relevance: Math.random() * 0.5 + 0.5, // Simulated relevance score
        context: [
          'Related research from recent GWAS studies',
          'Correlates with clinical trial outcomes',
          'Supported by mechanistic modeling'
        ]
      }))
      .sort((a, b) => b.relevance - a.relevance);

    setSearchResults(results);
  };

  const createKnowledgeTimeline = () => {
    if (!timelineRef.current) return;

    const svg = d3.select(timelineRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 60, left: 60 };

    svg.attr('width', width).attr('height', height);

    const xScale = d3.scaleTime()
      .domain(d3.extent(knowledgeItems, d => new Date(d.timestamp)) as [Date, Date])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height - margin.bottom, margin.top]);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).tickFormat(d => `${(d as number * 100).toFixed(0)}%`));

    // Add knowledge accumulation line
    const cumulativeData = knowledgeItems
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .map((item, index) => ({
        date: new Date(item.timestamp),
        cumulative: (index + 1) / knowledgeItems.length,
        confidence: item.confidence
      }));

    const line = d3.line<{ date: Date; cumulative: number }>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.cumulative));

    svg.append('path')
      .datum(cumulativeData)
      .attr('fill', 'none')
      .attr('stroke', '#1f77b4')
      .attr('stroke-width', 3)
      .attr('d', line);

    // Add confidence points
    svg.selectAll('.confidence-point')
      .data(cumulativeData)
      .enter()
      .append('circle')
      .attr('class', 'confidence-point')
      .attr('cx', d => xScale(d.date))
      .attr('cy', d => yScale(d.cumulative))
      .attr('r', d => d.confidence * 8 + 4)
      .attr('fill', '#ff7f0e')
      .attr('opacity', 0.7)
      .append('title')
      .text(d => `Confidence: ${(d.confidence * 100).toFixed(1)}%`);
  };

  const createPatternDashboard = () => {
    if (!patternRef.current) return;

    const data = [{
      x: patterns.map(p => p.pattern),
      y: patterns.map(p => p.frequency),
      type: 'bar' as const,
      name: 'Frequency',
      marker: {
        color: patterns.map(p =>
          p.trend === 'increasing' ? '#4CAF50' :
          p.trend === 'decreasing' ? '#F44336' : '#FF9800'
        )
      }
    }];

    const layout = {
      title: 'Research Pattern Recognition',
      xaxis: {
        title: 'Pattern',
        tickangle: -45
      },
      yaxis: {
        title: 'Frequency'
      },
      showlegend: false,
      width: 600,
      height: 400,
      margin: { t: 50, r: 50, l: 60, b: 100 }
    };

    Plotly.newPlot(patternRef.current, data as any, layout as any);
  };

  const createEnrichmentVisualization = () => {
    if (!enrichmentRef.current) return;

    const svg = d3.select(enrichmentRef.current);
    svg.selectAll('*').remove();

    const width = 600;
    const height = 300;

    svg.attr('width', width).attr('height', height);

    // Create a simple network showing knowledge enrichment
    const nodes = [
      { id: 'query', label: 'Research Query', x: 100, y: 150 },
      { id: 'context', label: 'Context Enrichment', x: 250, y: 100 },
      { id: 'reasoning', label: 'Enhanced Reasoning', x: 400, y: 150 },
      { id: 'output', label: 'Improved Output', x: 550, y: 150 }
    ];

    const links = [
      { source: 'query', target: 'context' },
      { source: 'context', target: 'reasoning' },
      { source: 'reasoning', target: 'output' }
    ];

    // Draw links
    svg.selectAll('.enrichment-link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'enrichment-link')
      .attr('x1', d => nodes.find(n => n.id === d.source)!.x)
      .attr('y1', d => nodes.find(n => n.id === d.source)!.y)
      .attr('x2', d => nodes.find(n => n.id === d.target)!.x)
      .attr('y2', d => nodes.find(n => n.id === d.target)!.y)
      .attr('stroke', '#666')
      .attr('stroke-width', 3)
      .attr('marker-end', 'url(#enrichment-arrow)');

    // Draw nodes
    const nodeGroups = svg.selectAll('.enrichment-node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'enrichment-node')
      .attr('transform', d => `translate(${d.x - 50}, ${d.y - 20})`);

    nodeGroups.append('rect')
      .attr('width', 100)
      .attr('height', 40)
      .attr('fill', '#3498db')
      .attr('stroke', '#2980b9')
      .attr('rx', 5);

    nodeGroups.append('text')
      .attr('x', 50)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .attr('font-size', '11px')
      .text(d => d.label);

    // Add arrow marker
    svg.append('defs')
      .append('marker')
      .attr('id', 'enrichment-arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 8)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#666');
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Knowledge Base Navigator</h1>
      <p className="text-gray-600 mb-8">
        Explore the evolving knowledge ecosystem of the self-learning research platform.
        Witness how insights accumulate, patterns emerge, and knowledge continuously improves.
      </p>

      {/* Semantic Search Interface */}
      <div className="bg-gray-50 p-6 rounded-lg mb-8">
        <h2 className="text-xl font-semibold mb-4">Semantic Search with RAG</h2>
        <div className="flex space-x-4 mb-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && performSemanticSearch(searchQuery)}
            placeholder="Search knowledge base (e.g., 'amyloid tau interaction', 'therapeutic strategy')..."
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={() => performSemanticSearch(searchQuery)}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Search
          </button>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-700">Search Results ({searchResults.length})</h3>
            {searchResults.map((result, _index) => (
              <div
                key={result.item.id}
                className="bg-white p-4 rounded-lg border cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => setSelectedItem(result.item)}
              >
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-gray-800">{result.item.content}</h4>
                  <span className="text-sm text-green-600 font-medium">
                    {(result.relevance * 100).toFixed(1)}% relevant
                  </span>
                </div>
                <div className="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                  <span>Source: {result.item.source}</span>
                  <span>Category: {result.item.category}</span>
                  <span>Confidence: {(result.item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="text-sm text-gray-500">
                  <strong>Context:</strong> {result.context.join(' • ')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Knowledge Evolution Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Knowledge Evolution Timeline</h2>
          <div ref={timelineRef} className="w-full h-80"></div>
          <p className="text-sm text-gray-600 mt-2">
            Shows how knowledge accumulates over time. Circle size indicates confidence level.
            The system never forgets - only improves upon existing knowledge.
          </p>
        </div>

        <div className="bg-white p-4 rounded-lg border">
          <h2 className="text-xl font-semibold mb-4">Pattern Recognition Dashboard</h2>
          <div ref={patternRef} className="w-full h-80"></div>
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span>Increasing</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-orange-500 rounded"></div>
              <span>Stable</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span>Decreasing</span>
            </div>
          </div>
        </div>
      </div>

      {/* Context Enrichment Viewer */}
      <div className="bg-white p-4 rounded-lg border mb-8">
        <h2 className="text-xl font-semibold mb-4">Context Enrichment Process</h2>
        <div ref={enrichmentRef} className="w-full h-60 flex justify-center items-center"></div>
        <p className="text-sm text-gray-600 mt-2">
          Illustrates how raw queries are enriched with learned context before processing,
          enabling more sophisticated reasoning and better research outcomes.
        </p>
      </div>

      {/* Selected Item Details */}
      {selectedItem && (
        <div className="bg-blue-50 p-6 rounded-lg border-l-4 border-blue-500">
          <h2 className="text-xl font-semibold mb-4">Knowledge Item Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-gray-700 mb-2">Content</h3>
              <p className="text-gray-800 mb-4">{selectedItem.content}</p>

              <h3 className="font-semibold text-gray-700 mb-2">Metadata</h3>
              <div className="space-y-1 text-sm text-gray-600">
                <p><strong>Source:</strong> {selectedItem.source}</p>
                <p><strong>Category:</strong> {selectedItem.category}</p>
                <p><strong>Confidence:</strong> {(selectedItem.confidence * 100).toFixed(1)}%</p>
                <p><strong>Timestamp:</strong> {new Date(selectedItem.timestamp).toLocaleString()}</p>
                {selectedItem.supersededBy && (
                  <p className="text-orange-600">
                    <strong>Superseded by:</strong> {selectedItem.supersededBy}
                  </p>
                )}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-gray-700 mb-2">Knowledge Evolution</h3>
              <div className="bg-white p-4 rounded border">
                <p className="text-sm text-gray-600 mb-2">
                  This knowledge item represents the current state of understanding.
                  The system continuously refines and improves upon previous insights.
                </p>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>Active and improving</span>
                </div>
                <div className="mt-2 text-xs text-gray-500">
                  Last updated: {new Date(selectedItem.timestamp).toLocaleDateString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeBaseNavigator;