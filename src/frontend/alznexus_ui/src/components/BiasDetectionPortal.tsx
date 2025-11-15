import React, { useState, useEffect } from 'react';
import {
  submitBiasAnalysis,
  getBiasReports,
  getBiasReport,
  BiasAnalysisRequest,
  BiasReport,
  BiasAnalysisResponse
} from '../api/alznexusApi';
import { AxiosError } from 'axios';

function BiasDetectionPortal() {
  const [content, setContent] = useState('');
  const [analysisType, setAnalysisType] = useState<'text' | 'dataset' | 'model'>('text');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reports, setReports] = useState<BiasReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<BiasReport | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);

  useEffect(() => {
    loadBiasReports();
  }, []);

  const loadBiasReports = async () => {
    try {
      const response = await getBiasReports();
      setReports(response);
    } catch (err) {
      console.error('Failed to load bias reports:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const request: BiasAnalysisRequest = {
        content,
        analysis_type: analysisType,
        include_recommendations: true
      };

      const response: BiasAnalysisResponse = await submitBiasAnalysis(request);

      // Add the new report to the list
      setReports(prev => [response.report, ...prev]);

      // Clear form
      setContent('');
    } catch (err) {
      setError((err as AxiosError).message || 'Failed to submit bias analysis');
    } finally {
      setLoading(false);
    }
  };

  const viewReport = async (reportId: string) => {
    try {
      const report = await getBiasReport(reportId);
      setSelectedReport(report);
      setShowReportModal(true);
    } catch (err) {
      setError('Failed to load report details');
    }
  };

  const getBiasLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getBiasLevelIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return '✓';
      case 'medium': return '⚠';
      case 'high': return '⚠';
      default: return '?';
    }
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Bias Detection Portal</h2>
      <p className="text-gray-600 mb-6">
        Analyze content, datasets, and models for potential biases in Alzheimer's research.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Analysis Form */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Submit Content for Analysis</h3>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="analysis-type" className="block text-sm font-medium text-gray-700 mb-1">
                Analysis Type
              </label>
              <select
                id="analysis-type"
                value={analysisType}
                onChange={(e) => setAnalysisType(e.target.value as 'text' | 'dataset' | 'model')}
                className="input-field"
              >
                <option value="text">Text Content</option>
                <option value="dataset">Dataset</option>
                <option value="model">AI Model</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Content to Analyze
              </label>
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder={
                  analysisType === 'text'
                    ? "Enter text content to analyze for bias..."
                    : analysisType === 'dataset'
                    ? "Enter dataset description or metadata..."
                    : "Enter model description and parameters..."
                }
                rows={8}
                className="input-field"
                required
              />
            </div>

            <button
              type="submit"
              className="btn-primary w-full"
              disabled={loading || !content.trim()}
            >
              {loading ? 'Analyzing...' : 'Submit for Analysis'}
            </button>
          </form>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <p className="text-red-800">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Recent Reports */}
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Recent Bias Reports</h3>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {reports.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                No bias reports available yet.
              </p>
            ) : (
              reports.map((report) => (
                <div
                  key={report.id}
                  className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
                  onClick={() => viewReport(report.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm">
                      {report.analysis_type.charAt(0).toUpperCase() + report.analysis_type.slice(1)} Analysis
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBiasLevelColor(report.bias_level || 'low')}`}>
                      {getBiasLevelIcon(report.bias_level || 'low')} {report.bias_level || 'low'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                    {report.content_preview || 'No preview available'}
                  </p>
                  <div className="text-xs text-gray-500">
                    {new Date(report.created_at).toLocaleDateString()}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Report Modal */}
      {showReportModal && selectedReport && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Bias Analysis Report</h3>
                <button
                  onClick={() => setShowReportModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-6">
                {/* Report Header */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <span className="text-sm font-medium text-gray-500">Analysis Type</span>
                    <p className="text-sm">{selectedReport.analysis_type}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Bias Level</span>
                    <p className={`text-sm font-medium ${getBiasLevelColor(selectedReport.bias_level || 'low')}`}>
                      {selectedReport.bias_level || 'low'}
                    </p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-500">Created</span>
                    <p className="text-sm">{new Date(selectedReport.created_at).toLocaleString()}</p>
                  </div>
                </div>

                {/* Content Preview */}
                <div>
                  <h4 className="font-medium mb-2">Content Analyzed</h4>
                  <div className="bg-gray-50 p-4 rounded-lg max-h-32 overflow-y-auto">
                    <p className="text-sm">{selectedReport.content_preview || 'No preview available'}</p>
                  </div>
                </div>

                {/* Bias Categories */}
                <div>
                  <h4 className="font-medium mb-2">Bias Categories Detected</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {selectedReport.bias_categories && selectedReport.bias_categories.length > 0 ? (
                      selectedReport.bias_categories.map((category, index) => (
                        <div key={index} className="bg-red-50 border border-red-200 rounded-lg p-3">
                          <div className="font-medium text-red-800">{category.name}</div>
                          <div className="text-sm text-red-600">{category.description}</div>
                          <div className="text-xs text-red-500 mt-1">
                            Confidence: {(category.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="col-span-full text-center text-gray-500 py-4">
                        No bias categories detected
                      </div>
                    )}
                  </div>
                </div>

                {/* Recommendations */}
                {selectedReport.recommendations && selectedReport.recommendations.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Recommendations</h4>
                    <ul className="space-y-2">
                      {selectedReport.recommendations.map((rec, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-blue-600 mt-1">•</span>
                          <span className="text-sm">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Mitigation Strategies */}
                {selectedReport.mitigation_strategies && selectedReport.mitigation_strategies.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2">Mitigation Strategies</h4>
                    <ul className="space-y-2">
                      {selectedReport.mitigation_strategies.map((strategy, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-green-600 mt-1">•</span>
                          <span className="text-sm">{strategy}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default BiasDetectionPortal;