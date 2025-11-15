import React, { useState } from 'react';
import { chatWithLLM, LLMResponse } from '../api/alznexusApi';
import { AxiosError } from 'axios';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: any;
}

function LLMChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState('gpt-4');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(500);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response: LLMResponse = await chatWithLLM({
        prompt: input,
        model,
        max_tokens: maxTokens,
        temperature
      });

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response_text,
        timestamp: new Date(),
        metadata: {
          model: response.model_name,
          tokens: response.usage_tokens,
          confidence: response.confidence_score,
          ethical_flags: response.ethical_flags
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError((err as AxiosError).message || 'Failed to get LLM response');
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">LLM Chat Interface</h2>
      <p className="text-gray-600 mb-6">
        Interact directly with Large Language Models for research assistance and analysis.
      </p>

      {/* Settings */}
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <h3 className="font-medium mb-3">Model Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="input-field"
            >
              <option value="gpt-4">GPT-4</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="gemini-pro">Gemini Pro</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature
            </label>
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="input-field"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Tokens
            </label>
            <input
              type="number"
              min="50"
              max="2000"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="input-field"
            />
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="border border-gray-200 rounded-lg h-96 overflow-y-auto mb-4 p-4 bg-gray-50">
        {messages.length === 0 ? (
          <p className="text-gray-500 text-center mt-20">
            Start a conversation with the LLM...
          </p>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-200'
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  {message.metadata && (
                    <div className="mt-2 text-xs opacity-75">
                      <div>Model: {message.metadata.model}</div>
                      <div>Tokens: {message.metadata.tokens}</div>
                      {message.metadata.ethical_flags.length > 0 && (
                        <div className="text-yellow-600">
                          Ethical flags: {message.metadata.ethical_flags.join(', ')}
                        </div>
                      )}
                    </div>
                  )}
                  <div className="text-xs opacity-50 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-sm text-gray-600">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the LLM anything about Alzheimer's research..."
          className="input-field flex-grow"
          disabled={loading}
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !input.trim()}
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
        <button
          type="button"
          onClick={clearChat}
          className="btn-secondary"
          disabled={loading}
        >
          Clear
        </button>
      </form>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mt-4">
          <p className="text-red-800">Error: {error}</p>
        </div>
      )}
    </div>
  );
}

export default LLMChat;