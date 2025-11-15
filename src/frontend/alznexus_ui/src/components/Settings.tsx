import React, { useState, useEffect } from 'react';

interface APIKey {
  service: string;
  key: string;
  isValid: boolean;
  lastValidated?: Date;
}

function Settings() {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([
    { service: 'OpenAI', key: '', isValid: false },
    { service: 'Anthropic', key: '', isValid: false },
    { service: 'Google AI', key: '', isValid: false },
    { service: 'AD Workbench', key: '', isValid: false },
    { service: 'Bias Detection', key: '', isValid: false }
  ]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [notifications, setNotifications] = useState({
    taskCompletion: true,
    errorAlerts: true,
    debateResolution: true,
    systemUpdates: false
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = () => {
    // Load settings from localStorage or environment
    const savedKeys = localStorage.getItem('alznexus_api_keys');
    if (savedKeys) {
      try {
        const parsed = JSON.parse(savedKeys);
        setApiKeys(prev => prev.map(apiKey => ({
          ...apiKey,
          key: parsed[apiKey.service] || '',
          isValid: parsed[`${apiKey.service}_valid`] || false
        })));
      } catch (err) {
        console.error('Failed to load saved API keys:', err);
      }
    }

    const savedTheme = localStorage.getItem('alznexus_theme') as 'light' | 'dark';
    if (savedTheme) setTheme(savedTheme);

    const savedNotifications = localStorage.getItem('alznexus_notifications');
    if (savedNotifications) {
      try {
        setNotifications(JSON.parse(savedNotifications));
      } catch (err) {
        console.error('Failed to load notification settings:', err);
      }
    }
  };

  const saveSettings = () => {
    const keysToSave: { [key: string]: string | boolean } = {};
    apiKeys.forEach(apiKey => {
      keysToSave[apiKey.service] = apiKey.key;
      keysToSave[`${apiKey.service}_valid`] = apiKey.isValid;
    });

    localStorage.setItem('alznexus_api_keys', JSON.stringify(keysToSave));
    localStorage.setItem('alznexus_theme', theme);
    localStorage.setItem('alznexus_notifications', JSON.stringify(notifications));

    setMessage({ type: 'success', text: 'Settings saved successfully!' });
    setTimeout(() => setMessage(null), 3000);
  };

  const validateAPIKey = async (service: string, key: string) => {
    if (!key.trim()) return;

    setLoading(true);
    try {
      // This would typically make a call to validate the API key
      // For now, we'll simulate validation
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Simple validation - in real implementation, this would call backend validation
      const isValid = key.length > 10; // Mock validation

      setApiKeys(prev => prev.map(apiKey =>
        apiKey.service === service
          ? { ...apiKey, isValid, lastValidated: new Date() }
          : apiKey
      ));

      setMessage({
        type: isValid ? 'success' : 'error',
        text: isValid ? `${service} API key validated successfully!` : `Invalid ${service} API key.`
      });
    } catch (err) {
      setMessage({ type: 'error', text: `Failed to validate ${service} API key.` });
    } finally {
      setLoading(false);
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const updateAPIKey = (service: string, key: string) => {
    setApiKeys(prev => prev.map(apiKey =>
      apiKey.service === service
        ? { ...apiKey, key, isValid: false }
        : apiKey
    ));
  };

  const exportSettings = () => {
    const settings = {
      apiKeys: apiKeys.reduce((acc, apiKey) => {
        acc[apiKey.service] = apiKey.key;
        return acc;
      }, {} as { [key: string]: string }),
      theme,
      notifications
    };

    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'alznexus_settings.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const importSettings = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const settings = JSON.parse(e.target?.result as string);

        if (settings.apiKeys) {
          setApiKeys(prev => prev.map(apiKey => ({
            ...apiKey,
            key: settings.apiKeys[apiKey.service] || '',
            isValid: false
          })));
        }

        if (settings.theme) setTheme(settings.theme);
        if (settings.notifications) setNotifications(settings.notifications);

        setMessage({ type: 'success', text: 'Settings imported successfully!' });
        setTimeout(() => setMessage(null), 3000);
      } catch (err) {
        setMessage({ type: 'error', text: 'Failed to import settings. Invalid file format.' });
        setTimeout(() => setMessage(null), 3000);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-semibold mb-4">Settings & Configuration</h2>
      <p className="text-gray-600 mb-6">
        Configure API keys, preferences, and system settings for the AlzNexus platform.
      </p>

      {message && (
        <div className={`mb-6 p-4 rounded-md ${
          message.type === 'success'
            ? 'bg-green-50 border border-green-200 text-green-800'
            : 'bg-red-50 border border-red-200 text-red-800'
        }`}>
          <p>{message.text}</p>
        </div>
      )}

      <div className="space-y-8">
        {/* API Keys Section */}
        <div>
          <h3 className="text-lg font-medium mb-4">API Keys</h3>
          <div className="space-y-4">
            {apiKeys.map((apiKey) => (
              <div key={apiKey.service} className="flex items-center space-x-4 p-4 border border-gray-200 rounded-lg">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {apiKey.service} API Key
                  </label>
                  <input
                    type="password"
                    value={apiKey.key}
                    onChange={(e) => updateAPIKey(apiKey.service, e.target.value)}
                    placeholder={`Enter ${apiKey.service} API key...`}
                    className="input-field w-full"
                  />
                  {apiKey.lastValidated && (
                    <p className="text-xs text-gray-500 mt-1">
                      Last validated: {apiKey.lastValidated.toLocaleString()}
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    apiKey.isValid ? 'text-green-600 bg-green-100' : 'text-red-600 bg-red-100'
                  }`}>
                    {apiKey.isValid ? '✓ Valid' : '✗ Invalid'}
                  </span>
                  <button
                    onClick={() => validateAPIKey(apiKey.service, apiKey.key)}
                    className="btn-secondary text-xs"
                    disabled={loading || !apiKey.key.trim()}
                  >
                    {loading ? 'Validating...' : 'Validate'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Appearance Section */}
        <div>
          <h3 className="text-lg font-medium mb-4">Appearance</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Theme
              </label>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
                className="input-field"
                aria-label="Application theme selection"
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>
          </div>
        </div>

        {/* Notifications Section */}
        <div>
          <h3 className="text-lg font-medium mb-4">Notifications</h3>
          <div className="space-y-3">
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={notifications.taskCompletion}
                onChange={(e) => setNotifications(prev => ({ ...prev, taskCompletion: e.target.checked }))}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Task completion notifications</span>
            </label>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={notifications.errorAlerts}
                onChange={(e) => setNotifications(prev => ({ ...prev, errorAlerts: e.target.checked }))}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Error and failure alerts</span>
            </label>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={notifications.debateResolution}
                onChange={(e) => setNotifications(prev => ({ ...prev, debateResolution: e.target.checked }))}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Agent debate resolution requests</span>
            </label>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={notifications.systemUpdates}
                onChange={(e) => setNotifications(prev => ({ ...prev, systemUpdates: e.target.checked }))}
                className="rounded border-gray-300"
              />
              <span className="text-sm">System updates and maintenance notifications</span>
            </label>
          </div>
        </div>

        {/* Import/Export Section */}
        <div>
          <h3 className="text-lg font-medium mb-4">Import/Export Settings</h3>
          <div className="flex space-x-4">
            <button
              onClick={exportSettings}
              className="btn-secondary"
            >
              Export Settings
            </button>
            <label className="btn-secondary cursor-pointer">
              Import Settings
              <input
                type="file"
                accept=".json"
                onChange={importSettings}
                className="hidden"
              />
            </label>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Export your settings to a JSON file or import settings from a previously exported file.
          </p>
        </div>

        {/* System Information */}
        <div>
          <h3 className="text-lg font-medium mb-4">System Information</h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-500">Version:</span> v1.0.0
              </div>
              <div>
                <span className="font-medium text-gray-500">Environment:</span> Development
              </div>
              <div>
                <span className="font-medium text-gray-500">Frontend:</span> React 18 + TypeScript
              </div>
              <div>
                <span className="font-medium text-gray-500">Backend:</span> FastAPI + Microservices
              </div>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="pt-6 border-t border-gray-200">
          <button
            onClick={saveSettings}
            className="btn-primary w-full"
          >
            Save All Settings
          </button>
        </div>
      </div>
    </div>
  );
}

export default Settings;