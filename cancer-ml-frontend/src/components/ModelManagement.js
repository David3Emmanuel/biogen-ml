import React, { useState, useEffect } from 'react';
import { Save, Upload, Trash2, AlertCircle, CheckCircle, FileText } from 'lucide-react';
import axios from 'axios';

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [modelName, setModelName] = useState('');
  const [modelDescription, setModelDescription] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/models');
      setModels(response.data.models || []);
    } catch (err) {
      setError('Failed to fetch models. Please check if the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveModel = async () => {
    if (!modelName.trim()) {
      setError('Please enter a model name');
      return;
    }

    setActionLoading('save');
    setError(null);
    setSuccess(null);

    try {
      await axios.post('http://localhost:8000/save_model', {
        name: modelName.trim(),
        description: modelDescription.trim()
      });

      setSuccess(`Model "${modelName}" saved successfully!`);
      setModelName('');
      setModelDescription('');
      fetchModels(); // Refresh the list
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to save model');
    } finally {
      setActionLoading(null);
    }
  };

  const handleLoadModel = async (modelName) => {
    setActionLoading(`load-${modelName}`);
    setError(null);
    setSuccess(null);

    try {
      await axios.post('http://localhost:8000/load_model', {
        name: modelName
      });

      setSuccess(`Model "${modelName}" loaded successfully!`);
    } catch (err) {
      setError(err.response?.data?.detail || `Failed to load model "${modelName}"`);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDeleteModel = async (modelName) => {
    if (!window.confirm(`Are you sure you want to delete the model "${modelName}"?`)) {
      return;
    }

    setActionLoading(`delete-${modelName}`);
    setError(null);
    setSuccess(null);

    try {
      await axios.delete(`http://localhost:8000/models/${modelName}`);

      setSuccess(`Model "${modelName}" deleted successfully!`);
      fetchModels(); // Refresh the list
    } catch (err) {
      setError(err.response?.data?.detail || `Failed to delete model "${modelName}"`);
    } finally {
      setActionLoading(null);
    }
  };

  const clearMessages = () => {
    setError(null);
    setSuccess(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Management</h1>
        <p className="text-gray-600">Save, load, and manage trained models</p>
      </div>

      {/* Status Messages */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <span className="text-red-700">{error}</span>
            </div>
            <button
              onClick={clearMessages}
              className="text-red-500 hover:text-red-700"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
              <span className="text-green-700">{success}</span>
            </div>
            <button
              onClick={clearMessages}
              className="text-green-500 hover:text-green-700"
            >
              ×
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Save Model */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Save Current Model</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Name
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g., cancer_risk_v1"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={modelDescription}
                onChange={(e) => setModelDescription(e.target.value)}
                placeholder="Brief description of the model..."
                rows={3}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <button
              onClick={handleSaveModel}
              disabled={actionLoading === 'save'}
              className="w-full flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {actionLoading === 'save' ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Model
                </>
              )}
            </button>
          </div>
        </div>

        {/* Model List */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Saved Models</h2>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="text-gray-500 mt-2">Loading models...</p>
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Saved Models</h3>
              <p className="text-gray-500">Train and save your first model to see it here</p>
            </div>
          ) : (
            <div className="space-y-4">
              {models.map((model, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="text-sm font-medium text-gray-900">{model.name}</h3>
                      {model.description && (
                        <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                      )}
                      <div className="flex items-center mt-2 text-xs text-gray-500">
                        <span>Saved: {new Date(model.created_at).toLocaleString()}</span>
                        {model.size && (
                          <>
                            <span className="mx-2">•</span>
                            <span>Size: {(model.size / 1024 / 1024).toFixed(2)} MB</span>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="flex space-x-2 ml-4">
                      <button
                        onClick={() => handleLoadModel(model.name)}
                        disabled={actionLoading === `load-${model.name}`}
                        className="flex items-center px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                      >
                        {actionLoading === `load-${model.name}` ? (
                          <div className="animate-spin rounded-full h-3 w-3 border-b border-blue-700"></div>
                        ) : (
                          <Upload className="h-3 w-3 mr-1" />
                        )}
                        Load
                      </button>

                      <button
                        onClick={() => handleDeleteModel(model.name)}
                        disabled={actionLoading === `delete-${model.name}`}
                        className="flex items-center px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                      >
                        {actionLoading === `delete-${model.name}` ? (
                          <div className="animate-spin rounded-full h-3 w-3 border-b border-red-700"></div>
                        ) : (
                          <Trash2 className="h-3 w-3 mr-1" />
                        )}
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Management Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
              <Save className="h-4 w-4 mr-2 text-green-600" />
              Saving Models
            </h4>
            <p className="text-sm text-gray-700">
              Save trained models with descriptive names for later use. Include version numbers
              and training details in descriptions.
            </p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
              <Upload className="h-4 w-4 mr-2 text-blue-600" />
              Loading Models
            </h4>
            <p className="text-sm text-gray-700">
              Load previously saved models to make predictions or continue training.
              Only one model can be active at a time.
            </p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2 flex items-center">
              <Trash2 className="h-4 w-4 mr-2 text-red-600" />
              Managing Models
            </h4>
            <p className="text-sm text-gray-700">
              Delete outdated models to free up storage space. Be careful as deleted
              models cannot be recovered.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;