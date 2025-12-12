import React, { useState } from 'react';
import { Upload, Plus, Send, AlertCircle, CheckCircle, BookOpen } from 'lucide-react';
import axios from 'axios';

const Training = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [tabularData, setTabularData] = useState({
    age: '',
    num_pregnancies: '',
    age_first_intercourse: '',
    num_sexual_partners: '',
    smokes: '',
    smokes_years: '',
    hormonal_contraceptives: '',
    hormonal_contra_years: '',
    iud: '',
    iud_years: '',
    stds_any: '',
    stds_num: '',
    hpv_positive: '',
    pap_abnormal_history: '',
    immunosuppressed: ''
  });
  const [targets, setTargets] = useState({
    cancer_1yr: '',
    cancer_3yr: ''
  });
  const [training, setTraining] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleTabularChange = (field, value) => {
    setTabularData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleTargetChange = (field, value) => {
    setTargets(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an image');
      return;
    }

    // Check if all tabular fields are filled
    const emptyTabularFields = Object.entries(tabularData).filter(([key, value]) => value === '');
    if (emptyTabularFields.length > 0) {
      setError(`Please fill in all tabular fields: ${emptyTabularFields.map(([key]) => key.replace('_', ' ')).join(', ')}`);
      return;
    }

    // Check if targets are filled
    const emptyTargetFields = Object.entries(targets).filter(([key, value]) => value === '');
    if (emptyTargetFields.length > 0) {
      setError(`Please fill in target values: ${emptyTargetFields.map(([key]) => key.replace('_', ' ')).join(', ')}`);
      return;
    }

    setTraining(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('tabular', JSON.stringify(Object.values(tabularData).map(v => parseFloat(v))));
      formData.append('targets', JSON.stringify(Object.values(targets).map(v => parseFloat(v))));

      const response = await axios.post('http://localhost:8000/train', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Training failed. Please check if the API is running.');
    } finally {
      setTraining(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setImagePreview(null);
    setTabularData({
      age: '',
      num_pregnancies: '',
      age_first_intercourse: '',
      num_sexual_partners: '',
      smokes: '',
      smokes_years: '',
      hormonal_contraceptives: '',
      hormonal_contra_years: '',
      iud: '',
      iud_years: '',
      stds_any: '',
      stds_num: '',
      hpv_positive: '',
      pap_abnormal_history: '',
      immunosuppressed: ''
    });
    setTargets({
      cancer_1yr: '',
      cancer_3yr: ''
    });
    setResults(null);
    setError(null);
  };

  const fillSampleData = () => {
    setTabularData({
      age: '45',
      num_pregnancies: '2',
      age_first_intercourse: '18',
      num_sexual_partners: '3',
      smokes: '0',
      smokes_years: '0',
      hormonal_contraceptives: '1',
      hormonal_contra_years: '8',
      iud: '0',
      iud_years: '0',
      stds_any: '0',
      stds_num: '0',
      hpv_positive: '1',
      pap_abnormal_history: '0',
      immunosuppressed: '0'
    });
    setTargets({
      cancer_1yr: '0',
      cancer_3yr: '0'
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
        <p className="text-gray-600">Add training samples to improve the multimodal cancer risk prediction model</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Training Sample</h2>
            <button
              onClick={fillSampleData}
              className="flex items-center px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition"
            >
              <BookOpen className="h-4 w-4 mr-1" />
              Sample Data
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Image Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Medical Image
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition">
                {imagePreview ? (
                  <div className="space-y-4">
                    <img
                      src={imagePreview}
                      alt="Preview"
                      className="max-w-full h-32 object-contain mx-auto rounded"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setImage(null);
                        setImagePreview(null);
                      }}
                      className="text-sm text-red-600 hover:text-red-800"
                    >
                      Remove image
                    </button>
                  </div>
                ) : (
                  <div>
                    <Upload className="mx-auto h-8 w-8 text-gray-400" />
                    <div className="mt-2">
                      <label htmlFor="train-image-upload" className="cursor-pointer">
                        <span className="mt-1 block text-sm font-medium text-gray-900">
                          Upload training image
                        </span>
                      </label>
                      <input
                        id="train-image-upload"
                        name="image"
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="sr-only"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Tabular Data */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-4">
                Patient Features (Cervical Cancer)
              </label>
              <div className="grid grid-cols-2 gap-3 max-h-64 overflow-y-auto">
                {Object.entries(tabularData).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={value}
                      onChange={(e) => handleTabularChange(key, e.target.value)}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-blue-500 focus:border-transparent"
                      placeholder="0"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Target Values */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-4">
                Target Labels
              </label>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(targets).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </label>
                    <select
                      value={value}
                      onChange={(e) => handleTargetChange(key, e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select</option>
                      <option value="0">Negative (0)</option>
                      <option value="1">Positive (1)</option>
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex space-x-4">
              <button
                type="submit"
                disabled={training}
                className="flex-1 flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {training ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Training...
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Training Sample
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition"
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Results & Info */}
        <div className="space-y-6">
          {/* Training Results */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Training Results</h2>

            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                  <span className="text-red-700">{error}</span>
                </div>
              </div>
            )}

            {results ? (
              <div className="space-y-4">
                <div className="text-center">
                  <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Training Step Complete</h3>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-900 mb-3">Training Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Loss:</span>
                      <span className="text-sm font-medium text-gray-900">{results.loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Predicted 1-Year Risk:</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(results.predictions[0] * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Predicted 3-Year Risk:</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(results.predictions[1] * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Target 1-Year:</span>
                      <span className="text-sm font-medium text-gray-900">
                        {results.targets[0] === 1 ? 'Positive' : 'Negative'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Target 3-Year:</span>
                      <span className="text-sm font-medium text-gray-900">
                        {results.targets[1] === 1 ? 'Positive' : 'Negative'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="text-center">
                  <p className="text-sm text-gray-600">
                    Sample added to training. The model parameters have been updated.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Training Yet</h3>
                <p className="text-gray-500">Add training samples to improve model performance</p>
              </div>
            )}
          </div>

          {/* Training Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-blue-900 mb-2">Training Information</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Each sample performs one gradient descent step</li>
              <li>• Model parameters are updated immediately</li>
              <li>• Use multiple samples for better training</li>
              <li>• Monitor loss to ensure model is learning</li>
              <li>• Save model checkpoints periodically</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;