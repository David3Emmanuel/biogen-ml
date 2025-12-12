import React, { useState } from 'react';
import { Upload, Image as ImageIcon, FileText, Send, AlertCircle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const Prediction = () => {
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
  const [loading, setLoading] = useState(false);
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an image');
      return;
    }

    // Check if all tabular fields are filled
    const emptyFields = Object.entries(tabularData).filter(([key, value]) => value === '');
    if (emptyFields.length > 0) {
      setError(`Please fill in all tabular fields: ${emptyFields.map(([key]) => key.replace('_', ' ')).join(', ')}`);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('tabular', JSON.stringify(Object.values(tabularData).map(v => parseFloat(v))));

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please check if the API is running.');
    } finally {
      setLoading(false);
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
    setResults(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Cancer Risk Prediction</h1>
        <p className="text-gray-600">Upload medical imaging and patient data for multimodal risk assessment</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Input Data</h2>

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
                      className="max-w-full h-48 object-contain mx-auto rounded"
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
                    <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="mt-4">
                      <label htmlFor="image-upload" className="cursor-pointer">
                        <span className="mt-2 block text-sm font-medium text-gray-900">
                          Upload medical image
                        </span>
                        <span className="mt-1 block text-sm text-gray-500">
                          PNG, JPG up to 10MB
                        </span>
                      </label>
                      <input
                        id="image-upload"
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
                Patient Information (Cervical Cancer)
              </label>
              <div className="grid grid-cols-2 gap-4">
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
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="0"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex space-x-4">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4 mr-2" />
                    Predict Risk
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

        {/* Results */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Results</h2>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-red-700">{error}</span>
              </div>
            </div>
          )}

          {results ? (
            <div className="space-y-6">
              <div className="text-center">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Analysis Complete</h3>
              </div>

              {/* Risk Scores */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-blue-900 mb-2">1-Year Risk</h4>
                  <p className="text-2xl font-bold text-blue-600">
                    {(results.predictions['1_year_risk'] * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-blue-700 mt-1">
                    Logit: {results.logits[0].toFixed(4)}
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-purple-900 mb-2">3-Year Risk</h4>
                  <p className="text-2xl font-bold text-purple-600">
                    {(results.predictions['3_year_risk'] * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-purple-700 mt-1">
                    Logit: {results.logits[1].toFixed(4)}
                  </p>
                </div>
              </div>

              {/* Risk Interpretation */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Risk Interpretation</h4>
                <div className="text-sm text-gray-700 space-y-1">
                  <p>
                    <strong>1-Year Risk:</strong> {
                      results.predictions['1_year_risk'] > 0.5 ? 'High' :
                      results.predictions['1_year_risk'] > 0.2 ? 'Moderate' : 'Low'
                    }
                  </p>
                  <p>
                    <strong>3-Year Risk:</strong> {
                      results.predictions['3_year_risk'] > 0.5 ? 'High' :
                      results.predictions['3_year_risk'] > 0.2 ? 'Moderate' : 'Low'
                    }
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex space-x-3">
                <button
                  onClick={() => window.open('/explanation', '_blank')}
                  className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
                >
                  View Explanation
                </button>
                <button
                  onClick={() => navigator.clipboard.writeText(JSON.stringify(results, null, 2))}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition"
                >
                  Copy Results
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
              <p className="text-gray-500">Upload an image and patient data to get predictions</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Prediction;