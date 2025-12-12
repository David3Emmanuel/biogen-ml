import React, { useState } from 'react';
import { Image as ImageIcon, BarChart3, AlertCircle, Eye } from 'lucide-react';
import axios from 'axios';

const Explanation = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [targetOutput, setTargetOutput] = useState(0);
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setError('Please upload an image');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', image);

      const response = await axios.post(`http://localhost:8000/explain?target_output=${targetOutput}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Explanation failed. Please check if the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Explanation</h1>
        <p className="text-gray-600">Visualize what the model sees using Grad-CAM heatmaps</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Generate Explanation</h2>

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
                      <label htmlFor="explain-image-upload" className="cursor-pointer">
                        <span className="mt-2 block text-sm font-medium text-gray-900">
                          Upload image for explanation
                        </span>
                        <span className="mt-1 block text-sm text-gray-500">
                          PNG, JPG up to 10MB
                        </span>
                      </label>
                      <input
                        id="explain-image-upload"
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

            {/* Target Output Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Explain Prediction For
              </label>
              <select
                value={targetOutput}
                onChange={(e) => setTargetOutput(parseInt(e.target.value))}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={0}>1-Year Cancer Risk</option>
                <option value={1}>3-Year Cancer Risk</option>
              </select>
            </div>

            {/* Submit Button */}
            <div className="flex space-x-4">
              <button
                type="submit"
                disabled={loading}
                className="flex-1 flex items-center justify-center px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 mr-2" />
                    Generate Explanation
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
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Explanation Results</h2>

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
                <BarChart3 className="h-12 w-12 text-purple-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Grad-CAM Explanation</h3>
                <p className="text-sm text-gray-600">
                  Explaining: {targetOutput === 0 ? '1-Year Risk' : '3-Year Risk'}
                </p>
              </div>

              {/* Visualization */}
              {results.visualization_base64 && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-gray-900">Heatmap Overlay</h4>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <img
                      src={`data:image/png;base64,${results.visualization_base64}`}
                      alt="Grad-CAM Visualization"
                      className="max-w-full h-auto rounded-lg shadow-sm"
                    />
                  </div>
                  <p className="text-xs text-gray-600 text-center">
                    Red areas indicate regions that most influenced the model's prediction
                  </p>
                </div>
              )}

              {/* Raw Data */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-900 mb-3">Technical Details</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Target Output:</span>
                    <span className="font-medium">{results.target_output}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Heatmap Shape:</span>
                    <span className="font-medium">
                      {results.grayscale_cam ? `${results.grayscale_cam.length} × ${results.grayscale_cam[0]?.length || 0}` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Max Activation:</span>
                    <span className="font-medium">
                      {results.grayscale_cam ? Math.max(...results.grayscale_cam.flat()).toFixed(4) : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Interpretation Guide */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="text-sm font-medium text-blue-900 mb-2">How to Interpret</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• <strong>Red areas:</strong> Most important for the prediction</li>
                  <li>• <strong>Blue areas:</strong> Less influential regions</li>
                  <li>• Focus on clinically relevant anatomical structures</li>
                  <li>• Compare with domain expert interpretation</li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Explanation Yet</h3>
              <p className="text-gray-500">Upload an image to generate Grad-CAM explanations</p>
            </div>
          )}
        </div>
      </div>

      {/* Information Panel */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">About Grad-CAM Explanations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2">What is Grad-CAM?</h4>
            <p className="text-sm text-gray-700">
              Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes the regions of an image
              that are most important for a model's prediction. It uses gradients from the final convolutional
              layer to create a heatmap overlay.
            </p>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-2">Clinical Applications</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• Identify predictive image features</li>
              <li>• Validate model attention patterns</li>
              <li>• Support clinical decision making</li>
              <li>• Quality assurance and bias detection</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Explanation;