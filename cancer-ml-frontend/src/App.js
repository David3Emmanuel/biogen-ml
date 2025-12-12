import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate } from 'react-router-dom';
import { Brain, Upload, BookOpen, BarChart3, Settings, Home } from 'lucide-react';
import Dashboard from './components/Dashboard';
import Prediction from './components/Prediction';
import Training from './components/Training';
import Explanation from './components/Explanation';
import ModelManagement from './components/ModelManagement';
import './App.css';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation Header */}
        <nav className="bg-white shadow-lg border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-blue-600 mr-3" />
                <h1 className="text-xl font-bold text-gray-900">BioGen ML</h1>
                <span className="ml-2 text-sm text-gray-500">Cancer Risk Prediction</span>
              </div>
              <div className="flex space-x-8">
                <Link
                  to="/"
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-md transition"
                >
                  <Home className="h-4 w-4 mr-2" />
                  Dashboard
                </Link>
                <Link
                  to="/prediction"
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-md transition"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Prediction
                </Link>
                <Link
                  to="/training"
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-md transition"
                >
                  <BookOpen className="h-4 w-4 mr-2" />
                  Training
                </Link>
                <Link
                  to="/explanation"
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-md transition"
                >
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Explanation
                </Link>
                <Link
                  to="/management"
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-blue-600 hover:bg-gray-50 rounded-md transition"
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Management
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/prediction" element={<Prediction />} />
            <Route path="/training" element={<Training />} />
            <Route path="/explanation" element={<Explanation />} />
            <Route path="/management" element={<ModelManagement />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
