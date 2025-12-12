import React, { useState, useEffect } from 'react';
import { Activity, Users, TrendingUp, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import axios from 'axios';

const Dashboard = () => {
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get('http://localhost:8000/health');
      setHealthStatus(response.data);
    } catch (error) {
      setHealthStatus({ status: 'error', message: 'API not reachable' });
    } finally {
      setLoading(false);
    }
  };

  const stats = [
    {
      title: 'Model Status',
      value: healthStatus?.status === 'healthy' ? 'Online' : 'Offline',
      icon: healthStatus?.status === 'healthy' ? CheckCircle : AlertTriangle,
      color: healthStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600',
      bgColor: healthStatus?.status === 'healthy' ? 'bg-green-50' : 'bg-red-50'
    },
    {
      title: 'Active Sessions',
      value: '1',
      icon: Activity,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      title: 'Predictions Today',
      value: '0',
      icon: TrendingUp,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    },
    {
      title: 'Training Samples',
      value: '0',
      icon: Users,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50'
    }
  ];

  const recentActivity = [
    { id: 1, action: 'Model loaded successfully', time: '2 minutes ago', type: 'success' },
    { id: 2, action: 'API server started', time: '5 minutes ago', type: 'info' },
    { id: 3, action: 'Dataset validation completed', time: '10 minutes ago', type: 'success' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">Monitor your multimodal cancer risk prediction system</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`h-6 w-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Status */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">System Status</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">API Server</span>
              <div className="flex items-center">
                {loading ? (
                  <Clock className="h-4 w-4 text-yellow-500 mr-2" />
                ) : healthStatus?.status === 'healthy' ? (
                  <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-red-500 mr-2" />
                )}
                <span className={`text-sm font-medium ${
                  loading ? 'text-yellow-600' :
                  healthStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {loading ? 'Checking...' :
                   healthStatus?.status === 'healthy' ? 'Online' : 'Offline'}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Model Loaded</span>
              <div className="flex items-center">
                <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                <span className="text-sm font-medium text-green-600">FusedModel</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Memory Usage</span>
              <span className="text-sm font-medium text-gray-900">~2.1 GB</span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Uptime</span>
              <span className="text-sm font-medium text-gray-900">00:15:32</span>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Activity</h2>
          <div className="space-y-3">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start space-x-3">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  activity.type === 'success' ? 'bg-green-500' :
                  activity.type === 'error' ? 'bg-red-500' : 'bg-blue-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{activity.action}</p>
                  <p className="text-xs text-gray-500">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
            <Activity className="h-5 w-5 mr-2" />
            Run Health Check
          </button>
          <button className="flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition">
            <TrendingUp className="h-5 w-5 mr-2" />
            View Predictions
          </button>
          <button className="flex items-center justify-center px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition">
            <Users className="h-5 w-5 mr-2" />
            Train Model
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;