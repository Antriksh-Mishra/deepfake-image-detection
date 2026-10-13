import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { apiService } from '../services/api';
import { DashboardStats } from '../types/api';
import { Scan, ShieldCheck, ShieldAlert, Activity, ArrowRight, Clock, FileCheck } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const data = await apiService.getDashboardStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load dashboard stats:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      
      {/* Hero Banner */}
      <div className="relative overflow-hidden bg-gradient-to-r from-slate-900 via-slate-900 to-sky-950/40 border border-slate-800 rounded-2xl p-8 shadow-2xl">
        <div className="max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-sky-500/10 border border-sky-500/20 text-sky-400 rounded-full text-xs font-semibold">
            <Scan className="w-3.5 h-3.5 animate-spin" />
            <span>AI Deepfake Forensic Studio</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-100">
            Automated Deepfake Image Detection
          </h1>
          <p className="text-slate-400 text-sm leading-relaxed">
            Upload images for real-time deepfake analysis, facial region cropping, PyTorch classification, and Grad-CAM visual explainability.
          </p>

          <div className="pt-2">
            <NavLink
              to="/detect"
              className="inline-flex items-center gap-2 px-6 py-3 bg-sky-500 hover:bg-sky-400 text-white font-semibold rounded-xl transition-all shadow-lg shadow-sky-500/20"
            >
              <Scan className="w-5 h-5" />
              <span>Launch Detection Studio</span>
              <ArrowRight className="w-4 h-4" />
            </NavLink>
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        
        {/* Total Analyses */}
        <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl flex items-center justify-between">
          <div>
            <div className="text-xs font-medium text-slate-400">Total Analyses</div>
            <div className="text-2xl font-bold font-mono text-slate-100 mt-1">
              {loading ? '...' : stats?.total_analyses || 0}
            </div>
          </div>
          <div className="p-3 bg-sky-500/10 text-sky-400 rounded-xl border border-sky-500/20">
            <FileCheck className="w-6 h-6" />
          </div>
        </div>

        {/* Real Images */}
        <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl flex items-center justify-between">
          <div>
            <div className="text-xs font-medium text-slate-400">Authentic / Real</div>
            <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">
              {loading ? '...' : stats?.real_count || 0}
            </div>
          </div>
          <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
            <ShieldCheck className="w-6 h-6" />
          </div>
        </div>

        {/* Deepfakes Detected */}
        <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl flex items-center justify-between">
          <div>
            <div className="text-xs font-medium text-slate-400">Deepfakes Detected</div>
            <div className="text-2xl font-bold font-mono text-rose-400 mt-1">
              {loading ? '...' : stats?.deepfake_count || 0}
            </div>
          </div>
          <div className="p-3 bg-rose-500/10 text-rose-400 rounded-xl border border-rose-500/20">
            <ShieldAlert className="w-6 h-6" />
          </div>
        </div>

        {/* Average Confidence */}
        <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl flex items-center justify-between">
          <div>
            <div className="text-xs font-medium text-slate-400">Avg Confidence</div>
            <div className="text-2xl font-bold font-mono text-purple-400 mt-1">
              {loading ? '...' : `${((stats?.avg_confidence || 0) * 100).toFixed(1)}%`}
            </div>
          </div>
          <div className="p-3 bg-purple-500/10 text-purple-400 rounded-xl border border-purple-500/20">
            <Activity className="w-6 h-6" />
          </div>
        </div>

      </div>

      {/* Recent Activity Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Clock className="w-5 h-5 text-sky-400" />
            Recent Predictions
          </h2>
          <NavLink
            to="/history"
            className="text-xs font-semibold text-sky-400 hover:text-sky-300 flex items-center gap-1"
          >
            <span>View All History</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </NavLink>
        </div>

        {loading ? (
          <div className="text-center py-10 text-slate-400 text-sm">Loading recent activity...</div>
        ) : !stats || stats.recent_predictions.length === 0 ? (
          <div className="text-center py-10 text-slate-400 text-sm">No analysis history recorded yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-slate-400 uppercase bg-slate-800/50">
                <tr>
                  <th className="px-4 py-3 rounded-l-lg">File Name</th>
                  <th className="px-4 py-3">Result</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3 rounded-r-lg">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {stats.recent_predictions.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-4 py-3 font-mono font-medium text-slate-200">{item.image_name}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${
                          item.prediction === 'DEEPFAKE'
                            ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                        }`}
                      >
                        {item.prediction}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-300">{(item.confidence * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-400">{item.model}</td>
                    <td className="px-4 py-3 text-xs text-slate-400">{item.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
};
