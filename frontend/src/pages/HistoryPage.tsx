import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import { PredictionResponse, PredictionHistoryResponse } from '../types/api';
import { ResultView } from '../components/ResultView';
import { ToastContainer, ToastMessage } from '../components/Toast';
import { History as HistoryIcon, Search, Filter, Trash2, Eye, ChevronLeft, ChevronRight, X } from 'lucide-react';

export const HistoryPage: React.FC = () => {
  const [data, setData] = useState<PredictionHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<string>('ALL');
  const [search, setSearch] = useState<string>('');
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionResponse | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const addToast = (type: 'success' | 'error' | 'warning', title: string, message?: string) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, title, message }]);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  useEffect(() => {
    fetchHistory();
  }, [page, filter]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const res = await apiService.getHistory(page, 10, filter, search);
      setData(res);
    } catch (err) {
      console.error('Failed to load prediction history:', err);
      addToast('error', 'Error Loading History', 'Could not retrieve prediction records.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchHistory();
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm(`Are you sure you want to delete prediction #${id}?`)) return;
    try {
      await apiService.deletePrediction(id);
      addToast('success', 'Record Deleted', `Prediction #${id} removed successfully.`);
      fetchHistory();
      if (selectedPrediction?.id === id) {
        setSelectedPrediction(null);
      }
    } catch (err) {
      addToast('error', 'Delete Failed', 'Could not delete prediction record.');
    }
  };

  return (
    <div className="space-y-6">
      <ToastContainer toasts={toasts} onDismiss={removeToast} />

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <HistoryIcon className="w-6 h-6 text-sky-400" />
          Prediction History Log
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Search, filter, inspect, and manage past deepfake detection reports.
        </p>
      </div>

      {/* Controls Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl">
        
        {/* Search */}
        <form onSubmit={handleSearchSubmit} className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search image name..."
            className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
          />
        </form>

        {/* Filter Pills */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          {['ALL', 'REAL', 'DEEPFAKE'].map((f) => (
            <button
              key={f}
              onClick={() => {
                setFilter(f);
                setPage(1);
              }}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                filter === f
                  ? 'bg-sky-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* History Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
        {loading ? (
          <div className="text-center py-16 text-slate-400 text-sm">Loading history records...</div>
        ) : !data || data.items.length === 0 ? (
          <div className="text-center py-16 text-slate-400 text-sm">No predictions found matching criteria.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs text-slate-400 uppercase bg-slate-950/60 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3.5">ID</th>
                  <th className="px-4 py-3.5">Image Name</th>
                  <th className="px-4 py-3.5">Prediction</th>
                  <th className="px-4 py-3.5">Confidence</th>
                  <th className="px-4 py-3.5">Model</th>
                  <th className="px-4 py-3.5">Latency</th>
                  <th className="px-4 py-3.5">Date</th>
                  <th className="px-4 py-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {data.items.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="px-4 py-3.5 font-mono text-xs text-slate-400">#{item.id}</td>
                    <td className="px-4 py-3.5 font-mono font-medium text-slate-200">{item.image_name}</td>
                    <td className="px-4 py-3.5">
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
                    <td className="px-4 py-3.5 font-mono text-slate-300">{(item.confidence * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3.5 font-mono text-xs text-slate-400">{item.model}</td>
                    <td className="px-4 py-3.5 font-mono text-xs text-slate-400">{item.processing_time_ms}ms</td>
                    <td className="px-4 py-3.5 text-xs text-slate-400">{item.created_at}</td>
                    <td className="px-4 py-3.5 text-right space-x-2">
                      <button
                        onClick={() => setSelectedPrediction(item)}
                        className="p-1.5 text-sky-400 hover:text-sky-300 hover:bg-sky-500/10 rounded-lg transition-colors"
                        title="View Report"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(item.id)}
                        className="p-1.5 text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 rounded-lg transition-colors"
                        title="Delete Record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Footer */}
        {data && data.total_pages > 1 && (
          <div className="p-4 bg-slate-950/40 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <div>
              Showing page <span className="font-semibold text-slate-200">{data.page}</span> of{' '}
              <span className="font-semibold text-slate-200">{data.total_pages}</span> ({data.total} total items)
            </div>
            <div className="flex items-center gap-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="p-2 bg-slate-800 disabled:opacity-40 text-slate-200 rounded-lg border border-slate-700 hover:bg-slate-700"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                disabled={page >= data.total_pages}
                onClick={() => setPage((p) => p + 1)}
                className="p-2 bg-slate-800 disabled:opacity-40 text-slate-200 rounded-lg border border-slate-700 hover:bg-slate-700"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Modal Detail Drawer */}
      {selectedPrediction && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="relative max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setSelectedPrediction(null)}
              className="absolute top-4 right-4 z-10 p-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-full border border-slate-700"
            >
              <X className="w-5 h-5" />
            </button>
            <ResultView prediction={selectedPrediction} onReset={() => setSelectedPrediction(null)} />
          </div>
        </div>
      )}

    </div>
  );
};
