import React, { useState } from 'react';
import { PredictionResponse } from '../types/api';
import { AlertCircle, Cpu, Clock, Info, CheckCircle2, XCircle, Layers } from 'lucide-react';

interface ResultViewProps {
  prediction: PredictionResponse;
  onReset?: () => void;
}

export const ResultView: React.FC<ResultViewProps> = ({ prediction, onReset }) => {
  const [activeTab, setActiveTab] = useState<'overlay' | 'heatmap' | 'crop' | 'original'>('overlay');

  const isDeepfake = prediction.prediction === 'DEEPFAKE';
  const confidencePercent = (prediction.confidence * 100).toFixed(2);

  const getImageUrl = (url?: string) => {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    return `${url}`;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
      
      {/* Top Header & Result Badge */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 mb-1">
            <span>ID #{prediction.id}</span>
            <span>•</span>
            <span>{prediction.created_at}</span>
          </div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Analysis Report: <span className="font-mono text-slate-300">{prediction.image_name}</span>
          </h2>
        </div>

        <div className="flex items-center gap-3">
          {isDeepfake ? (
            <div className="flex items-center gap-2 px-4 py-2 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-xl font-bold tracking-wider text-sm shadow-lg shadow-rose-950/40">
              <XCircle className="w-5 h-5" />
              <span>DEEPFAKE DETECTED</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-xl font-bold tracking-wider text-sm shadow-lg shadow-emerald-950/40">
              <CheckCircle2 className="w-5 h-5" />
              <span>AUTHENTIC / REAL</span>
            </div>
          )}

          {onReset && (
            <button
              onClick={onReset}
              className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-xl transition-colors border border-slate-700"
            >
              New Analysis
            </button>
          )}
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Confidence Meter */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2">
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Classification Confidence</span>
            <span className="font-mono font-semibold text-slate-200">{confidencePercent}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-2.5 rounded-full transition-all duration-500 ${
                isDeepfake ? 'bg-rose-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
        </div>

        {/* Processing Time */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl flex items-center gap-3">
          <div className="p-2.5 bg-slate-800 text-sky-400 rounded-lg">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">Processing Time</div>
            <div className="text-sm font-bold text-slate-100 font-mono">{prediction.processing_time_ms} ms</div>
          </div>
        </div>

        {/* Model Architecture */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl flex items-center gap-3">
          <div className="p-2.5 bg-slate-800 text-purple-400 rounded-lg">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">Deploy Model</div>
            <div className="text-sm font-bold text-slate-100 font-mono">
              {prediction.model} <span className="text-xs font-normal text-slate-400">v{prediction.model_version}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Grad-CAM Visual Inspector Studio */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-sky-400" />
            Grad-CAM Visual Explainability Inspector
          </h3>

          {/* Inspector Tabs */}
          <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/60 text-xs">
            <button
              onClick={() => setActiveTab('overlay')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'overlay'
                  ? 'bg-sky-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Overlay
            </button>
            <button
              onClick={() => setActiveTab('heatmap')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'heatmap'
                  ? 'bg-sky-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Heatmap
            </button>
            <button
              onClick={() => setActiveTab('crop')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'crop'
                  ? 'bg-sky-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Face Crop
            </button>
            <button
              onClick={() => setActiveTab('original')}
              className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
                activeTab === 'original'
                  ? 'bg-sky-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Original Upload
            </button>
          </div>
        </div>

        {/* Display Frame */}
        <div className="relative aspect-video max-h-[420px] w-full bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex items-center justify-center">
          {activeTab === 'overlay' && prediction.overlay_url && (
            <img
              src={getImageUrl(prediction.overlay_url)}
              alt="Grad-CAM Overlay"
              className="h-full w-full object-contain"
            />
          )}

          {activeTab === 'heatmap' && prediction.heatmap_url && (
            <img
              src={getImageUrl(prediction.heatmap_url)}
              alt="Grad-CAM Heatmap"
              className="h-full w-full object-contain"
            />
          )}

          {activeTab === 'crop' && prediction.processed_image_url && (
            <img
              src={getImageUrl(prediction.processed_image_url)}
              alt="Processed Face Crop"
              className="h-full w-full object-contain"
            />
          )}

          {activeTab === 'original' && prediction.original_image_url && (
            <img
              src={getImageUrl(prediction.original_image_url)}
              alt="Original Image"
              className="h-full w-full object-contain"
            />
          )}
        </div>
      </div>

      {/* Grad-CAM Disclaimer Banner */}
      <div className="p-4 bg-sky-500/5 border border-sky-500/20 rounded-xl flex items-start gap-3">
        <Info className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
        <p className="text-xs text-sky-200/80 leading-relaxed">
          <strong>Explainability Notice:</strong> Grad-CAM highlights specific visual regions that strongly influenced the model's classification score. Warmer colors (red/yellow) indicate higher feature activation weights. Note that highlighted regions represent decision influence, not direct proof that a standalone region is fake.
        </p>
      </div>

    </div>
  );
};
