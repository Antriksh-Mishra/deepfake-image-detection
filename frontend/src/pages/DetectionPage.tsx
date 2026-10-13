import React, { useState, useRef } from 'react';
import { apiService } from '../services/api';
import { PredictionResponse } from '../types/api';
import { ResultView } from '../components/ResultView';
import { ToastContainer, ToastMessage } from '../components/Toast';
import { UploadCloud, FileImage, Scan, AlertCircle, X, CheckCircle2 } from 'lucide-react';

export const DetectionPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addToast = (type: 'success' | 'error' | 'warning', title: string, message?: string) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, type, title, message }]);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const handleFileSelect = (file: File) => {
    const validFormats = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validFormats.includes(file.type)) {
      addToast('error', 'Unsupported File Format', 'Please select a JPG, PNG, or WEBP image.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      addToast('error', 'File Too Large', 'Maximum file size allowed is 10 MB.');
      return;
    }

    setSelectedFile(file);
    setResult(null);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    try {
      setLoading(true);
      const data = await apiService.predictImage(selectedFile);
      setResult(data);
      addToast('success', 'Analysis Complete', `Classified as ${data.prediction} with ${(data.confidence * 100).toFixed(1)}% confidence.`);
    } catch (err: any) {
      const detail = err.response?.data?.detail || 'An unexpected error occurred during analysis.';
      const errorCode = err.response?.data?.error_code || 'PREDICTION_FAILED';
      addToast('error', `Analysis Failed (${errorCode})`, detail);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <ToastContainer toasts={toasts} onDismiss={removeToast} />

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
          <Scan className="w-6 h-6 text-sky-400" />
          Detection Studio
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Upload an image to detect AI manipulations, face crops, and generate Grad-CAM explainability heatmaps.
        </p>
      </div>

      {result ? (
        <ResultView prediction={result} onReset={handleReset} />
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          
          {/* Dropzone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
              isDragOver
                ? 'border-sky-500 bg-sky-500/10'
                : selectedFile
                ? 'border-slate-700 bg-slate-950/40'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/20'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            {previewUrl ? (
              <div className="space-y-4">
                <div className="relative max-h-64 mx-auto inline-block rounded-xl overflow-hidden border border-slate-700 shadow-xl">
                  <img src={previewUrl} alt="Preview" className="max-h-64 object-contain" />
                </div>
                <div className="flex items-center justify-center gap-2 text-xs font-mono text-slate-400">
                  <FileImage className="w-4 h-4 text-sky-400" />
                  <span>{selectedFile?.name}</span>
                  <span>({((selectedFile?.size || 0) / (1024 * 1024)).toFixed(2)} MB)</span>
                </div>
              </div>
            ) : (
              <div className="space-y-3 py-6">
                <div className="p-4 bg-sky-500/10 text-sky-400 rounded-full inline-block border border-sky-500/20">
                  <UploadCloud className="w-8 h-8" />
                </div>
                <div>
                  <p className="text-base font-semibold text-slate-200">
                    Drag & Drop Image or <span className="text-sky-400 underline">Browse Files</span>
                  </p>
                  <p className="text-xs text-slate-400 mt-1">Supports JPG, PNG, WEBP (Max 10 MB)</p>
                </div>
              </div>
            )}
          </div>

          {/* Action Trigger */}
          {selectedFile && (
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={handleReset}
                disabled={loading}
                className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium rounded-xl transition-colors"
              >
                Clear File
              </button>

              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="px-6 py-2.5 bg-sky-500 hover:bg-sky-400 disabled:opacity-50 text-white font-semibold text-sm rounded-xl transition-all shadow-lg shadow-sky-500/20 flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <Scan className="w-4 h-4 animate-spin" />
                    <span>Analyzing Image...</span>
                  </>
                ) : (
                  <>
                    <Scan className="w-4 h-4" />
                    <span>Analyze Image</span>
                  </>
                )}
              </button>
            </div>
          )}

        </div>
      )}

    </div>
  );
};
