export interface PredictionResponse {
  id: number;
  image_name: str;
  prediction: 'REAL' | 'DEEPFAKE';
  confidence: number;
  model: string;
  model_version: string;
  processing_time_ms: number;
  original_image_url?: string;
  processed_image_url?: string;
  heatmap_url?: string;
  overlay_url?: string;
  created_at: string;
}

export interface PredictionHistoryResponse {
  items: PredictionResponse[];
  total: number;
  page: number;
  size: number;
  total_pages: number;
}

export interface DashboardStats {
  total_analyses: number;
  real_count: number;
  deepfake_count: number;
  avg_confidence: number;
  recent_predictions: PredictionResponse[];
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  image_size: number;
  confidence_threshold: number;
  supported_formats: string[];
  max_upload_size_mb: number;
}

export interface ApiError {
  detail: string;
  error_code?: string;
  status_code?: number;
}
