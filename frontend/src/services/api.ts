import axios from 'axios';
import {
  PredictionResponse,
  PredictionHistoryResponse,
  DashboardStats,
  ModelInfo
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const getMediaUrl = (url?: string): string => {
  if (!url) return '';
  if (url.startsWith('http')) return url;
  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  if (baseUrl.startsWith('http')) {
    const host = baseUrl.replace(/\/api.*$/, '');
    return `${host}${url.startsWith('/') ? '' : '/'}${url}`;
  }
  return url;
};

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Accept': 'application/json',
  },
});

export const apiService = {
  async predictImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<PredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getHistory(
    page: number = 1,
    size: number = 10,
    filter?: string,
    search?: string
  ): Promise<PredictionHistoryResponse> {
    const params: Record<string, any> = { page, size };
    if (filter && filter !== 'ALL') params.filter = filter;
    if (search) params.search = search;

    const response = await apiClient.get<PredictionHistoryResponse>('/predictions', { params });
    return response.data;
  },

  async getPredictionById(id: number): Promise<PredictionResponse> {
    const response = await apiClient.get<PredictionResponse>(`/predictions/${id}`);
    return response.data;
  },

  async deletePrediction(id: number): Promise<void> {
    await apiClient.delete(`/predictions/${id}`);
  },

  async getDashboardStats(): Promise<DashboardStats> {
    const response = await apiClient.get<DashboardStats>('/stats');
    return response.data;
  },

  async getModelInfo(): Promise<ModelInfo> {
    const response = await apiClient.get<ModelInfo>('/model');
    return response.data;
  },

  async getHealth(): Promise<{ status: string; model: string }> {
    const response = await apiClient.get('/health');
    return response.data;
  },
};
