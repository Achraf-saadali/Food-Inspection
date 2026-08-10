/**
 * Axios HTTP client for the Food Inspection FastAPI backend.
 *
 * Backend runs at: uvicorn api:app --host 0.0.0.0 --port 8000
 * Configure VITE_API_BASE_URL in .env to override the default.
 *
 * Design: Industrial Precision — clean API abstraction layer
 */

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  // 120s timeout for the polling calls — individual poll requests are fast,
  // but the overall inspection can take 60-90s for many VLM detections.
  // The /inspect endpoint itself returns immediately with a job_id.
  timeout: 120_000,
  headers: {
    Accept: 'application/json',
  },
});

// Request interceptor — log outgoing requests in dev
apiClient.interceptors.request.use((config) => {
  if (import.meta.env.DEV) {
    console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
  }
  return config;
});

// Response interceptor — normalize errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'Unknown API error';
    return Promise.reject(new Error(message));
  }
);

export default apiClient;
