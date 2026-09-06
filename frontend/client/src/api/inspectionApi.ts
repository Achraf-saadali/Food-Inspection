/**
 * Inspection API service for the FastAPI application in backend/api.py.
 * Full inspections are persisted locally and report functions return only
 * completed SQLite-backed results; this module contains no mock data.
 */

import type {
  FarmerReportResponse,
  HealthStatus,
  InspectionResult,
  InspectionStats,
  ModelInfo,
} from '../types/inspection';
import apiClient from './client';

export async function detectImage(file: File, signal?: AbortSignal): Promise<InspectionResult> {
  const form = new FormData();
  form.append('file', file);
  // Do not set Content-Type manually: the browser adds the multipart boundary.
  const { data } = await apiClient.post<InspectionResult>('/detect', form, { signal });
  return data;
}

export async function submitInspectJob(
  file: File,
  options?: { confidence_gate?: number; vlm_backend?: string; vlm_model?: string }
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append('file', file);
  const params = new URLSearchParams();
  if (options?.confidence_gate !== undefined) {
    params.set('confidence_gate', String(options.confidence_gate));
  }
  if (options?.vlm_backend) params.set('vlm_backend', options.vlm_backend);
  if (options?.vlm_model) params.set('vlm_model', options.vlm_model);
  const { data } = await apiClient.post<{ job_id: string; status: string }>(
    `/inspect?${params.toString()}`,
    form
  );
  return data;
}

export async function pollInspectStatus(jobId: string, signal?: AbortSignal): Promise<{
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: InspectionResult;
  error?: string;
}> {
  const { data } = await apiClient.get(`/inspect/status/${jobId}`, { signal });
  return data;
}

/** Submit a full inspection and poll until the persisted result is ready. */
export async function inspectImage(
  file: File,
  options?: { confidence_gate?: number; vlm_backend?: string; vlm_model?: string; signal?: AbortSignal },
  onProgress?: (stage: string) => void,
): Promise<InspectionResult> {
  onProgress?.('Téléversement de l’image…');
  const { job_id } = await submitInspectJob(file, options);
  onProgress?.('Détection YOLO…');

  const pollIntervalMs = 1000;
  const maxWaitMs = 120_000;
  const startDate = Date.now();

  while (Date.now() - startDate < maxWaitMs) {
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(resolve, pollIntervalMs);
      options?.signal?.addEventListener('abort', () => {
        window.clearTimeout(timeout);
        reject(new DOMException('Inspection annulée', 'AbortError'));
      }, { once: true });
    });
    const response = await pollInspectStatus(job_id, options?.signal);
    if (response.status === 'processing') {
      onProgress?.('Analyse avec le VLM…');
      continue;
    }
    if (response.status === 'completed') {
      onProgress?.('Terminé');
      if (!response.result) throw new Error('L’inspection est terminée sans résultat.');
      return response.result;
    }
    if (response.status === 'failed') {
      throw new Error(response.error || 'La tâche d’inspection a échoué sur le serveur.');
    }
  }

  throw new Error('L’inspection a dépassé deux minutes. Le serveur est peut-être surchargé.');
}

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const { data } = await apiClient.get<{ status: string; model_loaded?: boolean }>('/health');
    return {
      status: data.status === 'ok' ? 'ok' : 'error',
      model_loaded: Boolean(data.model_loaded),
      backend_url: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    };
  } catch {
    return {
      status: 'error',
      model_loaded: false,
      backend_url: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    };
  }
}

/** Return real completed reports saved by the full inspection endpoint. */
export async function getInspectionHistory(
  limit = 100,
  offset = 0,
  options?: { status?: string; search?: string }
): Promise<InspectionResult[]> {
  const { data } = await apiClient.get<InspectionResult[]>('/reports', {
    params: { limit, offset, ...options },
  });
  return data;
}

/** Return the farmer-facing report for one persisted inspection. */
export async function getFarmerReport(reportId: string): Promise<FarmerReportResponse> {
  const { data } = await apiClient.get<FarmerReportResponse>(
    `/reports/${encodeURIComponent(reportId)}/farmer-report`,
  );
  return data;
}

/** Return dashboard metrics calculated only from persisted reports. */
export async function getInspectionStats(): Promise<InspectionStats> {
  const { data } = await apiClient.get<InspectionStats>('/reports/summary');
  return data;
}

/** Return actual metadata exposed by the backend rather than local placeholder values. */
export async function getModelInfo(): Promise<ModelInfo> {
  const { data } = await apiClient.get<ModelInfo>('/model-info');
  return data;
}

/** Download the complete persisted report set as a JSON file. */
export async function downloadReportExport(): Promise<void> {
  const { data } = await apiClient.get<InspectionResult[]>('/reports/export');
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `food-inspection-reports-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
