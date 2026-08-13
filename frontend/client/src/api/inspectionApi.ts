/**
 * Inspection API service.
 *
 * Wraps all FastAPI endpoints from backend/api.py:
 *   POST /detect                    — YOLO detection only (synchronous)
 *   POST /inspect                   — Async YOLO + VLM: returns { job_id }
 *   GET  /inspect/status/{job_id}   — Poll for job result
 *   GET  /health                    — health check
 *
 * Placeholder endpoints (to be implemented in backend):
 *   GET  /api/history — inspection history log
 *   GET  /api/stats   — aggregated statistics
 *   GET  /api/model   — model metadata
 */

import type { HealthStatus, InspectionResult, InspectionStats, ModelInfo } from '../types/inspection';
import apiClient from './client';

/**
 * POST /detect
 * Fast detection-only endpoint. No VLM call. Fully synchronous.
 */
export async function detectImage(file: File): Promise<InspectionResult> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await apiClient.post<InspectionResult>('/detect', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/**
 * POST /inspect
 * Async pipeline: submits job, returns job_id immediately.
 */
export async function submitInspectJob(
  file: File,
  options?: {
    confidence_gate?: number;
  }
): Promise<{ job_id: string; status: string }> {
  const form = new FormData();
  form.append('file', file);
  const params = new URLSearchParams();
  if (options?.confidence_gate !== undefined)
    params.set('confidence_gate', String(options.confidence_gate));
  const { data } = await apiClient.post<{ job_id: string; status: string }>(
    `/inspect?${params.toString()}`,
    form,
    { headers: { 'Content-Type': 'multipart/form-data' } }
  );
  return data;
}

/**
 * GET /inspect/status/{job_id}
 * Poll for the status of an async inspection job.
 */
export async function pollInspectStatus(jobId: string): Promise<{
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: InspectionResult;
  error?: string;
}> {
  const { data } = await apiClient.get(`/inspect/status/${jobId}`);
  return data;
}

/**
 * Full async inspect flow: submit job, poll until complete.
 * Calls onProgress with the current status string for UI feedback.
 */
export async function inspectImage(
  file: File,
  options?: {
    confidence_gate?: number;
  },
  onProgress?: (stage: string) => void
): Promise<InspectionResult> {
  onProgress?.('Uploading image...');
  const { job_id } = await submitInspectJob(file, options);

  onProgress?.('YOLO detecting...');

  const POLL_INTERVAL_MS = 1000;
  const MAX_WAIT_MS = 120_000; // 2 minutes max
  const startTime = Date.now();

  while (Date.now() - startTime < MAX_WAIT_MS) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    const statusResponse = await pollInspectStatus(job_id);

    if (statusResponse.status === 'processing') {
      onProgress?.('Analyzing with VLM...');
    } else if (statusResponse.status === 'completed') {
      onProgress?.('Complete');
      if (!statusResponse.result) throw new Error('Job completed but no result returned');
      return statusResponse.result;
    } else if (statusResponse.status === 'failed') {
      throw new Error(statusResponse.error || 'Inspection job failed on the server');
    }
    // else still 'pending' — keep polling
  }

  throw new Error('Inspection timed out after 2 minutes. The server may be overloaded.');
}

/**
 * GET /health
 * Backend health check.
 */
export async function checkHealth(): Promise<HealthStatus> {
  try {
    const { data } = await apiClient.get<{ status: string }>('/health');
    return {
      status: data.status === 'ok' ? 'ok' : 'error',
      model_loaded: true,
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

/**
 * GET /api/history
 * PLACEHOLDER — Backend endpoint not yet implemented.
 * Returns mock data for frontend development.
 *
 * TODO: Implement in FastAPI:
 *   @app.get("/api/history", response_model=List[InspectionResult])
 *   async def get_history(limit: int = 50, offset: int = 0): ...
 */
export async function getInspectionHistory(
  _limit = 50,
  _offset = 0
): Promise<InspectionResult[]> {
  // Return mock data — replace with real API call when backend is ready
  return getMockHistory();
}

/**
 * GET /api/stats
 * PLACEHOLDER — Backend endpoint not yet implemented.
 *
 * TODO: Implement in FastAPI:
 *   @app.get("/api/stats", response_model=InspectionStats)
 *   async def get_stats(): ...
 */
export async function getInspectionStats(): Promise<InspectionStats> {
  return getMockStats();
}

/**
 * GET /api/model
 * PLACEHOLDER — Backend endpoint not yet implemented.
 *
 * TODO: Implement in FastAPI:
 *   @app.get("/api/model", response_model=ModelInfo)
 *   async def get_model_info(): ...
 */
export async function getModelInfo(): Promise<ModelInfo> {
  return {
    name: 'FoodScan YOLOv9c',
    version: '1.0.0',
    architecture: 'YOLOv9c (Compact)',
    num_classes: 63,
    input_size: '640 × 640',
    training_epochs: 30,
    map50: 23.8,
    precision: 34.3,
    recall: 25.3,
    vlm_backends: ['qwen2.5-vl (local)', 'gpt-4o (OpenAI)', 'qwen-vl-max (DashScope)', 'gemini-flash (OpenRouter)'],
  };
}

// ─── Mock data helpers ────────────────────────────────────────────────────────

const LABELS = [
  'apple', 'banana', 'tomato', 'broccoli', 'carrot', 'zucchini/courgette',
  'pickle', 'strawberry', 'onion', 'potato', 'orange/orange fruit',
  'bell pepper/capsicum', 'lettuce', 'cucumber', 'lemon',
];
const STATUSES: Array<'ok' | 'defect' | 'uncertain' | 'skipped'> = ['ok', 'ok', 'ok', 'defect', 'uncertain', 'skipped'];
const ACTIONS: Array<'none' | 'flag_for_review' | 'remove'> = ['none', 'none', 'flag_for_review', 'remove'];
const DEFECTS = ['minor_bruising', 'mold_spot', 'discoloration', 'shriveling', 'cracking', 'soft_spots'];

function rand(min: number, max: number) { return Math.random() * (max - min) + min; }
function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)]; }

function makeMockItem(label?: string): import('../types/inspection').InspectionItem {
  const lbl = label || pick(LABELS);
  const conf = rand(0.4, 0.98);
  const status = pick(STATUSES);
  const x1 = rand(50, 200), y1 = rand(50, 200);
  const x2 = x1 + rand(80, 200), y2 = y1 + rand(80, 200);
  const w = 640, h = 480;
  const defects = status === 'defect' ? [pick(DEFECTS)] : [];
  const qualityMetrics: Record<string, number> = {
    ripeness: rand(0.3, 1.0),
    freshness: rand(0.3, 1.0),
    bruising: rand(0.0, 0.5),
    mold: rand(0.0, 0.3),
  };
  return {
    detection: {
      label: lbl,
      class_id: LABELS.indexOf(lbl),
      confidence: conf,
      bbox_xyxy: [x1, y1, x2, y2],
      bbox_normalized: [x1 / w, y1 / h, x2 / w, y2 / h],
    },
    quality: {
      status,
      overall_quality_score: status === 'skipped' ? null : rand(0.3, 1.0),
      quality_metrics: status === 'skipped' ? {} : qualityMetrics,
      defects,
      explanation: status === 'ok'
        ? 'Item appears fresh with no visible defects.'
        : status === 'defect'
        ? `Visible ${defects[0]?.replace('_', ' ')} detected on surface.`
        : status === 'uncertain'
        ? 'Lighting conditions make assessment inconclusive.'
        : 'VLM reasoning skipped (low confidence or VLM disabled).',
      required_action: status === 'defect' ? pick(['flag_for_review', 'remove'] as const) : 'none',
      vlm_backend: pick(['gpt-4o', 'qwen-api', 'openrouter', 'none']),
      latency_ms: status === 'skipped' ? null : rand(80, 800),
    },
  };
}

function makeMockResult(frameId: number, hoursAgo: number): InspectionResult {
  const numItems = Math.floor(rand(0, 4));
  const items = Array.from({ length: numItems }, () => makeMockItem());
  const ts = new Date(Date.now() - hoursAgo * 3_600_000);
  return {
    frame_id: frameId,
    timestamp: ts.toISOString(),
    source: 'webcam_0',
    image_size: { width: 640, height: 480 },
    num_detections: numItems,
    items,
  };
}

function getMockHistory(): InspectionResult[] {
  return Array.from({ length: 40 }, (_, i) => makeMockResult(1000 - i, i * 0.5));
}

function getMockStats(): InspectionStats {
  const history = getMockHistory();
  const allItems = history.flatMap((r) => r.items);
  const counts = allItems.reduce(
    (acc, item) => {
      acc[item.quality.status] = (acc[item.quality.status] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );
  const labelCounts = allItems.reduce((acc, item) => {
    acc[item.detection.label] = (acc[item.detection.label] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  const topClasses = Object.entries(labelCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label, count]) => ({ label, count }));
  const avgConf =
    allItems.length > 0
      ? allItems.reduce((s, i) => s + i.detection.confidence, 0) / allItems.length
      : 0;
  return {
    total_inspections: history.length,
    total_detections: allItems.length,
    ok_count: counts['ok'] || 0,
    defect_count: counts['defect'] || 0,
    uncertain_count: counts['uncertain'] || 0,
    skipped_count: counts['skipped'] || 0,
    avg_confidence: avgConf,
    top_classes: topClasses,
  };
}
