/**
 * TypeScript interfaces mirroring the Pydantic schemas from backend/schemas.py
 * Design: Industrial Precision — data types match the real backend contract exactly
 */

export type InspectionStatus = 'ok' | 'defect' | 'uncertain' | 'skipped';

export type RequiredAction = 'none' | 'flag_for_review' | 'remove';

export interface ImageSize {
  width: number;
  height: number;
}

export interface Detection {
  label: string;
  display_label?: string;
  class_id: number;
  confidence: number;
  /** Absolute pixel coordinates [x1, y1, x2, y2] */
  bbox_xyxy: [number, number, number, number];
  /** Normalized coordinates [x1, y1, x2, y2] in range [0, 1] */
  bbox_normalized: [number, number, number, number];
}

export interface QualityAssessment {
  status: InspectionStatus;
  detected_class?: string | null;
  overall_quality_score: number | null;
  /** Class-specific quality metrics, e.g. { ripeness: 0.9, bruising: 0.05 } */
  quality_metrics: Record<string, number>;
  defects: string[];
  explanation: string;
  /** Short farmer-facing interpretation derived from score, defects, and action. */
  commentary: string;
  required_action: RequiredAction;
  vlm_backend: string;
  latency_ms: number | null;
}

export interface InspectionItem {
  detection: Detection;
  quality: QualityAssessment;
}

export interface InspectionResult {
  /** Stable SQLite identifier, present for completed full inspections. */
  report_id: string | null;
  frame_id: number;
  timestamp: string;
  source: string;
  image_size: ImageSize;
  num_detections: number;
  items: InspectionItem[];
}

/** Legacy detection-only format from backend/live_inference.py / session logs */
export interface LegacyInspectionResult {
  frame_id: number;
  timestamp: string;
  source: string;
  image_size: ImageSize;
  num_detections: number;
  detections: Detection[];
}

/** Dashboard statistics derived from inspection history */
export interface InspectionStats {
  total_inspections: number;
  total_detections: number;
  ok_count: number;
  defect_count: number;
  uncertain_count: number;
  skipped_count: number;
  avg_confidence: number;
  avg_quality_score: number | null;
  top_classes: Array<{ label: string; count: number }>;
  defect_breakdown: Array<{ defect: string; count: number }>;
  action_counts: Record<RequiredAction, number>;
}

/** Model metadata */
export interface ModelInfo {
  name: string;
  version: string;
  architecture: string;
  num_classes: number;
  class_names: string[];
  input_size: string;
  training_epochs: number;
  map50: number;
  precision: number;
  recall: number;
  vlm_backends: string[];
}

/** API health check response */
export interface HealthStatus {
  status: 'ok' | 'error';
  model_loaded: boolean;
  backend_url: string;
}
