/**
 * Utility helpers for inspection data display.
 * Design: Industrial Precision — status colors match factory HMI conventions
 */

import type { InspectionStatus, RequiredAction } from '../types/inspection';

export function getStatusColor(status: InspectionStatus): string {
  switch (status) {
    case 'ok': return '#22c55e';        // emerald green
    case 'defect': return '#ef4444';    // crimson red
    case 'uncertain': return '#f59e0b'; // amber
    case 'skipped': return '#6b7280';   // slate gray
  }
}

export function getStatusLabel(status: InspectionStatus): string {
  switch (status) {
    case 'ok': return 'PASS';
    case 'defect': return 'DEFECT';
    case 'uncertain': return 'UNCERTAIN';
    case 'skipped': return 'SKIPPED';
  }
}

export function getStatusClass(status: InspectionStatus): string {
  switch (status) {
    case 'ok': return 'status-ok';
    case 'defect': return 'status-defect';
    case 'uncertain': return 'status-uncertain';
    case 'skipped': return 'status-skipped';
  }
}

export function getStatusBgClass(status: InspectionStatus): string {
  switch (status) {
    case 'ok': return 'status-bg-ok';
    case 'defect': return 'status-bg-defect';
    case 'uncertain': return 'status-bg-uncertain';
    case 'skipped': return 'status-bg-skipped';
  }
}

export function getStatusBorderClass(status: InspectionStatus): string {
  switch (status) {
    case 'ok': return 'status-border-ok';
    case 'defect': return 'status-border-defect';
    case 'uncertain': return 'status-border-uncertain';
    case 'skipped': return 'status-border-skipped';
  }
}

export function getActionLabel(action: RequiredAction): string {
  switch (action) {
    case 'none': return 'No Action';
    case 'flag_for_review': return 'Flag for Review';
    case 'remove': return 'Remove from Line';
  }
}

export function getActionColor(action: RequiredAction): string {
  switch (action) {
    case 'none': return '#6b7280';
    case 'flag_for_review': return '#f59e0b';
    case 'remove': return '#ef4444';
  }
}

export function formatConfidence(conf: number): string {
  return `${(conf * 100).toFixed(1)}%`;
}

export function formatScore(score: number | null): string {
  if (score === null) return 'N/A';
  return score.toFixed(3);
}

export function formatLatency(ms: number | null): string {
  if (ms === null) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatTimestamp(ts: string): string {
  const d = new Date(ts);
  return d.toLocaleString('en-US', {
    month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

export function formatBbox(bbox: [number, number, number, number]): string {
  return `[${bbox.map(v => v.toFixed(1)).join(', ')}]`;
}

/**
 * Scale normalized bbox coordinates to canvas/image pixel coordinates.
 * @param bbox_normalized [x1, y1, x2, y2] in [0, 1]
 * @param width  Rendered image width in pixels
 * @param height Rendered image height in pixels
 */
export function scaleBbox(
  bbox_normalized: [number, number, number, number],
  width: number,
  height: number
): { x: number; y: number; w: number; h: number } {
  const [nx1, ny1, nx2, ny2] = bbox_normalized;
  return {
    x: nx1 * width,
    y: ny1 * height,
    w: (nx2 - nx1) * width,
    h: (ny2 - ny1) * height,
  };
}
