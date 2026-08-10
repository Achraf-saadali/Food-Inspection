/**
 * Card displaying a single detected item with quality assessment.
 * Design: Industrial Precision — status border strip, monospace data
 */

import { cn } from '@/lib/utils';
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, HelpCircle, SkipForward } from 'lucide-react';
import { useState } from 'react';
import type { InspectionItem } from '../../types/inspection';
import {
  getStatusBorderClass,
  getStatusColor,
  getStatusLabel,
  getActionLabel,
  getActionColor,
  formatConfidence,
  formatScore,
  formatLatency,
  formatBbox,
} from '../../utils/inspection';
import { StatusBadge } from './StatusBadge';
import { ConfidenceBar } from './ConfidenceBar';

interface DetectionCardProps {
  item: InspectionItem;
  index: number;
  className?: string;
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'ok': return <CheckCircle className="w-4 h-4 text-[#22c55e]" />;
    case 'defect': return <AlertTriangle className="w-4 h-4 text-[#ef4444]" />;
    case 'uncertain': return <HelpCircle className="w-4 h-4 text-[#f59e0b]" />;
    default: return <SkipForward className="w-4 h-4 text-[#6b7280]" />;
  }
}

export function DetectionCard({ item, index, className }: DetectionCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { detection: det, quality: q } = item;
  const borderClass = getStatusBorderClass(q.status);
  const statusColor = getStatusColor(q.status);
  const metricEntries = Object.entries(q.quality_metrics);

  return (
    <div
      className={cn(
        'rounded border border-border bg-card border-l-2 overflow-hidden transition-all duration-200',
        borderClass,
        className
      )}
      style={{ animationDelay: `${index * 40}ms` }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-secondary/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <StatusIcon status={q.status} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-foreground capitalize truncate" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
              {det.label}
            </span>
            <span className="text-xs text-muted-foreground font-mono">#{det.class_id}</span>
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-xs font-mono text-muted-foreground">
              conf: <span style={{ color: statusColor }}>{formatConfidence(det.confidence)}</span>
            </span>
            {q.overall_quality_score !== null && (
              <span className="text-xs font-mono text-muted-foreground">
                score: <span style={{ color: statusColor }}>{formatScore(q.overall_quality_score)}</span>
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusBadge status={q.status} size="sm" pulse={q.status === 'defect'} />
          {expanded ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-border">
          {/* Explanation */}
          {q.explanation && (
            <div className="pt-3">
              <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">Assessment</p>
              <p className="text-sm text-foreground leading-relaxed">{q.explanation}</p>
            </div>
          )}

          {/* Required action */}
          {q.required_action !== 'none' && (
            <div className="flex items-center gap-2 px-3 py-2 rounded border" style={{ borderColor: `${getActionColor(q.required_action)}40`, backgroundColor: `${getActionColor(q.required_action)}10` }}>
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" style={{ color: getActionColor(q.required_action) }} />
              <span className="text-xs font-semibold" style={{ color: getActionColor(q.required_action) }}>
                {getActionLabel(q.required_action)}
              </span>
            </div>
          )}

          {/* Defects */}
          {q.defects.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Defects Detected</p>
              <div className="flex flex-wrap gap-1.5">
                {q.defects.map((d) => (
                  <span key={d} className="text-xs px-2 py-0.5 rounded font-mono bg-[#ef444420] text-[#ef4444] border border-[#ef444440]">
                    {d.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Quality metrics */}
          {metricEntries.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground mb-2 uppercase tracking-wider">Quality Metrics</p>
              <div className="space-y-2">
                {metricEntries.map(([key, val]) => (
                  <ConfidenceBar key={key} value={val} label={key} size="sm" />
                ))}
              </div>
            </div>
          )}

          {/* Bounding box & metadata */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">Bounding Box</p>
              <p className="text-xs font-mono text-foreground">{formatBbox(det.bbox_xyxy)}</p>
              <p className="text-xs font-mono text-muted-foreground mt-0.5">
                norm: {formatBbox(det.bbox_normalized)}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">Pipeline</p>
              <p className="text-xs font-mono text-foreground">VLM: {q.vlm_backend}</p>
              <p className="text-xs font-mono text-muted-foreground mt-0.5">
                latency: {formatLatency(q.latency_ms)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
