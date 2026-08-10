/**
 * Confidence score bar visualization.
 * Design: Industrial Precision — thin progress bar with monospace percentage
 */

import { cn } from '@/lib/utils';

interface ConfidenceBarProps {
  value: number; // 0.0 to 1.0
  label?: string;
  showPercent?: boolean;
  className?: string;
  size?: 'sm' | 'md';
}

function getBarColor(value: number): string {
  if (value >= 0.7) return '#22c55e';
  if (value >= 0.4) return '#f59e0b';
  return '#ef4444';
}

export function ConfidenceBar({ value, label, showPercent = true, className, size = 'md' }: ConfidenceBarProps) {
  const pct = Math.round(value * 100);
  const color = getBarColor(value);

  return (
    <div className={cn('space-y-1', className)}>
      {(label || showPercent) && (
        <div className="flex items-center justify-between">
          {label && (
            <span className="text-xs text-muted-foreground capitalize">
              {label.replace(/_/g, ' ')}
            </span>
          )}
          {showPercent && (
            <span className="text-xs font-mono" style={{ color }}>
              {pct}%
            </span>
          )}
        </div>
      )}
      <div className={cn('w-full rounded-full bg-secondary overflow-hidden', size === 'sm' ? 'h-1' : 'h-1.5')}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
