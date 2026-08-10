/**
 * Status badge for inspection results.
 * Design: Industrial Precision — color-coded, monospace label
 */

import { cn } from '@/lib/utils';
import type { InspectionStatus } from '../../types/inspection';
import { getStatusColor, getStatusLabel, getStatusBgClass } from '../../utils/inspection';

interface StatusBadgeProps {
  status: InspectionStatus;
  size?: 'sm' | 'md' | 'lg';
  pulse?: boolean;
  className?: string;
}

export function StatusBadge({ status, size = 'md', pulse = false, className }: StatusBadgeProps) {
  const color = getStatusColor(status);
  const label = getStatusLabel(status);
  const bgClass = getStatusBgClass(status);

  const sizeClasses = {
    sm: 'text-xs px-1.5 py-0.5',
    md: 'text-xs px-2 py-1',
    lg: 'text-sm px-3 py-1.5',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded font-mono font-medium border',
        sizeClasses[size],
        bgClass,
        pulse && status === 'defect' ? 'defect-pulse' : '',
        className
      )}
      style={{ color, borderColor: `${color}40` }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full shrink-0"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}
