/**
 * SVG overlay for drawing bounding boxes on top of an image.
 * Uses bbox_normalized for resolution-independent rendering.
 * Design: Industrial Precision — colored boxes with status-coded labels
 */

import type { InspectionItem } from '../../types/inspection';
import { getStatusColor, getStatusLabel, formatConfidence } from '../../utils/inspection';

interface BoundingBoxOverlayProps {
  items: InspectionItem[];
  width: number;
  height: number;
}

export function BoundingBoxOverlay({ items, width, height }: BoundingBoxOverlayProps) {
  if (!items.length) return null;

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ overflow: 'visible' }}
    >
      {items.map((item, i) => {
        const { detection: det, quality: q } = item;
        const [nx1, ny1, nx2, ny2] = det.bbox_normalized;
        const x = nx1 * width;
        const y = ny1 * height;
        const w = (nx2 - nx1) * width;
        const h = (ny2 - ny1) * height;
        const color = getStatusColor(q.status);
        const label = `${det.label} ${formatConfidence(det.confidence)} [${getStatusLabel(q.status)}]`;
        const labelY = y > 24 ? y - 6 : y + h + 16;

        return (
          <g key={i}>
            {/* Box */}
            <rect
              x={x} y={y} width={w} height={h}
              fill={`${color}18`}
              stroke={color}
              strokeWidth={2}
              rx={2}
            />
            {/* Label background */}
            <rect
              x={x} y={labelY - 14}
              width={Math.min(label.length * 6.5 + 8, width - x)}
              height={18}
              fill={color}
              rx={2}
            />
            {/* Label text */}
            <text
              x={x + 4}
              y={labelY}
              fill="#000"
              fontSize={10}
              fontFamily="'JetBrains Mono', monospace"
              fontWeight="500"
            >
              {label}
            </text>
            {/* Corner markers */}
            <line x1={x} y1={y} x2={x + 8} y2={y} stroke={color} strokeWidth={2.5} />
            <line x1={x} y1={y} x2={x} y2={y + 8} stroke={color} strokeWidth={2.5} />
            <line x1={x + w} y1={y} x2={x + w - 8} y2={y} stroke={color} strokeWidth={2.5} />
            <line x1={x + w} y1={y} x2={x + w} y2={y + 8} stroke={color} strokeWidth={2.5} />
            <line x1={x} y1={y + h} x2={x + 8} y2={y + h} stroke={color} strokeWidth={2.5} />
            <line x1={x} y1={y + h} x2={x} y2={y + h - 8} stroke={color} strokeWidth={2.5} />
            <line x1={x + w} y1={y + h} x2={x + w - 8} y2={y + h} stroke={color} strokeWidth={2.5} />
            <line x1={x + w} y1={y + h} x2={x + w} y2={y + h - 8} stroke={color} strokeWidth={2.5} />
          </g>
        );
      })}
    </svg>
  );
}

