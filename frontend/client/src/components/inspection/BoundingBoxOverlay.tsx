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
  sourceWidth?: number;
  sourceHeight?: number;
}

export function BoundingBoxOverlay({ items, width, height, sourceWidth = width, sourceHeight = height }: BoundingBoxOverlayProps) {
  if (!items.length) return null;

  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const contentWidth = sourceWidth * scale;
  const contentHeight = sourceHeight * scale;
  const offsetX = (width - contentWidth) / 2;
  const offsetY = (height - contentHeight) / 2;

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
        const x = offsetX + nx1 * contentWidth;
        const y = offsetY + ny1 * contentHeight;
        const w = (nx2 - nx1) * contentWidth;
        const h = (ny2 - ny1) * contentHeight;
        const color = getStatusColor(q.status);
        const label = `${det.display_label || det.label} ${formatConfidence(det.confidence)} [${getStatusLabel(q.status)}]`;
        const labelY = y > 24 ? y - 6 : y + h + 16;
        const commentary = q.commentary || q.explanation;
        const commentaryLines = commentary.match(/.{1,48}(?:\s|$)/g)?.slice(0, 3) ?? [];
        const commentaryY = y + h + 34;

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
            {commentaryLines.map((line, lineIndex) => (
              <text
                key={`${i}-commentary-${lineIndex}`}
                x={x + 4}
                y={commentaryY + lineIndex * 13}
                fill={color}
                fontSize={9}
                fontFamily="'JetBrains Mono', monospace"
              >
                {line.trim()}
              </text>
            ))}
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

