import React from 'react';
import type { CerebroBubble } from '../types/injection';

interface CerebroBubbleGraphProps {
  bubbles: CerebroBubble[];
  onSelectBubble?: (bubble: CerebroBubble) => void;
}

function getPosition(index: number, total: number, radius: number): { x: number; y: number } {
  const angle = (2 * Math.PI * index) / Math.max(1, total);
  return {
    x: radius + radius * Math.cos(angle),
    y: radius + radius * Math.sin(angle)
  };
}

export function CerebroBubbleGraph({ bubbles, onSelectBubble }: CerebroBubbleGraphProps) {
  const radius = 120;
  const maxActivation = Math.max(...bubbles.map((b) => b.activation), 1);

  return (
    <div className="cerebro-bubble-graph">
      <svg
        width={radius * 2 + 40}
        height={radius * 2 + 40}
        viewBox={`0 0 ${radius * 2 + 40} ${radius * 2 + 40}`}
      >
        <circle
          cx={radius + 20}
          cy={radius + 20}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeDasharray="6 6"
        />
        {bubbles.map((bubble, idx) => {
          const pos = getPosition(idx, bubbles.length, radius);
          const size = 6 + 10 * (bubble.activation / maxActivation);
          const tagColor =
            bubble.tag === 'risk'
              ? '#ef4444'
              : bubble.tag === 'value'
                ? '#10b981'
                : bubble.tag === 'action'
                  ? '#3b82f6'
                  : bubble.tag === 'desire'
                    ? '#a855f7'
                    : '#6b7280';

          return (
            <g
              key={bubble.id}
              transform={`translate(${pos.x + 20}, ${pos.y + 20})`}
              onClick={() => onSelectBubble?.(bubble)}
              style={{ cursor: 'pointer' }}
            >
              <circle r={size} fill={tagColor} opacity={0.75} />
              <text textAnchor="middle" y={size + 12} fontSize="10" fill="#111827">
                {bubble.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend mt-2 text-xs text-gray-600">
        <div className="flex gap-2 flex-wrap">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-red-400" /> Risk
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-green-400" /> Value
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-blue-400" /> Action
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-purple-400" /> Desire
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-gray-500" /> Other
          </span>
        </div>
      </div>
    </div>
  );
}
