/**
 * InformationTheoryPanel - Visualize Information Bottleneck metrics during training
 *
 * Displays:
 * - I(X;Z) vs I(Z;Y) information plane
 * - Compression-prediction trade-off curve
 * - Current beta value and schedule
 * - Entropy metrics H(Z), H(Z|X)
 */

import React from 'react';
import type { InformationMetrics } from '../losses/information_bottleneck';

export interface InformationTheoryPanelProps {
  /** History of information metrics collected during training */
  metricsHistory: InformationMetrics[];
  /** Whether training is currently active */
  isTraining: boolean;
  /** Current epoch number */
  currentEpoch: number;
  /** Total epochs */
  totalEpochs: number;
  /** Direction for RTL support */
  direction?: 'ltr' | 'rtl';
}

/**
 * Format number with fixed precision
 */
function formatNumber(value: number, precision = 4): string {
  return value.toFixed(precision);
}

/**
 * Get color for compression-prediction trade-off visualization
 */
function getTradeoffColor(compressionMI: number, predictionMI: number): string {
  // More red = more compression (low I(X;Z))
  // More blue = more prediction (high I(Z;Y))
  const compressionIntensity = Math.min(1, compressionMI / 2);
  const predictionIntensity = Math.min(1, predictionMI / 2);

  const r = Math.floor(255 * (1 - compressionIntensity));
  const g = 100;
  const b = Math.floor(255 * predictionIntensity);

  return `rgb(${r}, ${g}, ${b})`;
}

export function InformationTheoryPanel({
  metricsHistory,
  isTraining,
  currentEpoch,
  totalEpochs,
  direction = 'ltr'
}: InformationTheoryPanelProps) {
  // Get latest metrics
  const latestMetrics =
    metricsHistory.length > 0 ? metricsHistory[metricsHistory.length - 1] : null;

  // Compute min/max for scaling
  const allCompressionMI = metricsHistory.map((m) => m.compressionMI);
  const allPredictionMI = metricsHistory.map((m) => m.predictionMI);
  const minCompression = Math.min(...allCompressionMI, 0);
  const maxCompression = Math.max(...allCompressionMI, 1);
  const minPrediction = Math.min(...allPredictionMI, 0);
  const maxPrediction = Math.max(...allPredictionMI, 1);

  return (
    <div
      role="region"
      aria-label="Information Theory Metrics"
      style={{
        background: 'rgba(30, 41, 59, 0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        marginTop: 16,
        direction
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 16 }}>
        <div
          style={{
            fontSize: 14,
            fontWeight: 700,
            color: '#a78bfa',
            marginBottom: 4,
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}
        >
          <span>📊 Information Bottleneck Metrics</span>
          {isTraining && (
            <span style={{ fontSize: 11, color: '#10b981', fontWeight: 600 }}>● ACTIVE</span>
          )}
        </div>
        <div style={{ fontSize: 11, color: '#94a3b8' }}>
          Information-theoretic analysis of representation learning
        </div>
      </div>

      {/* Current Metrics */}
      {latestMetrics && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: 12,
            marginBottom: 16
          }}
        >
          {/* I(X;Z) - Compression */}
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#fca5a5', marginBottom: 4, fontWeight: 600 }}>
              I(X;Z) Compression
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#ef4444' }}>
              {formatNumber(latestMetrics.compressionMI, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>Input-Hidden MI</div>
          </div>

          {/* I(Z;Y) - Prediction */}
          <div
            style={{
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#93c5fd', marginBottom: 4, fontWeight: 600 }}>
              I(Z;Y) Prediction
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#3b82f6' }}>
              {formatNumber(latestMetrics.predictionMI, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>Hidden-Output MI</div>
          </div>

          {/* IB Loss */}
          <div
            style={{
              background: 'rgba(168, 85, 247, 0.1)',
              border: '1px solid rgba(168, 85, 247, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#c084fc', marginBottom: 4, fontWeight: 600 }}>
              IB Loss
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#a855f7' }}>
              {formatNumber(latestMetrics.ibLoss, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>-I(Z;Y) + β·I(X;Z)</div>
          </div>

          {/* Beta */}
          <div
            style={{
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#6ee7b7', marginBottom: 4, fontWeight: 600 }}>
              β (Beta)
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#10b981' }}>
              {formatNumber(latestMetrics.beta, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>Trade-off param</div>
          </div>

          {/* H(Z) */}
          <div
            style={{
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#fcd34d', marginBottom: 4, fontWeight: 600 }}>
              H(Z) Entropy
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#f59e0b' }}>
              {formatNumber(latestMetrics.representationEntropy, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>Hidden entropy</div>
          </div>

          {/* H(Z|X) */}
          <div
            style={{
              background: 'rgba(139, 92, 246, 0.1)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: 8,
              padding: 12
            }}
          >
            <div style={{ fontSize: 10, color: '#c4b5fd', marginBottom: 4, fontWeight: 600 }}>
              H(Z|X) Conditional
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#8b5cf6' }}>
              {formatNumber(latestMetrics.conditionalEntropy, 3)}
            </div>
            <div style={{ fontSize: 9, color: '#94a3b8', marginTop: 2 }}>H(Z) - I(X;Z)</div>
          </div>
        </div>
      )}

      {/* Information Plane Visualization */}
      {metricsHistory.length > 0 && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid #1e293b',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: '#cbd5e1', marginBottom: 12 }}>
            📈 Information Plane: I(X;Z) vs I(Z;Y)
          </div>

          {/* Simple ASCII-style visualization */}
          <div
            style={{
              position: 'relative',
              height: 200,
              background: 'rgba(0, 0, 0, 0.3)',
              borderRadius: 8,
              padding: 8,
              fontFamily: 'monospace',
              fontSize: 10
            }}
          >
            {/* Y-axis label */}
            <div
              style={{
                position: 'absolute',
                left: 4,
                top: '50%',
                transform: 'rotate(-90deg) translateX(-50%)',
                transformOrigin: 'left center',
                fontSize: 9,
                color: '#94a3b8',
                whiteSpace: 'nowrap'
              }}
            >
              I(Z;Y) Prediction →
            </div>

            {/* X-axis label */}
            <div
              style={{
                position: 'absolute',
                bottom: 4,
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: 9,
                color: '#94a3b8'
              }}
            >
              I(X;Z) Compression →
            </div>

            {/* Plot points */}
            <div style={{ position: 'relative', height: '100%', width: '100%' }}>
              {metricsHistory.map((metrics, idx) => {
                // Normalize to 0-100% range
                const x =
                  ((metrics.compressionMI - minCompression) / (maxCompression - minCompression)) *
                  100;
                const y =
                  ((metrics.predictionMI - minPrediction) / (maxPrediction - minPrediction)) * 100;

                // Color based on epoch (gradient from purple to green)
                const epochProgress = idx / Math.max(1, metricsHistory.length - 1);
                const color = getTradeoffColor(metrics.compressionMI, metrics.predictionMI);

                return (
                  <div
                    key={idx}
                    style={{
                      position: 'absolute',
                      left: `${x}%`,
                      bottom: `${y}%`,
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: color,
                      border: idx === metricsHistory.length - 1 ? '2px solid #fbbf24' : 'none',
                      opacity: 0.6 + 0.4 * epochProgress,
                      transform: 'translate(-50%, 50%)',
                      boxShadow:
                        idx === metricsHistory.length - 1
                          ? '0 0 8px rgba(251, 191, 36, 0.6)'
                          : 'none'
                    }}
                    title={`Epoch ${idx + 1}: I(X;Z)=${metrics.compressionMI.toFixed(3)}, I(Z;Y)=${metrics.predictionMI.toFixed(3)}`}
                  />
                );
              })}
            </div>
          </div>

          <div style={{ fontSize: 9, color: '#64748b', marginTop: 8, textAlign: 'center' }}>
            Each point represents one epoch. Latest epoch highlighted in gold.
          </div>
        </div>
      )}

      {/* Theory Explanation */}
      <div
        style={{
          background: 'rgba(139, 92, 246, 0.1)',
          border: '1px solid rgba(139, 92, 246, 0.2)',
          borderRadius: 8,
          padding: 12,
          fontSize: 11,
          color: '#cbd5e1',
          lineHeight: 1.6
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 6, color: '#c4b5fd' }}>
          💡 Information Bottleneck Principle
        </div>
        <div style={{ fontSize: 10, color: '#94a3b8' }}>
          The IB principle finds representations Z that compress input X (minimize I(X;Z)) while
          preserving information relevant for prediction Y (maximize I(Z;Y)).
          <br />
          <br />
          <strong style={{ color: '#a78bfa' }}>Loss:</strong> L = -I(Z;Y) + β·I(X;Z)
          <br />
          <strong style={{ color: '#a78bfa' }}>Beta (β):</strong> Controls compression-prediction
          trade-off
          <br />• β → 0: Maximum compression (minimal info)
          <br />• β → ∞: Maximum prediction (all info)
        </div>
      </div>

      {/* Training Progress */}
      {isTraining && (
        <div style={{ marginTop: 12, fontSize: 10, color: '#64748b', textAlign: 'center' }}>
          Training epoch {currentEpoch + 1} / {totalEpochs}
        </div>
      )}
    </div>
  );
}
