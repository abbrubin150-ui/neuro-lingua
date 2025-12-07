import React from 'react';
import { formatTimestamp, createTrainingHistoryCsv, downloadBlob } from '../lib/utils';
import { EXPORT_FILENAMES } from '../config/constants';
import type { GPUMetrics } from '../backend/gpu_neural_ops';
import type { EdgeLearningDiagnostics } from '../backend/edgeLearning';
import { EDGE_LEARNING_INFO } from '../backend/edgeLearning';
import type { Architecture } from './TrainingPanel';
import type { ModelMeta, ModelMetaStore } from '../types/modelMeta';

interface ModelMetricsProps {
  stats: { loss: number; acc: number; ppl: number; lossEMA: number; tokensPerSec: number };
  info: { V: number; P: number };
  activeArchitecture: Architecture;
  activeModelMeta: ModelMeta | null;
  modelComparisons: ModelMetaStore;
  trainingHistory: { loss: number; accuracy: number; timestamp: number }[];
  gpuMetrics?: GPUMetrics | null;
  edgeLearningDiagnostics?: EdgeLearningDiagnostics | null;
  onMessage: (message: string) => void;
}

type ComparisonEntry = { architecture: Architecture; meta: ModelMeta };

const ARCHITECTURE_LABELS: Record<Architecture, string> = {
  feedforward: 'Standard (ProNeural)',
  advanced: 'AdvancedNeuralLM',
  transformer: 'TransformerLM'
};

/**
 * TrainingChart displays loss and accuracy progression over epochs
 */
function TrainingChart({
  history
}: {
  history: { loss: number; accuracy: number; timestamp: number }[];
}) {
  if (history.length === 0) return null;
  const maxLoss = Math.max(...history.map((h) => h.loss), 1e-6);
  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.9)',
        borderRadius: 12,
        padding: 16,
        marginTop: 16,
        border: '1px solid #334155'
      }}
    >
      <h4 style={{ color: '#a78bfa', margin: '0 0 12px 0' }}>📈 Training Progress</h4>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, height: 140 }}>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Loss</div>
          <div
            style={{
              display: 'flex',
              alignItems: 'end',
              gap: 2,
              height: 70,
              borderLeft: '1px solid #334155',
              borderBottom: '1px solid #334155',
              padding: '4px 0'
            }}
          >
            {history.map((h, i) => (
              <div
                key={i}
                title={`Epoch ${i + 1}: ${h.loss.toFixed(4)}`}
                style={{
                  flex: 1,
                  height: `${(h.loss / maxLoss) * 100}%`,
                  background: 'linear-gradient(to top, #ef4444, #dc2626)',
                  borderRadius: 2,
                  minHeight: 1
                }}
              />
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Accuracy</div>
          <div
            style={{
              display: 'flex',
              alignItems: 'end',
              gap: 2,
              height: 70,
              borderLeft: '1px solid #334155',
              borderBottom: '1px solid #334155',
              padding: '4px 0'
            }}
          >
            {history.map((h, i) => (
              <div
                key={i}
                title={`Epoch ${i + 1}: ${(h.accuracy * 100).toFixed(1)}%`}
                style={{
                  flex: 1,
                  height: `${h.accuracy * 100}%`,
                  background: 'linear-gradient(to top, #10b981, #059669)',
                  borderRadius: 2,
                  minHeight: 1
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * WebGPU browser compatibility info
 */
const WEBGPU_COMPATIBILITY = {
  supported: [
    { browser: 'Chrome', version: '113+', status: 'Full support' },
    { browser: 'Edge', version: '113+', status: 'Full support' },
    { browser: 'Opera', version: '99+', status: 'Full support' }
  ],
  partial: [{ browser: 'Firefox', version: '127+', status: 'Behind flag' }],
  unsupported: [
    { browser: 'Safari', version: 'N/A', status: 'Not supported (WebGPU via Metal planned)' }
  ]
};

/**
 * Get friendly reason for GPU unavailability
 */
function getGPUUnavailableReason(metrics: GPUMetrics): string {
  if (metrics.lastError) {
    if (metrics.lastError.includes('not supported')) {
      return 'WebGPU is not supported in this browser.';
    }
    if (metrics.lastError.includes('adapter')) {
      return 'No compatible GPU adapter found.';
    }
    if (metrics.lastError.includes('device')) {
      return 'GPU device initialization failed.';
    }
    return metrics.lastError;
  }
  return 'WebGPU is not available in this browser.';
}

/**
 * GPUStatusPanel shows GPU status when not available
 */
function GPUStatusPanel({ metrics }: { metrics: GPUMetrics }) {
  const reason = getGPUUnavailableReason(metrics);

  return (
    <div
      style={{
        background: 'rgba(100, 116, 139, 0.1)',
        border: '1px solid rgba(100, 116, 139, 0.3)',
        borderRadius: 12,
        padding: 16,
        marginTop: 16
      }}
    >
      <h4
        style={{
          color: '#94a3b8',
          margin: '0 0 12px 0',
          display: 'flex',
          alignItems: 'center',
          gap: 8
        }}
      >
        ⚡ GPU Acceleration
        <span
          style={{
            fontSize: 11,
            padding: '2px 8px',
            background: '#64748b',
            borderRadius: 12,
            color: 'white'
          }}
        >
          UNAVAILABLE
        </span>
      </h4>

      <div
        style={{
          background: 'rgba(251, 191, 36, 0.1)',
          border: '1px solid rgba(251, 191, 36, 0.3)',
          borderRadius: 8,
          padding: '10px 12px',
          marginBottom: 12
        }}
      >
        <div style={{ fontSize: 12, color: '#fbbf24', marginBottom: 4 }}>⚠️ {reason}</div>
        <div style={{ fontSize: 11, color: '#94a3b8' }}>
          Training will use CPU (slower but fully functional).
        </div>
      </div>

      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
          Browser Compatibility:
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {WEBGPU_COMPATIBILITY.supported.map((b) => (
            <div key={b.browser} style={{ fontSize: 11, display: 'flex', alignItems: 'center' }}>
              <span style={{ color: '#10b981', marginRight: 6 }}>✓</span>
              <span style={{ color: '#cbd5f5' }}>
                {b.browser} {b.version}
              </span>
            </div>
          ))}
          {WEBGPU_COMPATIBILITY.partial.map((b) => (
            <div key={b.browser} style={{ fontSize: 11, display: 'flex', alignItems: 'center' }}>
              <span style={{ color: '#fbbf24', marginRight: 6 }}>◐</span>
              <span style={{ color: '#94a3b8' }}>
                {b.browser} {b.version} ({b.status})
              </span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ fontSize: 10, color: '#64748b', marginTop: 8 }}>
        💡 Tip: Use Chrome or Edge for 2-5x faster training with GPU acceleration.
      </div>
    </div>
  );
}

/**
 * GPUMetricsPanel displays GPU acceleration metrics
 */
function GPUMetricsPanel({ metrics }: { metrics: GPUMetrics }) {
  if (!metrics.available) {
    return <GPUStatusPanel metrics={metrics} />;
  }

  const utilization = Math.max(0, Math.min(100, metrics.utilizationPercent ?? 0));

  return (
    <div
      style={{
        background: 'rgba(59, 130, 246, 0.1)',
        border: '1px solid rgba(59, 130, 246, 0.3)',
        borderRadius: 12,
        padding: 16,
        marginTop: 16
      }}
    >
      <h4
        style={{
          color: '#60a5fa',
          margin: '0 0 12px 0',
          display: 'flex',
          alignItems: 'center',
          gap: 8
        }}
      >
        ⚡ GPU Acceleration
        {metrics.enabled ? (
          <span
            style={{
              fontSize: 11,
              padding: '2px 8px',
              background: '#10b981',
              borderRadius: 12,
              color: 'white'
            }}
          >
            ACTIVE
          </span>
        ) : (
          <span
            style={{
              fontSize: 11,
              padding: '2px 8px',
              background: '#64748b',
              borderRadius: 12,
              color: 'white'
            }}
          >
            DISABLED
          </span>
        )}
      </h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Total Operations</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#60a5fa' }}>
            {metrics.totalOperations.toLocaleString()}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Total Time</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#3b82f6' }}>
            {metrics.totalTimeMs.toFixed(1)}ms
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Avg Time/Op</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#2563eb' }}>
            {metrics.averageTimeMs.toFixed(2)}ms
          </div>
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Utilization</div>
        <div
          style={{
            height: 10,
            borderRadius: 999,
            background: 'rgba(59, 130, 246, 0.25)',
            overflow: 'hidden'
          }}
          aria-label={`GPU utilization ${utilization.toFixed(0)} percent`}
        >
          <div
            style={{
              width: `${utilization}%`,
              height: '100%',
              borderRadius: 999,
              background: 'linear-gradient(90deg, #22d3ee, #3b82f6)',
              transition: 'width 0.3s ease'
            }}
          />
        </div>
        <div style={{ fontSize: 12, color: '#cbd5f5', marginTop: 4 }}>
          {utilization.toFixed(0)}% of training time spent on GPU kernels
        </div>
      </div>
      {metrics.deviceInfo && (
        <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8', textAlign: 'center' }}>
          Device: {metrics.deviceInfo}
        </div>
      )}
      {!metrics.enabled && metrics.lastError && (
        <div
          style={{
            marginTop: 12,
            fontSize: 11,
            color: '#fca5a5',
            background: 'rgba(239, 68, 68, 0.08)',
            borderRadius: 8,
            padding: '8px 10px'
          }}
        >
          ⚠️ GPU fallback activated: {metrics.lastError}
        </div>
      )}
    </div>
  );
}

/**
 * ModelMetrics displays training statistics, model info, and training history chart
 */
export function ModelMetrics({
  stats,
  info,
  activeArchitecture,
  activeModelMeta,
  modelComparisons,
  trainingHistory,
  gpuMetrics,
  edgeLearningDiagnostics,
  onMessage
}: ModelMetricsProps) {
  const handleDownloadCsv = () => {
    if (trainingHistory.length === 0) {
      onMessage('ℹ️ Train the model to generate history before exporting CSV.');
      return;
    }
    const blob = createTrainingHistoryCsv(trainingHistory);
    downloadBlob(blob, EXPORT_FILENAMES.TRAINING_HISTORY);
  };

  const comparisonEntries = Object.entries(modelComparisons).reduce<ComparisonEntry[]>(
    (entries, [architecture, meta]) => {
      if (meta) entries.push({ architecture: architecture as Architecture, meta });
      return entries;
    },
    []
  );

  const bestEntry = comparisonEntries.reduce<ComparisonEntry | null>((best, entry) => {
    const entryPpl = typeof entry.meta.perplexity === 'number' ? entry.meta.perplexity : Infinity;
    if (!best) return entry;
    const bestPpl = typeof best.meta.perplexity === 'number' ? best.meta.perplexity : Infinity;
    return entryPpl < bestPpl ? entry : best;
  }, null);

  const activeArchitectureLabel = ARCHITECTURE_LABELS[activeArchitecture];

  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <h3 style={{ color: '#34d399', marginTop: 0, marginBottom: 16 }}>📊 Advanced Statistics</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Loss (Avg)</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#ef4444' }}>
            {stats.loss.toFixed(4)}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Loss (EMA)</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#f87171' }}>
            {stats.lossEMA > 0 ? stats.lossEMA.toFixed(4) : '—'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Accuracy</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#10b981' }}>
            {(stats.acc * 100).toFixed(1)}%
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Perplexity</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#f59e0b' }}>
            {stats.ppl.toFixed(2)}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Tokens/sec</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#06b6d4' }}>
            {stats.tokensPerSec > 0 ? stats.tokensPerSec.toFixed(1) : '—'}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Vocab Size</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#a78bfa' }}>{info.V}</div>
        </div>
        <div style={{ textAlign: 'center', gridColumn: 'span 3' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Parameters</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#60a5fa' }}>
            {info.P.toLocaleString()}
          </div>
        </div>
      </div>
      <div
        style={{
          marginTop: 12,
          fontSize: 12,
          color: '#94a3b8',
          display: 'flex',
          flexWrap: 'wrap',
          gap: 12,
          justifyContent: 'space-between'
        }}
      >
        <div>
          Active architecture:{' '}
          <span style={{ fontWeight: 600, color: '#a78bfa' }}>{activeArchitectureLabel}</span>
        </div>
        <div style={{ color: '#cbd5f5' }}>
          {activeModelMeta
            ? `Last update: ${formatTimestamp(activeModelMeta.timestamp)} • Vocab ${activeModelMeta.vocab}`
            : 'Last update: No trained model yet.'}
        </div>
      </div>
      <TrainingChart history={trainingHistory} />
      {comparisonEntries.length > 0 && (
        <div
          style={{
            marginTop: 16,
            background: 'rgba(167, 139, 250, 0.08)',
            border: '1px solid rgba(167, 139, 250, 0.2)',
            borderRadius: 12,
            padding: 16
          }}
        >
          <h4 style={{ color: '#c4b5fd', margin: '0 0 12px 0' }}>🏗️ Architecture Comparison</h4>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: 12
            }}
          >
            {comparisonEntries.map(({ architecture, meta }) => {
              const isBest =
                bestEntry?.architecture === architecture && meta.perplexity !== undefined;
              const isActive = architecture === activeArchitecture;
              return (
                <div
                  key={architecture}
                  style={{
                    padding: 12,
                    borderRadius: 10,
                    border: `1px solid ${isActive ? '#a78bfa' : 'rgba(148,163,184,0.4)'}`,
                    background: isActive ? 'rgba(167, 139, 250, 0.12)' : 'rgba(15,23,42,0.6)'
                  }}
                >
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#e0e7ff', marginBottom: 6 }}>
                    {ARCHITECTURE_LABELS[architecture]}
                    {isBest && (
                      <span
                        style={{
                          marginLeft: 6,
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 999,
                          background: '#facc15',
                          color: '#0f172a'
                        }}
                      >
                        BEST PPL
                      </span>
                    )}
                    {isActive && (
                      <span
                        style={{
                          marginLeft: 6,
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 999,
                          background: '#a78bfa',
                          color: '#0f172a'
                        }}
                      >
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5f5' }}>
                    Perplexity:{' '}
                    <strong>
                      {typeof meta.perplexity === 'number' ? meta.perplexity.toFixed(2) : '—'}
                    </strong>
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5f5' }}>
                    Accuracy:{' '}
                    <strong>
                      {typeof meta.accuracy === 'number'
                        ? `${(meta.accuracy * 100).toFixed(1)}%`
                        : '—'}
                    </strong>
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5f5' }}>
                    Tokens/sec:{' '}
                    <strong>
                      {typeof meta.tokensPerSec === 'number' ? meta.tokensPerSec.toFixed(1) : '—'}
                    </strong>
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5f5' }}>
                    Train time:{' '}
                    <strong>
                      {typeof meta.trainingDurationMs === 'number'
                        ? `${(meta.trainingDurationMs / 1000).toFixed(1)}s`
                        : '—'}
                    </strong>
                  </div>
                </div>
              );
            })}
          </div>
          {comparisonEntries.length === 1 && (
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
              Train a second architecture to unlock full comparisons.
            </div>
          )}
        </div>
      )}
      {gpuMetrics && <GPUMetricsPanel metrics={gpuMetrics} />}
      {edgeLearningDiagnostics && (
        <div
          style={{
            background: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: 12,
            padding: 16,
            marginTop: 16
          }}
        >
          <h4
            style={{
              color: '#22c55e',
              margin: '0 0 12px 0',
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}
          >
            📊 Edge Learning Diagnostics
            {edgeLearningDiagnostics.status === 'success' && (
              <span
                style={{
                  fontSize: 11,
                  padding: '2px 8px',
                  background: '#10b981',
                  borderRadius: 12,
                  color: 'white'
                }}
              >
                COMPUTED
              </span>
            )}
            <span
              title="Edge Learning Theory: Information-theoretic framework analyzing learning efficiency based on Fisher Information, Cramér-Rao bounds, and statistical efficiency. Higher efficiency values indicate the model is learning close to theoretical limits."
              style={{
                cursor: 'help',
                fontSize: 14,
                opacity: 0.7,
                marginLeft: 'auto'
              }}
            >
              ℹ️
            </span>
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.fisherInformation}
              >
                Fisher Info ℹ️
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#22c55e' }}>
                {edgeLearningDiagnostics.fisherInformation.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.efficiency}
              >
                Efficiency ℹ️
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>
                {(edgeLearningDiagnostics.efficiency * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.variance}
              >
                Variance ℹ️
              </div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#059669' }}>
                {edgeLearningDiagnostics.variance.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.entropy}
              >
                Entropy ℹ️
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#34d399' }}>
                {edgeLearningDiagnostics.entropy.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.estimatorCovariance}
              >
                Est. Covariance ℹ️
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#6ee7b7' }}>
                {edgeLearningDiagnostics.estimatorCovariance.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div
                style={{ fontSize: 11, color: '#94a3b8', cursor: 'help' }}
                title={EDGE_LEARNING_INFO.cramerRaoBound}
              >
                CRB ℹ️
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#a7f3d0' }}>
                {edgeLearningDiagnostics.cramerRaoBound.toExponential(2)}
              </div>
            </div>
          </div>
          <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 12, fontStyle: 'italic' }}>
            💡 Hover over metrics for explanations. Edge Learning quantifies how efficiently the
            model learns relative to information-theoretic bounds.
          </div>
        </div>
      )}
      <button
        onClick={handleDownloadCsv}
        style={{
          marginTop: 12,
          padding: '10px 14px',
          background: 'linear-gradient(90deg, #1d4ed8, #3b82f6)',
          border: 'none',
          borderRadius: 10,
          color: 'white',
          fontWeight: 600,
          cursor: 'pointer',
          alignSelf: 'flex-start'
        }}
      >
        ⬇️ Export history CSV
      </button>
    </div>
  );
}
