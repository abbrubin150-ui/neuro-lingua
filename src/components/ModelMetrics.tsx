import React from 'react';
import { formatTimestamp, createTrainingHistoryCsv, downloadBlob } from '../lib/utils';
import { EXPORT_FILENAMES } from '../config/constants';
import type { GPUMetrics } from '../backend/gpu_neural_ops';
import type { EdgeLearningDiagnostics } from '../backend/edgeLearning';

interface ModelMetricsProps {
  stats: { loss: number; acc: number; ppl: number; lossEMA: number; tokensPerSec: number };
  info: { V: number; P: number };
  lastModelUpdate: { timestamp: number; vocab: number } | null;
  trainingHistory: { loss: number; accuracy: number; timestamp: number }[];
  gpuMetrics?: GPUMetrics | null;
  edgeLearningDiagnostics?: EdgeLearningDiagnostics | null;
  onMessage: (message: string) => void;
}

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
 * GPUMetricsPanel displays GPU acceleration metrics
 */
function GPUMetricsPanel({ metrics }: { metrics: GPUMetrics }) {
  if (!metrics.available) {
    return null;
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
  lastModelUpdate,
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
      <div style={{ marginTop: 12, fontSize: 12, color: '#cbd5f5' }}>
        {lastModelUpdate
          ? `Last update: ${formatTimestamp(lastModelUpdate.timestamp)} • Vocab ${lastModelUpdate.vocab}`
          : 'Last update: No trained model yet.'}
      </div>
      <TrainingChart history={trainingHistory} />
      {gpuMetrics && gpuMetrics.available && <GPUMetricsPanel metrics={gpuMetrics} />}
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
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Fisher Info</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#22c55e' }}>
                {edgeLearningDiagnostics.fisherInformation.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Efficiency</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>
                {(edgeLearningDiagnostics.efficiency * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Variance</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#059669' }}>
                {edgeLearningDiagnostics.variance.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Entropy</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#34d399' }}>
                {edgeLearningDiagnostics.entropy.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>Est. Covariance</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#6ee7b7' }}>
                {edgeLearningDiagnostics.estimatorCovariance.toFixed(4)}
              </div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>CRB</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#a7f3d0' }}>
                {edgeLearningDiagnostics.cramerRaoBound.toExponential(2)}
              </div>
            </div>
          </div>
          <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 8 }}>
            💡 Edge Learning: Information-theoretic analysis of model learning efficiency
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
