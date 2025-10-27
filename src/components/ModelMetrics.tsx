import React from 'react';
import { formatTimestamp, createTrainingHistoryCsv, downloadBlob } from '../lib/utils';
import { EXPORT_FILENAMES } from '../config/constants';

interface ModelMetricsProps {
  stats: { loss: number; acc: number; ppl: number; lossEMA: number; tokensPerSec: number };
  info: { V: number; P: number };
  lastModelUpdate: { timestamp: number; vocab: number } | null;
  trainingHistory: { loss: number; accuracy: number; timestamp: number }[];
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
 * ModelMetrics displays training statistics, model info, and training history chart
 */
export function ModelMetrics({
  stats,
  info,
  lastModelUpdate,
  trainingHistory,
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
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Loss</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#ef4444' }}>
            {stats.loss.toFixed(4)}
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
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Vocab Size</div>
          <div style={{ fontSize: 24, fontWeight: 800, color: '#a78bfa' }}>{info.V}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
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
