import React from 'react';
import type { Architecture } from './TrainingPanel';
import type { ModelMeta } from '../types/modelMeta';
import { formatTimestamp } from '../lib/utils';

interface ModelSnapshotProps {
  meta: ModelMeta | null;
  architecture: Architecture;
  title: string;
  lastUpdatedLabel: string;
  vocabLabel: string;
  emptyLabel: string;
  hint: string;
}

const ARCHITECTURE_BADGES: Record<Architecture, string> = {
  feedforward: 'ProNeural Snapshot',
  advanced: 'Advanced Snapshot',
  transformer: 'Transformer Snapshot'
};

export function ModelSnapshot({
  meta,
  architecture,
  title,
  lastUpdatedLabel,
  vocabLabel,
  emptyLabel,
  hint
}: ModelSnapshotProps) {
  return (
    <div
      style={{
        border: '1px solid rgba(148,163,184,0.4)',
        borderRadius: 12,
        padding: 12,
        background: 'rgba(15,23,42,0.6)',
        marginBottom: 12
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 8
        }}
      >
        <div style={{ fontWeight: 600, color: '#f8fafc' }}>{title}</div>
        <span
          style={{
            fontSize: 10,
            padding: '2px 8px',
            borderRadius: 999,
            background: 'rgba(96,165,250,0.15)',
            color: '#bfdbfe'
          }}
        >
          {ARCHITECTURE_BADGES[architecture]}
        </span>
      </div>
      {meta ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ fontSize: 12, color: '#e2e8f0' }}>
            <strong>{lastUpdatedLabel}:</strong> {formatTimestamp(meta.timestamp)}
          </div>
          <div style={{ fontSize: 12, color: '#e2e8f0' }}>
            <strong>{vocabLabel}:</strong> {meta.vocab.toLocaleString()}
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>{hint}</div>
        </div>
      ) : (
        <div style={{ fontSize: 12, color: '#cbd5f5' }}>{emptyLabel}</div>
      )}
    </div>
  );
}
