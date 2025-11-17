/**
 * DecisionLedgerEditor - UI for editing Decision Ledger
 * Implements Σ-SIG / EXACT1 governance framework
 */

import React, { useState } from 'react';
import type { DecisionLedger, ExecutionStatus } from '../types/project';
import { computeExecutionStatus } from '../types/project';

interface DecisionLedgerEditorProps {
  ledger: DecisionLedger;
  onChange: (ledger: DecisionLedger) => void;
  direction?: 'ltr' | 'rtl';
}

function getStatusColor(status: ExecutionStatus): string {
  switch (status) {
    case 'EXECUTE':
      return '#10b981'; // green
    case 'HOLD':
      return '#f59e0b'; // amber
    case 'ESCALATE':
      return '#ef4444'; // red
  }
}

function getStatusLabel(status: ExecutionStatus): string {
  switch (status) {
    case 'EXECUTE':
      return '✅ EXECUTE - Training permitted';
    case 'HOLD':
      return '⏸️ HOLD - Training paused or expired';
    case 'ESCALATE':
      return '🚨 ESCALATE - Review required';
  }
}

export function DecisionLedgerEditor({
  ledger,
  onChange,
  direction = 'ltr'
}: DecisionLedgerEditorProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const status = computeExecutionStatus(ledger);

  return (
    <div
      style={{
        background: 'rgba(99, 102, 241, 0.1)',
        border: `2px solid ${getStatusColor(status)}`,
        borderRadius: 12,
        padding: 16,
        marginBottom: 16,
        direction
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 700, color: '#a78bfa' }}>
          📋 Decision Ledger (Σ-SIG)
        </div>
        <div
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: getStatusColor(status),
            padding: '4px 12px',
            background: `${getStatusColor(status)}22`,
            borderRadius: 6
          }}
        >
          {getStatusLabel(status)}
        </div>
      </div>

      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          width: '100%',
          padding: '8px 12px',
          background: '#334155',
          border: '1px solid #475569',
          borderRadius: 8,
          color: '#e2e8f0',
          fontSize: 12,
          fontWeight: 600,
          cursor: 'pointer',
          textAlign: direction === 'rtl' ? 'right' : 'left',
          marginBottom: isExpanded ? 12 : 0
        }}
      >
        {isExpanded ? '▼' : '▶'} {isExpanded ? 'Hide Details' : 'Show Details'}
      </button>

      {isExpanded && (
        <div style={{ display: 'grid', gap: 12 }}>
          {/* Rationale */}
          <div>
            <label
              htmlFor="ledger-rationale"
              style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
            >
              Rationale (Why?)
            </label>
            <textarea
              id="ledger-rationale"
              value={ledger.rationale}
              onChange={(e) => onChange({ ...ledger, rationale: e.target.value })}
              placeholder="Why is this training run necessary? What is the purpose?"
              style={{
                width: '100%',
                minHeight: 60,
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: 8,
                color: '#e2e8f0',
                fontSize: 12,
                fontFamily: 'inherit',
                resize: 'vertical'
              }}
            />
          </div>

          {/* Witness */}
          <div>
            <label
              htmlFor="ledger-witness"
              style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
            >
              Witness (Who?)
            </label>
            <input
              id="ledger-witness"
              type="text"
              value={ledger.witness}
              onChange={(e) => onChange({ ...ledger, witness: e.target.value })}
              placeholder="Who authorized this training? (e.g., local-user, researcher@domain)"
              style={{
                width: '100%',
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: 8,
                color: '#e2e8f0',
                fontSize: 12
              }}
            />
          </div>

          {/* Expiry */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <label
                htmlFor="ledger-expiry"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                Expiry Date (Optional)
              </label>
              <input
                id="ledger-expiry"
                type="date"
                value={
                  ledger.expiry ? new Date(ledger.expiry).toISOString().split('T')[0] : ''
                }
                onChange={(e) =>
                  onChange({
                    ...ledger,
                    expiry: e.target.value ? new Date(e.target.value).toISOString() : null
                  })
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: '#e2e8f0',
                  fontSize: 12
                }}
              />
            </div>

            {/* Rollback */}
            <div>
              <label
                htmlFor="ledger-rollback"
                style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 4 }}
              >
                After Expiry
              </label>
              <select
                id="ledger-rollback"
                value={ledger.rollback}
                onChange={(e) =>
                  onChange({
                    ...ledger,
                    rollback: e.target.value as 'keep' | 'delete-after-expiry' | 'archive'
                  })
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: '#e2e8f0',
                  fontSize: 12
                }}
              >
                <option value="keep">Keep</option>
                <option value="delete-after-expiry">Delete</option>
                <option value="archive">Archive</option>
              </select>
            </div>
          </div>

          {/* Status explanation */}
          <div
            style={{
              fontSize: 11,
              color: '#94a3b8',
              padding: 8,
              background: 'rgba(0,0,0,0.2)',
              borderRadius: 6
            }}
          >
            {status === 'EXECUTE' && (
              <>
                ✅ <strong>EXECUTE:</strong> All governance checks passed. Training is permitted.
              </>
            )}
            {status === 'HOLD' && (
              <>
                ⏸️ <strong>HOLD:</strong> This run has expired. Update the expiry date to continue.
              </>
            )}
            {status === 'ESCALATE' && (
              <>
                🚨 <strong>ESCALATE:</strong> Missing rationale or witness. Complete all fields before
                training.
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
