/**
 * RunComparisonPanel - Side-by-side comparison of training runs
 * Shows hyperparameter and metrics differences with visual indicators
 */

import React, { useState, useMemo } from 'react';
import { useProjects } from '../contexts/ProjectContext';
import { computeRunDiff } from '../lib/experimentComparison';
import type { Run } from '../types/project';
import type { RunDiff, FieldDiff, DiffType } from '../types/experiment';

interface RunComparisonPanelProps {
  /** Selected project ID to filter runs */
  projectId?: string;
  /** Pre-selected run IDs to compare */
  preselectedRunIds?: string[];
  /** Callback when panel is closed */
  onClose?: () => void;
  /** Direction for RTL support */
  direction?: 'ltr' | 'rtl';
}

/**
 * Get color for diff type
 */
function getDiffColor(type: DiffType): string {
  switch (type) {
    case 'added':
      return '#10b981';
    case 'removed':
      return '#ef4444';
    case 'changed':
      return '#f59e0b';
    case 'unchanged':
      return '#6b7280';
    default:
      return '#6b7280';
  }
}

/**
 * Get icon for diff type
 */
function getDiffIcon(type: DiffType): string {
  switch (type) {
    case 'added':
      return '+';
    case 'removed':
      return '−';
    case 'changed':
      return '↔';
    case 'unchanged':
      return '=';
    default:
      return '?';
  }
}

/**
 * Format field value for display
 */
function formatFieldValue(value: unknown): string {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'number') {
    return value.toFixed(4);
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  return String(value);
}

/**
 * Render a single field diff
 */
function FieldDiffRow({ label, diff }: { label: string; diff: FieldDiff<any> }) {
  const color = getDiffColor(diff.type);
  const icon = getDiffIcon(diff.type);
  const isSignificant = diff.percentChange !== undefined && Math.abs(diff.percentChange) > 10;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 40px 1fr',
        gap: 12,
        padding: '8px 12px',
        background: isSignificant ? 'rgba(245, 158, 11, 0.08)' : 'transparent',
        borderRadius: 6,
        border: isSignificant ? '1px solid rgba(245, 158, 11, 0.3)' : '1px solid transparent',
        marginBottom: 4
      }}
    >
      {/* Base value */}
      <div style={{ textAlign: 'right', color: '#94a3b8', fontSize: 13 }}>
        {formatFieldValue(diff.oldValue)}
      </div>

      {/* Diff indicator */}
      <div
        style={{
          textAlign: 'center',
          color,
          fontWeight: 700,
          fontSize: 14
        }}
        title={`${label}: ${diff.type}`}
      >
        {icon}
        {diff.percentChange !== undefined && (
          <div style={{ fontSize: 10, marginTop: 2 }}>
            {diff.percentChange > 0 ? '+' : ''}
            {diff.percentChange.toFixed(1)}%
          </div>
        )}
      </div>

      {/* Compare value */}
      <div style={{ textAlign: 'left', color: '#e2e8f0', fontSize: 13, fontWeight: 600 }}>
        {formatFieldValue(diff.newValue)}
      </div>
    </div>
  );
}

/**
 * Render improvement direction badge
 */
function ImprovementBadge({ direction }: { direction: 'better' | 'worse' | 'mixed' | 'unknown' }) {
  const config = {
    better: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)', icon: '↑', label: 'Improved' },
    worse: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)', icon: '↓', label: 'Degraded' },
    mixed: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)', icon: '↕', label: 'Mixed' },
    unknown: { color: '#6b7280', bg: 'rgba(107, 114, 128, 0.15)', icon: '?', label: 'Unknown' }
  };

  const { color, bg, icon, label } = config[direction];

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 12px',
        background: bg,
        borderRadius: 999,
        color,
        fontSize: 13,
        fontWeight: 600
      }}
    >
      <span style={{ fontSize: 16 }}>{icon}</span>
      {label}
    </div>
  );
}

/**
 * Render run diff comparison
 */
function RunDiffView({ diff }: { diff: RunDiff }) {
  const { baseRun, compareRun, hyperparameters, metrics, corpusChanged, summary } = diff;

  return (
    <div
      style={{
        background: 'rgba(30, 41, 59, 0.6)',
        border: '1px solid #334155',
        borderRadius: 12,
        padding: 16,
        marginBottom: 16
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16,
          paddingBottom: 12,
          borderBottom: '1px solid #334155'
        }}
      >
        <div>
          <h4 style={{ margin: '0 0 4px 0', color: '#a78bfa', fontSize: '1.1rem' }}>
            Comparing: {baseRun.name} vs {compareRun.name}
          </h4>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            {summary.totalChanges} parameter{summary.totalChanges !== 1 ? 's' : ''} changed
            {corpusChanged && ' • Corpus modified'}
          </div>
        </div>
        <ImprovementBadge direction={summary.improvementDirection} />
      </div>

      {/* Significant Changes Summary */}
      {summary.significantChanges.length > 0 && (
        <div
          style={{
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: 8,
            padding: 12,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 600, color: '#fbbf24', marginBottom: 8 }}>
            ⚡ Significant Changes (&gt;10%)
          </div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#94a3b8' }}>
            {summary.significantChanges.map((change, i) => (
              <li key={i}>{change}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Hyperparameters Comparison */}
      <div style={{ marginBottom: 16 }}>
        <h5 style={{ margin: '0 0 12px 0', color: '#c4b5fd', fontSize: '0.95rem' }}>
          📊 Hyperparameters
        </h5>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '140px 1fr',
            gap: 8,
            fontSize: 12
          }}
        >
          <div style={{ color: '#6b7280', fontWeight: 600 }}>Architecture</div>
          <FieldDiffRow label="Architecture" diff={hyperparameters.architecture} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Hidden Size</div>
          <FieldDiffRow label="Hidden Size" diff={hyperparameters.hiddenSize} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Epochs</div>
          <FieldDiffRow label="Epochs" diff={hyperparameters.epochs} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Learning Rate</div>
          <FieldDiffRow label="Learning Rate" diff={hyperparameters.learningRate} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Optimizer</div>
          <FieldDiffRow label="Optimizer" diff={hyperparameters.optimizer} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Dropout</div>
          <FieldDiffRow label="Dropout" diff={hyperparameters.dropout} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Context Size</div>
          <FieldDiffRow label="Context Size" diff={hyperparameters.contextSize} />

          {hyperparameters.activation && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Activation</div>
              <FieldDiffRow label="Activation" diff={hyperparameters.activation} />
            </>
          )}

          {hyperparameters.useLayerNorm && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Layer Norm</div>
              <FieldDiffRow label="Layer Norm" diff={hyperparameters.useLayerNorm} />
            </>
          )}

          {hyperparameters.numHeads && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Num Heads</div>
              <FieldDiffRow label="Num Heads" diff={hyperparameters.numHeads} />
            </>
          )}

          {hyperparameters.numLayers && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Num Layers</div>
              <FieldDiffRow label="Num Layers" diff={hyperparameters.numLayers} />
            </>
          )}
        </div>
      </div>

      {/* Metrics Comparison */}
      <div>
        <h5 style={{ margin: '0 0 12px 0', color: '#c4b5fd', fontSize: '0.95rem' }}>
          📈 Performance Metrics
        </h5>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '140px 1fr',
            gap: 8,
            fontSize: 12
          }}
        >
          <div style={{ color: '#6b7280', fontWeight: 600 }}>Loss</div>
          <FieldDiffRow label="Loss" diff={metrics.finalLoss} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Accuracy</div>
          <FieldDiffRow label="Accuracy" diff={metrics.finalAccuracy} />

          <div style={{ color: '#6b7280', fontWeight: 600 }}>Perplexity</div>
          <FieldDiffRow label="Perplexity" diff={metrics.finalPerplexity} />

          {metrics.trainingTime && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Training Time</div>
              <FieldDiffRow label="Training Time" diff={metrics.trainingTime} />
            </>
          )}

          {metrics.modelSize && (
            <>
              <div style={{ color: '#6b7280', fontWeight: 600 }}>Model Size</div>
              <FieldDiffRow label="Model Size" diff={metrics.modelSize} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Main comparison panel component
 */
export function RunComparisonPanel({
  projectId,
  preselectedRunIds = [],
  onClose,
  direction = 'ltr'
}: RunComparisonPanelProps) {
  const { runs, getRunsByProject, projects: _projects } = useProjects();

  // Filter runs by project if specified
  const availableRuns = useMemo(() => {
    if (projectId) {
      return getRunsByProject(projectId).filter((r) => r.status === 'completed');
    }
    return runs.filter((r) => r.status === 'completed');
  }, [projectId, runs, getRunsByProject]);

  // Selected runs for comparison
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>(
    preselectedRunIds.length > 0 ? preselectedRunIds : []
  );

  // Compute diffs
  const comparisonResult = useMemo(() => {
    if (selectedRunIds.length < 2 || selectedRunIds.length > 3) return null;

    const selectedRuns = selectedRunIds
      .map((id) => availableRuns.find((r) => r.id === id))
      .filter((r): r is Run => r !== undefined);

    if (selectedRuns.length !== selectedRunIds.length) return null;

    // Compute pairwise diffs against the first run (baseline)
    const baseRun = selectedRuns[0];
    const diffs: RunDiff[] = selectedRuns.slice(1).map((run) => computeRunDiff(baseRun, run));

    return { baseRun, diffs, runs: selectedRuns };
  }, [selectedRunIds, availableRuns]);

  // Toggle run selection
  const toggleRunSelection = (runId: string) => {
    setSelectedRunIds((prev) => {
      if (prev.includes(runId)) {
        return prev.filter((id) => id !== runId);
      } else if (prev.length < 3) {
        return [...prev, runId];
      }
      return prev;
    });
  };

  return (
    <div
      role="button"
      tabIndex={0}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1001,
        padding: 20,
        direction,
        overflow: 'auto'
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape' || e.key === 'Enter') {
          onClose();
        }
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        style={{
          background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          borderRadius: 16,
          padding: 24,
          maxWidth: 1100,
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '2px solid #475569'
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 20
          }}
        >
          <h2 style={{ margin: 0, color: '#a78bfa', fontSize: '1.5rem' }}>
            🔬 Run Comparison Explorer
          </h2>
          <button
            onClick={onClose}
            style={{
              background: '#374151',
              border: '1px solid #4b5563',
              borderRadius: 8,
              color: '#e5e7eb',
              padding: '8px 16px',
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            ✕ Close
          </button>
        </div>

        {/* Run Selection */}
        <div
          style={{
            background: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 20
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 600, color: '#a78bfa', marginBottom: 12 }}>
            Select 2-3 runs to compare ({selectedRunIds.length}/3 selected)
          </div>
          <div style={{ display: 'grid', gap: 8 }}>
            {availableRuns.length === 0 && (
              <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: 20 }}>
                No completed runs available for comparison
              </div>
            )}
            {availableRuns.map((run) => {
              const isSelected = selectedRunIds.includes(run.id);
              const isBaseline = selectedRunIds[0] === run.id;

              return (
                <div
                  key={run.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => toggleRunSelection(run.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleRunSelection(run.id);
                    }
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: 12,
                    background: isSelected ? 'rgba(139, 92, 246, 0.15)' : 'rgba(30, 41, 59, 0.5)',
                    border: isSelected ? '2px solid #a78bfa' : '1px solid #475569',
                    borderRadius: 8,
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => {}}
                    style={{ cursor: 'pointer' }}
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>
                      {run.name}
                      {isBaseline && (
                        <span
                          style={{
                            marginLeft: 8,
                            fontSize: 10,
                            padding: '2px 6px',
                            background: '#10b981',
                            borderRadius: 4,
                            color: 'white'
                          }}
                        >
                          BASELINE
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>
                      {run.config.architecture} • {run.config.optimizer} • Hidden:{' '}
                      {run.config.hiddenSize} • Loss: {run.results?.finalLoss.toFixed(4)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Comparison Results */}
        {comparisonResult ? (
          <div>
            {comparisonResult.diffs.map((diff, index) => (
              <RunDiffView key={index} diff={diff} />
            ))}
          </div>
        ) : (
          <div
            style={{
              textAlign: 'center',
              padding: 40,
              color: '#64748b',
              fontSize: 14
            }}
          >
            {selectedRunIds.length === 0
              ? 'Select at least 2 runs to begin comparison'
              : selectedRunIds.length === 1
                ? 'Select one more run to compare'
                : 'Select 2-3 runs to compare (maximum 3)'}
          </div>
        )}
      </div>
    </div>
  );
}
