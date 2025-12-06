import React, { useMemo, useRef, useState } from 'react';
import type { CerebroBubble, InjectionEvent, InjectionProposal } from '../types/injection';
import type { InjectableLayer } from '../lib/expandable/InjectableLayer';
import { InjectionEngine } from '../lib/expandable/InjectionEngine';
import { InjectionRunSession, createLedgerAdapter } from '../training/injection_hooks';
import { CerebroBubbleGraph } from './CerebroBubbleGraph';

interface CerebroPanelProps {
  layer: InjectableLayer;
  bubbles?: CerebroBubble[];
  engine?: InjectionEngine;
}

function createPlaceholderBubbles(dModel: number): CerebroBubble[] {
  return Array.from({ length: 12 }, (_, i) => {
    const angle = (2 * Math.PI * i) / 12;
    const padding = Math.max(0, dModel - 2);
    return {
      id: `bubble-${i}`,
      label: `b${i}`,
      activation: 0.4 + 0.6 * Math.random(),
      embedding: [
        Math.cos(angle),
        Math.sin(angle),
        ...Array.from({ length: padding }, () => Math.random() * 0.1)
      ],
      tag: ['body', 'desire', 'risk', 'value', 'action'][i % 5] as CerebroBubble['tag'],
      ts: Date.now() - i * 1000
    };
  });
}

function formatNumber(value: number): string {
  if (Number.isNaN(value)) return '0.00';
  return value.toFixed(4);
}

export function CerebroPanel({ layer, bubbles, engine }: CerebroPanelProps) {
  const target = layer.getTarget();
  const bubbleSet = useMemo(
    () => bubbles ?? createPlaceholderBubbles(target.dModel),
    [bubbles, target.dModel]
  );
  const ledgerRef = useRef<InjectionEvent[]>([]);
  const session = useMemo(
    () =>
      new InjectionRunSession(
        layer,
        createLedgerAdapter(ledgerRef.current),
        engine ?? new InjectionEngine()
      ),
    [engine, layer]
  );

  const [proposal, setProposal] = useState<InjectionProposal | null>(null);
  const [lastEvent, setLastEvent] = useState<InjectionEvent | null>(null);
  const [selectedBubble, setSelectedBubble] = useState<CerebroBubble | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handlePropose = () => {
    const suggestion = session.propose(bubbleSet);
    setProposal(suggestion);
  };

  const handleInject = () => {
    if (!proposal) return;
    const event = session.inject(proposal, bubbleSet);
    setLastEvent(event);
  };

  const handleUndo = () => {
    const undone = session.undoLast();
    if (undone) {
      setLastEvent(undone);
    }
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow cerebro-panel">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500">Cerebro Mode</p>
          <h2 className="text-2xl font-semibold">Neuron Injection Console</h2>
        </div>
        <div className="text-sm text-gray-600">
          <p>Model: {target.modelId}</p>
          <p>Layer: {target.layerId}</p>
          <p>Hidden size: {target.hiddenSize}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <CerebroBubbleGraph bubbles={bubbleSet} onSelectBubble={setSelectedBubble} />
          {selectedBubble ? (
            <div className="text-sm text-gray-700 border border-indigo-100 bg-indigo-50 p-3 rounded">
              <div className="font-semibold">Focus bubble</div>
              <div className="text-xs text-gray-600">{selectedBubble.id}</div>
              <div>Activation: {formatNumber(selectedBubble.activation)}</div>
              <div>Tag: {selectedBubble.tag ?? 'other'}</div>
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          <div className="border border-gray-200 rounded p-3 bg-gray-50">
            <h3 className="font-semibold mb-2">Observe → Propose → Inject</h3>
            <p className="text-sm text-gray-700 mb-3">
              Cerebro scans bubble residual energy and proposes orthogonal neurons to add.
              The ledger keeps every injection and allows rollback.
            </p>
            <div className="flex gap-2 mb-2">
              <button className="px-3 py-2 bg-blue-600 text-white rounded" onClick={handlePropose}>
                Propose
              </button>
              <button
                className="px-3 py-2 bg-green-600 text-white rounded disabled:opacity-50"
                disabled={!proposal}
                onClick={handleInject}
              >
                Inject
              </button>
              <button className="px-3 py-2 bg-gray-200 text-gray-900 rounded" onClick={handleUndo}>
                Undo last
              </button>
            </div>
            <div className="text-sm text-gray-800">
              <div>k suggestion: {proposal?.k ?? '–'}</div>
              <div>Method: {proposal?.method ?? '–'}</div>
              <div>ε threshold: {proposal ? formatNumber(proposal.epsilon) : '–'}</div>
            </div>
          </div>

          <div className="border border-gray-200 rounded p-3">
            <h4 className="font-semibold mb-2">Ledger Snapshot</h4>
            {lastEvent ? (
              <div className="text-sm text-gray-800">
                <div className="flex justify-between">
                  <span>Accepted</span>
                  <span>{lastEvent.accepted ? '✅' : '⚠️'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Δ residual</span>
                  <span>{formatNumber(lastEvent.delta.meanResidual ?? 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Δ trace⊥</span>
                  <span>{formatNumber(lastEvent.delta.tracePerp ?? 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Seed</span>
                  <span>{lastEvent.seed}</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-600">No injections yet.</p>
            )}
          </div>

          <div className="border border-dashed border-indigo-200 rounded p-3 bg-indigo-50">
            <h4 className="font-semibold mb-2">Cerebro Metrics</h4>
            <ul className="text-sm text-gray-800 space-y-1">
              <li>Bubble coherence ↑</li>
              <li>Residual energy Δ</li>
              <li>Training stability (grad variance)</li>
              <li>Perplexity Δ (proxy)</li>
            </ul>
          </div>

          <div>
            <button
              className="text-indigo-600 text-sm underline"
              onClick={() => setShowAdvanced((s) => !s)}
            >
              {showAdvanced ? 'Hide advanced' : 'Advanced controls'}
            </button>
            {showAdvanced ? (
              <div className="mt-2 text-sm text-gray-800 space-y-1">
                <div>
                  Orth penalty: {proposal ? formatNumber(proposal.orthPenalty) : formatNumber(0.1)}
                </div>
                <div>
                  Min gain: {proposal ? formatNumber(proposal.minGain) : formatNumber(0.01)}
                </div>
                <div>Run ID: {proposal?.target.modelId ?? target.modelId}</div>
                <div>Layer type: {target.type}</div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
