import React, { useEffect, useMemo, useState } from 'react';

type BrainFacet = 'creativity' | 'stability' | 'mood' | 'vocabulary';

type FacetSelection = BrainFacet | 'auto';

type EventLogEntry = {
  id: number;
  timestamp: number;
  facet: FacetSelection;
  label: string;
  detail: string;
};

type BrainMetrics = Record<BrainFacet, number>;

const FACET_LABELS: Record<BrainFacet, string> = {
  creativity: 'Creativity',
  stability: 'Stability',
  mood: 'Mood',
  vocabulary: 'Vocabulary'
};

const FACET_COLORS: Record<BrainFacet, string> = {
  creativity: '#c084fc',
  stability: '#38bdf8',
  mood: '#f97316',
  vocabulary: '#22c55e'
};

const clamp = (value: number, min = 0, max = 100) => Math.min(Math.max(value, min), max);

function useAutonomousTicks(
  enabled: boolean,
  onTick: () => void,
  intervalMs = 5000
): void {
  useEffect(() => {
    if (!enabled) return undefined;
    const interval = setInterval(onTick, intervalMs);
    return () => clearInterval(interval);
  }, [enabled, intervalMs, onTick]);
}

function facetFromText(text: string): BrainFacet {
  const normalized = text.toLowerCase();
  if (/\bcreative|idea|imagine|paint|compose\b/.test(normalized)) return 'creativity';
  if (/\bstable|steady|reliable|consistent\b/.test(normalized)) return 'stability';
  if (/\bjoy|sad|mood|feel|emotion\b/.test(normalized)) return 'mood';
  if (/\bwords|lexicon|vocab|dictionary|phrase\b/.test(normalized)) return 'vocabulary';
  return 'creativity';
}

function adjustMetrics(metrics: BrainMetrics, facet: BrainFacet, delta: number): BrainMetrics {
  const next: BrainMetrics = { ...metrics };
  next[facet] = clamp(metrics[facet] + delta);
  return next;
}

export interface BrainTelemetryPanelProps {
  initialMetrics?: BrainMetrics;
  autonomousIntervalMs?: number;
}

export function BrainTelemetryPanel({
  initialMetrics = { creativity: 72, stability: 64, mood: 58, vocabulary: 69 },
  autonomousIntervalMs = 4500
}: BrainTelemetryPanelProps) {
  const [metrics, setMetrics] = useState<BrainMetrics>(initialMetrics);
  const [facetSelection, setFacetSelection] = useState<FacetSelection>('auto');
  const [feedText, setFeedText] = useState('');
  const [autonomous, setAutonomous] = useState(false);
  const [events, setEvents] = useState<EventLogEntry[]>([]);

  const idleTick = useMemo(
    () => () => {
      const target: BrainFacet = ['creativity', 'stability', 'mood', 'vocabulary'][
        Math.floor(Math.random() * 4)
      ] as BrainFacet;
      setMetrics((prev) => adjustMetrics(prev, target, Math.random() * 4 - 2));
      const entry: EventLogEntry = {
        id: Date.now(),
        timestamp: Date.now(),
        facet: target,
        label: 'Autonomous IDLE tick',
        detail: `Stabilized ${FACET_LABELS[target]} background activity.`
      };
      setEvents((prev) => [entry, ...prev].slice(0, 30));
    },
    []
  );

  useAutonomousTicks(autonomous, idleTick, autonomousIntervalMs);

  const handleFeedSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = feedText.trim();
    if (!text) return;

    const facet = facetSelection === 'auto' ? facetFromText(text) : facetSelection;
    const emphasis = Math.min(8, Math.max(3, Math.round(text.length / 40)));

    setMetrics((prev) => adjustMetrics(prev, facet, emphasis));
    const entry: EventLogEntry = {
      id: Date.now(),
      timestamp: Date.now(),
      facet,
      label: facetSelection === 'auto' ? 'Auto-faceted Feed' : `${FACET_LABELS[facet]} Feed`,
      detail: text
    };

    setEvents((prev) => [entry, ...prev].slice(0, 30));

    setFeedText('');
  };

  const handleAutonomousToggle = (checked: boolean) => {
    setAutonomous(checked);
    const entry: EventLogEntry = {
      id: Date.now(),
      timestamp: Date.now(),
      facet: 'auto',
      label: checked ? 'Autonomous mode enabled' : 'Autonomous mode disabled',
      detail: checked
        ? 'Initiating autonomous IDLE ticks for background calibration.'
        : 'Halting autonomous adjustments.'
    };

    setEvents((prev) => [entry, ...prev].slice(0, 30));
  };

  const renderMeter = (facet: BrainFacet) => (
    <div
      key={facet}
      style={{
        background: 'rgba(15,23,42,0.8)',
        border: '1px solid #1f2937',
        borderRadius: 12,
        padding: 16,
        boxShadow: '0 10px 30px rgba(0,0,0,0.25)'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ color: '#cbd5e1', fontWeight: 600 }}>{FACET_LABELS[facet]}</div>
        <div style={{ color: FACET_COLORS[facet], fontWeight: 700, fontSize: 18 }}>
          {Math.round(metrics[facet])}
        </div>
      </div>
      <div
        style={{
          marginTop: 10,
          height: 10,
          borderRadius: 999,
          background: 'rgba(51,65,85,0.7)',
          overflow: 'hidden'
        }}
      >
        <div
          style={{
            width: `${clamp(metrics[facet])}%`,
            background: `linear-gradient(to right, ${FACET_COLORS[facet]}, #0ea5e9)`,
            height: '100%',
            transition: 'width 0.3s ease'
          }}
        />
      </div>
    </div>
  );

  const renderEvent = (entry: EventLogEntry) => (
    <div
      key={entry.id}
      style={{
        padding: '10px 12px',
        borderRadius: 12,
        border: '1px solid #1e293b',
        background: 'rgba(15,23,42,0.7)',
        display: 'grid',
        gridTemplateColumns: '120px 1fr',
        gap: 8
      }}
    >
      <div style={{ color: '#94a3b8', fontSize: 12 }}>
        {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
      <div>
        <div style={{ color: '#e2e8f0', fontWeight: 600 }}>{entry.label}</div>
        <div style={{ color: '#cbd5e1', fontSize: 13 }}>{entry.detail}</div>
      </div>
    </div>
  );

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1.25fr 1fr',
        gap: 20,
        background: 'radial-gradient(circle at 20% 20%, rgba(148,163,184,0.15), transparent 30%), radial-gradient(circle at 80% 0%, rgba(167,139,250,0.18), transparent 25%)',
        padding: 20,
        borderRadius: 20,
        border: '1px solid #1f2937'
      }}
    >
      <div style={{ display: 'grid', gridTemplateRows: 'auto auto 1fr', gap: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
          {(['creativity', 'stability', 'mood', 'vocabulary'] as BrainFacet[]).map(renderMeter)}
        </div>

        <form
          onSubmit={handleFeedSubmit}
          style={{
            background: 'rgba(15,23,42,0.85)',
            border: '1px solid #1e293b',
            borderRadius: 16,
            padding: 16,
            display: 'grid',
            gap: 10,
            boxShadow: '0 15px 35px rgba(0,0,0,0.25)'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Feed console</div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#cbd5e1', fontSize: 13 }}>
              <input
                type="checkbox"
                checked={autonomous}
                onChange={(e) => handleAutonomousToggle(e.target.checked)}
                style={{ width: 16, height: 16 }}
              />
              Autonomous mode (IDLE ticks)
            </label>
          </div>
          <textarea
            value={feedText}
            onChange={(e) => setFeedText(e.target.value)}
            placeholder="Inject cognitive stimulus or prompt (auto-faceted by default)"
            style={{
              background: '#0b1220',
              color: '#e5e7eb',
              borderRadius: 12,
              border: '1px solid #1f2937',
              padding: 12,
              minHeight: 90,
              fontSize: 14,
              resize: 'vertical'
            }}
          />
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <select
              value={facetSelection}
              onChange={(e) => setFacetSelection(e.target.value as FacetSelection)}
              style={{
                background: '#0b1220',
                color: '#e5e7eb',
                borderRadius: 12,
                border: '1px solid #1f2937',
                padding: '10px 12px',
                minWidth: 180
              }}
            >
              <option value="auto">Auto-detect facet</option>
              <option value="creativity">Creativity</option>
              <option value="stability">Stability</option>
              <option value="mood">Mood</option>
              <option value="vocabulary">Vocabulary</option>
            </select>
            <button
              type="submit"
              style={{
                background: 'linear-gradient(135deg, #6366f1, #22d3ee)',
                color: '#0b1220',
                border: 'none',
                borderRadius: 12,
                padding: '10px 16px',
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 10px 25px rgba(45,212,191,0.25)'
              }}
            >
              Send feed
            </button>
          </div>
        </form>

        <div
          style={{
            background: 'rgba(15,23,42,0.85)',
            border: '1px solid #1e293b',
            borderRadius: 16,
            padding: 16,
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            maxHeight: 260,
            overflow: 'auto'
          }}
        >
          <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Event log</div>
          {events.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: 13 }}>
              Awaiting stimuli. Feed the model or enable autonomous mode to generate activity.
            </div>
          ) : (
            events.map(renderEvent)
          )}
        </div>
      </div>

      <div
        style={{
          background: 'rgba(15,23,42,0.92)',
          border: '1px solid #1f2937',
          borderRadius: 18,
          padding: 16,
          display: 'grid',
          gap: 14,
          boxShadow: '0 18px 40px rgba(0,0,0,0.25)'
        }}
      >
        <div>
          <div style={{ color: '#cbd5e1', fontSize: 13, marginBottom: 4 }}>Mode</div>
          <div style={{ color: '#e2e8f0', fontWeight: 700 }}>Brain telemetry monitor</div>
          <div style={{ color: '#94a3b8', fontSize: 13 }}>
            Track cognitive vitals, inject targeted feeds, and observe autonomous IDLE balancing ticks.
          </div>
        </div>
        <div
          style={{
            background: 'rgba(79,70,229,0.1)',
            border: '1px dashed rgba(99,102,241,0.7)',
            borderRadius: 14,
            padding: 14,
            color: '#cbd5e1',
            fontSize: 14
          }}
        >
          <div style={{ fontWeight: 700, color: '#a5b4fc', marginBottom: 6 }}>How it works</div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.4 }}>
            <li>Faceted feeds reinforce targeted cognitive pathways.</li>
            <li>Auto-detect mode guesses the best facet from your text.</li>
            <li>Autonomous mode issues periodic IDLE ticks to keep the system balanced.</li>
          </ul>
        </div>
        <div style={{ display: 'grid', gap: 10 }}>
          {(['creativity', 'stability', 'mood', 'vocabulary'] as BrainFacet[]).map((facet) => (
            <div key={facet} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: 999,
                  background: FACET_COLORS[facet],
                  boxShadow: `0 0 0 6px ${FACET_COLORS[facet]}22`
                }}
              />
              <div style={{ color: '#cbd5e1' }}>{FACET_LABELS[facet]}</div>
              <div style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 12 }}>
                Trending {metrics[facet] >= 65 ? '↑' : '→'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
