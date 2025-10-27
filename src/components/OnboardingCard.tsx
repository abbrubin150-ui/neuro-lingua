import React from 'react';
import { STORAGE_KEYS } from '../config/constants';

interface OnboardingCardProps {
  show: boolean;
  onDismiss: () => void;
}

/**
 * OnboardingCard displays welcome information to new users
 * Shows tips about session persistence, import/export, and pause/resume
 */
export function OnboardingCard({ show, onDismiss }: OnboardingCardProps) {
  if (!show) return null;

  const handleDismiss = () => {
    try {
      localStorage.setItem(STORAGE_KEYS.ONBOARDING_DISMISSED, 'true');
    } catch (err) {
      console.warn('Failed to persist onboarding dismissal', err);
    }
    onDismiss();
  };

  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.95)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        marginBottom: 24,
        display: 'flex',
        flexDirection: 'column',
        gap: 12
      }}
    >
      <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#60a5fa' }}>
        👋 Welcome! Here is how Neuro-Lingua keeps your session in sync
      </div>

      <div
        style={{
          background: 'rgba(239,68,68,0.15)',
          border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 8,
          padding: 12,
          marginTop: 4
        }}
      >
        <div style={{ fontWeight: 700, fontSize: 13, color: '#fca5a5', marginBottom: 6 }}>
          ⚠️ Privacy Warning
        </div>
        <div style={{ fontSize: 12, color: '#fecaca', lineHeight: 1.5 }}>
          <strong>DO NOT train with sensitive data.</strong> This app stores everything in browser
          localStorage (unencrypted). Never use PII, passwords, financial data, medical records, or
          confidential information.
        </div>
      </div>

      <ul
        style={{
          margin: 0,
          paddingLeft: 20,
          fontSize: 13,
          color: '#cbd5f5',
          lineHeight: 1.5
        }}
      >
        <li>
          <strong>Pause / Resume:</strong> use the Stop button to pause training. With{' '}
          <em>Resume training</em> enabled we pick up from the latest checkpoint when you train
          again.
        </li>
        <li>
          <strong>Import / Export:</strong> save models and tokenizer presets to JSON for
          safekeeping or sharing. Importing immediately refreshes the charts and metadata.
        </li>
        <li>
          <strong>Session Persistence:</strong> hyperparameters, corpus text, and tokenizer choices
          live in localStorage, so a refresh restores your workspace automatically.
        </li>
      </ul>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <button
          onClick={handleDismiss}
          style={{
            padding: '10px 16px',
            background: '#2563eb',
            border: 'none',
            borderRadius: 10,
            color: 'white',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          Got it
        </button>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          You can reopen this info from localStorage by clearing the flag.
        </div>
      </div>
    </div>
  );
}
