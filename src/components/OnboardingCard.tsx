import React from 'react';
import { STORAGE_KEYS } from '../config/constants';

export interface OnboardingCardStrings {
  welcomeTitle: string;
  privacyWarningTitle: string;
  privacyWarningLead: string;
  privacyWarningBody: string;
  bulletPauseResume: string;
  bulletImportExport: string;
  bulletPersistence: string;
  gotIt: string;
  reopenInfo: string;
}

interface OnboardingCardProps {
  show: boolean;
  onDismiss: () => void;
  strings: OnboardingCardStrings;
  direction: 'ltr' | 'rtl';
}

/**
 * OnboardingCard displays welcome information to new users
 * Shows tips about session persistence, import/export, and pause/resume
 */
export function OnboardingCard({ show, onDismiss, strings, direction }: OnboardingCardProps) {
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
        gap: 12,
        direction
      }}
    >
      <div style={{ fontWeight: 700, fontSize: '1.05rem', color: '#60a5fa' }}>{strings.welcomeTitle}</div>

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
          ⚠️ {strings.privacyWarningTitle}
        </div>
        <div style={{ fontSize: 12, color: '#fecaca', lineHeight: 1.5 }}>
          <strong>{strings.privacyWarningLead}</strong> {strings.privacyWarningBody}
        </div>
      </div>

      <ul
        style={{
          margin: 0,
          paddingInlineStart: 20,
          fontSize: 13,
          color: '#cbd5f5',
          lineHeight: 1.5
        }}
      >
        <li>{strings.bulletPauseResume}</li>
        <li>{strings.bulletImportExport}</li>
        <li>{strings.bulletPersistence}</li>
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
          {strings.gotIt}
        </button>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          {strings.reopenInfo}
        </div>
      </div>
    </div>
  );
}
