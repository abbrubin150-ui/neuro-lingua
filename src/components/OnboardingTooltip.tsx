import React, { useState } from 'react';
import type { OnboardingCardStrings } from './OnboardingCard';

interface OnboardingTooltipProps {
  label: string;
  closeLabel: string;
  direction: 'ltr' | 'rtl';
  strings: Pick<
    OnboardingCardStrings,
    'bulletPauseResume' | 'bulletImportExport' | 'bulletPersistence'
  >;
}

export function OnboardingTooltip({
  label,
  closeLabel,
  direction,
  strings
}: OnboardingTooltipProps) {
  const [open, setOpen] = useState(false);
  const alignmentStyle = direction === 'rtl' ? { right: 0 } : { left: 0 };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        aria-label={open ? closeLabel : label}
        style={{
          padding: '6px 12px',
          borderRadius: 999,
          border: '1px solid rgba(148,163,184,0.4)',
          background: open ? 'rgba(37,99,235,0.2)' : 'rgba(15,23,42,0.6)',
          color: '#bfdbfe',
          cursor: 'pointer',
          fontSize: 12,
          fontWeight: 600
        }}
      >
        {open ? closeLabel : label}
      </button>
      {open && (
        <div
          style={{
            position: 'absolute',
            top: 'calc(100% + 8px)',
            minWidth: 240,
            background: 'rgba(15,23,42,0.95)',
            border: '1px solid #334155',
            borderRadius: 12,
            padding: 12,
            boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
            zIndex: 10,
            ...alignmentStyle
          }}
          role="dialog"
          aria-label="Onboarding tips"
        >
          <div style={{ fontWeight: 600, color: '#e0e7ff', marginBottom: 8 }}>⚡ Quick Tips</div>
          <ul
            style={{
              margin: 0,
              paddingInlineStart: direction === 'rtl' ? 18 : 20,
              color: '#cbd5f5',
              fontSize: 12,
              lineHeight: 1.5
            }}
          >
            <li>{strings.bulletPauseResume}</li>
            <li>{strings.bulletImportExport}</li>
            <li>{strings.bulletPersistence}</li>
          </ul>
        </div>
      )}
    </div>
  );
}
