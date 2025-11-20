import React from 'react';

export interface Message {
  type: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

export interface ChatInterfaceStrings {
  title: string;
  replies: (count: number) => string;
  regionAria: string;
  logAria: string;
  inputAria: string;
  generateAria: string;
  placeholderReady: string;
  placeholderEmpty: string;
  generate: string;
  tip: string;
  userLabel: string;
  assistantLabel: string;
  systemLabel: string;
  userRole: string;
  assistantRole: string;
  systemRole: string;
  messageSuffix: string;
}

interface ChatInterfaceProps {
  messages: Message[];
  input: string;
  modelExists: boolean;
  onInputChange: (value: string) => void;
  onGenerate: () => void;
  strings: ChatInterfaceStrings;
  direction: 'ltr' | 'rtl';
  locale: 'en' | 'he';
  // Bayesian inference
  useBayesian?: boolean;
  onUseBayesianChange?: (value: boolean) => void;
  confidence?: number | null;
}

/**
 * ChatInterface provides a chat UI for interacting with the trained model
 */
export function ChatInterface({
  messages,
  input,
  modelExists,
  onInputChange,
  onGenerate,
  strings,
  direction,
  locale,
  useBayesian = false,
  onUseBayesianChange,
  confidence = null
}: ChatInterfaceProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onGenerate();
    }
  };

  const assistantReplies = messages.filter((m) => m.type === 'assistant').length;

  return (
    <div
      role="region"
      aria-label={strings.regionAria}
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        display: 'flex',
        flexDirection: 'column',
        height: 600,
        direction
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16
        }}
      >
        <h3 style={{ color: '#60a5fa', margin: 0 }}>{strings.title}</h3>
        <div style={{ fontSize: 12, color: '#94a3b8' }} aria-live="polite">
          {strings.replies(assistantReplies)}
        </div>
      </div>

      <div
        role="log"
        aria-label={strings.logAria}
        aria-live="polite"
        aria-atomic="false"
        style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          marginBottom: 16
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            role="article"
            aria-label={`${
              m.type === 'user'
                ? strings.userRole
                : m.type === 'assistant'
                  ? strings.assistantRole
                  : strings.systemRole
            } ${strings.messageSuffix}`}
            style={{
              padding: '12px 16px',
              borderRadius: 12,
              background:
                m.type === 'user'
                  ? 'linear-gradient(90deg, #3730a3, #5b21b6)'
                  : m.type === 'assistant'
                    ? 'linear-gradient(90deg, #1e293b, #334155)'
                    : 'linear-gradient(90deg, #065f46, #059669)',
              border: '1px solid #475569',
              wordWrap: 'break-word',
              position: 'relative'
            }}
          >
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
              {m.type === 'user'
                ? strings.userLabel
                : m.type === 'assistant'
                  ? strings.assistantLabel
                  : strings.systemLabel}
              {m.timestamp && (
                <span style={{ marginInlineStart: 8 }}>
                  {new Date(m.timestamp).toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US')}
                </span>
              )}
            </div>
            <div
              style={{
                fontFamily:
                  m.type === 'assistant'
                    ? "'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', monospace"
                    : 'inherit'
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>

      {/* Bayesian Inference Toggle */}
      {onUseBayesianChange && (
        <div
          style={{
            marginBottom: 12,
            padding: '10px 12px',
            background: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            borderRadius: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 12
          }}
        >
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: 'pointer',
              flex: 1
            }}
          >
            <input
              type="checkbox"
              checked={useBayesian}
              onChange={(e) => onUseBayesianChange(e.target.checked)}
              style={{
                width: 16,
                height: 16,
                cursor: 'pointer'
              }}
            />
            <span
              style={{ fontSize: 13, color: '#a78bfa', fontWeight: 600, cursor: 'help' }}
              title="Bayesian Inference uses Monte Carlo sampling to generate multiple predictions and quantify uncertainty. Higher confidence means predictions are more consistent across samples."
            >
              🎲 Bayesian Inference ℹ️
            </span>
            {confidence !== null && useBayesian && (
              <span
                style={{
                  marginInlineStart: 'auto',
                  fontSize: 12,
                  background: 'rgba(34, 197, 94, 0.2)',
                  color: '#34d399',
                  padding: '4px 10px',
                  borderRadius: 6,
                  fontWeight: 600,
                  cursor: 'help'
                }}
                title={`Confidence: ${(confidence * 100).toFixed(0)}%. This represents prediction consistency across Monte Carlo samples. Values close to 1.0 indicate high agreement, values close to 0.0 indicate high uncertainty.`}
              >
                Confidence: {confidence.toFixed(2)}
              </span>
            )}
          </label>
        </div>
      )}
      {useBayesian && (
        <div style={{ fontSize: 11, color: '#cbd5f5', marginBottom: 8 }}>
          💡 Monte Carlo dropout enabled: generating multiple samples for uncertainty quantification
        </div>
      )}

      <div style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
        <textarea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={modelExists ? strings.placeholderReady : strings.placeholderEmpty}
          aria-label={strings.inputAria}
          style={{
            flex: 1,
            padding: '12px 16px',
            background: '#1e293b',
            border: '1px solid #475569',
            borderRadius: 12,
            color: '#e2e8f0',
            fontSize: 14,
            resize: 'none',
            minHeight: 60,
            fontFamily: 'inherit'
          }}
          rows={2}
        />
        <button
          onClick={onGenerate}
          disabled={!modelExists}
          aria-label={strings.generateAria}
          aria-disabled={!modelExists}
          style={{
            padding: '12px 20px',
            background: modelExists ? 'linear-gradient(90deg, #2563eb, #4f46e5)' : '#475569',
            border: 'none',
            borderRadius: 12,
            color: 'white',
            fontWeight: 700,
            cursor: modelExists ? 'pointer' : 'not-allowed',
            alignSelf: 'flex-end',
            minWidth: 100
          }}
        >
          {strings.generate}
        </button>
      </div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>{strings.tip}</div>
    </div>
  );
}
