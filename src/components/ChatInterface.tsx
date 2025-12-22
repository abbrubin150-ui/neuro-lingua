import React, { useState } from 'react';

export interface Message {
  type: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

/**
 * Format messages as plain text for export/copy
 */
function formatMessagesAsText(messages: Message[], locale: 'en' | 'he'): string {
  if (messages.length === 0) {
    return locale === 'he' ? 'אין הודעות לייצא' : 'No messages to export';
  }

  const lines: string[] = [];
  lines.push(
    locale === 'he' ? '=== שיחה עם Neuro-Lingua ===' : '=== Neuro-Lingua Chat Conversation ==='
  );
  lines.push(
    locale === 'he'
      ? `ייצוא בתאריך: ${new Date().toLocaleString('he-IL')}`
      : `Exported: ${new Date().toLocaleString('en-US')}`
  );
  lines.push('');

  messages.forEach((msg, idx) => {
    const roleLabel =
      msg.type === 'user'
        ? locale === 'he'
          ? 'משתמש'
          : 'User'
        : msg.type === 'assistant'
          ? locale === 'he'
            ? 'עוזר'
            : 'Assistant'
          : locale === 'he'
            ? 'מערכת'
            : 'System';

    const timestamp = msg.timestamp
      ? ` [${new Date(msg.timestamp).toLocaleTimeString(locale === 'he' ? 'he-IL' : 'en-US')}]`
      : '';

    lines.push(`${roleLabel}${timestamp}:`);
    lines.push(msg.content);
    if (idx < messages.length - 1) {
      lines.push('');
    }
  });

  return lines.join('\n');
}

/**
 * Download a blob as a file
 */
function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
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
  // Generation controls
  temperature?: number;
  onTemperatureChange?: (value: number) => void;
  maxTokens?: number;
  onMaxTokensChange?: (value: number) => void;
  samplingMode?: 'off' | 'topk' | 'topp' | 'typical' | 'mirostat';
  onSamplingModeChange?: (value: 'off' | 'topk' | 'topp' | 'typical' | 'mirostat') => void;
  mirostatEnabled?: boolean;
  onMirostatEnabledChange?: (value: boolean) => void;
  topK?: number;
  onTopKChange?: (value: number) => void;
  topP?: number;
  onTopPChange?: (value: number) => void;
  typicalTau?: number;
  onTypicalTauChange?: (value: number) => void;
  mirostatMu?: number;
  onMirostatMuChange?: (value: number) => void;
  mirostatTau?: number;
  onMirostatTauChange?: (value: number) => void;
  mirostatEta?: number;
  onMirostatEtaChange?: (value: number) => void;
  frequencyPenalty?: number;
  onFrequencyPenaltyChange?: (value: number) => void;
  presencePenalty?: number;
  onPresencePenaltyChange?: (value: number) => void;
  useBeamSearch?: boolean;
  onUseBeamSearchChange?: (value: boolean) => void;
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
  confidence = null,
  temperature = 0.8,
  onTemperatureChange,
  maxTokens = 25,
  onMaxTokensChange,
  samplingMode = 'topp',
  onSamplingModeChange,
  mirostatEnabled = samplingMode === 'mirostat',
  onMirostatEnabledChange,
  topK = 20,
  onTopKChange,
  topP = 0.9,
  onTopPChange,
  typicalTau = 0.9,
  onTypicalTauChange,
  mirostatMu = 10,
  onMirostatMuChange,
  mirostatTau = 5,
  onMirostatTauChange,
  mirostatEta = 0.1,
  onMirostatEtaChange,
  frequencyPenalty = 0,
  onFrequencyPenaltyChange,
  presencePenalty = 0,
  onPresencePenaltyChange,
  useBeamSearch = false,
  onUseBeamSearchChange
}: ChatInterfaceProps) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onGenerate();
    }
  };

  const handleCopyChat = async () => {
    if (messages.length === 0) return;

    try {
      const text = formatMessagesAsText(messages, locale);
      await navigator.clipboard.writeText(text);
      setCopyStatus('success');
      setTimeout(() => setCopyStatus('idle'), 2000);
    } catch (error) {
      console.error('Failed to copy chat:', error);
      setCopyStatus('error');
      setTimeout(() => setCopyStatus('idle'), 2000);
    }
  };

  const handleExportText = () => {
    if (messages.length === 0) return;

    const text = formatMessagesAsText(messages, locale);
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const filename = `neuro-lingua-chat-${new Date().toISOString().slice(0, 10)}.txt`;
    downloadBlob(blob, filename);
  };

  const handleExportJSON = () => {
    if (messages.length === 0) return;

    const exportData = {
      exported: new Date().toISOString(),
      messageCount: messages.length,
      locale,
      messages: messages.map((msg) => ({
        type: msg.type,
        content: msg.content,
        timestamp: msg.timestamp
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json;charset=utf-8'
    });
    const filename = `neuro-lingua-chat-${new Date().toISOString().slice(0, 10)}.json`;
    downloadBlob(blob, filename);
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
          marginBottom: 16,
          gap: 12
        }}
      >
        <h3 style={{ color: '#60a5fa', margin: 0 }}>{strings.title}</h3>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }} aria-live="polite">
            {strings.replies(assistantReplies)}
          </div>

          {messages.length > 0 && (
            <div style={{ display: 'flex', gap: 4 }}>
              {/* Copy Chat Button */}
              <button
                onClick={handleCopyChat}
                title={locale === 'he' ? 'העתק שיחה ללוח' : 'Copy chat to clipboard'}
                style={{
                  padding: '6px 12px',
                  background:
                    copyStatus === 'success'
                      ? 'rgba(34, 197, 94, 0.2)'
                      : copyStatus === 'error'
                        ? 'rgba(239, 68, 68, 0.2)'
                        : 'rgba(59, 130, 246, 0.2)',
                  border: `1px solid ${
                    copyStatus === 'success'
                      ? 'rgba(34, 197, 94, 0.4)'
                      : copyStatus === 'error'
                        ? 'rgba(239, 68, 68, 0.4)'
                        : 'rgba(59, 130, 246, 0.4)'
                  }`,
                  borderRadius: 8,
                  color:
                    copyStatus === 'success'
                      ? '#34d399'
                      : copyStatus === 'error'
                        ? '#f87171'
                        : '#60a5fa',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  whiteSpace: 'nowrap'
                }}
              >
                {copyStatus === 'success'
                  ? locale === 'he'
                    ? '✓ הועתק'
                    : '✓ Copied'
                  : copyStatus === 'error'
                    ? locale === 'he'
                      ? '✗ שגיאה'
                      : '✗ Error'
                    : locale === 'he'
                      ? '📋 העתק'
                      : '📋 Copy'}
              </button>

              {/* Export Dropdown */}
              <details style={{ position: 'relative', display: 'inline-block' }}>
                <summary
                  style={{
                    padding: '6px 12px',
                    background: 'rgba(139, 92, 246, 0.2)',
                    border: '1px solid rgba(139, 92, 246, 0.4)',
                    borderRadius: 8,
                    color: '#a78bfa',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    listStyle: 'none',
                    whiteSpace: 'nowrap',
                    userSelect: 'none'
                  }}
                  title={locale === 'he' ? 'ייצא שיחה' : 'Export chat'}
                >
                  💾 {locale === 'he' ? 'ייצוא' : 'Export'}
                </summary>
                <div
                  style={{
                    position: 'absolute',
                    top: '100%',
                    [direction === 'rtl' ? 'left' : 'right']: 0,
                    marginTop: 4,
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 8,
                    padding: 4,
                    zIndex: 100,
                    minWidth: 140,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
                  }}
                >
                  <button
                    onClick={handleExportText}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '8px 12px',
                      background: 'transparent',
                      border: 'none',
                      borderRadius: 6,
                      color: '#e2e8f0',
                      fontSize: 12,
                      textAlign: direction === 'rtl' ? 'right' : 'left',
                      cursor: 'pointer',
                      transition: 'background 0.2s'
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)')
                    }
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    📄 {locale === 'he' ? 'קובץ טקסט (.txt)' : 'Text file (.txt)'}
                  </button>
                  <button
                    onClick={handleExportJSON}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '8px 12px',
                      background: 'transparent',
                      border: 'none',
                      borderRadius: 6,
                      color: '#e2e8f0',
                      fontSize: 12,
                      textAlign: direction === 'rtl' ? 'right' : 'left',
                      cursor: 'pointer',
                      transition: 'background 0.2s'
                    }}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)')
                    }
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    📦 {locale === 'he' ? 'קובץ JSON (.json)' : 'JSON file (.json)'}
                  </button>
                </div>
              </details>
            </div>
          )}
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

      {/* Generation Controls */}
      <details
        open
        style={{
          marginBottom: 12,
          padding: '12px 14px',
          background: 'rgba(59, 130, 246, 0.1)',
          border: '1px solid rgba(59, 130, 246, 0.3)',
          borderRadius: 10
        }}
      >
        <summary
          style={{
            cursor: 'pointer',
            fontWeight: 600,
            color: '#60a5fa',
            fontSize: 13,
            marginBottom: 8
          }}
        >
          ⚙️ Generation Settings
        </summary>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px' }}>
          {/* Sampling Mode */}
          {onSamplingModeChange && (
            <div>
              <label
                htmlFor="sampling-mode-select"
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
              >
                Sampling Mode
              </label>
              <select
                id="sampling-mode-select"
                value={samplingMode}
                onChange={(e) =>
                  onSamplingModeChange(
                    e.target.value as 'off' | 'topk' | 'topp' | 'typical' | 'mirostat'
                  )
                }
                style={{
                  width: '100%',
                  padding: '6px 8px',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  color: '#e2e8f0',
                  fontSize: 12
                }}
              >
                <option value="off">Greedy (deterministic)</option>
                <option value="topk">Top-K</option>
                <option value="topp">Top-P (Nucleus)</option>
                <option value="typical">Typical (entropy-based)</option>
                <option value="mirostat">Mirostat v2</option>
              </select>
            </div>
          )}

          {onMirostatEnabledChange && (
            <div style={{ alignSelf: 'flex-start' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: 'pointer',
                  fontSize: 12,
                  color: '#cbd5e1'
                }}
              >
                <input
                  type="checkbox"
                  checked={mirostatEnabled}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    onMirostatEnabledChange(enabled);
                    if (onSamplingModeChange) {
                      onSamplingModeChange(enabled ? 'mirostat' : 'topp');
                    }
                  }}
                  style={{ width: 14, height: 14, cursor: 'pointer' }}
                />
                <span title="Enable adaptive Mirostat v2 sampling and expose its parameters">
                  Toggle Mirostat v2
                </span>
              </label>
            </div>
          )}

          {/* Max Tokens */}
          {onMaxTokensChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
              >
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range"
                min="1"
                max="200"
                value={maxTokens}
                onChange={(e) => onMaxTokensChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Temperature */}
          {onTemperatureChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
              >
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.05"
                max="2"
                step="0.05"
                value={temperature}
                onChange={(e) => onTemperatureChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Top-K (shown when topk mode) */}
          {samplingMode === 'topk' && onTopKChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
              >
                Top-K: {topK}
              </label>
              <input
                type="range"
                min="1"
                max="100"
                value={topK}
                onChange={(e) => onTopKChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Top-P (shown when topp mode) */}
          {samplingMode === 'topp' && onTopPChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
              >
                Top-P: {topP.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.99"
                step="0.01"
                value={topP}
                onChange={(e) => onTopPChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Typical Tau (shown when typical mode) */}
          {samplingMode === 'typical' && onTypicalTauChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
                title="Threshold for typicality - keeps tokens with information content close to entropy"
              >
                Typical Tau: {typicalTau.toFixed(2)} ℹ️
              </label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={typicalTau}
                onChange={(e) => onTypicalTauChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Mirostat v2 controls */}
          {samplingMode === 'mirostat' &&
            onMirostatTauChange &&
            onMirostatEtaChange &&
            onMirostatMuChange && (
            <div
              style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}
            >
              <div>
                <label
                  style={{
                    display: 'block',
                    fontSize: 11,
                    color: '#94a3b8',
                    marginBottom: 4,
                    fontWeight: 600
                  }}
                  title="Initial truncation window size μ (higher = more exploration)."
                >
                  Initial μ: {mirostatMu.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="2"
                  max="16"
                  step="0.5"
                  value={mirostatMu}
                  onChange={(e) => onMirostatMuChange(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label
                  style={{
                    display: 'block',
                    fontSize: 11,
                    color: '#94a3b8',
                    marginBottom: 4,
                    fontWeight: 600
                  }}
                >
                  Target Surprise (τ): {mirostatTau.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  value={mirostatTau}
                  onChange={(e) => onMirostatTauChange(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <label
                  style={{
                    display: 'block',
                    fontSize: 11,
                    color: '#94a3b8',
                    marginBottom: 4,
                    fontWeight: 600
                  }}
                >
                  Adapt Rate (η): {mirostatEta.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={mirostatEta}
                  onChange={(e) => onMirostatEtaChange(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}

          {/* Frequency Penalty */}
          {onFrequencyPenaltyChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
                title="Penalize tokens based on how often they appear (reduces repetition)"
              >
                Frequency Penalty: {frequencyPenalty.toFixed(2)} ℹ️
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={frequencyPenalty}
                onChange={(e) => onFrequencyPenaltyChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Presence Penalty */}
          {onPresencePenaltyChange && (
            <div>
              <label
                style={{
                  display: 'block',
                  fontSize: 11,
                  color: '#94a3b8',
                  marginBottom: 4,
                  fontWeight: 600
                }}
                title="Penalize tokens that have appeared at all (encourages diversity)"
              >
                Presence Penalty: {presencePenalty.toFixed(2)} ℹ️
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={presencePenalty}
                onChange={(e) => onPresencePenaltyChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {/* Beam Search Toggle */}
          {onUseBeamSearchChange && (
            <div style={{ gridColumn: '1 / -1' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: 'pointer',
                  fontSize: 12,
                  color: '#cbd5e1'
                }}
              >
                <input
                  type="checkbox"
                  checked={useBeamSearch}
                  onChange={(e) => onUseBeamSearchChange(e.target.checked)}
                  style={{ width: 14, height: 14, cursor: 'pointer' }}
                />
                <span>Use Beam Search (AdvancedNeuralLM only)</span>
              </label>
            </div>
          )}
        </div>
      </details>

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
