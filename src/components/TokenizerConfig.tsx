import React, { useRef } from 'react';
import type { TokenizerConfig as TokenizerConfigType, TokenizerMode } from '../lib/ProNeuralLM';
import { isValidRegex, parseTokenizerConfig, downloadBlob } from '../lib/utils';
import { DEFAULT_CUSTOM_TOKENIZER_PATTERN, EXPORT_FILENAMES } from '../config/constants';

interface TokenizerConfigProps {
  config: TokenizerConfigType;
  customPattern: string;
  error: string | null;
  onConfigChange: (config: TokenizerConfigType) => void;
  onCustomPatternChange: (pattern: string) => void;
  onError: (error: string | null) => void;
  onMessage: (message: string) => void;
}

/**
 * TokenizerConfig manages tokenizer mode selection and custom pattern configuration
 */
export function TokenizerConfig({
  config,
  customPattern,
  error,
  onConfigChange,
  onCustomPatternChange,
  onError,
  onMessage
}: TokenizerConfigProps) {
  const importRef = useRef<HTMLInputElement>(null);

  const handleModeChange = (mode: TokenizerMode) => {
    if (mode === 'custom') {
      const pattern = customPattern || DEFAULT_CUSTOM_TOKENIZER_PATTERN;
      if (isValidRegex(pattern)) {
        onError(null);
        onConfigChange({ mode: 'custom', pattern });
      } else {
        onError('Provide a valid regular expression (without slashes) for custom mode.');
      }
      return;
    }
    onError(null);
    onConfigChange({ mode });
  };

  const handlePatternChange = (value: string) => {
    onCustomPatternChange(value);
    if (!value.trim()) {
      onError('Enter a regular expression pattern to enable custom tokenization.');
      return;
    }
    if (isValidRegex(value)) {
      onError(null);
      onConfigChange({ mode: 'custom', pattern: value });
    } else {
      onError('Invalid regex. The previous tokenizer remains active.');
    }
  };

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    downloadBlob(blob, EXPORT_FILENAMES.TOKENIZER);
  };

  const handleImport = (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const raw = JSON.parse(String(reader.result));
        const parsed = parseTokenizerConfig(raw);
        if (parsed.mode === 'custom' && parsed.pattern) {
          if (!isValidRegex(parsed.pattern)) {
            throw new Error('Invalid regex in tokenizer config');
          }
          onCustomPatternChange(parsed.pattern);
        } else if (parsed.mode === 'custom') {
          onCustomPatternChange('');
        }
        onError(null);
        onConfigChange(parsed);
        onMessage('🧩 Tokenizer configuration imported.');
      } catch (err) {
        console.warn('Failed to import tokenizer config', err);
        onError('Failed to import tokenizer config. Check the file structure.');
        onMessage('❌ Tokenizer import failed. Please verify the file.');
      }
    };
    reader.readAsText(file);
    ev.target.value = '';
  };

  return (
    <>
      <div
        style={{
          marginTop: 16,
          display: 'grid',
          gridTemplateColumns: config.mode === 'custom' ? '1fr 1fr' : '1fr',
          gap: 12,
          alignItems: 'end'
        }}
      >
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Tokenizer Mode</div>
          <select
            aria-label="Tokenizer mode"
            value={config.mode}
            onChange={(e) => handleModeChange(e.target.value as TokenizerMode)}
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          >
            <option value="unicode">Unicode (all scripts)</option>
            <option value="ascii">ASCII (a-z, digits)</option>
            <option value="custom">Custom RegExp</option>
          </select>
        </div>
        {config.mode === 'custom' && (
          <div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>Custom pattern (JS RegExp)</div>
            <input
              aria-label="Custom tokenizer pattern"
              type="text"
              value={customPattern}
              onChange={(e) => handlePatternChange(e.target.value)}
              placeholder="e.g. [^\\p{L}\\d\\s'-]"
              style={{
                width: '100%',
                background: '#1e293b',
                border: '1px solid #475569',
                borderRadius: 6,
                padding: 8,
                color: 'white'
              }}
            />
          </div>
        )}
      </div>
      {error && <div style={{ fontSize: 12, color: '#f87171', marginTop: 6 }}>{error}</div>}
      <div
        style={{
          display: 'flex',
          gap: 8,
          flexWrap: 'wrap',
          marginTop: 8,
          alignItems: 'center'
        }}
      >
        <button
          onClick={handleExport}
          style={{
            padding: '8px 12px',
            background: '#14b8a6',
            border: 'none',
            borderRadius: 8,
            color: 'white',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          📤 Export tokenizer
        </button>
        <button
          onClick={() => importRef.current?.click()}
          style={{
            padding: '8px 12px',
            background: '#6366f1',
            border: 'none',
            borderRadius: 8,
            color: 'white',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          📥 Import tokenizer
        </button>
        <input
          ref={importRef}
          type="file"
          accept="application/json"
          onChange={handleImport}
          style={{ display: 'none' }}
        />
        <span style={{ fontSize: 12, color: '#94a3b8' }}>
          Tip: Unicode captures multilingual corpora. Switch to ASCII for code-like datasets.
        </span>
      </div>
    </>
  );
}
