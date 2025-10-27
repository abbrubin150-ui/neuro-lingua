import React from 'react';

export interface Message {
  type: 'system' | 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

interface ChatInterfaceProps {
  messages: Message[];
  input: string;
  modelExists: boolean;
  onInputChange: (value: string) => void;
  onGenerate: () => void;
}

/**
 * ChatInterface provides a chat UI for interacting with the trained model
 */
export function ChatInterface({
  messages,
  input,
  modelExists,
  onInputChange,
  onGenerate
}: ChatInterfaceProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onGenerate();
    }
  };

  return (
    <div
      role="region"
      aria-label="Chat interface"
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20,
        display: 'flex',
        flexDirection: 'column',
        height: 600
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
        <h3 style={{ color: '#60a5fa', margin: 0 }}>💬 Chat Console</h3>
        <div style={{ fontSize: 12, color: '#94a3b8' }} aria-live="polite">
          {messages.filter((m) => m.type === 'assistant').length} replies
        </div>
      </div>

      <div
        role="log"
        aria-label="Chat messages"
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
            aria-label={`${m.type === 'user' ? 'User' : m.type === 'assistant' ? 'Model' : 'System'} message`}
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
              {m.type === 'user' ? '👤 You' : m.type === 'assistant' ? '🤖 Model' : '⚙️ System'}
              {m.timestamp && (
                <span style={{ marginLeft: 8 }}>
                  {new Date(m.timestamp).toLocaleTimeString('en-US')}
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

      <div style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
        <textarea
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={modelExists ? 'Type a message for the model…' : 'Train the model first…'}
          aria-label="Chat prompt"
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
          aria-label="Generate text from model"
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
          ✨ Generate
        </button>
      </div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>
        💡 Press Shift+Enter to add a new line. Responses reflect the active sampling mode and
        temperature.
      </div>
    </div>
  );
}
