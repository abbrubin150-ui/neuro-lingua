import React, { Component, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * ErrorBoundary catches and displays React errors gracefully
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
            color: '#e2e8f0',
            padding: 20,
            fontFamily: "'Segoe UI', system-ui, sans-serif",
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <div
            style={{
              background: 'rgba(30,41,59,0.95)',
              border: '1px solid #334155',
              borderRadius: 16,
              padding: 40,
              maxWidth: 600,
              textAlign: 'center'
            }}
          >
            <h1 style={{ color: '#ef4444', marginTop: 0 }}>⚠️ Something went wrong</h1>
            <p style={{ color: '#94a3b8', fontSize: 14, lineHeight: 1.6 }}>
              The application encountered an unexpected error. Please try refreshing the page or
              resetting your model.
            </p>
            {this.state.error && (
              <details style={{ marginTop: 20, textAlign: 'left' }}>
                <summary
                  style={{ cursor: 'pointer', color: '#60a5fa', fontSize: 13, marginBottom: 10 }}
                >
                  Error Details
                </summary>
                <pre
                  style={{
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 8,
                    padding: 12,
                    fontSize: 12,
                    color: '#f87171',
                    overflow: 'auto',
                    maxHeight: 200
                  }}
                >
                  {this.state.error.toString()}
                  {'\n'}
                  {this.state.error.stack}
                </pre>
              </details>
            )}
            <button
              onClick={() => window.location.reload()}
              style={{
                marginTop: 20,
                padding: '12px 20px',
                background: '#2563eb',
                border: 'none',
                borderRadius: 10,
                color: 'white',
                fontWeight: 600,
                cursor: 'pointer',
                fontSize: 14
              }}
            >
              🔄 Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
