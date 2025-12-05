/**
 * GovernanceBoard - Monitoring board for autonomous governance
 * Displays alerts, calibration history, and governance ledger
 */

import React, { useState } from 'react';
import type { BoardAlert, CalibrationAction, GovernanceLedgerEntry } from '../types/governance';
import { useProjects } from '../contexts/ProjectContext';

interface GovernanceBoardProps {
  /** Whether to show the board */
  visible?: boolean;
}

/**
 * GovernanceBoard component
 */
export function GovernanceBoard({ visible = true }: GovernanceBoardProps) {
  const {
    getActiveAlerts,
    acknowledgeAlert,
    clearAlerts,
    getCalibrationHistory,
    getGovernanceLedger
  } = useProjects();

  const [activeTab, setActiveTab] = useState<'alerts' | 'calibration' | 'ledger'>('alerts');

  const alerts = getActiveAlerts();
  const calibrationHistory = getCalibrationHistory();
  const ledger = getGovernanceLedger();

  if (!visible) {
    return null;
  }

  return (
    <div className="governance-board" style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🛡️ Governance Board</h3>
        <div style={styles.tabs}>
          <button
            style={activeTab === 'alerts' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('alerts')}
          >
            Alerts {alerts.length > 0 && <span style={styles.badge}>{alerts.length}</span>}
          </button>
          <button
            style={activeTab === 'calibration' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('calibration')}
          >
            Calibration
          </button>
          <button
            style={activeTab === 'ledger' ? styles.tabActive : styles.tab}
            onClick={() => setActiveTab('ledger')}
          >
            Ledger
          </button>
        </div>
      </div>

      <div style={styles.content}>
        {activeTab === 'alerts' && (
          <AlertsPanel
            alerts={alerts}
            onAcknowledge={acknowledgeAlert}
            onClearAll={clearAlerts}
          />
        )}

        {activeTab === 'calibration' && (
          <CalibrationPanel calibrationHistory={calibrationHistory} />
        )}

        {activeTab === 'ledger' && <LedgerPanel ledger={ledger} />}
      </div>
    </div>
  );
}

/**
 * Alerts panel
 */
function AlertsPanel({
  alerts,
  onAcknowledge,
  onClearAll
}: {
  alerts: BoardAlert[];
  onAcknowledge: (id: string) => void;
  onClearAll: () => void;
}) {
  if (alerts.length === 0) {
    return (
      <div style={styles.emptyState}>
        <p>✅ No active alerts</p>
        <p style={styles.emptyStateSubtext}>System is operating normally</p>
      </div>
    );
  }

  return (
    <div>
      <div style={styles.panelHeader}>
        <span style={styles.panelTitle}>{alerts.length} Active Alert{alerts.length > 1 ? 's' : ''}</span>
        <button style={styles.clearButton} onClick={onClearAll}>
          Clear All
        </button>
      </div>

      <div style={styles.alertsList}>
        {alerts.map((alert) => (
          <AlertCard key={alert.id} alert={alert} onAcknowledge={onAcknowledge} />
        ))}
      </div>
    </div>
  );
}

/**
 * Alert card
 */
function AlertCard({
  alert,
  onAcknowledge
}: {
  alert: BoardAlert;
  onAcknowledge: (id: string) => void;
}) {
  const severityColors = {
    info: '#2196F3',
    warning: '#FF9800',
    critical: '#F44336'
  };

  const severityIcons = {
    info: 'ℹ️',
    warning: '⚠️',
    critical: '🚨'
  };

  return (
    <div
      style={{
        ...styles.alertCard,
        borderLeftColor: severityColors[alert.severity]
      }}
    >
      <div style={styles.alertHeader}>
        <span style={styles.alertIcon}>{severityIcons[alert.severity]}</span>
        <span style={styles.alertType}>{alert.type.toUpperCase()}</span>
        <span style={styles.alertSeverity}>{alert.severity}</span>
      </div>

      <div style={styles.alertMessage}>{alert.message}</div>

      {alert.metric && (
        <div style={styles.alertMetric}>
          Metric: {alert.metric}
          {alert.value !== undefined && ` = ${alert.value.toFixed(4)}`}
        </div>
      )}

      <div style={styles.alertFooter}>
        <span style={styles.alertTime}>
          {new Date(alert.timestamp).toLocaleTimeString()}
        </span>
        <button style={styles.acknowledgeButton} onClick={() => onAcknowledge(alert.id)}>
          Acknowledge
        </button>
      </div>
    </div>
  );
}

/**
 * Calibration panel
 */
function CalibrationPanel({
  calibrationHistory
}: {
  calibrationHistory: CalibrationAction[];
}) {
  if (calibrationHistory.length === 0) {
    return (
      <div style={styles.emptyState}>
        <p>📊 No calibrations yet</p>
        <p style={styles.emptyStateSubtext}>
          Governor will calibrate parameters after 2-3 training sessions
        </p>
      </div>
    );
  }

  // Show most recent first
  const sortedHistory = [...calibrationHistory].reverse();

  return (
    <div>
      <div style={styles.panelHeader}>
        <span style={styles.panelTitle}>
          {calibrationHistory.length} Calibration{calibrationHistory.length > 1 ? 's' : ''}
        </span>
      </div>

      <div style={styles.calibrationList}>
        {sortedHistory.map((action) => (
          <CalibrationCard key={action.id} action={action} />
        ))}
      </div>
    </div>
  );
}

/**
 * Calibration card
 */
function CalibrationCard({ action }: { action: CalibrationAction }) {
  const change = action.newValue - action.previousValue;
  const changePercent = ((change / action.previousValue) * 100).toFixed(1);
  const isIncrease = change > 0;

  return (
    <div style={styles.calibrationCard}>
      <div style={styles.calibrationHeader}>
        <span style={styles.calibrationParam}>{action.parameter}</span>
        <span style={styles.calibrationChange}>
          {isIncrease ? '↑' : '↓'} {Math.abs(parseFloat(changePercent))}%
        </span>
      </div>

      <div style={styles.calibrationValues}>
        <span>
          {action.previousValue.toFixed(4)} → {action.newValue.toFixed(4)}
        </span>
      </div>

      <div style={styles.calibrationReason}>{action.reason}</div>

      <div style={styles.calibrationFooter}>
        <span style={styles.calibrationMetric}>Trigger: {action.triggeringMetric}</span>
        <span style={styles.calibrationTime}>
          {new Date(action.timestamp).toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}

/**
 * Ledger panel
 */
function LedgerPanel({ ledger }: { ledger: GovernanceLedgerEntry[] }) {
  if (ledger.length === 0) {
    return (
      <div style={styles.emptyState}>
        <p>📝 No ledger entries</p>
        <p style={styles.emptyStateSubtext}>
          All governance decisions will be recorded here
        </p>
      </div>
    );
  }

  // Show most recent first
  const sortedLedger = [...ledger].reverse();

  return (
    <div>
      <div style={styles.panelHeader}>
        <span style={styles.panelTitle}>
          {ledger.length} Ledger Entr{ledger.length === 1 ? 'y' : 'ies'}
        </span>
      </div>

      <div style={styles.ledgerList}>
        {sortedLedger.map((entry) => (
          <LedgerCard key={entry.id} entry={entry} />
        ))}
      </div>
    </div>
  );
}

/**
 * Ledger card
 */
function LedgerCard({ entry }: { entry: GovernanceLedgerEntry }) {
  const typeIcons = {
    calibration: '⚙️',
    alert: '⚠️',
    decision: '📋',
    'no-action': '⏸️'
  };

  return (
    <div style={styles.ledgerCard}>
      <div style={styles.ledgerHeader}>
        <span style={styles.ledgerIcon}>{typeIcons[entry.type]}</span>
        <span style={styles.ledgerType}>{entry.type.toUpperCase()}</span>
      </div>

      <div style={styles.ledgerDescription}>{entry.description}</div>

      <div style={styles.ledgerFooter}>
        <span style={styles.ledgerSession}>Session: {entry.sessionId.substring(0, 8)}...</span>
        <span style={styles.ledgerTime}>
          {new Date(entry.timestamp).toLocaleString()}
        </span>
      </div>
    </div>
  );
}

/**
 * Styles
 */
const styles = {
  container: {
    border: '1px solid #ddd',
    borderRadius: '8px',
    backgroundColor: '#fff',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    marginBottom: '1rem',
    overflow: 'hidden'
  } as React.CSSProperties,

  header: {
    backgroundColor: '#f5f5f5',
    padding: '1rem',
    borderBottom: '1px solid #ddd'
  } as React.CSSProperties,

  title: {
    margin: '0 0 0.5rem 0',
    fontSize: '1.2rem',
    fontWeight: 'bold'
  } as React.CSSProperties,

  tabs: {
    display: 'flex',
    gap: '0.5rem'
  } as React.CSSProperties,

  tab: {
    padding: '0.5rem 1rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '0.9rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  } as React.CSSProperties,

  tabActive: {
    padding: '0.5rem 1rem',
    border: '1px solid #2196F3',
    borderRadius: '4px',
    backgroundColor: '#E3F2FD',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  } as React.CSSProperties,

  badge: {
    backgroundColor: '#F44336',
    color: '#fff',
    borderRadius: '10px',
    padding: '2px 6px',
    fontSize: '0.75rem',
    fontWeight: 'bold'
  } as React.CSSProperties,

  content: {
    padding: '1rem',
    maxHeight: '400px',
    overflowY: 'auto' as const
  } as React.CSSProperties,

  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem'
  } as React.CSSProperties,

  panelTitle: {
    fontWeight: 'bold',
    fontSize: '1rem'
  } as React.CSSProperties,

  clearButton: {
    padding: '0.25rem 0.75rem',
    backgroundColor: '#f44336',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.85rem'
  } as React.CSSProperties,

  emptyState: {
    textAlign: 'center' as const,
    padding: '2rem',
    color: '#666'
  } as React.CSSProperties,

  emptyStateSubtext: {
    fontSize: '0.9rem',
    color: '#999',
    marginTop: '0.5rem'
  } as React.CSSProperties,

  alertsList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.75rem'
  } as React.CSSProperties,

  alertCard: {
    border: '1px solid #ddd',
    borderLeft: '4px solid',
    borderRadius: '4px',
    padding: '0.75rem',
    backgroundColor: '#fafafa'
  } as React.CSSProperties,

  alertHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem'
  } as React.CSSProperties,

  alertIcon: {
    fontSize: '1.2rem'
  } as React.CSSProperties,

  alertType: {
    fontWeight: 'bold',
    fontSize: '0.85rem'
  } as React.CSSProperties,

  alertSeverity: {
    fontSize: '0.75rem',
    color: '#666',
    marginLeft: 'auto'
  } as React.CSSProperties,

  alertMessage: {
    fontSize: '0.95rem',
    marginBottom: '0.5rem',
    lineHeight: '1.4'
  } as React.CSSProperties,

  alertMetric: {
    fontSize: '0.85rem',
    color: '#666',
    fontFamily: 'monospace',
    marginBottom: '0.5rem'
  } as React.CSSProperties,

  alertFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '0.5rem',
    paddingTop: '0.5rem',
    borderTop: '1px solid #ddd'
  } as React.CSSProperties,

  alertTime: {
    fontSize: '0.75rem',
    color: '#999'
  } as React.CSSProperties,

  acknowledgeButton: {
    padding: '0.25rem 0.75rem',
    backgroundColor: '#4CAF50',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.8rem'
  } as React.CSSProperties,

  calibrationList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.75rem'
  } as React.CSSProperties,

  calibrationCard: {
    border: '1px solid #ddd',
    borderRadius: '4px',
    padding: '0.75rem',
    backgroundColor: '#fafafa'
  } as React.CSSProperties,

  calibrationHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.5rem'
  } as React.CSSProperties,

  calibrationParam: {
    fontWeight: 'bold',
    fontSize: '0.95rem'
  } as React.CSSProperties,

  calibrationChange: {
    fontSize: '0.9rem',
    fontWeight: 'bold',
    color: '#2196F3'
  } as React.CSSProperties,

  calibrationValues: {
    fontSize: '0.9rem',
    fontFamily: 'monospace',
    marginBottom: '0.5rem',
    color: '#444'
  } as React.CSSProperties,

  calibrationReason: {
    fontSize: '0.9rem',
    color: '#666',
    marginBottom: '0.5rem',
    lineHeight: '1.4'
  } as React.CSSProperties,

  calibrationFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.75rem',
    color: '#999',
    paddingTop: '0.5rem',
    borderTop: '1px solid #ddd'
  } as React.CSSProperties,

  calibrationMetric: {
    fontFamily: 'monospace'
  } as React.CSSProperties,

  calibrationTime: {} as React.CSSProperties,

  ledgerList: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '0.75rem'
  } as React.CSSProperties,

  ledgerCard: {
    border: '1px solid #ddd',
    borderRadius: '4px',
    padding: '0.75rem',
    backgroundColor: '#fafafa'
  } as React.CSSProperties,

  ledgerHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.5rem'
  } as React.CSSProperties,

  ledgerIcon: {
    fontSize: '1.1rem'
  } as React.CSSProperties,

  ledgerType: {
    fontWeight: 'bold',
    fontSize: '0.85rem'
  } as React.CSSProperties,

  ledgerDescription: {
    fontSize: '0.9rem',
    marginBottom: '0.5rem',
    lineHeight: '1.4'
  } as React.CSSProperties,

  ledgerFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.75rem',
    color: '#999',
    paddingTop: '0.5rem',
    borderTop: '1px solid #ddd'
  } as React.CSSProperties,

  ledgerSession: {
    fontFamily: 'monospace'
  } as React.CSSProperties,

  ledgerTime: {} as React.CSSProperties
};
