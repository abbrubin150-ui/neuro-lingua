/**
 * Autonomous Governance Types for Neuro-Lingua
 * Implements minimalist autonomous governance with Σ-SIG compliance
 */

/**
 * Metric snapshot from a training session
 */
export interface MetricSnapshot {
  /** Training session ID */
  sessionId: string;
  /** Epoch number */
  epoch: number;
  /** Training loss */
  trainLoss: number;
  /** Validation loss (if available) */
  valLoss?: number;
  /** Training accuracy */
  trainAccuracy: number;
  /** Validation accuracy (if available) */
  valAccuracy?: number;
  /** Perplexity */
  perplexity: number;
  /** Timestamp */
  timestamp: number;
}

/**
 * Board alert types
 */
export type AlertType =
  | 'plateau' // No improvement detected
  | 'overfitting' // Train/val gap increasing
  | 'underfitting' // Both train and val loss high
  | 'divergence' // Loss increasing
  | 'oscillation' // High variance in loss
  | 'calibration'; // Parameter adjusted by governor

/**
 * Alert severity
 */
export type AlertSeverity = 'info' | 'warning' | 'critical';

/**
 * Monitoring board alert
 */
export interface BoardAlert {
  /** Unique ID */
  id: string;
  /** Alert type */
  type: AlertType;
  /** Severity level */
  severity: AlertSeverity;
  /** Human-readable message */
  message: string;
  /** Metric that triggered the alert */
  metric?: string;
  /** Value that triggered */
  value?: number;
  /** Timestamp */
  timestamp: number;
  /** Whether alert was acknowledged */
  acknowledged: boolean;
}

/**
 * Parameter change type
 */
export type ParameterType = 'learningRate' | 'dropout' | 'batchSize' | 'other';

/**
 * Calibration action taken by governor
 */
export interface CalibrationAction {
  /** Unique ID */
  id: string;
  /** Parameter that was changed */
  parameter: ParameterType;
  /** Previous value */
  previousValue: number;
  /** New value */
  newValue: number;
  /** Reason for change */
  reason: string;
  /** Metric that triggered the change */
  triggeringMetric: string;
  /** Timestamp */
  timestamp: number;
  /** Session ID where change was applied */
  appliedToSession: string;
}

/**
 * Governance ledger entry
 */
export interface GovernanceLedgerEntry {
  /** Unique ID */
  id: string;
  /** Entry type */
  type: 'calibration' | 'alert' | 'decision' | 'no-action';
  /** Description */
  description: string;
  /** Related calibration action (if type is calibration) */
  calibrationAction?: CalibrationAction;
  /** Related alert (if type is alert) */
  alert?: BoardAlert;
  /** Timestamp */
  timestamp: number;
  /** Session ID */
  sessionId: string;
  /** Project ID */
  projectId: string;
  /** Run ID (if applicable) */
  runId?: string;
}

/**
 * Governor configuration
 */
export interface GovernorConfig {
  /** Enable autonomous calibration */
  enabled: boolean;
  /** Check interval (number of training sessions) */
  checkInterval: number;
  /** Probabilistic activation (0-1) */
  activationProbability: number;
  /** Minimum improvement threshold (percentage) */
  improvementThreshold: number;
  /** Learning rate bounds */
  learningRate: {
    min: number;
    max: number;
    decreaseFactor: number;
    increaseFactor: number;
  };
  /** Dropout bounds */
  dropout: {
    min: number;
    max: number;
    increaseStep: number;
    decreaseStep: number;
  };
  /** Overfitting detection threshold (train/val gap %) */
  overfittingThreshold: number;
  /** Underfitting detection threshold (absolute loss) */
  underfittingThreshold: number;
  /** Plateau detection window (number of sessions) */
  plateauWindow: number;
}

/**
 * Governor state
 */
export interface GovernorState {
  /** Configuration */
  config: GovernorConfig;
  /** Metric history */
  metricHistory: MetricSnapshot[];
  /** Active alerts */
  alerts: BoardAlert[];
  /** Calibration history */
  calibrationHistory: CalibrationAction[];
  /** Full ledger */
  ledger: GovernanceLedgerEntry[];
  /** Session counter */
  sessionCount: number;
  /** Last check session */
  lastCheckSession: number;
}

/**
 * Governance analysis result
 */
export interface GovernanceAnalysis {
  /** Whether plateau detected */
  plateauDetected: boolean;
  /** Whether overfitting detected */
  overfittingDetected: boolean;
  /** Whether underfitting detected */
  underfittingDetected: boolean;
  /** Whether divergence detected */
  divergenceDetected: boolean;
  /** Whether oscillation detected */
  oscillationDetected: boolean;
  /** Recommended actions */
  recommendedActions: CalibrationAction[];
  /** Alerts to create */
  alertsToCreate: BoardAlert[];
}
