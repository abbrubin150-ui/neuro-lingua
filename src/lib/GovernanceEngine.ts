/**
 * GovernanceEngine - Autonomous parameter calibration for Neuro-Lingua
 * Implements minimalist autonomous governance with Σ-SIG compliance
 *
 * Key principles:
 * 1. Consistency: Complete state analysis before decisions
 * 2. Controlled Change: Small, gradual, single-parameter adjustments
 * 3. Documented Decision: Full ledger of all changes and rationale
 */

import type {
  MetricSnapshot,
  BoardAlert,
  CalibrationAction,
  GovernanceLedgerEntry,
  GovernorConfig,
  GovernorState,
  GovernanceAnalysis,
  AlertType,
  AlertSeverity,
  ParameterType
} from '../types/governance';

/**
 * Default governor configuration
 */
export const DEFAULT_GOVERNOR_CONFIG: GovernorConfig = {
  enabled: true,
  checkInterval: 2, // Check every 2 training sessions
  activationProbability: 0.5, // 50% chance on eligible sessions
  improvementThreshold: 1.0, // 1% minimum improvement
  learningRate: {
    min: 1e-6,
    max: 1.0,
    decreaseFactor: 0.8, // Reduce by 20%
    increaseFactor: 1.1  // Increase by 10%
  },
  dropout: {
    min: 0.0,
    max: 0.5,
    increaseStep: 0.05, // Increase by 5%
    decreaseStep: 0.05  // Decrease by 5%
  },
  overfittingThreshold: 10.0, // 10% train/val gap
  underfittingThreshold: 2.0, // Absolute loss > 2.0
  plateauWindow: 2 // Look back 2 sessions for plateau
};

/**
 * GovernanceEngine - Core autonomous governance system
 */
export class GovernanceEngine {
  private state: GovernorState;

  constructor(config?: Partial<GovernorConfig>) {
    this.state = {
      config: { ...DEFAULT_GOVERNOR_CONFIG, ...config },
      metricHistory: [],
      alerts: [],
      calibrationHistory: [],
      ledger: [],
      sessionCount: 0,
      lastCheckSession: 0
    };
  }

  /**
   * Record metrics from a training session
   */
  public recordMetrics(metrics: Omit<MetricSnapshot, 'timestamp'>): void {
    const snapshot: MetricSnapshot = {
      ...metrics,
      timestamp: Date.now()
    };

    this.state.metricHistory.push(snapshot);
    this.state.sessionCount++;
  }

  /**
   * Check if governor should activate on this session
   */
  public shouldActivate(): boolean {
    if (!this.state.config.enabled) {
      return false;
    }

    // Check interval condition
    const sessionsSinceLastCheck = this.state.sessionCount - this.state.lastCheckSession;
    if (sessionsSinceLastCheck < this.state.config.checkInterval) {
      return false;
    }

    // Probabilistic activation
    return Math.random() < this.state.config.activationProbability;
  }

  /**
   * Analyze current metrics and detect issues
   */
  public analyzeMetrics(): GovernanceAnalysis {
    const recentMetrics = this.getRecentMetrics();

    if (recentMetrics.length < 2) {
      // Not enough data for analysis
      return {
        plateauDetected: false,
        overfittingDetected: false,
        underfittingDetected: false,
        divergenceDetected: false,
        oscillationDetected: false,
        recommendedActions: [],
        alertsToCreate: []
      };
    }

    const plateauDetected = this.detectPlateau(recentMetrics);
    const overfittingDetected = this.detectOverfitting(recentMetrics);
    const underfittingDetected = this.detectUnderfitting(recentMetrics);
    const divergenceDetected = this.detectDivergence(recentMetrics);
    const oscillationDetected = this.detectOscillation(recentMetrics);

    const alertsToCreate: BoardAlert[] = [];
    const recommendedActions: CalibrationAction[] = [];

    // Plateau: Decrease learning rate
    if (plateauDetected) {
      alertsToCreate.push(this.createAlert('plateau', 'warning',
        'No improvement detected in recent sessions'));
    }

    // Overfitting: Increase dropout
    if (overfittingDetected) {
      alertsToCreate.push(this.createAlert('overfitting', 'warning',
        'Training/validation gap increasing - possible overfitting'));
    }

    // Underfitting: Both losses high
    if (underfittingDetected) {
      alertsToCreate.push(this.createAlert('underfitting', 'info',
        'Model may be underfitting - consider increasing capacity'));
    }

    // Divergence: Loss increasing
    if (divergenceDetected) {
      alertsToCreate.push(this.createAlert('divergence', 'critical',
        'Loss is increasing - training may be diverging'));
    }

    // Oscillation: High variance
    if (oscillationDetected) {
      alertsToCreate.push(this.createAlert('oscillation', 'warning',
        'High variance in loss - training is unstable'));
    }

    return {
      plateauDetected,
      overfittingDetected,
      underfittingDetected,
      divergenceDetected,
      oscillationDetected,
      recommendedActions,
      alertsToCreate
    };
  }

  /**
   * Calibrate parameters based on analysis
   * Returns calibration actions to apply
   */
  public calibrate(
    currentLearningRate: number,
    currentDropout: number,
    projectId: string,
    sessionId: string
  ): CalibrationAction[] {
    this.state.lastCheckSession = this.state.sessionCount;

    const analysis = this.analyzeMetrics();
    const actions: CalibrationAction[] = [];

    // Add alerts to state
    analysis.alertsToCreate.forEach(alert => {
      this.state.alerts.push(alert);
      this.addLedgerEntry('alert', alert.message, projectId, sessionId, undefined, alert);
    });

    // Determine calibration action (only one per session - controlled change)
    let actionTaken = false;

    // Priority 1: Handle divergence (most critical)
    if (analysis.divergenceDetected && !actionTaken) {
      const newLR = this.decreaseLearningRate(currentLearningRate);
      if (newLR !== currentLearningRate) {
        const action = this.createCalibrationAction(
          'learningRate',
          currentLearningRate,
          newLR,
          'Loss diverging - reducing learning rate',
          'trainLoss',
          sessionId
        );
        actions.push(action);
        actionTaken = true;
      }
    }

    // Priority 2: Handle oscillation
    if (analysis.oscillationDetected && !actionTaken) {
      const newLR = this.decreaseLearningRate(currentLearningRate);
      if (newLR !== currentLearningRate) {
        const action = this.createCalibrationAction(
          'learningRate',
          currentLearningRate,
          newLR,
          'High variance detected - stabilizing with lower learning rate',
          'trainLoss',
          sessionId
        );
        actions.push(action);
        actionTaken = true;
      }
    }

    // Priority 3: Handle overfitting
    if (analysis.overfittingDetected && !actionTaken) {
      const newDropout = this.increaseDropout(currentDropout);
      if (newDropout !== currentDropout) {
        const action = this.createCalibrationAction(
          'dropout',
          currentDropout,
          newDropout,
          'Overfitting detected - increasing dropout regularization',
          'valLoss',
          sessionId
        );
        actions.push(action);
        actionTaken = true;
      }
    }

    // Priority 4: Handle plateau
    if (analysis.plateauDetected && !actionTaken) {
      const newLR = this.decreaseLearningRate(currentLearningRate);
      if (newLR !== currentLearningRate) {
        const action = this.createCalibrationAction(
          'learningRate',
          currentLearningRate,
          newLR,
          'Plateau detected - fine-tuning with lower learning rate',
          'trainLoss',
          sessionId
        );
        actions.push(action);
        actionTaken = true;
      }
    }

    // Priority 5: Handle underfitting (only if losses very high at start)
    if (analysis.underfittingDetected && !actionTaken && this.state.sessionCount <= 3) {
      const newDropout = this.decreaseDropout(currentDropout);
      if (newDropout !== currentDropout && newDropout < currentDropout) {
        const action = this.createCalibrationAction(
          'dropout',
          currentDropout,
          newDropout,
          'Underfitting at start - reducing dropout to allow learning',
          'trainLoss',
          sessionId
        );
        actions.push(action);
        actionTaken = true;
      }
    }

    // If no action taken, log decision not to act
    if (!actionTaken && (analysis.plateauDetected || analysis.overfittingDetected ||
                         analysis.underfittingDetected || analysis.divergenceDetected ||
                         analysis.oscillationDetected)) {
      this.addLedgerEntry(
        'no-action',
        'Issues detected but no calibration needed (parameters at safety limits)',
        projectId,
        sessionId
      );
    }

    // Record all actions
    actions.forEach(action => {
      this.state.calibrationHistory.push(action);
      this.addLedgerEntry('calibration', action.reason, projectId, sessionId, action);

      // Create calibration alert
      const alert = this.createAlert(
        'calibration',
        'info',
        `Parameter calibrated: ${action.parameter} from ${action.previousValue.toFixed(4)} to ${action.newValue.toFixed(4)}`
      );
      this.state.alerts.push(alert);
    });

    return actions;
  }

  /**
   * Detect plateau (no improvement)
   */
  private detectPlateau(metrics: MetricSnapshot[]): boolean {
    if (metrics.length < this.state.config.plateauWindow + 1) {
      return false;
    }

    const recent = metrics.slice(-this.state.config.plateauWindow);
    const baseline = metrics[metrics.length - this.state.config.plateauWindow - 1];

    // Check if any recent metric improved by threshold
    const threshold = this.state.config.improvementThreshold / 100;

    for (const metric of recent) {
      const improvement = (baseline.trainLoss - metric.trainLoss) / baseline.trainLoss;
      if (improvement > threshold) {
        return false; // Found improvement
      }
    }

    return true; // No improvement found
  }

  /**
   * Detect overfitting (train/val gap)
   */
  private detectOverfitting(metrics: MetricSnapshot[]): boolean {
    const latest = metrics[metrics.length - 1];

    if (!latest.valLoss) {
      return false; // No validation data
    }

    const gap = ((latest.valLoss - latest.trainLoss) / latest.trainLoss) * 100;
    return gap > this.state.config.overfittingThreshold;
  }

  /**
   * Detect underfitting (both losses high)
   */
  private detectUnderfitting(metrics: MetricSnapshot[]): boolean {
    const latest = metrics[metrics.length - 1];

    const trainHigh = latest.trainLoss > this.state.config.underfittingThreshold;
    const valHigh = latest.valLoss ?
      latest.valLoss > this.state.config.underfittingThreshold : false;

    return trainHigh && (latest.valLoss ? valHigh : true);
  }

  /**
   * Detect divergence (loss increasing)
   */
  private detectDivergence(metrics: MetricSnapshot[]): boolean {
    if (metrics.length < 2) {
      return false;
    }

    const latest = metrics[metrics.length - 1];
    const previous = metrics[metrics.length - 2];

    // Loss increased by more than 10%
    return latest.trainLoss > previous.trainLoss * 1.1;
  }

  /**
   * Detect oscillation (high variance)
   */
  private detectOscillation(metrics: MetricSnapshot[]): boolean {
    if (metrics.length < 3) {
      return false;
    }

    const recent = metrics.slice(-3);
    const losses = recent.map(m => m.trainLoss);

    const mean = losses.reduce((a, b) => a + b, 0) / losses.length;
    const variance = losses.reduce((sum, loss) => sum + Math.pow(loss - mean, 2), 0) / losses.length;
    const stdDev = Math.sqrt(variance);

    // High variance if std dev > 20% of mean
    return stdDev > mean * 0.2;
  }

  /**
   * Decrease learning rate with safety bounds
   */
  private decreaseLearningRate(current: number): number {
    const newLR = current * this.state.config.learningRate.decreaseFactor;
    return Math.max(newLR, this.state.config.learningRate.min);
  }

  /**
   * Increase learning rate with safety bounds
   */
  private increaseLearningRate(current: number): number {
    const newLR = current * this.state.config.learningRate.increaseFactor;
    return Math.min(newLR, this.state.config.learningRate.max);
  }

  /**
   * Increase dropout with safety bounds
   */
  private increaseDropout(current: number): number {
    const newDropout = current + this.state.config.dropout.increaseStep;
    return Math.min(newDropout, this.state.config.dropout.max);
  }

  /**
   * Decrease dropout with safety bounds
   */
  private decreaseDropout(current: number): number {
    const newDropout = current - this.state.config.dropout.decreaseStep;
    return Math.max(newDropout, this.state.config.dropout.min);
  }

  /**
   * Create a calibration action
   */
  private createCalibrationAction(
    parameter: ParameterType,
    previousValue: number,
    newValue: number,
    reason: string,
    triggeringMetric: string,
    sessionId: string
  ): CalibrationAction {
    return {
      id: this.generateId(),
      parameter,
      previousValue,
      newValue,
      reason,
      triggeringMetric,
      timestamp: Date.now(),
      appliedToSession: sessionId
    };
  }

  /**
   * Create a board alert
   */
  private createAlert(
    type: AlertType,
    severity: AlertSeverity,
    message: string,
    metric?: string,
    value?: number
  ): BoardAlert {
    return {
      id: this.generateId(),
      type,
      severity,
      message,
      metric,
      value,
      timestamp: Date.now(),
      acknowledged: false
    };
  }

  /**
   * Add entry to governance ledger
   */
  private addLedgerEntry(
    type: GovernanceLedgerEntry['type'],
    description: string,
    projectId: string,
    sessionId: string,
    calibrationAction?: CalibrationAction,
    alert?: BoardAlert
  ): void {
    const entry: GovernanceLedgerEntry = {
      id: this.generateId(),
      type,
      description,
      calibrationAction,
      alert,
      timestamp: Date.now(),
      sessionId,
      projectId
    };

    this.state.ledger.push(entry);
  }

  /**
   * Get recent metrics for analysis
   */
  private getRecentMetrics(window: number = 5): MetricSnapshot[] {
    return this.state.metricHistory.slice(-window);
  }

  /**
   * Generate unique ID
   */
  private generateId(): string {
    return `gov_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get current state (read-only)
   */
  public getState(): Readonly<GovernorState> {
    return this.state;
  }

  /**
   * Get active (unacknowledged) alerts
   */
  public getActiveAlerts(): BoardAlert[] {
    return this.state.alerts.filter(a => !a.acknowledged);
  }

  /**
   * Acknowledge an alert
   */
  public acknowledgeAlert(alertId: string): void {
    const alert = this.state.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
    }
  }

  /**
   * Clear all alerts
   */
  public clearAlerts(): void {
    this.state.alerts = [];
  }

  /**
   * Get calibration history
   */
  public getCalibrationHistory(): CalibrationAction[] {
    return [...this.state.calibrationHistory];
  }

  /**
   * Get governance ledger
   */
  public getLedger(): GovernanceLedgerEntry[] {
    return [...this.state.ledger];
  }

  /**
   * Export state for persistence
   */
  public exportState(): GovernorState {
    return JSON.parse(JSON.stringify(this.state));
  }

  /**
   * Import state from persistence
   */
  public importState(state: GovernorState): void {
    this.state = state;
  }

  /**
   * Reset governance state
   */
  public reset(): void {
    const config = this.state.config;
    this.state = {
      config,
      metricHistory: [],
      alerts: [],
      calibrationHistory: [],
      ledger: [],
      sessionCount: 0,
      lastCheckSession: 0
    };
  }
}
