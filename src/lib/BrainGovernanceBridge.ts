/**
 * BrainGovernanceBridge - Connects Brain Vitals and Governance systems
 *
 * This module bridges the soft feedback (Brain mood/vitals) with hard
 * parameter calibration (Governance), enabling:
 *
 * 1. Mood-modulated governance activation
 * 2. Adaptive decay rates based on brain state
 * 3. Cross-system event propagation
 * 4. Safe autonomous actions with escalation paths
 * 5. Priority-based suggestion system
 *
 * Safety guarantees:
 * - Never executes heavy operations without explicit user consent
 * - All autonomous actions are bounded and reversible
 * - Full audit trail via brain diary and governance ledger
 */

import type { BrainStats, Mood, BrainEvent, DiaryEntry } from './BrainEngine';
import type {
  GovernanceAnalysis,
  BoardAlert,
  CalibrationAction,
  GovernorConfig
} from '../types/governance';

// ============================================================================
// Types
// ============================================================================

/**
 * Autonomous action priority levels
 */
export type ActionPriority = 'critical' | 'high' | 'medium' | 'low' | 'informational';

/**
 * Autonomous action types
 */
export type AutonomousActionType =
  | 'adjust_learning_rate'
  | 'adjust_dropout'
  | 'suggest_training'
  | 'suggest_feeding'
  | 'suggest_rest' // Cooling period
  | 'alert_user'
  | 'log_observation';

/**
 * Autonomous action definition
 */
export interface AutonomousAction {
  id: string;
  type: AutonomousActionType;
  priority: ActionPriority;
  description: string;
  reason: string;
  timestamp: number;
  payload?: Record<string, any>;
  requiresConfirmation: boolean;
  executed: boolean;
  executedAt?: number;
  cancelled: boolean;
  cancelledReason?: string;
}

/**
 * Auto-pilot configuration
 */
export interface AutoPilotConfig {
  enabled: boolean;
  maxActionsPerHour: number;
  allowedPriorities: ActionPriority[];
  cooldownMs: number;
  requireConfirmationFor: ActionPriority[];
}

/**
 * Mood-based modulation factors
 */
export interface MoodModulators {
  activationProbabilityMultiplier: number;
  learningRateAdjustmentScale: number;
  dropoutAdjustmentScale: number;
  suggestionFrequencyMs: number;
  decayRateMultiplier: number;
}

/**
 * Bridge state
 */
export interface BridgeState {
  autoPilot: AutoPilotConfig;
  pendingActions: AutonomousAction[];
  executedActions: AutonomousAction[];
  cancelledActions: AutonomousAction[];
  lastActionTime: number;
  actionsThisHour: number;
  hourStartTime: number;
}

/**
 * Recovery plan for distressed brain states
 */
export interface RecoveryPlan {
  mood: Mood;
  actions: AutonomousAction[];
  estimatedDuration: number; // ms
  description: string;
}

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_AUTO_PILOT_CONFIG: AutoPilotConfig = {
  enabled: false,
  maxActionsPerHour: 10,
  allowedPriorities: ['informational', 'low'],
  cooldownMs: 60000, // 1 minute between actions
  requireConfirmationFor: ['critical', 'high', 'medium']
};

/**
 * Mood-based modulation factors
 * These affect how governance behaves based on brain mood
 */
export const MOOD_MODULATORS: Record<Mood, MoodModulators> = {
  FOCUSED: {
    activationProbabilityMultiplier: 0.7, // Less intervention when focused
    learningRateAdjustmentScale: 1.0, // Normal adjustments
    dropoutAdjustmentScale: 1.0,
    suggestionFrequencyMs: 120000, // 2 minutes - less frequent
    decayRateMultiplier: 0.5 // Slower decay
  },
  CALM: {
    activationProbabilityMultiplier: 1.0, // Baseline
    learningRateAdjustmentScale: 1.0,
    dropoutAdjustmentScale: 1.0,
    suggestionFrequencyMs: 60000, // 1 minute
    decayRateMultiplier: 1.0
  },
  DREAMY: {
    activationProbabilityMultiplier: 1.2, // Slightly more intervention
    learningRateAdjustmentScale: 0.8, // More conservative
    dropoutAdjustmentScale: 1.2, // Increase regularization
    suggestionFrequencyMs: 45000, // More frequent
    decayRateMultiplier: 1.2 // Faster decay
  },
  AGITATED: {
    activationProbabilityMultiplier: 1.5, // More intervention needed
    learningRateAdjustmentScale: 0.5, // Very conservative
    dropoutAdjustmentScale: 1.5, // Strong regularization
    suggestionFrequencyMs: 30000, // Frequent suggestions
    decayRateMultiplier: 0.3 // Reduce decay to help stabilize
  },
  BURNT_OUT: {
    activationProbabilityMultiplier: 0.3, // Less training intervention
    learningRateAdjustmentScale: 0.5, // Gentle when overtrained
    dropoutAdjustmentScale: 0.5, // Reduce regularization
    suggestionFrequencyMs: 30000, // Frequent to suggest feeding
    decayRateMultiplier: 1.5 // Faster creativity decay
  }
};

/**
 * Recovery strategies for each distressed mood
 */
export const RECOVERY_STRATEGIES: Record<'AGITATED' | 'BURNT_OUT' | 'DREAMY', RecoveryPlan> = {
  AGITATED: {
    mood: 'AGITATED',
    actions: [],
    estimatedDuration: 300000, // 5 minutes
    description:
      'Stability recovery: Short training sessions with conservative learning rate to rebuild stability without overwhelming the model.'
  },
  BURNT_OUT: {
    mood: 'BURNT_OUT',
    actions: [],
    estimatedDuration: 600000, // 10 minutes
    description:
      'Creativity recovery: Feed diverse new content to refresh the model, avoid heavy training until creativity recovers.'
  },
  DREAMY: {
    mood: 'DREAMY',
    actions: [],
    estimatedDuration: 180000, // 3 minutes
    description:
      'Balance recovery: Light training to increase stability while preserving creativity gains.'
  }
};

// ============================================================================
// Bridge Class
// ============================================================================

/**
 * BrainGovernanceBridge - Orchestrates autonomy between Brain and Governance
 */
export class BrainGovernanceBridge {
  private state: BridgeState;
  private actionIdCounter: number = 0;

  constructor(autoPilotConfig?: Partial<AutoPilotConfig>) {
    this.state = {
      autoPilot: { ...DEFAULT_AUTO_PILOT_CONFIG, ...autoPilotConfig },
      pendingActions: [],
      executedActions: [],
      cancelledActions: [],
      lastActionTime: 0,
      actionsThisHour: 0,
      hourStartTime: Date.now()
    };
  }

  // ==========================================================================
  // Mood-Based Governance Modulation
  // ==========================================================================

  /**
   * Get governance activation probability modified by brain mood
   */
  public getModulatedActivationProbability(baseConfig: GovernorConfig, brainMood: Mood): number {
    const modulator = MOOD_MODULATORS[brainMood];
    const modulatedProbability =
      baseConfig.activationProbability * modulator.activationProbabilityMultiplier;
    return Math.min(1.0, Math.max(0.0, modulatedProbability));
  }

  /**
   * Get learning rate adjustment scaled by brain mood
   */
  public getModulatedLRAdjustment(baseAdjustment: number, brainMood: Mood): number {
    const modulator = MOOD_MODULATORS[brainMood];
    return baseAdjustment * modulator.learningRateAdjustmentScale;
  }

  /**
   * Get dropout adjustment scaled by brain mood
   */
  public getModulatedDropoutAdjustment(baseAdjustment: number, brainMood: Mood): number {
    const modulator = MOOD_MODULATORS[brainMood];
    return baseAdjustment * modulator.dropoutAdjustmentScale;
  }

  /**
   * Get suggestion check interval based on mood
   */
  public getSuggestionInterval(brainMood: Mood): number {
    return MOOD_MODULATORS[brainMood].suggestionFrequencyMs;
  }

  /**
   * Get decay rate multiplier based on mood
   */
  public getDecayMultiplier(brainMood: Mood): number {
    return MOOD_MODULATORS[brainMood].decayRateMultiplier;
  }

  // ==========================================================================
  // Adaptive Decay System
  // ==========================================================================

  /**
   * Calculate adaptive decay for creativity and stability
   * Based on mood, activity history, and current vitals
   */
  public calculateAdaptiveDecay(
    brain: BrainStats,
    timeSinceLastActivityMs: number
  ): { creativityDecay: number; stabilityDecay: number } {
    const baseCreativityDecay = 0.3;
    const baseStabilityDecay = 0.1;
    const moodMultiplier = this.getDecayMultiplier(brain.mood);

    // Scale decay by time since last activity (more decay if idle longer)
    const timeScale = Math.min(timeSinceLastActivityMs / 60000, 5); // Cap at 5x

    // Reduce decay if vitals are already low (protection floor)
    const creativityProtection = brain.creativity < 20 ? 0.3 : 1.0;
    const stabilityProtection = brain.stability < 20 ? 0.3 : 1.0;

    return {
      creativityDecay: baseCreativityDecay * moodMultiplier * timeScale * creativityProtection,
      stabilityDecay: baseStabilityDecay * moodMultiplier * timeScale * stabilityProtection
    };
  }

  // ==========================================================================
  // Cross-System Event Propagation
  // ==========================================================================

  /**
   * Convert governance alert to brain diary entry
   */
  public alertToDiaryEntry(alert: BoardAlert): DiaryEntry {
    return {
      timestamp: alert.timestamp,
      type: 'SUGGESTION',
      message: `[Governance] ${alert.message}`,
      metadata: {
        alertId: alert.id,
        alertType: alert.type,
        severity: alert.severity,
        metric: alert.metric,
        value: alert.value
      }
    };
  }

  /**
   * Convert calibration action to brain diary entry
   */
  public calibrationToDiaryEntry(action: CalibrationAction): DiaryEntry {
    return {
      timestamp: action.timestamp,
      type: 'SUGGESTION',
      message: `[Auto-Calibration] ${action.parameter}: ${action.previousValue.toFixed(4)} → ${action.newValue.toFixed(4)}. Reason: ${action.reason}`,
      metadata: {
        calibrationId: action.id,
        parameter: action.parameter,
        previousValue: action.previousValue,
        newValue: action.newValue,
        reason: action.reason
      }
    };
  }

  /**
   * Generate brain event from governance analysis
   */
  public analysisToSuggestionEvent(
    analysis: GovernanceAnalysis,
    brain: BrainStats
  ): BrainEvent | null {
    // Generate appropriate suggestion based on analysis and brain state
    if (analysis.divergenceDetected && brain.mood !== 'BURNT_OUT') {
      return {
        type: 'MOOD_OVERRIDE',
        timestamp: Date.now(),
        payload: {
          mood: brain.mood,
          _suggestion: 'Training is diverging - consider reducing learning rate'
        }
      };
    }

    if (analysis.overfittingDetected && brain.stability < 50) {
      return {
        type: 'MOOD_OVERRIDE',
        timestamp: Date.now(),
        payload: {
          mood: brain.mood,
          _suggestion: 'Overfitting detected - consider adding more data or increasing dropout'
        }
      };
    }

    return null;
  }

  // ==========================================================================
  // Autonomous Action System
  // ==========================================================================

  /**
   * Create an autonomous action
   */
  public createAction(
    type: AutonomousActionType,
    priority: ActionPriority,
    description: string,
    reason: string,
    payload?: Record<string, any>
  ): AutonomousAction {
    const requiresConfirmation = this.state.autoPilot.requireConfirmationFor.includes(priority);

    return {
      id: `action_${++this.actionIdCounter}_${Date.now()}`,
      type,
      priority,
      description,
      reason,
      timestamp: Date.now(),
      payload,
      requiresConfirmation,
      executed: false,
      cancelled: false
    };
  }

  /**
   * Queue an action for execution
   */
  public queueAction(action: AutonomousAction): boolean {
    // Check if auto-pilot allows this action
    if (!this.canExecuteAction(action)) {
      action.cancelled = true;
      action.cancelledReason = 'Auto-pilot restrictions';
      this.state.cancelledActions.push(action);
      return false;
    }

    // Add to pending if requires confirmation
    if (action.requiresConfirmation) {
      this.state.pendingActions.push(action);
      return true;
    }

    // Execute immediately if allowed
    return this.executeAction(action.id);
  }

  /**
   * Check if an action can be executed under current auto-pilot rules
   */
  public canExecuteAction(action: AutonomousAction): boolean {
    if (!this.state.autoPilot.enabled) {
      return !action.requiresConfirmation;
    }

    // Check priority allowance
    if (!this.state.autoPilot.allowedPriorities.includes(action.priority)) {
      return false;
    }

    // Check cooldown
    const timeSinceLastAction = Date.now() - this.state.lastActionTime;
    if (timeSinceLastAction < this.state.autoPilot.cooldownMs) {
      return false;
    }

    // Check hourly limit
    this.updateHourlyCounter();
    if (this.state.actionsThisHour >= this.state.autoPilot.maxActionsPerHour) {
      return false;
    }

    return true;
  }

  /**
   * Execute a pending action
   */
  public executeAction(actionId: string): boolean {
    const actionIndex = this.state.pendingActions.findIndex((a) => a.id === actionId);

    if (actionIndex === -1) {
      return false;
    }

    const action = this.state.pendingActions.splice(actionIndex, 1)[0];

    action.executed = true;
    action.executedAt = Date.now();

    this.state.executedActions.push(action);
    this.state.lastActionTime = Date.now();
    this.state.actionsThisHour++;

    return true;
  }

  /**
   * Cancel a pending action
   */
  public cancelAction(actionId: string, reason: string): boolean {
    const actionIndex = this.state.pendingActions.findIndex((a) => a.id === actionId);

    if (actionIndex === -1) {
      return false;
    }

    const action = this.state.pendingActions.splice(actionIndex, 1)[0];

    action.cancelled = true;
    action.cancelledReason = reason;

    this.state.cancelledActions.push(action);

    return true;
  }

  /**
   * Update hourly action counter
   */
  private updateHourlyCounter(): void {
    const now = Date.now();
    const hourMs = 3600000;

    if (now - this.state.hourStartTime > hourMs) {
      this.state.hourStartTime = now;
      this.state.actionsThisHour = 0;
    }
  }

  // ==========================================================================
  // Priority-Based Suggestion System
  // ==========================================================================

  /**
   * Generate prioritized suggestions based on brain and governance state
   */
  public generatePrioritizedSuggestions(
    brain: BrainStats,
    analysis?: GovernanceAnalysis
  ): AutonomousAction[] {
    const suggestions: AutonomousAction[] = [];

    // Priority 1: Critical - Divergence or very low vitals
    if (analysis?.divergenceDetected) {
      suggestions.push(
        this.createAction(
          'alert_user',
          'critical',
          'Training is diverging!',
          'Loss is increasing significantly. Immediate attention required.',
          { issue: 'divergence' }
        )
      );
    }

    if (brain.stability < 10) {
      suggestions.push(
        this.createAction(
          'suggest_training',
          'critical',
          'Critical stability warning',
          'Stability is critically low. Training strongly recommended.',
          { stability: brain.stability }
        )
      );
    }

    // Priority 2: High - Overfitting or AGITATED mood
    if (analysis?.overfittingDetected) {
      suggestions.push(
        this.createAction(
          'adjust_dropout',
          'high',
          'Increase dropout to combat overfitting',
          'Validation loss is significantly higher than training loss.',
          { issue: 'overfitting' }
        )
      );
    }

    if (brain.mood === 'AGITATED') {
      suggestions.push(
        this.createAction(
          'suggest_training',
          'high',
          'Model needs stabilization',
          'Brain is agitated. Short training session recommended.',
          { mood: 'AGITATED', stability: brain.stability }
        )
      );
    }

    // Priority 3: Medium - BURNT_OUT or plateau
    if (brain.mood === 'BURNT_OUT') {
      suggestions.push(
        this.createAction(
          'suggest_feeding',
          'medium',
          'Model needs fresh content',
          'Brain is burnt out. Feed new diverse text to restore creativity.',
          { mood: 'BURNT_OUT', creativity: brain.creativity }
        )
      );
    }

    if (analysis?.plateauDetected) {
      suggestions.push(
        this.createAction(
          'adjust_learning_rate',
          'medium',
          'Reduce learning rate for fine-tuning',
          'Training has plateaued. Fine-tuning with lower learning rate may help.',
          { issue: 'plateau' }
        )
      );
    }

    // Priority 4: Low - DREAMY or oscillation
    if (brain.mood === 'DREAMY') {
      suggestions.push(
        this.createAction(
          'suggest_training',
          'low',
          'Light training to balance creativity',
          'High creativity but low stability. Light training can help balance.',
          { mood: 'DREAMY', creativity: brain.creativity, stability: brain.stability }
        )
      );
    }

    if (analysis?.oscillationDetected) {
      suggestions.push(
        this.createAction(
          'adjust_learning_rate',
          'low',
          'Stabilize training with lower learning rate',
          'Training shows high variance. Lower learning rate may stabilize.',
          { issue: 'oscillation' }
        )
      );
    }

    // Priority 5: Informational - General observations
    if (brain.creativity < 40 && brain.mood !== 'BURNT_OUT') {
      suggestions.push(
        this.createAction(
          'log_observation',
          'informational',
          'Creativity is below average',
          'Consider feeding new content to boost creativity.',
          { creativity: brain.creativity }
        )
      );
    }

    // Sort by priority
    const priorityOrder: ActionPriority[] = ['critical', 'high', 'medium', 'low', 'informational'];
    suggestions.sort(
      (a, b) => priorityOrder.indexOf(a.priority) - priorityOrder.indexOf(b.priority)
    );

    return suggestions;
  }

  // ==========================================================================
  // Recovery Planning
  // ==========================================================================

  /**
   * Generate a recovery plan for a distressed brain state
   */
  public generateRecoveryPlan(brain: BrainStats): RecoveryPlan | null {
    if (brain.mood === 'CALM' || brain.mood === 'FOCUSED') {
      return null; // No recovery needed
    }

    const basePlan = RECOVERY_STRATEGIES[brain.mood as 'AGITATED' | 'BURNT_OUT' | 'DREAMY'];
    if (!basePlan) {
      return null;
    }

    // Generate specific actions based on current state
    const actions: AutonomousAction[] = [];

    switch (brain.mood) {
      case 'AGITATED':
        actions.push(
          this.createAction(
            'suggest_training',
            'high',
            'Short training session (5-10 epochs)',
            'Rebuild stability with controlled training',
            { recommendedEpochs: Math.max(5, Math.floor((30 - brain.stability) / 3)) }
          )
        );
        if (brain.creativity > 50) {
          actions.push(
            this.createAction(
              'adjust_learning_rate',
              'medium',
              'Use conservative learning rate',
              'Lower learning rate during recovery to prevent instability',
              { recommendedScale: 0.5 }
            )
          );
        }
        break;

      case 'BURNT_OUT':
        actions.push(
          this.createAction(
            'suggest_feeding',
            'high',
            'Feed diverse new content',
            'Restore creativity with fresh material',
            { recommendedWords: Math.max(100, Math.floor((40 - brain.creativity) * 5)) }
          )
        );
        actions.push(
          this.createAction(
            'suggest_rest',
            'medium',
            'Avoid heavy training temporarily',
            'Let the model "rest" before intense training',
            { restDurationMs: 300000 }
          )
        );
        break;

      case 'DREAMY':
        actions.push(
          this.createAction(
            'suggest_training',
            'low',
            'Light training to increase stability',
            'Balance high creativity with improved stability',
            { recommendedEpochs: 5 }
          )
        );
        break;
    }

    return {
      ...basePlan,
      actions,
      estimatedDuration: basePlan.estimatedDuration * (1 + (100 - brain.stability) / 200)
    };
  }

  // ==========================================================================
  // State Management
  // ==========================================================================

  /**
   * Set auto-pilot configuration
   */
  public setAutoPilotConfig(config: Partial<AutoPilotConfig>): void {
    this.state.autoPilot = { ...this.state.autoPilot, ...config };
  }

  /**
   * Enable auto-pilot mode
   */
  public enableAutoPilot(): void {
    this.state.autoPilot.enabled = true;
  }

  /**
   * Disable auto-pilot mode
   */
  public disableAutoPilot(): void {
    this.state.autoPilot.enabled = false;
    // Move all pending actions to cancelled
    this.state.pendingActions.forEach((action) => {
      action.cancelled = true;
      action.cancelledReason = 'Auto-pilot disabled';
      this.state.cancelledActions.push(action);
    });
    this.state.pendingActions = [];
  }

  /**
   * Get current bridge state
   */
  public getState(): Readonly<BridgeState> {
    return this.state;
  }

  /**
   * Get pending actions
   */
  public getPendingActions(): AutonomousAction[] {
    return [...this.state.pendingActions];
  }

  /**
   * Get execution history
   */
  public getExecutedActions(): AutonomousAction[] {
    return [...this.state.executedActions];
  }

  /**
   * Export state for persistence
   */
  public exportState(): BridgeState {
    return JSON.parse(JSON.stringify(this.state));
  }

  /**
   * Import state from persistence
   */
  public importState(state: BridgeState): void {
    this.state = state;
    this.actionIdCounter = Math.max(
      0,
      ...this.state.executedActions.map((a) => parseInt(a.id.split('_')[1]) || 0),
      ...this.state.pendingActions.map((a) => parseInt(a.id.split('_')[1]) || 0)
    );
  }

  /**
   * Reset bridge state
   */
  public reset(): void {
    const autoPilot = this.state.autoPilot;
    this.state = {
      autoPilot,
      pendingActions: [],
      executedActions: [],
      cancelledActions: [],
      lastActionTime: 0,
      actionsThisHour: 0,
      hourStartTime: Date.now()
    };
    this.actionIdCounter = 0;
  }
}

// ============================================================================
// Singleton instance for app-wide use
// ============================================================================

let bridgeInstance: BrainGovernanceBridge | null = null;

/**
 * Get the singleton bridge instance
 */
export function getBridge(config?: Partial<AutoPilotConfig>): BrainGovernanceBridge {
  if (!bridgeInstance) {
    bridgeInstance = new BrainGovernanceBridge(config);
  }
  return bridgeInstance;
}

/**
 * Reset the singleton bridge instance
 */
export function resetBridge(): void {
  if (bridgeInstance) {
    bridgeInstance.reset();
  }
  bridgeInstance = null;
}
