/**
 * Tests for BrainGovernanceBridge - Brain-Governance integration
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  BrainGovernanceBridge,
  DEFAULT_AUTO_PILOT_CONFIG,
  MOOD_MODULATORS,
  getBridge,
  resetBridge
} from '../../src/lib/BrainGovernanceBridge';
import { createBrain, BrainStats, Mood } from '../../src/lib/BrainEngine';
import type { GovernorConfig } from '../../src/types/governance';

describe('BrainGovernanceBridge', () => {
  let bridge: BrainGovernanceBridge;
  let brain: BrainStats;

  beforeEach(() => {
    bridge = new BrainGovernanceBridge();
    brain = createBrain('test-model', 'Test Brain');
  });

  describe('Mood-Based Governance Modulation', () => {
    it('should return modulated activation probability based on mood', () => {
      const baseConfig: GovernorConfig = {
        enabled: true,
        checkInterval: 2,
        activationProbability: 0.5,
        improvementThreshold: 1.0,
        learningRate: { min: 1e-6, max: 1.0, decreaseFactor: 0.8, increaseFactor: 1.1 },
        dropout: { min: 0.0, max: 0.5, increaseStep: 0.05, decreaseStep: 0.05 },
        overfittingThreshold: 10.0,
        underfittingThreshold: 2.0,
        plateauWindow: 2
      };

      // FOCUSED mood should reduce intervention
      const focusedProb = bridge.getModulatedActivationProbability(baseConfig, 'FOCUSED');
      expect(focusedProb).toBeLessThan(baseConfig.activationProbability);
      expect(focusedProb).toBeCloseTo(0.5 * MOOD_MODULATORS.FOCUSED.activationProbabilityMultiplier);

      // AGITATED mood should increase intervention
      const agitatedProb = bridge.getModulatedActivationProbability(baseConfig, 'AGITATED');
      expect(agitatedProb).toBeGreaterThan(baseConfig.activationProbability);

      // CALM should be baseline
      const calmProb = bridge.getModulatedActivationProbability(baseConfig, 'CALM');
      expect(calmProb).toBe(baseConfig.activationProbability);
    });

    it('should scale learning rate adjustments based on mood', () => {
      const baseAdjustment = 0.1;

      // FOCUSED: normal adjustments
      const focusedAdj = bridge.getModulatedLRAdjustment(baseAdjustment, 'FOCUSED');
      expect(focusedAdj).toBe(baseAdjustment * MOOD_MODULATORS.FOCUSED.learningRateAdjustmentScale);

      // AGITATED: conservative adjustments
      const agitatedAdj = bridge.getModulatedLRAdjustment(baseAdjustment, 'AGITATED');
      expect(agitatedAdj).toBeLessThan(baseAdjustment);

      // BURNT_OUT: gentle adjustments
      const burntAdj = bridge.getModulatedLRAdjustment(baseAdjustment, 'BURNT_OUT');
      expect(burntAdj).toBeLessThan(baseAdjustment);
    });

    it('should return different suggestion intervals for different moods', () => {
      const focusedInterval = bridge.getSuggestionInterval('FOCUSED');
      const agitatedInterval = bridge.getSuggestionInterval('AGITATED');
      const calmInterval = bridge.getSuggestionInterval('CALM');

      // FOCUSED should have longer intervals (less frequent suggestions)
      expect(focusedInterval).toBeGreaterThan(calmInterval);

      // AGITATED should have shorter intervals (more frequent suggestions)
      expect(agitatedInterval).toBeLessThan(calmInterval);
    });
  });

  describe('Adaptive Decay System', () => {
    it('should calculate adaptive decay based on mood', () => {
      const now = Date.now();

      // Set up brain with different moods
      const calmBrain = { ...brain, mood: 'CALM' as Mood };
      const focusedBrain = { ...brain, mood: 'FOCUSED' as Mood };
      const burntOutBrain = { ...brain, mood: 'BURNT_OUT' as Mood };

      const calmDecay = bridge.calculateAdaptiveDecay(calmBrain, now);
      const focusedDecay = bridge.calculateAdaptiveDecay(focusedBrain, now);
      const burntOutDecay = bridge.calculateAdaptiveDecay(burntOutBrain, now);

      // FOCUSED should have slower decay
      expect(focusedDecay.creativityDecay).toBeLessThan(calmDecay.creativityDecay);
      expect(focusedDecay.stabilityDecay).toBeLessThan(calmDecay.stabilityDecay);

      // BURNT_OUT should have faster creativity decay
      expect(burntOutDecay.creativityDecay).toBeGreaterThan(calmDecay.creativityDecay);
    });

    it('should apply protection floor for low vitals', () => {
      const now = Date.now();

      // Low creativity brain
      const lowCreativityBrain = { ...brain, creativity: 10, mood: 'CALM' as Mood };
      const normalBrain = { ...brain, creativity: 50, mood: 'CALM' as Mood };

      const lowDecay = bridge.calculateAdaptiveDecay(lowCreativityBrain, now);
      const normalDecay = bridge.calculateAdaptiveDecay(normalBrain, now);

      // Creativity decay should be reduced for low vitals brain
      expect(lowDecay.creativityDecay).toBeLessThan(normalDecay.creativityDecay);
    });
  });

  describe('Autonomous Action System', () => {
    it('should create actions with correct properties', () => {
      const action = bridge.createAction(
        'suggest_training',
        'high',
        'Training needed',
        'Low stability detected',
        { stability: 25 }
      );

      expect(action.type).toBe('suggest_training');
      expect(action.priority).toBe('high');
      expect(action.description).toBe('Training needed');
      expect(action.reason).toBe('Low stability detected');
      expect(action.payload).toEqual({ stability: 25 });
      expect(action.executed).toBe(false);
      expect(action.cancelled).toBe(false);
    });

    it('should respect auto-pilot configuration when queuing actions', () => {
      // Enable auto-pilot with limited priorities
      bridge.enableAutoPilot();
      bridge.setAutoPilotConfig({
        enabled: true,
        allowedPriorities: ['informational', 'low'],
        cooldownMs: 0,
        requireConfirmationFor: [] // No confirmation required
      });

      // Low priority action should be allowed (doesn't require confirmation)
      const lowAction = bridge.createAction(
        'log_observation',
        'informational', // Use informational which is allowed
        'Test',
        'Test reason'
      );
      // Action should be created without requiresConfirmation since we cleared that list
      expect(lowAction.requiresConfirmation).toBe(false);

      // Check if action can be executed
      const canExecute = bridge.canExecuteAction(lowAction);
      expect(canExecute).toBe(true);

      // High priority action should be rejected
      const highAction = bridge.createAction(
        'alert_user',
        'high',
        'Test',
        'Test reason'
      );
      const canExecuteHigh = bridge.canExecuteAction(highAction);
      expect(canExecuteHigh).toBe(false);
    });

    it('should execute and track actions', () => {
      bridge.setAutoPilotConfig({
        enabled: true,
        allowedPriorities: ['low'],
        cooldownMs: 0
      });

      const action = bridge.createAction('log_observation', 'low', 'Test', 'Test');
      action.requiresConfirmation = true;

      bridge.queueAction(action);
      const pending = bridge.getPendingActions();
      expect(pending.length).toBe(1);

      const executed = bridge.executeAction(action.id);
      expect(executed).toBe(true);
      expect(bridge.getPendingActions().length).toBe(0);
      expect(bridge.getExecutedActions().length).toBe(1);
    });

    it('should cancel actions with reason', () => {
      bridge.setAutoPilotConfig({
        enabled: true,
        allowedPriorities: ['medium'],
        cooldownMs: 0,
        requireConfirmationFor: ['medium'] // This will require confirmation
      });

      const action = bridge.createAction('suggest_training', 'medium', 'Test', 'Test');
      // Action should require confirmation since medium is in requireConfirmationFor
      expect(action.requiresConfirmation).toBe(true);

      // Queue the action - it should go to pending since it requires confirmation
      bridge.queueAction(action);
      expect(bridge.getPendingActions().length).toBe(1);

      const cancelled = bridge.cancelAction(action.id, 'User cancelled');

      expect(cancelled).toBe(true);
      expect(bridge.getPendingActions().length).toBe(0);
    });
  });

  describe('Priority-Based Suggestion System', () => {
    it('should generate prioritized suggestions based on brain state', () => {
      // Brain with low stability
      const agitatedBrain = {
        ...brain,
        mood: 'AGITATED' as Mood,
        stability: 15,
        creativity: 50
      };

      const suggestions = bridge.generatePrioritizedSuggestions(agitatedBrain);

      // Should have at least one suggestion
      expect(suggestions.length).toBeGreaterThan(0);

      // First suggestion should be highest priority
      if (suggestions.length > 1) {
        const priorities = ['critical', 'high', 'medium', 'low', 'informational'];
        const firstPriorityIndex = priorities.indexOf(suggestions[0].priority);
        const secondPriorityIndex = priorities.indexOf(suggestions[1].priority);
        expect(firstPriorityIndex).toBeLessThanOrEqual(secondPriorityIndex);
      }
    });

    it('should suggest training for low stability', () => {
      // Create a complete brain state with v4.3 fields
      const lowStabilityBrain: BrainStats = {
        ...createBrain('low-stability'),
        mood: 'CALM' as Mood,
        stability: 5, // Must be < 10 to trigger critical
        creativity: 50
      };

      const suggestions = bridge.generatePrioritizedSuggestions(lowStabilityBrain);

      // Should have suggestions for critical low stability
      expect(suggestions.length).toBeGreaterThan(0);

      // First suggestion should be critical priority for low stability
      expect(suggestions[0].priority).toBe('critical');
      expect(suggestions[0].description).toContain('stability');
    });

    it('should suggest feeding for burnt out mood', () => {
      const burntOutBrain: BrainStats = {
        ...createBrain('burnt-out'),
        mood: 'BURNT_OUT' as Mood,
        stability: 70,
        creativity: 15
      };

      const suggestions = bridge.generatePrioritizedSuggestions(burntOutBrain);

      // Should have suggestions for burnt out state
      expect(suggestions.length).toBeGreaterThan(0);

      // Should have a suggestion about feeding/creativity
      const feedingSuggestion = suggestions.find(
        (s) => s.type === 'suggest_feeding' || s.description.toLowerCase().includes('creativ')
      );
      expect(feedingSuggestion).toBeDefined();
    });
  });

  describe('Recovery Planning', () => {
    it('should generate recovery plan for AGITATED mood', () => {
      const agitatedBrain: BrainStats = {
        ...createBrain('agitated-recovery'),
        mood: 'AGITATED' as Mood,
        stability: 20
      };

      const plan = bridge.generateRecoveryPlan(agitatedBrain);

      expect(plan).not.toBeNull();
      expect(plan?.mood).toBe('AGITATED');
      expect(plan?.actions.length).toBeGreaterThan(0);
      expect(plan?.description).toContain('stability');
    });

    it('should generate recovery plan for BURNT_OUT mood', () => {
      const burntOutBrain: BrainStats = {
        ...createBrain('burnt-out-recovery'),
        mood: 'BURNT_OUT' as Mood,
        creativity: 15
      };

      const plan = bridge.generateRecoveryPlan(burntOutBrain);

      expect(plan).not.toBeNull();
      expect(plan?.mood).toBe('BURNT_OUT');
      expect(plan?.description).toContain('creativity');
    });

    it('should return null for healthy moods', () => {
      const calmBrain: BrainStats = { ...createBrain('calm'), mood: 'CALM' as Mood };
      const focusedBrain: BrainStats = { ...createBrain('focused'), mood: 'FOCUSED' as Mood };

      expect(bridge.generateRecoveryPlan(calmBrain)).toBeNull();
      expect(bridge.generateRecoveryPlan(focusedBrain)).toBeNull();
    });
  });

  describe('State Management', () => {
    it('should enable/disable auto-pilot', () => {
      bridge.enableAutoPilot();
      expect(bridge.getState().autoPilot.enabled).toBe(true);

      bridge.disableAutoPilot();
      expect(bridge.getState().autoPilot.enabled).toBe(false);
    });

    it('should export and import state', () => {
      // Make some changes
      bridge.setAutoPilotConfig({ maxActionsPerHour: 20 });
      const action = bridge.createAction('log_observation', 'low', 'Test', 'Test');
      action.requiresConfirmation = true;
      bridge.queueAction(action);

      // Export
      const exported = bridge.exportState();

      // Create new bridge and import
      const newBridge = new BrainGovernanceBridge();
      newBridge.importState(exported);

      expect(newBridge.getState().autoPilot.maxActionsPerHour).toBe(20);
    });

    it('should reset state correctly', () => {
      bridge.setAutoPilotConfig({ maxActionsPerHour: 20 });
      const action = bridge.createAction('log_observation', 'low', 'Test', 'Test');
      action.requiresConfirmation = true;
      bridge.queueAction(action);

      bridge.reset();

      expect(bridge.getPendingActions().length).toBe(0);
      expect(bridge.getExecutedActions().length).toBe(0);
      expect(bridge.getState().actionsThisHour).toBe(0);
    });
  });

  describe('Singleton Management', () => {
    beforeEach(() => {
      resetBridge();
    });

    it('should return same instance from getBridge', () => {
      const bridge1 = getBridge();
      const bridge2 = getBridge();

      expect(bridge1).toBe(bridge2);
    });

    it('should reset singleton with resetBridge', () => {
      const bridge1 = getBridge();
      bridge1.setAutoPilotConfig({ maxActionsPerHour: 50 });

      resetBridge();

      const bridge2 = getBridge();
      expect(bridge2.getState().autoPilot.maxActionsPerHour).toBe(
        DEFAULT_AUTO_PILOT_CONFIG.maxActionsPerHour
      );
    });
  });
});

describe('MOOD_MODULATORS Configuration', () => {
  it('should have valid modulators for all moods', () => {
    const moods: Mood[] = ['CALM', 'FOCUSED', 'AGITATED', 'DREAMY', 'BURNT_OUT'];

    moods.forEach((mood) => {
      const modulator = MOOD_MODULATORS[mood];
      expect(modulator).toBeDefined();
      expect(modulator.activationProbabilityMultiplier).toBeGreaterThan(0);
      expect(modulator.learningRateAdjustmentScale).toBeGreaterThan(0);
      expect(modulator.dropoutAdjustmentScale).toBeGreaterThan(0);
      expect(modulator.suggestionFrequencyMs).toBeGreaterThan(0);
      expect(modulator.decayRateMultiplier).toBeGreaterThan(0);
    });
  });

  it('should have FOCUSED mood with reduced intervention', () => {
    expect(MOOD_MODULATORS.FOCUSED.activationProbabilityMultiplier).toBeLessThan(1);
    expect(MOOD_MODULATORS.FOCUSED.decayRateMultiplier).toBeLessThan(1);
  });

  it('should have AGITATED mood with increased intervention', () => {
    expect(MOOD_MODULATORS.AGITATED.activationProbabilityMultiplier).toBeGreaterThan(1);
    expect(MOOD_MODULATORS.AGITATED.learningRateAdjustmentScale).toBeLessThan(1);
  });
});
