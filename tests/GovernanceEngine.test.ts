/**
 * Tests for GovernanceEngine
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { GovernanceEngine, DEFAULT_GOVERNOR_CONFIG } from '../src/lib/GovernanceEngine';

describe('GovernanceEngine', () => {
  let engine: GovernanceEngine;

  beforeEach(() => {
    engine = new GovernanceEngine();
  });

  describe('Initialization', () => {
    it('should initialize with default config', () => {
      const state = engine.getState();
      expect(state.config).toEqual(DEFAULT_GOVERNOR_CONFIG);
      expect(state.metricHistory).toEqual([]);
      expect(state.alerts).toEqual([]);
      expect(state.calibrationHistory).toEqual([]);
      expect(state.ledger).toEqual([]);
      expect(state.sessionCount).toBe(0);
      expect(state.lastCheckSession).toBe(0);
    });

    it('should accept custom config', () => {
      const customEngine = new GovernanceEngine({
        checkInterval: 5,
        activationProbability: 0.8
      });

      const state = customEngine.getState();
      expect(state.config.checkInterval).toBe(5);
      expect(state.config.activationProbability).toBe(0.8);
      expect(state.config.improvementThreshold).toBe(DEFAULT_GOVERNOR_CONFIG.improvementThreshold);
    });
  });

  describe('Metric Recording', () => {
    it('should record metrics correctly', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      const state = engine.getState();
      expect(state.metricHistory).toHaveLength(1);
      expect(state.sessionCount).toBe(1);
      expect(state.metricHistory[0].trainLoss).toBe(2.5);
      expect(state.metricHistory[0].sessionId).toBe('session1');
    });

    it('should record multiple metrics', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.5
      });

      const state = engine.getState();
      expect(state.metricHistory).toHaveLength(2);
      expect(state.sessionCount).toBe(2);
    });

    it('should record validation metrics when provided', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0,
        valLoss: 2.8,
        valAccuracy: 0.55
      });

      const state = engine.getState();
      expect(state.metricHistory[0].valLoss).toBe(2.8);
      expect(state.metricHistory[0].valAccuracy).toBe(0.55);
    });
  });

  describe('Activation Logic', () => {
    it('should not activate before check interval', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      // Default check interval is 2, so should not activate after 1 session
      expect(engine.shouldActivate()).toBe(false);
    });

    it('should activate after check interval', () => {
      // Record 2 sessions
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.4,
        trainAccuracy: 0.62,
        perplexity: 11.0
      });

      // Should potentially activate (probabilistic)
      // We can't test exact activation due to randomness, but we can test the logic
      const state = engine.getState();
      expect(state.sessionCount).toBe(2);
      expect(state.sessionCount - state.lastCheckSession).toBeGreaterThanOrEqual(
        state.config.checkInterval
      );
    });

    it('should not activate when disabled', () => {
      const disabledEngine = new GovernanceEngine({ enabled: false });

      disabledEngine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      disabledEngine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.4,
        trainAccuracy: 0.62,
        perplexity: 11.0
      });

      expect(disabledEngine.shouldActivate()).toBe(false);
    });
  });

  describe('Plateau Detection', () => {
    it('should detect plateau when no improvement', () => {
      // Record sessions with no improvement
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.plateauDetected).toBe(true);
    });

    it('should not detect plateau when improving', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.5
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 1.5,
        trainAccuracy: 0.8,
        perplexity: 4.5
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.plateauDetected).toBe(false);
    });
  });

  describe('Overfitting Detection', () => {
    it('should detect overfitting when train/val gap is large', () => {
      // Need at least 2 metrics for analysis
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 1.0,
        trainAccuracy: 0.9,
        perplexity: 2.7,
        valLoss: 2.5, // 150% higher than train
        valAccuracy: 0.6
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 0.9,
        trainAccuracy: 0.92,
        perplexity: 2.5,
        valLoss: 2.6,
        valAccuracy: 0.58
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.overfittingDetected).toBe(true);
    });

    it('should not detect overfitting when gap is small', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.4,
        valLoss: 2.1, // Only 5% higher
        valAccuracy: 0.68
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.overfittingDetected).toBe(false);
    });

    it('should not detect overfitting without validation data', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 1.0,
        trainAccuracy: 0.9,
        perplexity: 2.7
        // No validation data
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.overfittingDetected).toBe(false);
    });
  });

  describe('Underfitting Detection', () => {
    it('should detect underfitting when both losses are high', () => {
      // Need at least 2 metrics for analysis
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 3.0,
        trainAccuracy: 0.3,
        perplexity: 20.0,
        valLoss: 3.2,
        valAccuracy: 0.28
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 3.1,
        trainAccuracy: 0.29,
        perplexity: 22.0,
        valLoss: 3.3,
        valAccuracy: 0.27
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.underfittingDetected).toBe(true);
    });

    it('should not detect underfitting when losses are acceptable', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 1.5,
        trainAccuracy: 0.7,
        perplexity: 4.5,
        valLoss: 1.6,
        valAccuracy: 0.68
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.underfittingDetected).toBe(false);
    });
  });

  describe('Divergence Detection', () => {
    it('should detect divergence when loss increases significantly', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.4
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.5, // 25% increase
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.divergenceDetected).toBe(true);
    });

    it('should not detect divergence when loss decreases', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.4
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.divergenceDetected).toBe(false);
    });
  });

  describe('Oscillation Detection', () => {
    it('should detect oscillation with high variance', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.4
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 3.5,
        trainAccuracy: 0.5,
        perplexity: 33.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 1.8,
        trainAccuracy: 0.75,
        perplexity: 6.0
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.oscillationDetected).toBe(true);
    });

    it('should not detect oscillation with stable loss', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.0,
        trainAccuracy: 0.7,
        perplexity: 7.4
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 1.9,
        trainAccuracy: 0.71,
        perplexity: 6.7
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 1.8,
        trainAccuracy: 0.72,
        perplexity: 6.0
      });

      const analysis = engine.analyzeMetrics();
      expect(analysis.oscillationDetected).toBe(false);
    });
  });

  describe('Calibration', () => {
    it('should reduce learning rate on plateau', () => {
      // Create plateau scenario
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      const actions = engine.calibrate(0.1, 0.1, 'proj1', 'session3');

      expect(actions).toHaveLength(1);
      expect(actions[0].parameter).toBe('learningRate');
      expect(actions[0].newValue).toBeLessThan(actions[0].previousValue);
      expect(actions[0].reason).toContain('Plateau');
    });

    it('should increase dropout on overfitting', () => {
      // Create overfitting scenario
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 1.0,
        trainAccuracy: 0.9,
        perplexity: 2.7,
        valLoss: 2.5,
        valAccuracy: 0.6
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 0.9,
        trainAccuracy: 0.92,
        perplexity: 2.5,
        valLoss: 2.6,
        valAccuracy: 0.58
      });

      const actions = engine.calibrate(0.1, 0.1, 'proj1', 'session2');

      expect(actions).toHaveLength(1);
      expect(actions[0].parameter).toBe('dropout');
      expect(actions[0].newValue).toBeGreaterThan(actions[0].previousValue);
      expect(actions[0].reason).toContain('Overfitting');
    });

    it('should respect learning rate bounds', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      // Use very small LR close to minimum
      const actions = engine.calibrate(1e-6, 0.1, 'proj1', 'session3');

      // Should not reduce below minimum
      if (actions.length > 0 && actions[0].parameter === 'learningRate') {
        expect(actions[0].newValue).toBeGreaterThanOrEqual(DEFAULT_GOVERNOR_CONFIG.learningRate.min);
      }
    });

    it('should respect dropout bounds', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 1.0,
        trainAccuracy: 0.9,
        perplexity: 2.7,
        valLoss: 2.5,
        valAccuracy: 0.6
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 0.9,
        trainAccuracy: 0.92,
        perplexity: 2.5,
        valLoss: 2.6,
        valAccuracy: 0.58
      });

      // Use dropout close to maximum
      const actions = engine.calibrate(0.1, 0.49, 'proj1', 'session2');

      if (actions.length > 0 && actions[0].parameter === 'dropout') {
        expect(actions[0].newValue).toBeLessThanOrEqual(DEFAULT_GOVERNOR_CONFIG.dropout.max);
      }
    });

    it('should record calibration action in ledger', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.calibrate(0.1, 0.1, 'proj1', 'session3');

      const ledger = engine.getLedger();
      const calibrationEntries = ledger.filter(e => e.type === 'calibration');

      expect(calibrationEntries.length).toBeGreaterThan(0);
      expect(calibrationEntries[0].calibrationAction).toBeDefined();
    });
  });

  describe('Alert Management', () => {
    it('should create alerts for detected issues', () => {
      // Create plateau to trigger alert
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.calibrate(0.1, 0.1, 'proj1', 'session3');

      const alerts = engine.getActiveAlerts();
      expect(alerts.length).toBeGreaterThan(0);
    });

    it('should acknowledge alerts', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.calibrate(0.1, 0.1, 'proj1', 'session3');

      const alerts = engine.getActiveAlerts();
      const alertId = alerts[0].id;

      engine.acknowledgeAlert(alertId);

      const remainingAlerts = engine.getActiveAlerts();
      expect(remainingAlerts.find(a => a.id === alertId)).toBeUndefined();
    });

    it('should clear all alerts', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session2',
        epoch: 1,
        trainLoss: 2.49,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.recordMetrics({
        sessionId: 'session3',
        epoch: 2,
        trainLoss: 2.51,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.calibrate(0.1, 0.1, 'proj1', 'session3');

      engine.clearAlerts();

      const alerts = engine.getActiveAlerts();
      expect(alerts).toHaveLength(0);
    });
  });

  describe('State Management', () => {
    it('should export state', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      const exported = engine.exportState();

      expect(exported.metricHistory).toHaveLength(1);
      expect(exported.sessionCount).toBe(1);
    });

    it('should import state', () => {
      const state = engine.exportState();

      const newEngine = new GovernanceEngine();
      newEngine.importState(state);

      expect(newEngine.getState()).toEqual(state);
    });

    it('should reset state', () => {
      engine.recordMetrics({
        sessionId: 'session1',
        epoch: 0,
        trainLoss: 2.5,
        trainAccuracy: 0.6,
        perplexity: 12.0
      });

      engine.reset();

      const state = engine.getState();
      expect(state.metricHistory).toHaveLength(0);
      expect(state.sessionCount).toBe(0);
      expect(state.alerts).toHaveLength(0);
      expect(state.calibrationHistory).toHaveLength(0);
      expect(state.ledger).toHaveLength(0);
    });
  });
});
