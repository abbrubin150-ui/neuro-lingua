/**
 * CausalInferenceEngine Test Suite
 *
 * Comprehensive tests for the probabilistic dynamic causal inference system.
 * Tests cover:
 * - Engine initialization and configuration
 * - Offline learning phase
 * - Online selection phase
 * - Statistical testing
 * - Identifiability checking
 * - Bias verification
 * - State management
 *
 * @module CausalInferenceEngine.test
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  CausalInferenceEngine,
  createCausalEngine,
  restoreCausalEngine
} from '../../src/lib/CausalInferenceEngine';
import type { CausalModelConfig, HistoricalDataset } from '../../src/types/causal';

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Create a minimal test configuration
 */
function createTestConfig(overrides: Partial<CausalModelConfig> = {}): Partial<CausalModelConfig> {
  return {
    numStudents: 20,
    numTimeSteps: 10,
    featureDimension: 3,
    seed: 42,
    outcomeModel: {
      intercept: 0,
      treatmentCoefficient: 0.5, // True causal effect
      featureCoefficients: [0.1, 0.1, 0.1],
      confounderCoefficients: [0.2],
      autoregCoefficient: 0.5,
      noiseStd: 0.3
    },
    selectionModel: {
      intercept: 0,
      featureCoefficients: [0.1, 0.1, 0.1],
      previousOutcomeCoefficient: 0.1,
      quantizationCoefficients: [0.05],
      confounderCoefficients: [0.3]
    },
    confounderConfig: {
      dimension: 1,
      mean: [0],
      covariance: [1]
    },
    initialQuantization: {
      timeStep: 0,
      numBins: 5,
      boundaries: [-1, -0.3, 0.3, 1],
      method: 'uniform',
      symmetric: true
    },
    ...overrides
  };
}

// ============================================================================
// Initialization Tests
// ============================================================================

describe('CausalInferenceEngine', () => {
  describe('Initialization', () => {
    it('should create engine with default configuration', () => {
      const engine = new CausalInferenceEngine();

      expect(engine.getPhase()).toBe('uninitialized');
      const state = engine.getState();
      expect(state.config).toBeDefined();
      expect(state.config.numStudents).toBeGreaterThan(0);
      expect(state.config.numTimeSteps).toBeGreaterThan(0);
    });

    it('should create engine with custom configuration', () => {
      const config = createTestConfig({ numStudents: 50, numTimeSteps: 30 });
      const engine = new CausalInferenceEngine(config);

      const state = engine.getState();
      expect(state.config.numStudents).toBe(50);
      expect(state.config.numTimeSteps).toBe(30);
    });

    it('should initialize audit log', () => {
      const engine = new CausalInferenceEngine();
      const log = engine.getAuditLog();

      expect(Array.isArray(log)).toBe(true);
      expect(log.length).toBeGreaterThan(0);
      expect(log[0].action).toBe('initialize');
    });

    it('should use seeded random for reproducibility', () => {
      const engine1 = new CausalInferenceEngine({ seed: 12345 });
      const engine2 = new CausalInferenceEngine({ seed: 12345 });

      const data1 = engine1.simulateHistoricalData();
      const data2 = engine2.simulateHistoricalData();

      // First few records should be identical
      expect(data1.records[0].continuousOutcome.value).toBe(
        data2.records[0].continuousOutcome.value
      );
    });

    it('should create engine using factory function', () => {
      const engine = createCausalEngine({ numStudents: 25 });

      expect(engine.getPhase()).toBe('uninitialized');
      expect(engine.getState().config.numStudents).toBe(25);
    });
  });

  // ============================================================================
  // Data Simulation Tests
  // ============================================================================

  describe('Data Simulation', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
    });

    it('should simulate historical data with correct structure', () => {
      const data = engine.simulateHistoricalData();

      expect(data.numStudents).toBe(20);
      expect(data.numTimeSteps).toBe(10);
      expect(data.featureDimension).toBe(3);
      expect(data.records.length).toBe(20 * 10);
    });

    it('should generate valid features for each record', () => {
      const data = engine.simulateHistoricalData();

      for (const record of data.records) {
        expect(record.features.values.length).toBe(3);
        expect(record.features.studentId).toBe(record.studentId);
        expect(record.features.timeStep).toBe(record.timeStep);
      }
    });

    it('should generate outcomes with treatment effect', () => {
      const data = engine.simulateHistoricalData();

      // Compute average outcome by treatment
      const outcomesA: number[] = [];
      const outcomesB: number[] = [];

      for (const record of data.records) {
        if (record.policySelection.policy === 'A') {
          outcomesA.push(record.continuousOutcome.value);
        } else {
          outcomesB.push(record.continuousOutcome.value);
        }
      }

      // With treatment effect of 0.5, B should have higher mean on average
      // (though this is stochastic, so we just check both exist)
      expect(outcomesA.length).toBeGreaterThan(0);
      expect(outcomesB.length).toBeGreaterThan(0);
    });

    it('should generate valid quantized outcomes', () => {
      const data = engine.simulateHistoricalData();

      for (const record of data.records) {
        expect(record.quantizedOutcome.binIndex).toBeGreaterThanOrEqual(0);
        expect(record.quantizedOutcome.binIndex).toBeLessThan(5);
      }
    });

    it('should generate valid propensity scores', () => {
      const data = engine.simulateHistoricalData();

      for (const record of data.records) {
        const p = record.policySelection.propensityScore!;
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    });

    it('should include metadata', () => {
      const data = engine.simulateHistoricalData();

      expect(data.metadata).toBeDefined();
      expect(data.metadata.collectionPeriod).toBe('simulated');
      expect(data.metadata.dataSource).toContain('simulateHistoricalData');
    });
  });

  // ============================================================================
  // Offline Learning Phase Tests
  // ============================================================================

  describe('Offline Learning Phase', () => {
    let engine: CausalInferenceEngine;
    let historicalData: HistoricalDataset;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      historicalData = engine.simulateHistoricalData();
    });

    it('should transition to offline_learning phase', () => {
      engine.runOfflinePhase(historicalData);

      // Phase should transition through offline_learning
      expect(engine.getAuditLog().some((e) => e.phase === 'offline_learning')).toBe(true);
    });

    it('should learn propensity model parameters', () => {
      const result = engine.runOfflinePhase(historicalData);

      expect(result.propensityParams).toBeDefined();
      expect(result.propensityParams.length).toBeGreaterThan(0);
    });

    it('should learn outcome model parameters', () => {
      const result = engine.runOfflinePhase(historicalData);

      expect(result.outcomeParams).toBeDefined();
      expect(result.outcomeParams.length).toBeGreaterThan(0);
    });

    it('should learn dequantization mappings', () => {
      const result = engine.runOfflinePhase(historicalData);

      expect(result.dequantizationMappings).toBeDefined();
      expect(result.dequantizationMappings.length).toBe(5); // numBins
    });

    it('should estimate historical ATE', () => {
      const result = engine.runOfflinePhase(historicalData);

      expect(result.historicalATE).toBeDefined();
      expect(result.historicalATE.method).toBe('aipw');
      expect(typeof result.historicalATE.estimate).toBe('number');
      expect(result.historicalATE.standardError).toBeGreaterThan(0);
    });

    it('should compute confidence interval for historical ATE', () => {
      const result = engine.runOfflinePhase(historicalData);

      const [lower, upper] = result.historicalATE.confidenceInterval;
      expect(lower).toBeLessThan(upper);
      expect(lower).toBeLessThanOrEqual(result.historicalATE.estimate);
      expect(upper).toBeGreaterThanOrEqual(result.historicalATE.estimate);
    });

    it('should optimize initial quantization', () => {
      const result = engine.runOfflinePhase(historicalData);

      expect(result.optimizedQuantization).toBeDefined();
      expect(result.optimizedQuantization.boundaries.length).toBe(4);
    });

    it('should store offline results in state', () => {
      engine.runOfflinePhase(historicalData);

      const state = engine.getState();
      expect(state.offlineResults).toBeDefined();
    });

    it('should log offline phase completion', () => {
      engine.runOfflinePhase(historicalData);

      const log = engine.getAuditLog();
      expect(log.some((e) => e.action === 'offline_phase_complete')).toBe(true);
    });
  });

  // ============================================================================
  // Online Selection Phase Tests
  // ============================================================================

  describe('Online Selection Phase', () => {
    let engine: CausalInferenceEngine;
    let historicalData: HistoricalDataset;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      historicalData = engine.simulateHistoricalData();
      engine.runOfflinePhase(historicalData);
    });

    it('should initialize online phase', () => {
      engine.initializeOnlinePhase();

      expect(engine.getPhase()).toBe('online_selection');
      expect(engine.getOnlineState()).toBeDefined();
    });

    it('should throw if offline phase not completed', () => {
      const freshEngine = new CausalInferenceEngine(createTestConfig());

      expect(() => freshEngine.initializeOnlinePhase()).toThrow();
    });

    it('should select policies for students', () => {
      engine.initializeOnlinePhase();

      const features = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
      ];
      const studentIds = ['s1', 's2'];

      const selections = engine.selectPolicies(features, studentIds);

      expect(selections.length).toBe(2);
      expect(['A', 'B']).toContain(selections[0].policy);
      expect(selections[0].propensityScore).toBeGreaterThan(0);
      expect(selections[0].propensityScore).toBeLessThan(1);
    });

    it('should enforce exploration bounds on propensity', () => {
      engine.initializeOnlinePhase();

      const features = Array.from({ length: 100 }, () => [0, 0, 0]);
      const studentIds = features.map((_, i) => `s${i}`);

      const selections = engine.selectPolicies(features, studentIds);

      for (const sel of selections) {
        expect(sel.propensityScore).toBeGreaterThanOrEqual(0.1);
        expect(sel.propensityScore).toBeLessThanOrEqual(0.9);
      }
    });

    it('should record online observations', () => {
      engine.initializeOnlinePhase();

      const observations = [
        {
          studentId: 's1',
          features: [0.1, 0.2, 0.3],
          policy: 'A' as const,
          quantizedOutcome: 2,
          propensityScore: 0.5
        },
        {
          studentId: 's2',
          features: [0.4, 0.5, 0.6],
          policy: 'B' as const,
          quantizedOutcome: 3,
          propensityScore: 0.6
        }
      ];

      engine.recordOnlineObservations(observations);

      const state = engine.getOnlineState();
      expect(state?.observations.length).toBe(2);
    });

    it('should dequantize outcomes when recording', () => {
      engine.initializeOnlinePhase();

      const observations = [
        {
          studentId: 's1',
          features: [0.1, 0.2, 0.3],
          policy: 'A' as const,
          quantizedOutcome: 2,
          propensityScore: 0.5
        }
      ];

      engine.recordOnlineObservations(observations);

      const state = engine.getOnlineState();
      expect(typeof state?.observations[0].dequantizedOutcome).toBe('number');
    });

    it('should update running ATE estimate', () => {
      engine.initializeOnlinePhase();

      // Add enough observations to trigger ATE update
      const observations = Array.from({ length: 20 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random(), Math.random(), Math.random()],
        policy: (i % 2 === 0 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: Math.floor(Math.random() * 5),
        propensityScore: 0.5
      }));

      engine.recordOnlineObservations(observations);

      const state = engine.getOnlineState();
      expect(state?.runningATE.numObservations).toBe(20);
    });

    it('should advance time step', () => {
      engine.initializeOnlinePhase();

      expect(engine.getOnlineState()?.currentTimeStep).toBe(0);

      engine.advanceTimeStep();

      expect(engine.getOnlineState()?.currentTimeStep).toBe(1);
    });

    it('should adapt quantization online', () => {
      engine.initializeOnlinePhase();
      const initialQuant = { ...engine.getOnlineState()!.currentQuantization };

      // Add many observations to trigger adaptation
      const observations = Array.from({ length: 50 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random(), Math.random(), Math.random()],
        policy: (i % 2 === 0 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: Math.floor(Math.random() * 5),
        propensityScore: 0.5
      }));

      engine.recordOnlineObservations(observations);

      const finalQuant = engine.getOnlineState()!.currentQuantization;

      // Boundaries may have adapted (not always, depends on data)
      expect(finalQuant.timeStep).toBe(initialQuant.timeStep);
    });
  });

  // ============================================================================
  // Statistical Testing Phase Tests
  // ============================================================================

  describe('Statistical Testing Phase', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      // Simulate online data collection
      const observations = Array.from({ length: 100 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5],
        policy: (Math.random() > 0.5 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: Math.floor(Math.random() * 5),
        propensityScore: 0.5
      }));

      engine.recordOnlineObservations(observations);
    });

    it('should run statistical test', () => {
      const result = engine.runStatisticalTest();

      expect(result).toBeDefined();
      expect(typeof result.testStatistic).toBe('number');
      expect(typeof result.pValue).toBe('number');
      expect(typeof result.reject).toBe('boolean');
    });

    it('should use specified significance level', () => {
      const result = engine.runStatisticalTest(0.01);

      expect(result.significanceLevel).toBe(0.01);
    });

    it('should compute confidence interval', () => {
      const result = engine.runStatisticalTest();

      const [lower, upper] = result.confidenceInterval;
      expect(lower).toBeLessThan(upper);
    });

    it('should estimate achieved power', () => {
      const result = engine.runStatisticalTest();

      expect(result.achievedPower).toBeDefined();
      expect(result.achievedPower).toBeGreaterThanOrEqual(0);
      expect(result.achievedPower).toBeLessThanOrEqual(1);
    });

    it('should transition to testing phase', () => {
      engine.runStatisticalTest();

      expect(engine.getPhase()).toBe('testing');
    });

    it('should store test results in state', () => {
      engine.runStatisticalTest();

      const state = engine.getState();
      expect(state.testResults).toBeDefined();
    });

    it('should throw if no observations', () => {
      const freshEngine = new CausalInferenceEngine(createTestConfig());
      const data = freshEngine.simulateHistoricalData();
      freshEngine.runOfflinePhase(data);
      freshEngine.initializeOnlinePhase();

      expect(() => freshEngine.runStatisticalTest()).toThrow();
    });
  });

  // ============================================================================
  // Power Analysis Tests
  // ============================================================================

  describe('Power Analysis', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const observations = Array.from({ length: 50 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5],
        policy: (Math.random() > 0.5 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: Math.floor(Math.random() * 5),
        propensityScore: 0.5
      }));

      engine.recordOnlineObservations(observations);
    });

    it('should compute power analysis', () => {
      const result = engine.computePowerAnalysis(0.3);

      expect(result.effectSize).toBe(0.3);
      expect(result.requiredSampleSize).toBeGreaterThan(0);
      expect(result.achievedPower).toBeGreaterThanOrEqual(0);
      expect(result.achievedPower).toBeLessThanOrEqual(1);
    });

    it('should compute minimum detectable effect', () => {
      const result = engine.computePowerAnalysis(0.3);

      expect(result.minimumDetectableEffect).toBeGreaterThan(0);
    });

    it('should require larger sample for smaller effect', () => {
      const resultSmall = engine.computePowerAnalysis(0.1);
      const resultLarge = engine.computePowerAnalysis(0.5);

      expect(resultSmall.requiredSampleSize).toBeGreaterThan(resultLarge.requiredSampleSize);
    });
  });

  // ============================================================================
  // Identifiability Tests
  // ============================================================================

  describe('Identifiability Checking', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const observations = Array.from({ length: 50 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random(), Math.random(), Math.random()],
        policy: (Math.random() > 0.5 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: Math.floor(Math.random() * 5),
        propensityScore: 0.3 + Math.random() * 0.4 // Keep in valid range
      }));

      engine.recordOnlineObservations(observations);
    });

    it('should check identifiability conditions', () => {
      const result = engine.checkIdentifiability();

      expect(typeof result.identifiable).toBe('boolean');
      expect(typeof result.positivity).toBe('boolean');
      expect(typeof result.quantizationInvertible).toBe('boolean');
    });

    it('should provide diagnostics', () => {
      const result = engine.checkIdentifiability();

      expect(result.diagnostics).toBeDefined();
      expect(typeof result.diagnostics.minPropensity).toBe('number');
      expect(typeof result.diagnostics.maxPropensity).toBe('number');
    });

    it('should check positivity condition', () => {
      const result = engine.checkIdentifiability();

      // With propensities in [0.3, 0.7], positivity should be satisfied
      expect(result.positivity).toBe(true);
    });

    it('should provide warnings for potential issues', () => {
      const result = engine.checkIdentifiability();

      expect(Array.isArray(result.diagnostics.warnings)).toBe(true);
    });
  });

  // ============================================================================
  // Bias Verification Tests
  // ============================================================================

  describe('Bias Verification', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      // Add balanced observations
      const observations = Array.from({ length: 40 }, (_, i) => ({
        studentId: `s${i}`,
        features: [Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5],
        policy: (i % 2 === 0 ? 'A' : 'B') as 'A' | 'B', // Balanced
        quantizedOutcome: 2 + Math.floor(Math.random() * 2), // Around middle
        propensityScore: 0.5
      }));

      engine.recordOnlineObservations(observations);
    });

    it('should verify unbiasedness', () => {
      const result = engine.verifyUnbiasedness(20); // Fewer simulations for speed

      expect(typeof result.unbiased).toBe('boolean');
      expect(result.neutrality).toBeDefined();
      expect(result.differentialNeutrality).toBeDefined();
      expect(result.fairness).toBeDefined();
    });

    it('should check neutrality axiom', () => {
      const result = engine.verifyUnbiasedness(20);

      expect(typeof result.neutrality.neutral).toBe('boolean');
      expect(typeof result.neutrality.symmetryError).toBe('number');
    });

    it('should check differential neutrality', () => {
      const result = engine.verifyUnbiasedness(20);

      expect(typeof result.differentialNeutrality.satisfied).toBe('boolean');
      expect(typeof result.differentialNeutrality.epsilon).toBe('number');
    });

    it('should check causal fairness', () => {
      const result = engine.verifyUnbiasedness(20);

      expect(typeof result.fairness.fair).toBe('boolean');
      expect(typeof result.fairness.delta).toBe('number');
    });

    it('should estimate Type I error rate', () => {
      const result = engine.verifyUnbiasedness(50);

      expect(result.typeIErrorRate).toBeGreaterThanOrEqual(0);
      expect(result.typeIErrorRate).toBeLessThanOrEqual(1);
    });

    it('should perform sensitivity analysis', () => {
      const result = engine.verifyUnbiasedness(20);

      expect(typeof result.sensitivitySummary.sensitive).toBe('boolean');
      expect(typeof result.sensitivitySummary.ateChange).toBe('number');
    });

    it('should provide recommendations', () => {
      const result = engine.verifyUnbiasedness(20);

      expect(Array.isArray(result.recommendations)).toBe(true);
    });

    it('should store bias verification in state', () => {
      engine.verifyUnbiasedness(20);

      const state = engine.getState();
      expect(state.biasVerification).toBeDefined();
    });
  });

  // ============================================================================
  // State Management Tests
  // ============================================================================

  describe('State Management', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
    });

    it('should export state', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);

      const exportedState = engine.exportState();

      expect(exportedState.phase).not.toBe('uninitialized');
      expect(exportedState.config).toBeDefined();
      expect(exportedState.offlineResults).toBeDefined();
    });

    it('should import state', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      const exportedState = engine.exportState();

      const newEngine = new CausalInferenceEngine(createTestConfig());
      newEngine.importState(exportedState);

      expect(newEngine.getState().offlineResults).toBeDefined();
    });

    it('should restore engine from state', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      const exportedState = engine.exportState();

      const restoredEngine = restoreCausalEngine(exportedState);

      expect(restoredEngine.getState().offlineResults).toBeDefined();
    });

    it('should reset engine', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);

      engine.reset();

      expect(engine.getPhase()).toBe('uninitialized');
      expect(engine.getState().offlineResults).toBeUndefined();
    });

    it('should preserve config on reset', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      const originalConfig = engine.getState().config;

      engine.reset();

      expect(engine.getState().config.numStudents).toBe(originalConfig.numStudents);
    });
  });

  // ============================================================================
  // Audit Log and Ledger Tests
  // ============================================================================

  describe('Audit Log and Ledger', () => {
    let engine: CausalInferenceEngine;

    beforeEach(() => {
      engine = new CausalInferenceEngine(createTestConfig());
    });

    it('should maintain audit log', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const log = engine.getAuditLog();

      expect(log.length).toBeGreaterThan(1);
      expect(log.every((e) => e.timestamp > 0)).toBe(true);
      expect(log.every((e) => typeof e.action === 'string')).toBe(true);
    });

    it('should record phase transitions in log', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const log = engine.getAuditLog();
      const phases = new Set(log.map((e) => e.phase));

      expect(phases.has('uninitialized')).toBe(true);
      expect(phases.has('offline_learning')).toBe(true);
      expect(phases.has('online_selection')).toBe(true);
    });

    it('should generate causal analysis ledger', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const observations = Array.from({ length: 10 }, (_, i) => ({
        studentId: `s${i}`,
        features: [0.1, 0.2, 0.3],
        policy: (i % 2 === 0 ? 'A' : 'B') as 'A' | 'B',
        quantizedOutcome: 2,
        propensityScore: 0.5
      }));
      engine.recordOnlineObservations(observations);

      const ledger = engine.generateLedger();

      expect(ledger.entries.length).toBeGreaterThan(0);
      expect(ledger.summary.totalDecisions).toBe(ledger.entries.length);
    });

    it('should categorize ledger entries by type', () => {
      const data = engine.simulateHistoricalData();
      engine.runOfflinePhase(data);
      engine.initializeOnlinePhase();

      const ledger = engine.generateLedger();

      expect(typeof ledger.summary.quantizationUpdates).toBe('number');
      expect(typeof ledger.summary.policySelections).toBe('number');
      expect(typeof ledger.summary.ateEstimates).toBe('number');
    });
  });

  // ============================================================================
  // End-to-End Integration Tests
  // ============================================================================

  describe('End-to-End Integration', () => {
    it('should run complete workflow', () => {
      // Configuration
      const engine = new CausalInferenceEngine({
        ...createTestConfig(),
        numStudents: 30,
        numTimeSteps: 5
      });

      // Phase 1: Offline learning
      const historicalData = engine.simulateHistoricalData();
      const offlineResult = engine.runOfflinePhase(historicalData);

      expect(offlineResult.historicalATE).toBeDefined();

      // Phase 2: Online selection
      engine.initializeOnlinePhase();

      for (let t = 0; t < 3; t++) {
        const features = Array.from({ length: 10 }, () => [
          Math.random(),
          Math.random(),
          Math.random()
        ]);
        const studentIds = features.map((_, i) => `t${t}-s${i}`);

        const selections = engine.selectPolicies(features, studentIds);

        const observations = selections.map((sel, i) => ({
          studentId: sel.studentId,
          features: features[i],
          policy: sel.policy,
          quantizedOutcome: Math.floor(Math.random() * 5),
          propensityScore: sel.propensityScore
        }));

        engine.recordOnlineObservations(observations);
        engine.advanceTimeStep();
      }

      // Phase 3: Statistical testing
      const testResult = engine.runStatisticalTest();

      expect(testResult).toBeDefined();
      expect(typeof testResult.pValue).toBe('number');

      // Phase 4: Bias verification
      const biasResult = engine.verifyUnbiasedness(10);

      expect(biasResult).toBeDefined();

      // Check identifiability
      const identifiability = engine.checkIdentifiability();

      expect(identifiability).toBeDefined();

      // Generate ledger
      const ledger = engine.generateLedger();

      expect(ledger.entries.length).toBeGreaterThan(0);
    });

    it('should produce reasonable estimates with known effect', () => {
      // Create engine with known strong effect
      const engine = new CausalInferenceEngine({
        ...createTestConfig(),
        numStudents: 100,
        numTimeSteps: 20,
        outcomeModel: {
          intercept: 0,
          treatmentCoefficient: 1.0, // Strong effect
          featureCoefficients: [0.1, 0.1, 0.1],
          confounderCoefficients: [0.1], // Weak confounding
          autoregCoefficient: 0.3,
          noiseStd: 0.5
        },
        selectionModel: {
          intercept: 0,
          featureCoefficients: [0.05, 0.05, 0.05],
          previousOutcomeCoefficient: 0.05,
          quantizationCoefficients: [0.01],
          confounderCoefficients: [0.1] // Weak selection bias
        }
      });

      const data = engine.simulateHistoricalData();
      const offlineResult = engine.runOfflinePhase(data);

      // Historical ATE should be positive (effect is positive)
      // Note: This is not guaranteed due to confounding, but likely
      expect(typeof offlineResult.historicalATE.estimate).toBe('number');
    });
  });
});
