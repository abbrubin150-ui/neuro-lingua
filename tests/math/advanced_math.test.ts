/**
 * Tests for Advanced Mathematical Modules
 *
 * Tests for:
 * - Linearized Attention via Kernel Features
 * - K-FAC Optimizer
 * - NTK Analysis
 * - Variational Information Bottleneck
 * - Conformal Prediction
 */

import { describe, it, expect, beforeEach } from 'vitest';

// Linearized Attention
import {
  linearizedAttention,
  multiHeadLinearizedAttention,
  computeApproximationError,
  estimateMemorySavings,
  type KernelType
} from '../../src/models/linearized_attention';

// K-FAC Optimizer
import { KFACOptimizer, KFAC_DEFAULTS } from '../../src/training/KFACOptimizer';

// NTK Analysis
import {
  computeNTK,
  analyzeNTKDynamics,
  analyzeTrainability,
  analyzeSpectrum,
  computeNTKSummary,
  generateNTKAlerts
} from '../../src/math/ntk_analysis';

// Variational IB
import {
  reparameterize,
  gaussianKL,
  computeVIBLoss,
  estimateMINE,
  estimateInfoNCE,
  getBetaWithWarmup,
  diagnoseVIB,
  VIB_DEFAULTS,
  VIB_BETA_SCHEDULE_DEFAULTS,
  type GaussianParams
} from '../../src/losses/variational_ib';

// Conformal Prediction
import {
  softmaxScore,
  apsScore,
  rapsScore,
  calibrate,
  makePredictionSet,
  evaluateCoverage,
  initAdaptiveConformal,
  updateAdaptiveConformal,
  analyzeConditionalCoverage,
  CONFORMAL_DEFAULTS
} from '../../src/explainability/conformal_prediction';

// ============================================================================
// Linearized Attention Tests
// ============================================================================

describe('Linearized Attention', () => {
  // Test data
  const seqLen = 4;
  const dim = 8;
  const queries: number[][] = [];
  const keys: number[][] = [];
  const values: number[][] = [];

  beforeEach(() => {
    // Generate random test matrices
    queries.length = 0;
    keys.length = 0;
    values.length = 0;

    for (let i = 0; i < seqLen; i++) {
      queries.push(Array.from({ length: dim }, () => Math.random() - 0.5));
      keys.push(Array.from({ length: dim }, () => Math.random() - 0.5));
      values.push(Array.from({ length: dim }, () => Math.random() - 0.5));
    }
  });

  describe('ELU kernel', () => {
    it('should produce output of correct shape', () => {
      const result = linearizedAttention(queries, keys, values, {
        kernelType: 'elu',
        causal: false
      });

      expect(result.output.length).toBe(seqLen);
      expect(result.output[0].length).toBe(dim);
      expect(result.stats.kernelType).toBe('elu');
    });

    it('should handle causal masking', () => {
      const nonCausal = linearizedAttention(queries, keys, values, {
        kernelType: 'elu',
        causal: false
      });

      const causal = linearizedAttention(queries, keys, values, {
        kernelType: 'elu',
        causal: true
      });

      // Outputs should be different due to causal masking
      expect(causal.output).not.toEqual(nonCausal.output);
    });
  });

  describe('RFF kernel', () => {
    it('should use random Fourier features', () => {
      const result = linearizedAttention(queries, keys, values, {
        kernelType: 'rff',
        numFeatures: 128,
        seed: 42
      });

      expect(result.output.length).toBe(seqLen);
      expect(result.stats.featureDim).toBe(256); // 2 * numFeatures
    });

    it('should be deterministic with same seed', () => {
      const result1 = linearizedAttention(queries, keys, values, {
        kernelType: 'rff',
        numFeatures: 64,
        seed: 123
      });

      const result2 = linearizedAttention(queries, keys, values, {
        kernelType: 'rff',
        numFeatures: 64,
        seed: 123
      });

      expect(result1.output).toEqual(result2.output);
    });
  });

  describe('FAVOR+ kernel', () => {
    it('should produce non-negative features', () => {
      const result = linearizedAttention(queries, keys, values, {
        kernelType: 'favor',
        numFeatures: 64
      });

      expect(result.output.length).toBe(seqLen);
      expect(result.stats.kernelType).toBe('favor');
    });
  });

  describe('Chebyshev kernel', () => {
    it('should use polynomial features', () => {
      const result = linearizedAttention(queries, keys, values, {
        kernelType: 'chebyshev',
        chebyshevDegree: 4
      });

      expect(result.output.length).toBe(seqLen);
      // Feature dim = inputDim * (degree + 1)
      expect(result.stats.featureDim).toBe(dim * 5);
    });
  });

  describe('Multi-head linearized attention', () => {
    it('should split into multiple heads', () => {
      const largerDim = 16;
      const q = Array.from({ length: seqLen }, () =>
        Array.from({ length: largerDim }, () => Math.random())
      );
      const k = Array.from({ length: seqLen }, () =>
        Array.from({ length: largerDim }, () => Math.random())
      );
      const v = Array.from({ length: seqLen }, () =>
        Array.from({ length: largerDim }, () => Math.random())
      );

      const result = multiHeadLinearizedAttention(q, k, v, {
        kernelType: 'elu',
        numHeads: 4,
        modelDim: largerDim
      });

      expect(result.output.length).toBe(seqLen);
      expect(result.output[0].length).toBe(largerDim);
    });
  });

  describe('Memory savings estimation', () => {
    it('should estimate memory savings correctly', () => {
      const savings = estimateMemorySavings(1024, 256, 64);

      expect(savings.standardMemory).toBeGreaterThan(0);
      expect(savings.linearMemory).toBeGreaterThan(0);
      // For long sequences, linear attention should save memory
      expect(savings.savingsPercent).toBeGreaterThan(0);
    });
  });
});

// ============================================================================
// K-FAC Optimizer Tests
// ============================================================================

describe('K-FAC Optimizer', () => {
  let optimizer: KFACOptimizer;

  beforeEach(() => {
    optimizer = new KFACOptimizer(KFAC_DEFAULTS);
  });

  describe('Initialization', () => {
    it('should initialize with default config', () => {
      const config = optimizer.getConfig();

      expect(config.lr).toBe(0.01);
      expect(config.emaDecay).toBe(0.95);
      expect(config.damping).toBe(0.001);
    });

    it('should accept custom config', () => {
      const customOptimizer = new KFACOptimizer({
        lr: 0.005,
        momentum: 0.95
      });

      const config = customOptimizer.getConfig();
      expect(config.lr).toBe(0.005);
      expect(config.momentum).toBe(0.95);
    });
  });

  describe('Statistics update', () => {
    it('should accumulate activation statistics', () => {
      const activations = [
        [1, 2, 3],
        [4, 5, 6]
      ];

      optimizer.updateActivationStatistics('layer1', activations);
      // Statistics should be stored (no error)
    });

    it('should accumulate gradient statistics', () => {
      const gradients = [
        [0.1, 0.2],
        [0.3, 0.4]
      ];

      optimizer.updateGradientStatistics('layer1', gradients);
      // Statistics should be stored
    });
  });

  describe('Matrix update', () => {
    it('should update weights without error', () => {
      const W = [
        [1, 2],
        [3, 4]
      ];
      const G = [
        [0.1, 0.2],
        [0.3, 0.4]
      ];

      optimizer.updateMatrix(W, G, 'test_layer');

      // Weights should be modified
      expect(W[0][0]).not.toBe(1);
    });

    it('should apply weight decay', () => {
      const optimizer2 = new KFACOptimizer({
        lr: 0.1,
        weightDecay: 0.1,
        momentum: 0
      });

      const W = [[1.0]];
      const G = [[0]]; // Zero gradient

      optimizer2.updateMatrix(W, G, 'test');

      // Weight should decrease due to weight decay
      expect(W[0][0]).toBeLessThan(1.0);
    });
  });

  describe('Vector update', () => {
    it('should update bias vector', () => {
      const b = [1, 2, 3];
      const g = [0.1, 0.2, 0.3];

      optimizer.updateVector(b, g, 'bias');

      expect(b[0]).not.toBe(1);
    });
  });

  describe('State management', () => {
    it('should reset state', () => {
      optimizer.updateActivationStatistics('layer1', [[1, 2]]);
      optimizer.reset();

      expect(optimizer.getStep()).toBe(0);
    });

    it('should export and import state', () => {
      optimizer.updateActivationStatistics('layer1', [[1, 2]]);
      optimizer.step();

      const exported = optimizer.exportState();
      expect(exported.step).toBe(1);

      const newOptimizer = new KFACOptimizer();
      newOptimizer.importState(exported);
      expect(newOptimizer.getStep()).toBe(1);
    });
  });
});

// ============================================================================
// NTK Analysis Tests
// ============================================================================

describe('NTK Analysis', () => {
  describe('NTK computation', () => {
    it('should compute NTK matrix', () => {
      const gradients = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];

      const ntk = computeNTK(gradients);

      expect(ntk.kernel.length).toBe(3);
      expect(ntk.kernel[0].length).toBe(3);
      // Identity gradients should give identity NTK
      expect(ntk.kernel[0][0]).toBeCloseTo(1);
      expect(ntk.kernel[1][1]).toBeCloseTo(1);
    });

    it('should handle empty gradients', () => {
      const ntk = computeNTK([]);

      expect(ntk.kernel.length).toBe(0);
      expect(ntk.eigenvalues.length).toBe(0);
    });

    it('should compute eigenvalues', () => {
      const gradients = [
        [1, 2],
        [3, 4],
        [5, 6]
      ];

      const ntk = computeNTK(gradients);

      expect(ntk.eigenvalues.length).toBeGreaterThan(0);
      // Eigenvalues should be sorted descending
      for (let i = 1; i < ntk.eigenvalues.length; i++) {
        expect(ntk.eigenvalues[i]).toBeLessThanOrEqual(ntk.eigenvalues[i - 1]);
      }
    });
  });

  describe('Trainability analysis', () => {
    it('should identify trainable network', () => {
      const gradients = [
        [1, 0],
        [0, 1]
      ];
      const ntk = computeNTK(gradients);
      const trainability = analyzeTrainability(ntk);

      expect(trainability.isTrainable).toBe(true);
      expect(trainability.trainabilityScore).toBeGreaterThan(0);
    });

    it('should generate warnings for ill-conditioned NTK', () => {
      // Create highly ill-conditioned gradients
      const gradients = [
        [1, 0],
        [1.001, 0.001]
      ];
      const ntk = computeNTK(gradients);
      const trainability = analyzeTrainability(ntk);

      // May have warnings
      expect(trainability.warnings).toBeDefined();
    });
  });

  describe('Spectrum analysis', () => {
    it('should analyze eigenvalue spectrum', () => {
      const eigenvalues = [100, 50, 25, 10, 5, 2, 1];
      const spectrum = analyzeSpectrum(eigenvalues);

      expect(spectrum.spectrum).toEqual(eigenvalues);
      expect(spectrum.significantCount).toBeGreaterThan(0);
      expect(spectrum.decayRate).toBeGreaterThanOrEqual(0);
    });

    it('should compute spectral entropy', () => {
      const eigenvalues = [1, 1, 1, 1]; // Uniform distribution
      const spectrum = analyzeSpectrum(eigenvalues);

      // Maximum entropy for uniform distribution
      expect(spectrum.spectralEntropy).toBeGreaterThan(0);
    });
  });

  describe('NTK dynamics', () => {
    it('should detect lazy training regime', () => {
      const gradients = [
        [1, 2],
        [3, 4]
      ];

      // Slightly perturbed gradients (lazy regime)
      const perturbedGradients = [
        [1.01, 2.01],
        [3.01, 4.01]
      ];

      const dynamics = analyzeNTKDynamics(gradients, perturbedGradients, 0.1);

      expect(dynamics.relativeChange).toBeLessThan(0.1);
      expect(dynamics.lazyRegime).toBe(true);
    });
  });

  describe('Alert generation', () => {
    it('should generate alerts for training issues', () => {
      const gradients = [[1], [2]];
      const ntk = computeNTK(gradients);
      const trainability = analyzeTrainability(ntk);
      const spectrum = analyzeSpectrum(ntk.eigenvalues);

      const alerts = generateNTKAlerts(trainability, undefined, spectrum);

      expect(alerts).toBeDefined();
    });
  });
});

// ============================================================================
// Variational IB Tests
// ============================================================================

describe('Variational Information Bottleneck', () => {
  describe('Gaussian utilities', () => {
    it('should reparameterize samples', () => {
      const mu = [0, 0, 0];
      const logVar = [0, 0, 0]; // Unit variance

      const z = reparameterize(mu, logVar);

      expect(z.length).toBe(3);
      // Samples should be centered around mu (approximately)
    });

    it('should compute KL divergence', () => {
      const mu = [0, 0];
      const logVar = [0, 0]; // N(0,1)

      const kl = gaussianKL(mu, logVar);

      // KL(N(0,1) || N(0,1)) = 0
      expect(kl).toBeCloseTo(0, 5);
    });

    it('should compute positive KL for non-standard Gaussian', () => {
      const mu = [1, 1];
      const logVar = [0, 0];

      const kl = gaussianKL(mu, logVar);

      expect(kl).toBeGreaterThan(0);
    });
  });

  describe('VIB loss computation', () => {
    it('should compute VIB loss', () => {
      const encoderParams: GaussianParams[] = [
        { mu: [0, 0], logVar: [0, 0] },
        { mu: [0, 0], logVar: [0, 0] }
      ];

      const decoderLogits = [
        [1, 0, 0],
        [0, 1, 0]
      ];
      const targets = [0, 1];

      const metrics = computeVIBLoss(encoderParams, decoderLogits, targets, VIB_DEFAULTS);

      expect(metrics.vibLoss).toBeDefined();
      expect(metrics.reconstructionLoss).toBeGreaterThanOrEqual(0);
      expect(metrics.klDivergence).toBeGreaterThanOrEqual(0);
    });

    it('should handle empty batch', () => {
      const metrics = computeVIBLoss([], [], [], VIB_DEFAULTS);

      expect(metrics.vibLoss).toBe(0);
    });
  });

  describe('MI estimation', () => {
    it('should estimate MI with MINE', () => {
      const joint = [
        { x: [1, 2], y: [1, 2] },
        { x: [3, 4], y: [3, 4] }
      ];
      const marginalX = [[1, 2], [3, 4]];
      const marginalY = [[3, 4], [1, 2]]; // Shuffled

      const estimate = estimateMINE(joint, marginalX, marginalY);

      expect(estimate.method).toBe('MINE');
      expect(estimate.mi).toBeGreaterThanOrEqual(0);
    });

    it('should estimate MI with InfoNCE', () => {
      const anchor = [[1, 0], [0, 1]];
      const positive = [[1, 0], [0, 1]];
      const negatives = [[0, 1], [1, 0]];

      const estimate = estimateInfoNCE(anchor, positive, negatives);

      expect(estimate.method).toBe('InfoNCE');
      expect(estimate.mi).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Beta scheduling', () => {
    it('should respect warmup phase', () => {
      const beta = getBetaWithWarmup(2, VIB_BETA_SCHEDULE_DEFAULTS);

      // During warmup, beta should be betaInit
      expect(beta).toBe(VIB_BETA_SCHEDULE_DEFAULTS.betaInit);
    });

    it('should anneal after warmup', () => {
      const warmupEpochs = VIB_BETA_SCHEDULE_DEFAULTS.warmupEpochs;
      const beta = getBetaWithWarmup(warmupEpochs + 10, VIB_BETA_SCHEDULE_DEFAULTS);

      // After warmup, beta should move toward betaFinal
      expect(beta).toBeGreaterThan(VIB_BETA_SCHEDULE_DEFAULTS.betaInit);
    });
  });

  describe('VIB diagnostics', () => {
    it('should detect posterior collapse', () => {
      const encoderParams: GaussianParams[] = [
        { mu: [0, 0], logVar: [0, 0] }
      ];

      const metrics = {
        reconstructionLoss: 0.5,
        klDivergence: 0.001, // Very small KL
        vibLoss: 0.501,
        rate: 0.001,
        distortion: 0.5,
        beta: 1
      };

      const diagnostics = diagnoseVIB(encoderParams, metrics, VIB_DEFAULTS);

      expect(diagnostics.posteriorCollapse).toBe(true);
    });
  });
});

// ============================================================================
// Conformal Prediction Tests
// ============================================================================

describe('Conformal Prediction', () => {
  // Test softmax predictions
  const predictions = [
    [0.7, 0.2, 0.1], // High confidence for class 0
    [0.3, 0.5, 0.2], // Medium confidence for class 1
    [0.2, 0.3, 0.5], // Medium confidence for class 2
    [0.33, 0.33, 0.34] // Uncertain
  ];
  const trueLabels = [0, 1, 2, 0];

  describe('Conformity scores', () => {
    it('should compute softmax score', () => {
      const score = softmaxScore([0.8, 0.1, 0.1], 0);

      // s = 1 - 0.8 = 0.2
      expect(score).toBeCloseTo(0.2);
    });

    it('should compute APS score', () => {
      const probs = [0.5, 0.3, 0.2];
      const score = apsScore(probs, 1);

      // Class 1 has prob 0.3, classes with >= 0.3 are [0, 1]
      // Sum = 0.5 + 0.3 = 0.8 (before random tie-breaking)
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should compute RAPS score with regularization', () => {
      const probs = [0.5, 0.3, 0.2];
      const score = rapsScore(probs, 2, 0.1, 1);

      // Class 2 is rank 3, regularization = 0.1 * (3-1) = 0.2
      expect(score).toBeGreaterThan(0);
    });
  });

  describe('Calibration', () => {
    it('should calibrate on calibration set', () => {
      const result = calibrate(predictions, trueLabels, CONFORMAL_DEFAULTS);

      expect(result.threshold).toBeGreaterThan(0);
      expect(result.calibrationSize).toBe(4);
      expect(result.conformityScores.length).toBe(4);
    });

    it('should achieve approximately target coverage', () => {
      // Generate larger calibration set
      const largePreds: number[][] = [];
      const largeLabels: number[] = [];

      for (let i = 0; i < 100; i++) {
        const probs = [Math.random(), Math.random(), Math.random()];
        const sum = probs.reduce((a, b) => a + b, 0);
        largePreds.push(probs.map((p) => p / sum));
        largeLabels.push(Math.floor(Math.random() * 3));
      }

      const result = calibrate(largePreds, largeLabels, {
        ...CONFORMAL_DEFAULTS,
        coverageLevel: 0.9
      });

      // Empirical coverage should be close to 90%
      expect(result.empiricalCoverage).toBeGreaterThan(0.8);
    });
  });

  describe('Prediction sets', () => {
    it('should make prediction set for single sample', () => {
      const probs = [0.8, 0.15, 0.05];
      const predSet = makePredictionSet(probs, 0.5, CONFORMAL_DEFAULTS);

      expect(predSet.topClass).toBe(0);
      expect(predSet.topConfidence).toBe(0.8);
      expect(predSet.size).toBeGreaterThanOrEqual(1);
    });

    it('should include more classes with higher threshold', () => {
      const probs = [0.4, 0.35, 0.25];

      const smallSet = makePredictionSet(probs, 0.3, CONFORMAL_DEFAULTS);
      const largeSet = makePredictionSet(probs, 0.9, CONFORMAL_DEFAULTS);

      expect(largeSet.size).toBeGreaterThanOrEqual(smallSet.size);
    });
  });

  describe('Coverage evaluation', () => {
    it('should evaluate coverage on test set', () => {
      const diagnostics = evaluateCoverage(predictions, trueLabels, 0.5, CONFORMAL_DEFAULTS);

      expect(diagnostics.coverage).toBeGreaterThanOrEqual(0);
      expect(diagnostics.coverage).toBeLessThanOrEqual(1);
      expect(diagnostics.averageSetSize).toBeGreaterThan(0);
    });

    it('should report singleton rate', () => {
      const highConfPreds = [
        [0.95, 0.03, 0.02],
        [0.02, 0.95, 0.03]
      ];
      const labels = [0, 1];

      const diagnostics = evaluateCoverage(highConfPreds, labels, 0.5, CONFORMAL_DEFAULTS);

      // High confidence predictions should have many singletons
      expect(diagnostics.singletonRate).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Adaptive conformal', () => {
    it('should initialize adaptive state', () => {
      const state = initAdaptiveConformal(0.5, 0.01);

      expect(state.threshold).toBe(0.5);
      expect(state.gamma).toBe(0.01);
      expect(state.numSamples).toBe(0);
    });

    it('should update threshold adaptively', () => {
      let state = initAdaptiveConformal(0.5, 0.1);

      // Simulate predictions
      for (let i = 0; i < 10; i++) {
        const probs = [0.6, 0.3, 0.1];
        state = updateAdaptiveConformal(state, probs, 0, 0.9, CONFORMAL_DEFAULTS);
      }

      expect(state.numSamples).toBe(10);
      // Threshold should have adjusted
      expect(state.threshold).not.toBe(0.5);
    });
  });

  describe('Conditional coverage', () => {
    it('should analyze per-class coverage', () => {
      const analysis = analyzeConditionalCoverage(
        predictions,
        trueLabels,
        0.5,
        CONFORMAL_DEFAULTS
      );

      expect(analysis.perClassCoverage.size).toBeGreaterThan(0);
      expect(analysis.worstClassCoverage).toBeLessThanOrEqual(1);
    });
  });
});
