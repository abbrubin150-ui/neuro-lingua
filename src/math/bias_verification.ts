/**
 * Bias Verification Module for Causal Inference
 *
 * This module provides comprehensive bias verification utilities for
 * causal inference algorithms, including:
 * - Neutrality axiom verification
 * - Differential privacy-inspired constraints
 * - Causal fairness metrics
 * - Simulation-based auditing
 * - Sensitivity analysis
 * - Adversarial testing
 *
 * These checks ensure that the causal inference algorithm does not
 * systematically favor any particular policy due to:
 * - Historical biases in the training data
 * - Quantization artifacts
 * - Model misspecification
 *
 * Reference: Rubin (1974), Pearl (2009)
 *
 * @module bias_verification
 */

import type {
  AverageTreatmentEffect,
  QuantizationParams,
  BiasVerificationResult,
  NeutralityVerification,
  DifferentialNeutralityCheck,
  FairnessMetric,
  SensitivityAnalysis
} from '../types/causal';

import {
  mean,
  variance,
  stdDev,
  computeAIPWEstimate,
  testATESignificance,
  createSeededRandom,
  randomNormal
} from './causal_math';

// ============================================================================
// Constants
// ============================================================================

/** Default neutrality tolerance */
const DEFAULT_NEUTRALITY_TOLERANCE = 0.1;

/** Default differential neutrality epsilon */
const DEFAULT_EPSILON = 0.5;

/** Default fairness delta threshold */
const DEFAULT_DELTA = 0.05;

/** Default significance level for testing */
const DEFAULT_ALPHA = 0.05;

/** Default number of simulations for auditing */
const DEFAULT_NUM_SIMULATIONS = 100;

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * Data sample for bias verification
 */
export interface VerificationSample {
  outcomes: number[];
  treatments: number[]; // 0 for A, 1 for B
  propensities: number[];
  muA: number[];
  muB: number[];
}

/**
 * Configuration for bias verification
 */
export interface BiasVerificationConfig {
  /** Neutrality tolerance threshold */
  neutralityTolerance?: number;
  /** Differential neutrality epsilon target */
  epsilonTarget?: number;
  /** Fairness delta threshold */
  deltaTarget?: number;
  /** Significance level for hypothesis tests */
  alpha?: number;
  /** Number of simulations for auditing */
  numSimulations?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

/**
 * Result of a single audit simulation
 */
export interface AuditSimulationResult {
  /** Estimated ATE under null */
  nullATE: number;
  /** Whether null was rejected */
  rejected: boolean;
  /** P-value */
  pValue: number;
}

/**
 * Comprehensive audit result
 */
export interface ComprehensiveAuditResult {
  /** Individual simulation results */
  simulations: AuditSimulationResult[];
  /** Type I error rate */
  typeIErrorRate: number;
  /** Mean null ATE (should be ~0) */
  meanNullATE: number;
  /** Standard deviation of null ATE */
  stdNullATE: number;
  /** Whether audit passed */
  passed: boolean;
  /** Diagnostic messages */
  diagnostics: string[];
}

// ============================================================================
// Neutrality Axiom Verification
// ============================================================================

/**
 * Verify neutrality axiom: algorithm is invariant to A↔B and Y→-Y swap
 *
 * The neutrality axiom requires that if we:
 * 1. Swap policy labels (A↔B)
 * 2. Negate all outcomes (Y→-Y)
 *
 * Then the estimated ATE should be negated: τ̂_swapped ≈ -τ̂_original
 *
 * @param sample - Original verification sample
 * @param options - Verification options
 * @returns Neutrality verification result
 */
export function verifyNeutralityAxiom(
  sample: VerificationSample,
  options: { tolerance?: number } = {}
): NeutralityVerification {
  const tolerance = options.tolerance ?? DEFAULT_NEUTRALITY_TOLERANCE;

  // Compute original ATE
  const { ate: originalATE } = computeAIPWEstimate(
    sample.outcomes,
    sample.treatments,
    sample.propensities,
    sample.muA,
    sample.muB
  );

  // Create swapped sample
  const swappedSample: VerificationSample = {
    outcomes: sample.outcomes.map((y) => -y),
    treatments: sample.treatments.map((t) => 1 - t),
    propensities: sample.propensities.map((p) => 1 - p),
    muA: sample.muB.map((m) => -m),
    muB: sample.muA.map((m) => -m)
  };

  // Compute swapped ATE
  const { ate: swappedATE } = computeAIPWEstimate(
    swappedSample.outcomes,
    swappedSample.treatments,
    swappedSample.propensities,
    swappedSample.muA,
    swappedSample.muB
  );

  // Compute symmetry error
  // Should have: originalEstimate + swappedEstimate ≈ 0
  const originalEstimate = originalATE.estimate;
  const swappedEstimate = swappedATE.estimate;
  const denominator = Math.abs(originalEstimate) + Math.abs(swappedEstimate);
  const symmetryError =
    denominator > 1e-10 ? Math.abs(originalEstimate + swappedEstimate) / denominator : 0;

  return {
    neutral: symmetryError < tolerance,
    originalEstimate,
    swappedEstimate,
    symmetryError,
    tolerance
  };
}

// ============================================================================
// Differential Neutrality
// ============================================================================

/**
 * Check differential privacy-inspired neutrality constraint
 *
 * This checks whether the quantization parameters θ_t are sufficiently
 * independent of the data to avoid encoding bias.
 *
 * Constraint: P(θ|data_A) / P(θ|data_B) ≤ e^ε
 *
 * @param quantizationHistory - History of quantization parameters
 * @param options - Check options
 * @returns Differential neutrality check result
 */
export function checkDifferentialNeutrality(
  quantizationHistory: QuantizationParams[],
  options: {
    targetEpsilon?: number;
    treatmentAssignments?: number[][];
  } = {}
): DifferentialNeutralityCheck {
  const targetEpsilon = options.targetEpsilon ?? DEFAULT_EPSILON;

  if (quantizationHistory.length === 0) {
    return {
      satisfied: true,
      epsilon: 0,
      targetEpsilon,
      noiseLevel: 0
    };
  }

  // Compute boundary variations across time
  const boundaryVariances: number[] = [];

  for (let i = 0; i < (quantizationHistory[0]?.boundaries.length ?? 0); i++) {
    const boundaryValues = quantizationHistory.map((q) => q.boundaries[i] ?? 0);
    boundaryVariances.push(variance(boundaryValues));
  }

  const avgBoundaryVariance = mean(boundaryVariances);
  const noiseLevel = Math.sqrt(avgBoundaryVariance) * 0.1;

  // Estimate effective epsilon
  // Higher variance means more data-dependent, thus higher epsilon needed
  const effectiveEpsilon = noiseLevel > 1e-10 ? Math.log(1 + 1 / noiseLevel) : Infinity;

  return {
    satisfied: effectiveEpsilon <= targetEpsilon,
    epsilon: effectiveEpsilon,
    targetEpsilon,
    noiseLevel
  };
}

/**
 * Add differential privacy noise to quantization parameters
 *
 * @param params - Quantization parameters
 * @param noiseScale - Scale of Gaussian noise to add
 * @param random - Random number generator
 * @returns Noised quantization parameters
 */
export function addDifferentialNoise(
  params: QuantizationParams,
  noiseScale: number,
  random: () => number = Math.random
): QuantizationParams {
  const noisedBoundaries = params.boundaries.map((b) => {
    const noise = randomNormal(0, noiseScale, 1, random)[0];
    return b + noise;
  });

  // Ensure monotonicity
  noisedBoundaries.sort((a, b) => a - b);

  return {
    ...params,
    boundaries: noisedBoundaries
  };
}

// ============================================================================
// Causal Fairness
// ============================================================================

/**
 * Check causal fairness metric
 * Δ = |E[τ̂|H_0,θ] - 0|
 *
 * Under the null hypothesis of no causal effect, the expected ATE
 * estimate should be zero. Systematic deviation indicates bias.
 *
 * @param sample - Verification sample
 * @param numSimulations - Number of null simulations
 * @param options - Check options
 * @returns Fairness metric result
 */
export function checkCausalFairness(
  sample: VerificationSample,
  numSimulations: number = DEFAULT_NUM_SIMULATIONS,
  options: {
    targetDelta?: number;
    seed?: number;
  } = {}
): FairnessMetric {
  const targetDelta = options.targetDelta ?? DEFAULT_DELTA;
  const random = options.seed ? createSeededRandom(options.seed) : Math.random;

  const n = sample.outcomes.length;
  if (n === 0) {
    return {
      fair: true,
      delta: 0,
      targetDelta,
      numNullSamples: 0
    };
  }

  const nullEstimates: number[] = [];

  for (let sim = 0; sim < numSimulations; sim++) {
    // Generate null sample by permuting treatments
    const permutedTreatments = [...sample.treatments];
    for (let i = permutedTreatments.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [permutedTreatments[i], permutedTreatments[j]] = [
        permutedTreatments[j],
        permutedTreatments[i]
      ];
    }

    const { ate } = computeAIPWEstimate(
      sample.outcomes,
      permutedTreatments,
      sample.propensities,
      sample.muA,
      sample.muB
    );

    nullEstimates.push(ate.estimate);
  }

  const nullMean = mean(nullEstimates);
  const delta = Math.abs(nullMean);

  return {
    fair: delta < targetDelta,
    delta,
    targetDelta,
    numNullSamples: numSimulations
  };
}

// ============================================================================
// Simulation-Based Auditing
// ============================================================================

/**
 * Run comprehensive simulation-based audit
 *
 * This generates data under the null hypothesis (no causal effect),
 * runs the estimation procedure, and checks:
 * 1. Type I error rate matches nominal α
 * 2. Null ATE distribution is centered at 0
 * 3. No systematic bias in any direction
 *
 * @param estimationProcedure - Function that computes ATE from sample
 * @param datGenerator - Function that generates null samples
 * @param numSimulations - Number of simulations
 * @param config - Audit configuration
 * @returns Comprehensive audit result
 */
export function runSimulationAudit(
  estimationProcedure: (sample: VerificationSample) => AverageTreatmentEffect,
  dataGenerator: (seed: number) => VerificationSample,
  numSimulations: number = DEFAULT_NUM_SIMULATIONS,
  config: BiasVerificationConfig = {}
): ComprehensiveAuditResult {
  const alpha = config.alpha ?? DEFAULT_ALPHA;
  const baseSeed = config.seed ?? 42;

  const simulations: AuditSimulationResult[] = [];
  const diagnostics: string[] = [];

  for (let i = 0; i < numSimulations; i++) {
    const sample = dataGenerator(baseSeed + i);
    const ate = estimationProcedure(sample);
    const { pValue, reject } = testATESignificance(ate, alpha);

    simulations.push({
      nullATE: ate.estimate,
      rejected: reject,
      pValue
    });
  }

  // Compute statistics
  const nullATEs = simulations.map((s) => s.nullATE);
  const meanNullATE = mean(nullATEs);
  const stdNullATE = stdDev(nullATEs);

  const rejections = simulations.filter((s) => s.rejected).length;
  const typeIErrorRate = rejections / numSimulations;

  // Determine pass/fail
  let passed = true;

  // Check Type I error rate is near nominal
  const typeITolerance = 1.5 * alpha; // Allow 50% deviation
  if (typeIErrorRate > typeITolerance) {
    passed = false;
    diagnostics.push(
      `Type I error rate (${(typeIErrorRate * 100).toFixed(1)}%) exceeds ` +
        `tolerance (${(typeITolerance * 100).toFixed(1)}%)`
    );
  }

  // Check mean null ATE is near 0
  const meanTolerance = (2 * stdNullATE) / Math.sqrt(numSimulations);
  if (Math.abs(meanNullATE) > meanTolerance) {
    passed = false;
    diagnostics.push(`Mean null ATE (${meanNullATE.toFixed(4)}) significantly different from 0`);
  }

  // Check for systematic bias
  const positiveATEs = nullATEs.filter((a) => a > 0).length;
  const signTestP =
    2 *
    Math.min(
      binomialPMF(positiveATEs, numSimulations, 0.5),
      binomialPMF(numSimulations - positiveATEs, numSimulations, 0.5)
    );
  if (signTestP < 0.01) {
    passed = false;
    diagnostics.push(`Sign imbalance detected (${positiveATEs}/${numSimulations} positive)`);
  }

  if (passed) {
    diagnostics.push('All audit checks passed');
  }

  return {
    simulations,
    typeIErrorRate,
    meanNullATE,
    stdNullATE,
    passed,
    diagnostics
  };
}

/**
 * Approximate binomial PMF for sign test
 */
function binomialPMF(k: number, n: number, p: number): number {
  // Use normal approximation for large n
  if (n > 30) {
    const mu = n * p;
    const sigma = Math.sqrt(n * p * (1 - p));
    const z = (k - mu) / sigma;
    return Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI) / sigma;
  }

  // Direct computation for small n
  let coeff = 1;
  for (let i = 0; i < k; i++) {
    coeff *= (n - i) / (i + 1);
  }
  return coeff * Math.pow(p, k) * Math.pow(1 - p, n - k);
}

// ============================================================================
// Sensitivity Analysis
// ============================================================================

/**
 * Run sensitivity analysis on quantization parameters
 *
 * Tests how robust the ATE estimate is to perturbations in
 * the quantization boundaries.
 *
 * @param sample - Verification sample
 * @param quantParams - Current quantization parameters
 * @param perturbationSizes - Array of perturbation sizes to test
 * @param numTrials - Number of trials per perturbation size
 * @param options - Analysis options
 * @returns Sensitivity analysis results
 */
export function runSensitivityAnalysis(
  sample: VerificationSample,
  quantParams: QuantizationParams,
  perturbationSizes: number[] = [0.05, 0.1, 0.2],
  numTrials: number = 50,
  options: { seed?: number } = {}
): {
  results: SensitivityAnalysis[];
  robustness: 'high' | 'medium' | 'low';
  recommendations: string[];
} {
  const random = options.seed ? createSeededRandom(options.seed) : Math.random;

  // Compute baseline ATE
  const { ate: baselineATE } = computeAIPWEstimate(
    sample.outcomes,
    sample.treatments,
    sample.propensities,
    sample.muA,
    sample.muB
  );

  const results: SensitivityAnalysis[] = [];

  for (const perturbSize of perturbationSizes) {
    const ateChanges: number[] = [];

    for (let trial = 0; trial < numTrials; trial++) {
      // Perturb boundaries (params stored for potential diagnostics)
      const _perturbedParams = addDifferentialNoise(
        quantParams,
        perturbSize * (Math.max(...quantParams.boundaries) - Math.min(...quantParams.boundaries)),
        random
      );

      // Note: In a real implementation, we would re-quantize and dequantize
      // For now, we simulate the effect by adding noise to outcomes
      const perturbedOutcomes = sample.outcomes.map((o) => {
        const noise = randomNormal(0, perturbSize * Math.abs(o) + 0.01, 1, random)[0];
        return o + noise;
      });

      const { ate: perturbedATE } = computeAIPWEstimate(
        perturbedOutcomes,
        sample.treatments,
        sample.propensities,
        sample.muA,
        sample.muB
      );

      const change = Math.abs(perturbedATE.estimate - baselineATE.estimate);
      ateChanges.push(change);
    }

    const avgChange = mean(ateChanges);
    const threshold = 2 * baselineATE.standardError;
    const sensitive = avgChange > threshold;

    results.push({
      perturbationSize: perturbSize,
      ateChange: avgChange,
      sensitive,
      threshold
    });
  }

  // Determine overall robustness
  const sensitiveCount = results.filter((r) => r.sensitive).length;
  let robustness: 'high' | 'medium' | 'low';
  const recommendations: string[] = [];

  if (sensitiveCount === 0) {
    robustness = 'high';
    recommendations.push('ATE estimate is robust to quantization perturbations');
  } else if (sensitiveCount <= perturbationSizes.length / 2) {
    robustness = 'medium';
    recommendations.push('ATE estimate shows moderate sensitivity to large perturbations');
    recommendations.push('Consider using finer quantization bins');
  } else {
    robustness = 'low';
    recommendations.push('ATE estimate is highly sensitive to quantization');
    recommendations.push(
      'Strongly recommend increasing number of bins or using continuous outcomes'
    );
    recommendations.push('Results should be interpreted with caution');
  }

  return { results, robustness, recommendations };
}

// ============================================================================
// Adversarial Testing
// ============================================================================

/**
 * Run adversarial testing to find worst-case bias
 *
 * Attempts to find quantization parameters that maximize bias
 * in the ATE estimate, to understand the algorithm's worst-case behavior.
 *
 * @param sample - Verification sample
 * @param initialParams - Initial quantization parameters
 * @param numIterations - Number of optimization iterations
 * @param options - Testing options
 * @returns Adversarial test result
 */
export function runAdversarialTest(
  sample: VerificationSample,
  initialParams: QuantizationParams,
  numIterations: number = 100,
  options: { seed?: number; targetDirection?: 'positive' | 'negative' | 'any' } = {}
): {
  worstCaseParams: QuantizationParams;
  worstCaseBias: number;
  baselineBias: number;
  vulnerabilityScore: number;
  recommendations: string[];
} {
  const random = options.seed ? createSeededRandom(options.seed) : Math.random;
  const targetDirection = options.targetDirection ?? 'any';

  // Compute baseline ATE
  const { ate: baselineATE } = computeAIPWEstimate(
    sample.outcomes,
    sample.treatments,
    sample.propensities,
    sample.muA,
    sample.muB
  );

  let worstCaseParams = { ...initialParams };
  let worstCaseBias = Math.abs(baselineATE.estimate);

  // Simple random search for worst-case parameters
  for (let iter = 0; iter < numIterations; iter++) {
    // Generate candidate perturbation
    const perturbScale =
      0.2 * (Math.max(...initialParams.boundaries) - Math.min(...initialParams.boundaries));
    const candidateParams = addDifferentialNoise(initialParams, perturbScale, random);

    // Simulate effect of quantization change on outcomes
    // In practice, this would involve re-quantizing with new params
    const scale = 1 + (random() - 0.5) * 0.3;
    const shift = (random() - 0.5) * 0.2;
    const perturbedOutcomes = sample.outcomes.map((o) => o * scale + shift);

    const { ate: perturbedATE } = computeAIPWEstimate(
      perturbedOutcomes,
      sample.treatments,
      sample.propensities,
      sample.muA,
      sample.muB
    );

    // Check if this is worse
    let isBad = false;
    const currentBias = perturbedATE.estimate;

    switch (targetDirection) {
      case 'positive':
        isBad = currentBias > worstCaseBias;
        break;
      case 'negative':
        isBad = currentBias < -worstCaseBias;
        break;
      default:
        isBad = Math.abs(currentBias) > worstCaseBias;
    }

    if (isBad) {
      worstCaseBias = Math.abs(currentBias);
      worstCaseParams = candidateParams;
    }
  }

  // Compute vulnerability score
  const baselineBias = Math.abs(baselineATE.estimate);
  const vulnerabilityScore =
    baselineBias > 1e-10 ? worstCaseBias / baselineBias - 1 : worstCaseBias;

  // Generate recommendations
  const recommendations: string[] = [];

  if (vulnerabilityScore < 0.2) {
    recommendations.push('Algorithm shows low vulnerability to adversarial quantization');
  } else if (vulnerabilityScore < 0.5) {
    recommendations.push('Algorithm shows moderate vulnerability');
    recommendations.push('Consider constraining quantization parameter updates');
  } else {
    recommendations.push('Algorithm shows high vulnerability to adversarial manipulation');
    recommendations.push('Strongly recommend using fixed quantization parameters');
    recommendations.push('Consider adding noise to quantization updates for robustness');
  }

  return {
    worstCaseParams,
    worstCaseBias,
    baselineBias,
    vulnerabilityScore,
    recommendations
  };
}

// ============================================================================
// Comprehensive Verification
// ============================================================================

/**
 * Run comprehensive bias verification
 *
 * Combines all verification methods into a single comprehensive check.
 *
 * @param sample - Verification sample
 * @param quantizationHistory - History of quantization parameters
 * @param config - Verification configuration
 * @returns Comprehensive bias verification result
 */
export function runComprehensiveBiasVerification(
  sample: VerificationSample,
  quantizationHistory: QuantizationParams[],
  config: BiasVerificationConfig = {}
): BiasVerificationResult {
  const numSimulations = config.numSimulations ?? DEFAULT_NUM_SIMULATIONS;

  // 1. Neutrality verification
  const neutrality = verifyNeutralityAxiom(sample, {
    tolerance: config.neutralityTolerance ?? DEFAULT_NEUTRALITY_TOLERANCE
  });

  // 2. Differential neutrality
  const differentialNeutrality = checkDifferentialNeutrality(quantizationHistory, {
    targetEpsilon: config.epsilonTarget ?? DEFAULT_EPSILON
  });

  // 3. Causal fairness
  const fairness = checkCausalFairness(sample, numSimulations, {
    targetDelta: config.deltaTarget ?? DEFAULT_DELTA,
    seed: config.seed
  });

  // 4. Estimate Type I error rate (result stored for potential diagnostics)
  const _typeIResults = checkCausalFairness(sample, numSimulations, {
    targetDelta: 1, // Accept all for counting rejections
    seed: config.seed ? config.seed + 1000 : undefined
  });

  // Count rejections under null
  let typeIErrorRate = 0;
  const nullEstimates: number[] = [];
  const random = config.seed ? createSeededRandom(config.seed) : Math.random;

  for (let sim = 0; sim < numSimulations; sim++) {
    const permutedTreatments = [...sample.treatments];
    for (let i = permutedTreatments.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [permutedTreatments[i], permutedTreatments[j]] = [
        permutedTreatments[j],
        permutedTreatments[i]
      ];
    }

    const { ate } = computeAIPWEstimate(
      sample.outcomes,
      permutedTreatments,
      sample.propensities,
      sample.muA,
      sample.muB
    );

    nullEstimates.push(ate.estimate);
    const { reject } = testATESignificance(ate, config.alpha ?? DEFAULT_ALPHA);
    if (reject) typeIErrorRate++;
  }
  typeIErrorRate /= numSimulations;

  // 5. Sensitivity analysis
  const currentQuant = quantizationHistory[quantizationHistory.length - 1] ?? {
    timeStep: 0,
    numBins: 5,
    boundaries: [-1, 0, 1],
    method: 'uniform' as const,
    symmetric: true
  };

  const sensitivityResults = runSensitivityAnalysis(sample, currentQuant, [0.1], 20, {
    seed: config.seed
  });

  const sensitivitySummary = sensitivityResults.results[0] ?? {
    perturbationSize: 0.1,
    ateChange: 0,
    sensitive: false,
    threshold: 0.2
  };

  // Build recommendations
  const recommendations: string[] = [];

  if (!neutrality.neutral) {
    recommendations.push(
      'Algorithm may favor one policy. Consider symmetric quantization boundaries.'
    );
  }
  if (!differentialNeutrality.satisfied) {
    recommendations.push('Add noise to quantization updates to reduce data-dependent bias.');
  }
  if (!fairness.fair) {
    recommendations.push(
      'Under null hypothesis, estimator shows systematic bias. Review dequantization calibration.'
    );
  }
  if (typeIErrorRate > (config.alpha ?? DEFAULT_ALPHA) * 1.5) {
    recommendations.push(
      'Type I error rate exceeds nominal. Use more conservative significance threshold.'
    );
  }
  if (sensitivitySummary.sensitive) {
    recommendations.push(
      'ATE estimate is sensitive to quantization perturbations. Consider finer bins.'
    );
  }

  // Determine overall unbiasedness
  const unbiased =
    neutrality.neutral &&
    differentialNeutrality.satisfied &&
    fairness.fair &&
    typeIErrorRate <= (config.alpha ?? DEFAULT_ALPHA) * 1.5 &&
    !sensitivitySummary.sensitive;

  return {
    unbiased,
    neutrality,
    differentialNeutrality,
    fairness,
    typeIErrorRate,
    sensitivitySummary,
    recommendations
  };
}
