/**
 * Conformal Prediction for Uncertainty Quantification
 *
 * Conformal prediction provides prediction sets with guaranteed coverage:
 *   P(Y ∈ C(X)) ≥ 1 - α
 *
 * for any desired confidence level 1-α, under exchangeability assumption.
 *
 * This module provides:
 * - Split conformal prediction (inductive)
 * - Adaptive prediction sets (APS)
 * - Regularized adaptive prediction sets (RAPS)
 * - Coverage diagnostics and calibration
 *
 * References:
 * - Vovk et al. (2005) "Algorithmic Learning in a Random World"
 * - Romano et al. (2020) "Classification with Valid and Adaptive Coverage"
 * - Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
 *
 * @module explainability/conformal_prediction
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface ConformalConfig {
  /** Desired coverage level (e.g., 0.9 for 90% coverage) */
  coverageLevel: number;
  /** Score function type */
  scoreType: 'softmax' | 'aps' | 'raps';
  /** RAPS regularization strength */
  rapsLambda?: number;
  /** RAPS size penalty threshold k_reg */
  rapsKreg?: number;
  /** Random seed for reproducibility */
  seed?: number;
}

export interface PredictionSet {
  /** Indices of classes in the prediction set */
  classes: number[];
  /** Conformity scores for included classes */
  scores: number[];
  /** Threshold used for this prediction */
  threshold: number;
  /** Set size */
  size: number;
  /** Most likely class (argmax) */
  topClass: number;
  /** Confidence for top class */
  topConfidence: number;
}

export interface CalibrationResult {
  /** Calibrated threshold for desired coverage */
  threshold: number;
  /** Empirical coverage on calibration set */
  empiricalCoverage: number;
  /** Average prediction set size */
  averageSetSize: number;
  /** Conformity scores on calibration set */
  conformityScores: number[];
  /** Number of calibration samples */
  calibrationSize: number;
}

export interface CoverageDiagnostics {
  /** Empirical coverage rate */
  coverage: number;
  /** Average prediction set size */
  averageSetSize: number;
  /** Median prediction set size */
  medianSetSize: number;
  /** Singleton rate (sets with exactly 1 class) */
  singletonRate: number;
  /** Empty set rate (should be 0 for valid conformal) */
  emptyRate: number;
  /** Coverage is valid (within tolerance of target) */
  isValid: boolean;
  /** Diagnostics message */
  message: string;
}

// ============================================================================
// Conformity Score Functions
// ============================================================================

/**
 * Simple softmax-based conformity score (THR method)
 *
 * s(x, y) = 1 - π_y(x)
 *
 * where π_y is the softmax probability for class y
 */
export function softmaxScore(softmaxProbs: number[], trueClass: number): number {
  return 1 - softmaxProbs[trueClass];
}

/**
 * Adaptive Prediction Sets (APS) score
 *
 * s(x, y) = Σ_{j: π_j ≥ π_y} π_j
 *
 * Sum of probabilities of classes more likely than true class
 */
export function apsScore(softmaxProbs: number[], trueClass: number): number {
  const trueProb = softmaxProbs[trueClass];

  // Sum probabilities of classes at least as likely as true class
  let score = 0;
  for (let i = 0; i < softmaxProbs.length; i++) {
    if (softmaxProbs[i] >= trueProb) {
      score += softmaxProbs[i];
    }
  }

  // Subtract random tie-breaking
  const u = Math.random();
  score -= u * trueProb;

  return score;
}

/**
 * Regularized Adaptive Prediction Sets (RAPS) score
 *
 * s(x, y) = Σ_{j: π_j ≥ π_y} π_j + λ·(o(y) - k_reg)⁺
 *
 * where o(y) is the rank of the true class
 */
export function rapsScore(
  softmaxProbs: number[],
  trueClass: number,
  lambda: number = 0.1,
  kReg: number = 5
): number {
  const trueProb = softmaxProbs[trueClass];

  // Get sorted indices by probability (descending)
  const indices = softmaxProbs.map((_, i) => i);
  indices.sort((a, b) => softmaxProbs[b] - softmaxProbs[a]);

  // Find rank of true class
  let rank = 0;
  let cumulativeProb = 0;
  for (let i = 0; i < indices.length; i++) {
    cumulativeProb += softmaxProbs[indices[i]];
    if (indices[i] === trueClass) {
      rank = i + 1;
      break;
    }
  }

  // Base APS score (cumulative probability up to true class)
  const u = Math.random();
  const baseScore = cumulativeProb - u * trueProb;

  // Regularization term
  const regularization = lambda * Math.max(0, rank - kReg);

  return baseScore + regularization;
}

// ============================================================================
// Calibration
// ============================================================================

/**
 * Compute conformity scores for calibration set
 */
export function computeConformityScores(
  predictions: number[][],
  trueLabels: number[],
  config: ConformalConfig
): number[] {
  const scores: number[] = [];

  for (let i = 0; i < predictions.length; i++) {
    const probs = predictions[i];
    const label = trueLabels[i];

    let score: number;
    switch (config.scoreType) {
      case 'aps':
        score = apsScore(probs, label);
        break;
      case 'raps':
        score = rapsScore(probs, label, config.rapsLambda ?? 0.1, config.rapsKreg ?? 5);
        break;
      case 'softmax':
      default:
        score = softmaxScore(probs, label);
    }

    scores.push(score);
  }

  return scores;
}

/**
 * Calibrate conformal predictor on a held-out calibration set
 *
 * Computes the (1-α)(1+1/n) quantile of conformity scores
 */
export function calibrate(
  predictions: number[][],
  trueLabels: number[],
  config: ConformalConfig
): CalibrationResult {
  const n = predictions.length;

  if (n === 0) {
    return {
      threshold: 1.0,
      empiricalCoverage: 0,
      averageSetSize: 0,
      conformityScores: [],
      calibrationSize: 0
    };
  }

  // Compute conformity scores
  const scores = computeConformityScores(predictions, trueLabels, config);

  // Sort scores
  const sortedScores = [...scores].sort((a, b) => a - b);

  // Compute quantile level: ⌈(n+1)(1-α)⌉ / n
  const alpha = 1 - config.coverageLevel;
  const quantileLevel = Math.ceil((n + 1) * (1 - alpha)) / n;
  const quantileIndex = Math.min(Math.floor(quantileLevel * n), n - 1);

  const threshold = sortedScores[quantileIndex];

  // Compute empirical coverage with this threshold
  let covered = 0;
  let totalSetSize = 0;

  for (let i = 0; i < n; i++) {
    const predSet = makePredictionSet(predictions[i], threshold, config);
    if (predSet.classes.includes(trueLabels[i])) {
      covered++;
    }
    totalSetSize += predSet.size;
  }

  return {
    threshold,
    empiricalCoverage: covered / n,
    averageSetSize: totalSetSize / n,
    conformityScores: scores,
    calibrationSize: n
  };
}

// ============================================================================
// Prediction Set Construction
// ============================================================================

/**
 * Construct prediction set for a single sample
 */
export function makePredictionSet(
  softmaxProbs: number[],
  threshold: number,
  config: ConformalConfig
): PredictionSet {
  const numClasses = softmaxProbs.length;

  // Get sorted indices by probability (descending)
  const indices = softmaxProbs.map((_, i) => i);
  indices.sort((a, b) => softmaxProbs[b] - softmaxProbs[a]);

  const topClass = indices[0];
  const topConfidence = softmaxProbs[topClass];

  // Build prediction set based on score type
  const includedClasses: number[] = [];
  const includedScores: number[] = [];

  switch (config.scoreType) {
    case 'softmax': {
      // Include classes with score ≤ threshold
      // s(y) = 1 - π_y, so include if π_y ≥ 1 - threshold
      const probThreshold = 1 - threshold;
      for (let i = 0; i < numClasses; i++) {
        if (softmaxProbs[i] >= probThreshold) {
          includedClasses.push(i);
          includedScores.push(1 - softmaxProbs[i]);
        }
      }
      break;
    }

    case 'aps':
    case 'raps': {
      // Include classes in order until cumulative probability exceeds threshold
      let cumProb = 0;
      for (const idx of indices) {
        cumProb += softmaxProbs[idx];
        includedClasses.push(idx);

        // For RAPS, add regularization penalty
        let score = cumProb;
        if (config.scoreType === 'raps') {
          const rank = includedClasses.length;
          const lambda = config.rapsLambda ?? 0.1;
          const kReg = config.rapsKreg ?? 5;
          score += lambda * Math.max(0, rank - kReg);
        }
        includedScores.push(score);

        if (score > threshold) {
          break;
        }
      }
      break;
    }
  }

  // Ensure at least one class is included
  if (includedClasses.length === 0) {
    includedClasses.push(topClass);
    includedScores.push(1 - topConfidence);
  }

  return {
    classes: includedClasses,
    scores: includedScores,
    threshold,
    size: includedClasses.length,
    topClass,
    topConfidence
  };
}

/**
 * Make prediction sets for a batch of samples
 */
export function makePredictionSetsBatch(
  predictions: number[][],
  threshold: number,
  config: ConformalConfig
): PredictionSet[] {
  return predictions.map((probs) => makePredictionSet(probs, threshold, config));
}

// ============================================================================
// Coverage Diagnostics
// ============================================================================

/**
 * Evaluate prediction sets on test data
 */
export function evaluateCoverage(
  predictions: number[][],
  trueLabels: number[],
  threshold: number,
  config: ConformalConfig
): CoverageDiagnostics {
  const n = predictions.length;

  if (n === 0) {
    return {
      coverage: 0,
      averageSetSize: 0,
      medianSetSize: 0,
      singletonRate: 0,
      emptyRate: 0,
      isValid: false,
      message: 'No test samples provided'
    };
  }

  let covered = 0;
  let singletons = 0;
  let empty = 0;
  const setSizes: number[] = [];

  for (let i = 0; i < n; i++) {
    const predSet = makePredictionSet(predictions[i], threshold, config);
    setSizes.push(predSet.size);

    if (predSet.classes.includes(trueLabels[i])) {
      covered++;
    }

    if (predSet.size === 0) {
      empty++;
    } else if (predSet.size === 1) {
      singletons++;
    }
  }

  const coverage = covered / n;
  const averageSetSize = setSizes.reduce((a, b) => a + b, 0) / n;

  // Compute median set size
  const sortedSizes = [...setSizes].sort((a, b) => a - b);
  const medianSetSize = sortedSizes[Math.floor(n / 2)];

  const singletonRate = singletons / n;
  const emptyRate = empty / n;

  // Check if coverage is valid (within 5% of target)
  const targetCoverage = config.coverageLevel;
  const tolerance = 0.05;
  const isValid = coverage >= targetCoverage - tolerance;

  let message = '';
  if (coverage < targetCoverage - tolerance) {
    message = `Coverage ${(coverage * 100).toFixed(1)}% below target ${(targetCoverage * 100).toFixed(1)}%`;
  } else if (coverage > targetCoverage + tolerance) {
    message = `Coverage ${(coverage * 100).toFixed(1)}% above target (conservative)`;
  } else {
    message = `Coverage ${(coverage * 100).toFixed(1)}% meets target ${(targetCoverage * 100).toFixed(1)}%`;
  }

  return {
    coverage,
    averageSetSize,
    medianSetSize,
    singletonRate,
    emptyRate,
    isValid,
    message
  };
}

// ============================================================================
// Adaptive Conformal Inference
// ============================================================================

export interface AdaptiveConformalState {
  /** Current threshold */
  threshold: number;
  /** Running coverage estimate */
  coverageEstimate: number;
  /** Number of samples seen */
  numSamples: number;
  /** Adaptation rate */
  gamma: number;
}

/**
 * Initialize adaptive conformal prediction state
 */
export function initAdaptiveConformal(
  initialThreshold: number,
  gamma: number = 0.01
): AdaptiveConformalState {
  return {
    threshold: initialThreshold,
    coverageEstimate: 0.9,
    numSamples: 0,
    gamma
  };
}

/**
 * Update adaptive conformal predictor after observing true label
 *
 * Adjusts threshold to maintain target coverage over time
 */
export function updateAdaptiveConformal(
  state: AdaptiveConformalState,
  prediction: number[],
  trueLabel: number,
  targetCoverage: number,
  config: ConformalConfig
): AdaptiveConformalState {
  const predSet = makePredictionSet(prediction, state.threshold, config);
  const covered = predSet.classes.includes(trueLabel) ? 1 : 0;

  // Update running coverage estimate
  const newCoverageEstimate =
    (1 - state.gamma) * state.coverageEstimate + state.gamma * covered;

  // Adjust threshold to maintain target coverage
  // If coverage too low, increase threshold (larger sets)
  // If coverage too high, decrease threshold (smaller sets)
  const alpha = 1 - targetCoverage;
  const coverageError = newCoverageEstimate - targetCoverage;

  // Learning rate for threshold adjustment
  const lr = state.gamma;
  let newThreshold = state.threshold - lr * coverageError;

  // Clamp threshold to valid range
  newThreshold = Math.max(0.01, Math.min(0.99, newThreshold));

  return {
    threshold: newThreshold,
    coverageEstimate: newCoverageEstimate,
    numSamples: state.numSamples + 1,
    gamma: state.gamma
  };
}

// ============================================================================
// Conditional Coverage Analysis
// ============================================================================

export interface ConditionalCoverage {
  /** Coverage per class */
  perClassCoverage: Map<number, number>;
  /** Coverage per confidence bin */
  perConfidenceCoverage: Map<string, number>;
  /** Worst-case coverage across classes */
  worstClassCoverage: number;
  /** Class with worst coverage */
  worstClass: number;
  /** Conditional coverage is balanced */
  isBalanced: boolean;
}

/**
 * Analyze conditional coverage (coverage conditioned on class or confidence)
 */
export function analyzeConditionalCoverage(
  predictions: number[][],
  trueLabels: number[],
  threshold: number,
  config: ConformalConfig
): ConditionalCoverage {
  const n = predictions.length;

  // Per-class coverage
  const classCounts = new Map<number, number>();
  const classCovered = new Map<number, number>();

  // Per-confidence-bin coverage
  const confBins = ['0-20', '20-40', '40-60', '60-80', '80-100'];
  const confCounts = new Map<string, number>();
  const confCovered = new Map<string, number>();

  for (const bin of confBins) {
    confCounts.set(bin, 0);
    confCovered.set(bin, 0);
  }

  for (let i = 0; i < n; i++) {
    const predSet = makePredictionSet(predictions[i], threshold, config);
    const covered = predSet.classes.includes(trueLabels[i]) ? 1 : 0;
    const label = trueLabels[i];
    const conf = predSet.topConfidence * 100;

    // Update per-class stats
    classCounts.set(label, (classCounts.get(label) ?? 0) + 1);
    classCovered.set(label, (classCovered.get(label) ?? 0) + covered);

    // Update per-confidence stats
    let bin: string;
    if (conf < 20) bin = '0-20';
    else if (conf < 40) bin = '20-40';
    else if (conf < 60) bin = '40-60';
    else if (conf < 80) bin = '60-80';
    else bin = '80-100';

    confCounts.set(bin, (confCounts.get(bin) ?? 0) + 1);
    confCovered.set(bin, (confCovered.get(bin) ?? 0) + covered);
  }

  // Compute per-class coverage
  const perClassCoverage = new Map<number, number>();
  let worstClassCoverage = 1;
  let worstClass = 0;

  for (const [classId, count] of classCounts) {
    const coverage = (classCovered.get(classId) ?? 0) / count;
    perClassCoverage.set(classId, coverage);

    if (coverage < worstClassCoverage) {
      worstClassCoverage = coverage;
      worstClass = classId;
    }
  }

  // Compute per-confidence coverage
  const perConfidenceCoverage = new Map<string, number>();
  for (const bin of confBins) {
    const count = confCounts.get(bin) ?? 0;
    if (count > 0) {
      perConfidenceCoverage.set(bin, (confCovered.get(bin) ?? 0) / count);
    }
  }

  // Check if balanced (all classes within 10% of target)
  const targetCoverage = config.coverageLevel;
  const isBalanced = worstClassCoverage >= targetCoverage - 0.1;

  return {
    perClassCoverage,
    perConfidenceCoverage,
    worstClassCoverage,
    worstClass,
    isBalanced
  };
}

// ============================================================================
// Utility: Prediction Set Formatting
// ============================================================================

/**
 * Format prediction set for display
 */
export function formatPredictionSet(
  predSet: PredictionSet,
  classNames?: string[]
): string {
  const names = predSet.classes.map((c) =>
    classNames ? classNames[c] : `Class ${c}`
  );

  if (names.length === 1) {
    return `{${names[0]}} (conf: ${(predSet.topConfidence * 100).toFixed(1)}%)`;
  }

  return `{${names.join(', ')}} (size: ${predSet.size})`;
}

// ============================================================================
// Defaults
// ============================================================================

export const CONFORMAL_DEFAULTS: ConformalConfig = {
  coverageLevel: 0.9,
  scoreType: 'aps',
  rapsLambda: 0.1,
  rapsKreg: 5,
  seed: 42
};

export const CONFORMAL_CONSERVATIVE: ConformalConfig = {
  coverageLevel: 0.95,
  scoreType: 'raps',
  rapsLambda: 0.2,
  rapsKreg: 3,
  seed: 42
};
