/**
 * Mathematical utilities for Causal Inference
 *
 * This module provides the mathematical foundations for the probabilistic
 * dynamic causal inference system, including:
 * - Propensity score estimation
 * - AIPW (Augmented Inverse Propensity Weighting) estimator
 * - Quantization and dequantization functions
 * - Statistical utilities for causal analysis
 *
 * Mathematical References:
 * - Rosenbaum & Rubin (1983) - Propensity scores
 * - Robins, Rotnitzky & Zhao (1994) - AIPW estimator
 * - Tishby et al. (1999) - Information bottleneck (for quantization)
 *
 * @module causal_math
 */

import type {
  QuantizationParams,
  DequantizationMapping,
  AverageTreatmentEffect,
  AIPWComponents,
  BinIndex
} from '../types/causal';
import {
  assertNormalizedProbabilities,
  assertSoftmaxInputBounds,
  logSumExp as stableLogSumExp
} from '../lib/MathUtils';

// ============================================================================
// Type Aliases
// ============================================================================

type Vector = number[];
type Matrix = number[][];

// ============================================================================
// Constants
// ============================================================================

/** Small value for numerical stability */
const EPSILON = 1e-10;

/** Maximum iterations for iterative algorithms */
const MAX_ITERATIONS = 1000;

/** Convergence tolerance */
const CONVERGENCE_TOL = 1e-6;

/** Propensity score clipping bounds */
const PROPENSITY_MIN = 0.01;
const PROPENSITY_MAX = 0.99;

// ============================================================================
// Basic Statistical Functions
// ============================================================================

/**
 * Sigmoid (logistic) function
 * σ(x) = 1 / (1 + exp(-x))
 *
 * @param x - Input value
 * @returns Sigmoid output in (0, 1)
 */
export function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  } else {
    const expX = Math.exp(x);
    return expX / (1 + expX);
  }
}

/**
 * Log-sum-exp trick for numerical stability
 * log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
 *
 * @param values - Input values
 * @returns log-sum-exp result
 */
export function logSumExp(values: Vector): number {
  return stableLogSumExp(values);
}

/**
 * Stable softmax function
 *
 * @param logits - Input logits
 * @returns Probability distribution
 */
export function stableSoftmax(logits: Vector, temperature = 1.0): Vector {
  if (logits.length === 0) return [];

  const T = Math.max(temperature, EPSILON);
  assertSoftmaxInputBounds(logits, T);

  const scaled = logits.map((x) => x / T);
  const normalization = logSumExp(scaled);
  const probabilities = scaled.map((x) => Math.exp(x - normalization));
  assertNormalizedProbabilities(probabilities);
  return probabilities;
}

/**
 * Compute mean of a vector
 *
 * @param values - Input values
 * @returns Mean
 */
export function mean(values: Vector): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Compute variance of a vector
 *
 * @param values - Input values
 * @param ddof - Delta degrees of freedom (default 1 for sample variance)
 * @returns Variance
 */
export function variance(values: Vector, ddof: number = 1): number {
  if (values.length <= ddof) return 0;
  const m = mean(values);
  const sumSq = values.reduce((sum, x) => sum + (x - m) ** 2, 0);
  return sumSq / (values.length - ddof);
}

/**
 * Compute standard deviation
 *
 * @param values - Input values
 * @param ddof - Delta degrees of freedom
 * @returns Standard deviation
 */
export function stdDev(values: Vector, ddof: number = 1): number {
  return Math.sqrt(variance(values, ddof));
}

/**
 * Compute covariance between two vectors
 *
 * @param x - First vector
 * @param y - Second vector
 * @returns Covariance
 */
export function covariance(x: Vector, y: Vector): number {
  if (x.length !== y.length || x.length < 2) return 0;
  const mx = mean(x);
  const my = mean(y);
  let cov = 0;
  for (let i = 0; i < x.length; i++) {
    cov += (x[i] - mx) * (y[i] - my);
  }
  return cov / (x.length - 1);
}

// ============================================================================
// Propensity Score Estimation
// ============================================================================

/**
 * Estimate propensity scores using logistic regression
 * P(Z=B|X) = σ(β^T X)
 *
 * Uses iterative reweighted least squares (IRLS) for optimization.
 *
 * @param features - Feature matrix (n x p)
 * @param treatments - Treatment indicators (n), 1 for B, 0 for A
 * @param options - Estimation options
 * @returns Estimated coefficients and propensity scores
 */
export function estimatePropensityScores(
  features: Matrix,
  treatments: Vector,
  options: {
    maxIterations?: number;
    tolerance?: number;
    regularization?: number;
  } = {}
): {
  coefficients: Vector;
  propensities: Vector;
  converged: boolean;
  iterations: number;
} {
  const {
    maxIterations = MAX_ITERATIONS,
    tolerance = CONVERGENCE_TOL,
    regularization = 0.01
  } = options;

  const n = features.length;
  const p = features[0]?.length || 0;

  if (n === 0 || p === 0) {
    return {
      coefficients: [],
      propensities: [],
      converged: false,
      iterations: 0
    };
  }

  // Add intercept column
  const X = features.map((row) => [1, ...row]);
  const pWithIntercept = p + 1;

  // Initialize coefficients
  let beta = new Array(pWithIntercept).fill(0);
  let prevLogLikelihood = -Infinity;
  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // Compute probabilities
    const probs = X.map((xi) => {
      const linearPred = dotProduct(xi, beta);
      return sigmoid(linearPred);
    });

    // Compute gradient and Hessian
    const gradient = new Array(pWithIntercept).fill(0);
    const hessian: Matrix = Array.from({ length: pWithIntercept }, () =>
      new Array(pWithIntercept).fill(0)
    );

    for (let i = 0; i < n; i++) {
      const pi = probs[i];
      const yi = treatments[i];
      const wi = pi * (1 - pi) + EPSILON;

      for (let j = 0; j < pWithIntercept; j++) {
        gradient[j] += (yi - pi) * X[i][j];

        for (let k = 0; k < pWithIntercept; k++) {
          hessian[j][k] -= wi * X[i][j] * X[i][k];
        }
      }
    }

    // Add L2 regularization
    for (let j = 1; j < pWithIntercept; j++) {
      gradient[j] -= regularization * beta[j];
      hessian[j][j] -= regularization;
    }

    // Newton-Raphson update: β_new = β - H^{-1} g
    const hessianInv = invertMatrix(hessian);
    if (!hessianInv) {
      // Fallback to gradient descent if Hessian is singular
      const stepSize = 0.01;
      beta = beta.map((b, j) => b + stepSize * gradient[j]);
    } else {
      const update = matrixVectorMultiply(hessianInv, gradient);
      beta = beta.map((b, j) => b - update[j]);
    }

    // Check convergence via log-likelihood
    const logLikelihood = computeLogLikelihood(probs, treatments, regularization, beta);

    if (Math.abs(logLikelihood - prevLogLikelihood) < tolerance) {
      converged = true;
      break;
    }
    prevLogLikelihood = logLikelihood;
  }

  // Compute final propensity scores with clipping
  const propensities = X.map((xi) => {
    const p = sigmoid(dotProduct(xi, beta));
    return Math.max(PROPENSITY_MIN, Math.min(PROPENSITY_MAX, p));
  });

  return {
    coefficients: beta,
    propensities,
    converged,
    iterations
  };
}

/**
 * Compute log-likelihood for logistic regression
 */
function computeLogLikelihood(
  probs: Vector,
  targets: Vector,
  regularization: number,
  beta: Vector
): number {
  let ll = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = Math.max(EPSILON, Math.min(1 - EPSILON, probs[i]));
    ll += targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p);
  }
  // L2 penalty (skip intercept)
  for (let j = 1; j < beta.length; j++) {
    ll -= (regularization / 2) * beta[j] ** 2;
  }
  return ll;
}

/**
 * Predict propensity scores for new data using learned coefficients
 *
 * @param features - New feature matrix
 * @param coefficients - Learned coefficients (including intercept)
 * @returns Propensity scores
 */
export function predictPropensity(features: Matrix, coefficients: Vector): Vector {
  return features.map((row) => {
    const withIntercept = [1, ...row];
    const linearPred = dotProduct(withIntercept, coefficients);
    const p = sigmoid(linearPred);
    return Math.max(PROPENSITY_MIN, Math.min(PROPENSITY_MAX, p));
  });
}

// ============================================================================
// AIPW Estimator
// ============================================================================

/**
 * Compute AIPW (Augmented Inverse Propensity Weighted) estimate of ATE
 *
 * τ̂_AIPW = (1/n) Σ [ Z(Y - μ̂_B)/π̂ + μ̂_B - ((1-Z)(Y - μ̂_A)/(1-π̂) + μ̂_A) ]
 *
 * This is a doubly robust estimator: consistent if either the propensity
 * model OR the outcome model is correctly specified.
 *
 * Reference: Robins, Rotnitzky & Zhao (1994)
 *
 * @param outcomes - Observed outcomes Y
 * @param treatments - Treatment indicators Z (1=B, 0=A)
 * @param propensities - Propensity scores P(Z=1|X)
 * @param muA - Outcome predictions under A: E[Y|Z=A,X]
 * @param muB - Outcome predictions under B: E[Y|Z=B,X]
 * @returns AIPW estimate with components
 */
export function computeAIPWEstimate(
  outcomes: Vector,
  treatments: Vector,
  propensities: Vector,
  muA: Vector,
  muB: Vector
): {
  ate: AverageTreatmentEffect;
  components: AIPWComponents[];
} {
  const n = outcomes.length;

  if (n === 0) {
    return {
      ate: {
        estimate: 0,
        standardError: 0,
        confidenceInterval: [0, 0],
        numObservations: 0,
        method: 'aipw'
      },
      components: []
    };
  }

  const components: AIPWComponents[] = [];
  const psiValues: Vector = []; // Influence function values for variance

  for (let i = 0; i < n; i++) {
    const Y = outcomes[i];
    const Z = treatments[i];
    const pi = propensities[i];
    const muA_i = muA[i];
    const muB_i = muB[i];

    // IPW terms
    const ipwB = Z === 1 ? (Y - muB_i) / pi : 0;
    const ipwA = Z === 0 ? (Y - muA_i) / (1 - pi) : 0;

    // AIPW components
    const termB = ipwB + muB_i;
    const termA = ipwA + muA_i;

    const psi = termB - termA; // Influence function value
    psiValues.push(psi);

    components.push({
      ipwTerm: Z === 1 ? ipwB : -ipwA,
      augmentationTerm: muB_i - muA_i,
      combined: psi,
      influenceFunction: psi
    });
  }

  // Point estimate
  const estimate = mean(psiValues);

  // Variance estimate using influence function
  const varianceEstimate = variance(psiValues) / n;
  const standardError = Math.sqrt(varianceEstimate);

  // 95% confidence interval
  const z95 = 1.96;
  const confidenceInterval: [number, number] = [
    estimate - z95 * standardError,
    estimate + z95 * standardError
  ];

  return {
    ate: {
      estimate,
      standardError,
      confidenceInterval,
      numObservations: n,
      method: 'aipw'
    },
    components
  };
}

/**
 * Compute IPW (Inverse Propensity Weighted) estimate of ATE
 * Simpler but less efficient than AIPW
 *
 * @param outcomes - Observed outcomes
 * @param treatments - Treatment indicators
 * @param propensities - Propensity scores
 * @returns IPW estimate
 */
export function computeIPWEstimate(
  outcomes: Vector,
  treatments: Vector,
  propensities: Vector
): AverageTreatmentEffect {
  const n = outcomes.length;
  if (n === 0) {
    return {
      estimate: 0,
      standardError: 0,
      confidenceInterval: [0, 0],
      numObservations: 0,
      method: 'ipw'
    };
  }

  let sumB = 0,
    sumA = 0;
  let weightSumB = 0,
    weightSumA = 0;
  const psiValues: Vector = [];

  for (let i = 0; i < n; i++) {
    const Y = outcomes[i];
    const Z = treatments[i];
    const pi = propensities[i];

    if (Z === 1) {
      sumB += Y / pi;
      weightSumB += 1 / pi;
    } else {
      sumA += Y / (1 - pi);
      weightSumA += 1 / (1 - pi);
    }

    // Normalized IPW
    const psi = Z === 1 ? Y / pi : -Y / (1 - pi);
    psiValues.push(psi);
  }

  // Normalized estimate
  const muB = weightSumB > 0 ? sumB / weightSumB : 0;
  const muA = weightSumA > 0 ? sumA / weightSumA : 0;
  const estimate = muB - muA;

  // Variance (simplified)
  const varianceEstimate = variance(psiValues) / n;
  const standardError = Math.sqrt(varianceEstimate);

  const z95 = 1.96;
  const confidenceInterval: [number, number] = [
    estimate - z95 * standardError,
    estimate + z95 * standardError
  ];

  return {
    estimate,
    standardError,
    confidenceInterval,
    numObservations: n,
    method: 'ipw'
  };
}

// ============================================================================
// Outcome Model Estimation
// ============================================================================

/**
 * Fit linear outcome model via OLS
 * Y = β_0 + β_Z Z + β_X^T X + ε
 *
 * @param outcomes - Observed outcomes
 * @param treatments - Treatment indicators
 * @param features - Feature matrix
 * @returns Fitted model parameters
 */
export function fitOutcomeModel(
  outcomes: Vector,
  treatments: Vector,
  features: Matrix
): {
  intercept: number;
  treatmentEffect: number;
  featureCoefficients: Vector;
  residualVariance: number;
} {
  const n = outcomes.length;
  const p = features[0]?.length || 0;

  // Build design matrix [1, Z, X]
  const X: Matrix = [];
  for (let i = 0; i < n; i++) {
    X.push([1, treatments[i], ...features[i]]);
  }

  // Solve via normal equations: β = (X^T X)^{-1} X^T y
  const XtX = matrixMultiply(transpose(X), X);
  const Xty = matrixVectorMultiply(transpose(X), outcomes);

  // Add small regularization for stability
  for (let i = 0; i < XtX.length; i++) {
    XtX[i][i] += 1e-6;
  }

  const XtXInv = invertMatrix(XtX);
  if (!XtXInv) {
    return {
      intercept: mean(outcomes),
      treatmentEffect: 0,
      featureCoefficients: new Array(p).fill(0),
      residualVariance: variance(outcomes)
    };
  }

  const beta = matrixVectorMultiply(XtXInv, Xty);

  // Compute residual variance
  const predictions = X.map((xi) => dotProduct(xi, beta));
  const residuals = outcomes.map((y, i) => y - predictions[i]);
  const residualVariance = variance(residuals);

  return {
    intercept: beta[0],
    treatmentEffect: beta[1],
    featureCoefficients: beta.slice(2),
    residualVariance
  };
}

/**
 * Predict outcomes for given treatment assignment
 *
 * @param features - Feature matrix
 * @param treatment - Treatment value to predict under
 * @param model - Fitted model parameters
 * @returns Predicted outcomes
 */
export function predictOutcome(
  features: Matrix,
  treatment: 0 | 1,
  model: {
    intercept: number;
    treatmentEffect: number;
    featureCoefficients: Vector;
  }
): Vector {
  return features.map((row) => {
    let pred = model.intercept + model.treatmentEffect * treatment;
    for (let j = 0; j < row.length; j++) {
      pred += model.featureCoefficients[j] * row[j];
    }
    return pred;
  });
}

// ============================================================================
// Quantization Functions
// ============================================================================

/**
 * Create uniform quantization parameters
 *
 * @param numBins - Number of bins K
 * @param range - Data range [min, max]
 * @param symmetric - Whether to make symmetric around zero
 * @returns Quantization parameters
 */
export function createUniformQuantization(
  numBins: number,
  range: [number, number],
  symmetric: boolean = true
): Omit<QuantizationParams, 'timeStep'> {
  const [minVal, maxVal] = range;
  const boundaries: number[] = [];

  if (symmetric) {
    // Symmetric bins around zero
    const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal));
    const binWidth = (2 * absMax) / numBins;

    for (let i = 1; i < numBins; i++) {
      boundaries.push(-absMax + i * binWidth);
    }
  } else {
    // Uniform bins
    const binWidth = (maxVal - minVal) / numBins;
    for (let i = 1; i < numBins; i++) {
      boundaries.push(minVal + i * binWidth);
    }
  }

  return {
    numBins,
    boundaries,
    method: 'uniform',
    symmetric
  };
}

/**
 * Create entropy-based quantization parameters
 * Optimizes bin boundaries to maximize mutual information
 *
 * @param values - Sample values for optimization
 * @param numBins - Number of bins K
 * @returns Quantization parameters
 */
export function createEntropyQuantization(
  values: Vector,
  numBins: number
): Omit<QuantizationParams, 'timeStep'> {
  if (values.length === 0) {
    return createUniformQuantization(numBins, [-1, 1], true);
  }

  // Sort values and find quantiles
  const sorted = [...values].sort((a, b) => a - b);
  const boundaries: number[] = [];

  for (let i = 1; i < numBins; i++) {
    const idx = Math.floor((i / numBins) * sorted.length);
    boundaries.push(sorted[Math.min(idx, sorted.length - 1)]);
  }

  // Check symmetry
  const symmetric =
    boundaries.length >= 2 &&
    Math.abs(boundaries[0] + boundaries[boundaries.length - 1]) <
      0.01 * (boundaries[boundaries.length - 1] - boundaries[0]);

  return {
    numBins,
    boundaries,
    method: 'entropy',
    symmetric
  };
}

/**
 * Quantize a continuous value
 * Q_θ(Y) = k where τ_{k-1} < Y ≤ τ_k
 *
 * @param value - Continuous value to quantize
 * @param params - Quantization parameters
 * @returns Bin index
 */
export function quantize(
  value: number,
  params: QuantizationParams | Omit<QuantizationParams, 'timeStep'>
): BinIndex {
  const { boundaries, numBins } = params;

  for (let k = 0; k < boundaries.length; k++) {
    if (value <= boundaries[k]) {
      return k;
    }
  }

  return numBins - 1;
}

/**
 * Batch quantize multiple values
 *
 * @param values - Values to quantize
 * @param params - Quantization parameters
 * @returns Bin indices
 */
export function batchQuantize(
  values: Vector,
  params: QuantizationParams | Omit<QuantizationParams, 'timeStep'>
): BinIndex[] {
  return values.map((v) => quantize(v, params));
}

/**
 * Learn dequantization mappings from historical data
 * φ_k(θ) = E[Y | Ỹ=k, θ]
 *
 * @param continuousValues - Original continuous values
 * @param quantizedValues - Corresponding quantized values
 * @param numBins - Number of bins
 * @returns Dequantization mappings for each bin
 */
export function learnDequantizationMappings(
  continuousValues: Vector,
  quantizedValues: BinIndex[],
  numBins: number
): DequantizationMapping[] {
  const mappings: DequantizationMapping[] = [];

  for (let k = 0; k < numBins; k++) {
    const binValues: number[] = [];

    for (let i = 0; i < continuousValues.length; i++) {
      if (quantizedValues[i] === k) {
        binValues.push(continuousValues[i]);
      }
    }

    const expectedValue = binValues.length > 0 ? mean(binValues) : 0;
    const binVariance = binValues.length > 1 ? variance(binValues) : 0;

    mappings.push({
      binIndex: k,
      expectedValue,
      variance: binVariance,
      sampleCount: binValues.length
    });
  }

  return mappings;
}

/**
 * Dequantize a binned value using learned mappings
 *
 * @param binIndex - Quantized bin index
 * @param mappings - Dequantization mappings
 * @returns Expected continuous value
 */
export function dequantize(binIndex: BinIndex, mappings: DequantizationMapping[]): number {
  const mapping = mappings.find((m) => m.binIndex === binIndex);
  return mapping ? mapping.expectedValue : 0;
}

/**
 * Batch dequantize multiple values
 *
 * @param binIndices - Quantized values
 * @param mappings - Dequantization mappings
 * @returns Dequantized values
 */
export function batchDequantize(binIndices: BinIndex[], mappings: DequantizationMapping[]): Vector {
  return binIndices.map((k) => dequantize(k, mappings));
}

// ============================================================================
// Adaptive Quantization
// ============================================================================

/**
 * Update quantization parameters adaptively based on observed data
 * θ_t = θ_{t-1} + α ∇_θ L(θ)
 *
 * Objective: minimize expected quantization error on effect-relevant regions
 *
 * @param currentParams - Current quantization parameters
 * @param observations - Recent observations
 * @param learningRate - Update step size
 * @returns Updated quantization parameters
 */
export function adaptQuantization(
  currentParams: QuantizationParams,
  observations: { continuous: number; quantized: BinIndex }[],
  learningRate: number = 0.1
): QuantizationParams {
  if (observations.length < 10) {
    return currentParams;
  }

  const { boundaries, numBins, symmetric } = currentParams;
  const newBoundaries = [...boundaries];

  // Compute bin-wise quantization errors
  const binErrors: number[][] = Array.from({ length: numBins }, () => []);

  for (const obs of observations) {
    const k = obs.quantized;
    const midpoint =
      k === 0
        ? boundaries[0] - (boundaries[1] - boundaries[0]) / 2
        : k === numBins - 1
          ? boundaries[numBins - 2] + (boundaries[numBins - 2] - (boundaries[numBins - 3] || 0)) / 2
          : (boundaries[k - 1] + boundaries[k]) / 2;
    binErrors[k].push(obs.continuous - midpoint);
  }

  // Adjust boundaries based on mean errors
  for (let k = 0; k < boundaries.length; k++) {
    const leftErrors = binErrors[k] || [];
    const rightErrors = binErrors[k + 1] || [];

    if (leftErrors.length > 0 || rightErrors.length > 0) {
      const leftMeanError = leftErrors.length > 0 ? mean(leftErrors) : 0;
      const rightMeanError = rightErrors.length > 0 ? mean(rightErrors) : 0;

      // Shift boundary towards reducing error
      const gradient = rightMeanError - leftMeanError;
      newBoundaries[k] += learningRate * gradient;
    }
  }

  // Enforce monotonicity
  for (let k = 1; k < newBoundaries.length; k++) {
    if (newBoundaries[k] <= newBoundaries[k - 1]) {
      newBoundaries[k] = newBoundaries[k - 1] + EPSILON;
    }
  }

  // Enforce symmetry if required
  if (symmetric && newBoundaries.length >= 2) {
    const mid = Math.floor(newBoundaries.length / 2);
    for (let i = 0; i < mid; i++) {
      const j = newBoundaries.length - 1 - i;
      const avg = (Math.abs(newBoundaries[i]) + Math.abs(newBoundaries[j])) / 2;
      newBoundaries[i] = -avg;
      newBoundaries[j] = avg;
    }
  }

  return {
    ...currentParams,
    boundaries: newBoundaries,
    method: 'adaptive'
  };
}

// ============================================================================
// Statistical Testing
// ============================================================================

/**
 * Perform two-sided t-test for ATE
 * H_0: τ = 0 vs H_1: τ ≠ 0
 *
 * @param ate - ATE estimate
 * @param significanceLevel - Significance level (default 0.05)
 * @returns Test result
 */
export function testATESignificance(
  ate: AverageTreatmentEffect,
  significanceLevel: number = 0.05
): {
  testStatistic: number;
  pValue: number;
  reject: boolean;
} {
  const { estimate, standardError, numObservations } = ate;

  if (standardError === 0 || numObservations < 2) {
    return {
      testStatistic: 0,
      pValue: 1,
      reject: false
    };
  }

  const testStatistic = estimate / standardError;

  // Approximate p-value using normal distribution for large n
  const pValue = 2 * (1 - normalCDF(Math.abs(testStatistic)));

  const criticalValue = normalQuantile(1 - significanceLevel / 2);
  const reject = Math.abs(testStatistic) > criticalValue;

  return {
    testStatistic,
    pValue,
    reject
  };
}

/**
 * Bootstrap confidence interval for ATE
 *
 * @param outcomes - Outcomes
 * @param treatments - Treatments
 * @param propensities - Propensity scores
 * @param muA - Outcome predictions under A
 * @param muB - Outcome predictions under B
 * @param nBootstrap - Number of bootstrap samples
 * @param confidenceLevel - Confidence level
 * @returns Bootstrap confidence interval
 */
export function bootstrapATEConfidenceInterval(
  outcomes: Vector,
  treatments: Vector,
  propensities: Vector,
  muA: Vector,
  muB: Vector,
  nBootstrap: number = 1000,
  confidenceLevel: number = 0.95
): [number, number] {
  const n = outcomes.length;
  const bootstrapEstimates: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    // Sample with replacement
    const indices = Array.from({ length: n }, () => Math.floor(Math.random() * n));

    const bootOutcomes = indices.map((i) => outcomes[i]);
    const bootTreatments = indices.map((i) => treatments[i]);
    const bootPropensities = indices.map((i) => propensities[i]);
    const bootMuA = indices.map((i) => muA[i]);
    const bootMuB = indices.map((i) => muB[i]);

    const { ate } = computeAIPWEstimate(
      bootOutcomes,
      bootTreatments,
      bootPropensities,
      bootMuA,
      bootMuB
    );

    bootstrapEstimates.push(ate.estimate);
  }

  // Sort and find percentiles
  bootstrapEstimates.sort((a, b) => a - b);
  const alpha = 1 - confidenceLevel;
  const lowerIdx = Math.floor((alpha / 2) * nBootstrap);
  const upperIdx = Math.floor((1 - alpha / 2) * nBootstrap);

  return [bootstrapEstimates[lowerIdx], bootstrapEstimates[upperIdx]];
}

// ============================================================================
// Power Analysis
// ============================================================================

/**
 * Estimate statistical power for detecting a given effect size
 *
 * @param effectSize - Minimum effect size to detect
 * @param standardError - Expected standard error
 * @param significanceLevel - Significance level
 * @returns Estimated power
 */
export function estimatePower(
  effectSize: number,
  standardError: number,
  significanceLevel: number = 0.05
): number {
  if (standardError <= 0) return 1;

  const criticalValue = normalQuantile(1 - significanceLevel / 2);
  const noncentrality = effectSize / standardError;

  // Power = P(|Z + ncp| > z_{α/2})
  const power =
    1 - normalCDF(criticalValue - noncentrality) + normalCDF(-criticalValue - noncentrality);

  return Math.max(0, Math.min(1, power));
}

/**
 * Calculate minimum detectable effect for given power
 *
 * @param power - Target power
 * @param standardError - Expected standard error
 * @param significanceLevel - Significance level
 * @returns Minimum detectable effect
 */
export function minimumDetectableEffect(
  power: number,
  standardError: number,
  significanceLevel: number = 0.05
): number {
  const zAlpha = normalQuantile(1 - significanceLevel / 2);
  const zBeta = normalQuantile(power);

  return (zAlpha + zBeta) * standardError;
}

// ============================================================================
// Matrix Operations (Helper Functions)
// ============================================================================

/**
 * Dot product of two vectors
 */
function dotProduct(a: Vector, b: Vector): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Matrix transpose
 */
function transpose(A: Matrix): Matrix {
  if (A.length === 0) return [];
  const m = A.length;
  const n = A[0].length;
  const result: Matrix = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      result[j][i] = A[i][j];
    }
  }
  return result;
}

/**
 * Matrix-vector multiplication
 */
function matrixVectorMultiply(A: Matrix, v: Vector): Vector {
  return A.map((row) => dotProduct(row, v));
}

/**
 * Matrix-matrix multiplication
 */
function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  const m = A.length;
  const n = B[0]?.length || 0;
  const k = B.length;
  const result: Matrix = Array.from({ length: m }, () => new Array(n).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      for (let l = 0; l < k; l++) {
        result[i][j] += A[i][l] * B[l][j];
      }
    }
  }
  return result;
}

/**
 * Matrix inversion using Gauss-Jordan elimination
 * Returns null if matrix is singular
 */
function invertMatrix(A: Matrix): Matrix | null {
  const n = A.length;
  if (n === 0) return null;

  // Create augmented matrix [A | I]
  const aug: Matrix = A.map((row, i) => {
    const identity = new Array(n).fill(0);
    identity[i] = 1;
    return [...row, ...identity];
  });

  // Forward elimination
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }

    // Swap rows
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    // Check for singular matrix
    if (Math.abs(aug[col][col]) < EPSILON) {
      return null;
    }

    // Scale pivot row
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row !== col) {
        const factor = aug[row][col];
        for (let j = 0; j < 2 * n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }
  }

  // Extract inverse from augmented matrix
  return aug.map((row) => row.slice(n));
}

// ============================================================================
// Distribution Functions
// ============================================================================

/**
 * Standard normal CDF approximation
 * Using Zelen & Severo (1964) approximation
 */
function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1.0 + sign * y);
}

/**
 * Standard normal quantile (inverse CDF) approximation
 * Using Abramowitz and Stegun approximation
 */
function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Rational approximation for lower region
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number, r: number;

  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    q = p - 0.5;
    r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}

// ============================================================================
// Random Number Generation (for simulations)
// ============================================================================

/**
 * Seeded random number generator (Mulberry32)
 */
export function createSeededRandom(seed: number): () => number {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Generate random normal samples using Box-Muller transform
 *
 * @param mean - Mean of distribution
 * @param std - Standard deviation
 * @param n - Number of samples
 * @param random - Random number generator (default Math.random)
 * @returns Normal samples
 */
export function randomNormal(
  mean: number,
  std: number,
  n: number = 1,
  random: () => number = Math.random
): Vector {
  const samples: number[] = [];

  for (let i = 0; i < n; i += 2) {
    const u1 = random();
    const u2 = random();
    const r = Math.sqrt(-2 * Math.log(u1 + EPSILON));
    const theta = 2 * Math.PI * u2;

    samples.push(mean + std * r * Math.cos(theta));
    if (i + 1 < n) {
      samples.push(mean + std * r * Math.sin(theta));
    }
  }

  return samples.slice(0, n);
}

/**
 * Generate random multivariate normal samples
 *
 * @param mean - Mean vector
 * @param covariance - Covariance matrix
 * @param n - Number of samples
 * @param random - Random number generator
 * @returns Matrix of samples (n x d)
 */
export function randomMultivariateNormal(
  mean: Vector,
  covariance: Matrix,
  n: number = 1,
  random: () => number = Math.random
): Matrix {
  const d = mean.length;

  // Cholesky decomposition of covariance
  const L = choleskyDecomposition(covariance);
  if (!L) {
    // Fallback to diagonal if Cholesky fails
    return Array.from({ length: n }, () =>
      mean.map((m, i) => m + Math.sqrt(covariance[i][i]) * randomNormal(0, 1, 1, random)[0])
    );
  }

  const samples: Matrix = [];
  for (let i = 0; i < n; i++) {
    const z = randomNormal(0, 1, d, random);
    const sample = new Array(d).fill(0);

    for (let j = 0; j < d; j++) {
      for (let k = 0; k <= j; k++) {
        sample[j] += L[j][k] * z[k];
      }
      sample[j] += mean[j];
    }

    samples.push(sample);
  }

  return samples;
}

/**
 * Cholesky decomposition of a positive-definite matrix
 * Returns lower triangular L such that A = LL^T
 */
function choleskyDecomposition(A: Matrix): Matrix | null {
  const n = A.length;
  const L: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;

      if (j === i) {
        for (let k = 0; k < j; k++) {
          sum += L[j][k] ** 2;
        }
        const diag = A[i][i] - sum;
        if (diag <= 0) return null; // Not positive definite
        L[i][j] = Math.sqrt(diag);
      } else {
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }

  return L;
}

// ============================================================================
// Exports for external use
// ============================================================================

export {
  dotProduct,
  transpose,
  matrixVectorMultiply,
  matrixMultiply,
  invertMatrix,
  normalCDF,
  normalQuantile,
  choleskyDecomposition
};
