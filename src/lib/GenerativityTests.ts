/**
 * Generativity Tests Implementation (B-module)
 *
 * Implements G1-G3 tests for determining whether a triadic construction
 * is genuinely generative versus taxonomic/aggregative.
 *
 * B01-B08: Generativity test framework
 */

import type {
  M0NonGenerativeModelClass,
  AggConstructor,
  LinCombConstructor,
  MonotoneTransformConstructor,
  G1Irreducibility,
  G2Mediation,
  G3DownwardClosure,
  GStatus,
  NonGenerativeMethod,
  StateSpaceRHS,
  TensionField
} from '../types/kernel';
import { computeTensionField, evaluateClosureCondition } from './KernelPrimitives';

// ============================================================================
// B01 - M0_NonGenerativeModelClass
// ============================================================================

/**
 * Creates the baseline class of non-generative constructions
 */
export function createM0NonGenerativeModelClass(): M0NonGenerativeModelClass {
  return {
    type: 'M0_NON_GENERATIVE',
    classId: 'M0',
    methods: ['aggregation', 'linear_combination', 'monotone_transform']
  };
}

/**
 * Tests if Z can be constructed from X, Y using non-generative methods
 */
export function testM0Membership(
  x: number[],
  y: number[],
  z: number[],
  tolerance: number = 0.1
): { isMember: boolean; method: NonGenerativeMethod | null; residual: number } {
  // Test aggregation
  const aggResult = testAggregation(x, y, z, tolerance);
  if (aggResult.fits) {
    return { isMember: true, method: 'aggregation', residual: aggResult.residual };
  }

  // Test linear combination
  const linResult = testLinearCombination(x, y, z, tolerance);
  if (linResult.fits) {
    return { isMember: true, method: 'linear_combination', residual: linResult.residual };
  }

  // Test monotone transform
  const monoResult = testMonotoneTransform(x, y, z, tolerance);
  if (monoResult.fits) {
    return { isMember: true, method: 'monotone_transform', residual: monoResult.residual };
  }

  return {
    isMember: false,
    method: null,
    residual: Math.min(aggResult.residual, linResult.residual, monoResult.residual)
  };
}

// ============================================================================
// B02 - AggConstructor
// ============================================================================

/**
 * Creates an aggregation constructor Z := X ∪ Y
 */
export function createAggConstructor(x: number[], y: number[]): AggConstructor {
  // Element-wise max (set union approximation for real vectors)
  const z = x.map((xi, i) => Math.max(xi, y[i]));

  return {
    type: 'AGG_CONSTRUCTOR',
    inputX: x,
    inputY: y,
    outputZ: z
  };
}

/**
 * Tests if Z can be constructed via aggregation
 */
function testAggregation(
  x: number[],
  y: number[],
  z: number[],
  tolerance: number
): { fits: boolean; residual: number } {
  // Try various aggregation operations
  const operations = [
    // Union-like: max
    x.map((xi, i) => Math.max(xi, y[i])),
    // Sum
    x.map((xi, i) => xi + y[i]),
    // Concatenation similarity (average)
    x.map((xi, i) => (xi + y[i]) / 2)
  ];

  let minResidual = Infinity;
  for (const result of operations) {
    const residual = computeResidual(result, z);
    minResidual = Math.min(minResidual, residual);
    if (residual < tolerance) {
      return { fits: true, residual };
    }
  }

  return { fits: false, residual: minResidual };
}

// ============================================================================
// B03 - LinCombConstructor
// ============================================================================

/**
 * Creates a linear combination constructor Z := aX + bY
 */
export function createLinCombConstructor(
  x: number[],
  y: number[],
  a: number,
  b: number
): LinCombConstructor {
  const z = x.map((xi, i) => a * xi + b * y[i]);

  return {
    type: 'LIN_COMB_CONSTRUCTOR',
    coeffA: a,
    coeffB: b,
    inputX: x,
    inputY: y,
    outputZ: z
  };
}

/**
 * Tests if Z can be constructed via linear combination
 */
function testLinearCombination(
  x: number[],
  y: number[],
  z: number[],
  tolerance: number
): { fits: boolean; residual: number; coeffA?: number; coeffB?: number } {
  // Solve least squares: minimize ||aX + bY - Z||^2
  // Using normal equations

  const n = x.length;
  let xxSum = 0,
    yySum = 0,
    xySum = 0,
    xzSum = 0,
    yzSum = 0;

  for (let i = 0; i < n; i++) {
    xxSum += x[i] * x[i];
    yySum += y[i] * y[i];
    xySum += x[i] * y[i];
    xzSum += x[i] * z[i];
    yzSum += y[i] * z[i];
  }

  // Solve 2x2 system
  const det = xxSum * yySum - xySum * xySum;
  if (Math.abs(det) < 1e-10) {
    return { fits: false, residual: Infinity };
  }

  const a = (yySum * xzSum - xySum * yzSum) / det;
  const b = (xxSum * yzSum - xySum * xzSum) / det;

  // Compute residual
  const predicted = x.map((xi, i) => a * xi + b * y[i]);
  const residual = computeResidual(predicted, z);

  return {
    fits: residual < tolerance,
    residual,
    coeffA: a,
    coeffB: b
  };
}

// ============================================================================
// B04 - MonotoneTransformConstructor
// ============================================================================

/**
 * Creates a monotone transform constructor Z := f(X, Y)
 */
export function createMonotoneTransformConstructor(
  transformId: string,
  preservesTopology: boolean
): MonotoneTransformConstructor {
  return {
    type: 'MONOTONE_TRANSFORM_CONSTRUCTOR',
    transformId,
    preservesTopology
  };
}

/**
 * Tests if Z can be constructed via monotone transform
 */
function testMonotoneTransform(
  x: number[],
  y: number[],
  z: number[],
  tolerance: number
): { fits: boolean; residual: number } {
  // Test various monotone transforms
  const transforms = [
    // Elementwise product
    x.map((xi, i) => xi * y[i]),
    // Softmax-like
    x.map((xi, i) => Math.exp(xi) / (Math.exp(xi) + Math.exp(y[i]))),
    // Polynomial (x^2 + y^2)
    x.map((xi, i) => xi * xi + y[i] * y[i]),
    // Min
    x.map((xi, i) => Math.min(xi, y[i]))
  ];

  let minResidual = Infinity;
  for (const result of transforms) {
    const residual = computeResidual(result, z);
    minResidual = Math.min(minResidual, residual);
    if (residual < tolerance) {
      return { fits: true, residual };
    }
  }

  // Also test if Z preserves order of X or Y (monotonicity)
  const xOrderPreserved = isOrderPreserved(x, z);
  const yOrderPreserved = isOrderPreserved(y, z);

  if (xOrderPreserved || yOrderPreserved) {
    // Monotone relationship exists - try to fit
    const bestFitResidual = fitMonotone(xOrderPreserved ? x : y, z);
    if (bestFitResidual < tolerance) {
      return { fits: true, residual: bestFitResidual };
    }
    minResidual = Math.min(minResidual, bestFitResidual);
  }

  return { fits: false, residual: minResidual };
}

/**
 * Checks if order is preserved between two vectors
 */
function isOrderPreserved(a: number[], b: number[]): boolean {
  let increasing = true;
  let decreasing = true;

  for (let i = 1; i < a.length; i++) {
    if (a[i] > a[i - 1] && b[i] < b[i - 1]) increasing = false;
    if (a[i] < a[i - 1] && b[i] > b[i - 1]) increasing = false;
    if (a[i] > a[i - 1] && b[i] > b[i - 1]) decreasing = false;
    if (a[i] < a[i - 1] && b[i] < b[i - 1]) decreasing = false;
  }

  return increasing || decreasing;
}

/**
 * Fits a monotone transformation
 */
function fitMonotone(input: number[], target: number[]): number {
  // Simple isotonic regression
  const pairs = input.map((v, i) => ({ x: v, y: target[i] })).sort((a, b) => a.x - b.x);

  // Pool Adjacent Violators Algorithm
  const n = pairs.length;
  const fitted: number[] = pairs.map((p) => p.y);

  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < n - 1; i++) {
      if (fitted[i] > fitted[i + 1]) {
        const avg = (fitted[i] + fitted[i + 1]) / 2;
        fitted[i] = avg;
        fitted[i + 1] = avg;
        changed = true;
      }
    }
  }

  // Map back and compute residual
  const predicted: number[] = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const originalIndex = input.indexOf(pairs[i].x);
    predicted[originalIndex] = fitted[i];
  }

  return computeResidual(predicted, target);
}

/**
 * Computes normalized residual between two vectors
 */
function computeResidual(predicted: number[], actual: number[]): number {
  const n = predicted.length;
  let sse = 0;
  let tss = 0;
  const mean = actual.reduce((a, b) => a + b, 0) / n;

  for (let i = 0; i < n; i++) {
    sse += Math.pow(predicted[i] - actual[i], 2);
    tss += Math.pow(actual[i] - mean, 2);
  }

  if (tss < 1e-10) return sse > 1e-10 ? Infinity : 0;
  return Math.sqrt(sse / tss);
}

// ============================================================================
// B05 - G1_Irreducibility Test
// ============================================================================

/**
 * Tests G1: Irreducibility - Z ∉ M0(X,Y)
 */
export function testG1Irreducibility(
  x: number[],
  y: number[],
  z: number[],
  tolerance: number = 0.1
): G1Irreducibility {
  const m0Result = testM0Membership(x, y, z, tolerance);

  // Compute irreducibility score based on residual
  // Higher residual from M0 methods = higher irreducibility
  const score = Math.min(1, m0Result.residual);

  return {
    type: 'G1_IRREDUCIBILITY',
    passes: !m0Result.isMember,
    testedMethods: ['aggregation', 'linear_combination', 'monotone_transform'],
    score
  };
}

/**
 * Extended G1 test with Phi (integrated information) proxy
 */
export function testG1WithPhi(
  x: number[],
  y: number[],
  z: number[],
  minPhi: number = 0.1
): G1Irreducibility & { phi: number } {
  const basicResult = testG1Irreducibility(x, y, z);

  // Compute Phi proxy (excess entropy)
  const phi = computePhiProxy(x, y, z);

  return {
    ...basicResult,
    passes: basicResult.passes && phi > minPhi,
    score: Math.min(basicResult.score, phi),
    phi
  };
}

/**
 * Computes Phi proxy via excess entropy approximation
 */
function computePhiProxy(x: number[], y: number[], z: number[]): number {
  // Mutual information approximation: I(X,Y;Z) - I(X;Z) - I(Y;Z)
  // Using correlation-based proxy

  const rxz = correlation(x, z);
  const ryz = correlation(y, z);
  const rxy = correlation(x, y);

  // Partial information: how much does knowing both X and Y tell us about Z
  // beyond what each tells individually
  const jointInfo = 0.5 * Math.log(1 + rxz * rxz + ryz * ryz + 2 * rxz * ryz * rxy);
  const xInfo = 0.5 * Math.log(1 + rxz * rxz);
  const yInfo = 0.5 * Math.log(1 + ryz * ryz);

  const phi = Math.max(0, jointInfo - Math.max(xInfo, yInfo));
  return Math.min(1, phi);
}

/**
 * Computes Pearson correlation
 */
function correlation(a: number[], b: number[]): number {
  const n = a.length;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;

  let num = 0,
    denA = 0,
    denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }

  const den = Math.sqrt(denA * denB);
  return den > 1e-10 ? num / den : 0;
}

// ============================================================================
// B06 - G2_Mediation Test
// ============================================================================

/**
 * Tests G2: Mediation - ∃π that reduces tension while preserving integrity
 */
export function testG2Mediation(state: StateSpaceRHS): G2Mediation {
  const tensionField = computeTensionField(state);
  const initialTension = tensionField.totalEnergy;

  // Try to find a mediation path
  const mediationResult = findMediationPath(state, tensionField);

  // Compute integrity preservation
  const integrityPreserved = checkIntegrityPreservation(state, mediationResult.path);

  const passes = mediationResult.tensionReduction > 0.1 && integrityPreserved;

  return {
    type: 'G2_MEDIATION',
    passes,
    tensionReduction: mediationResult.tensionReduction,
    integrityPreserved,
    mediationPath: mediationResult.path
  };
}

/**
 * Finds a mediation path that reduces tension
 */
function findMediationPath(
  state: StateSpaceRHS,
  tensionField: TensionField
): { tensionReduction: number; path: number[] } {
  const n = state.dimension;
  const path: number[] = [];
  let currentTension = tensionField.totalEnergy;
  let totalReduction = 0;

  // Greedy search for tension-reducing moves
  for (let step = 0; step < Math.min(n, 20); step++) {
    // Find the highest tension peak
    if (tensionField.peaks.length === 0) break;

    const peak = tensionField.peaks[step % tensionField.peaks.length];
    const peakIndex = peak.location[0];

    // Apply mediation at this location
    const mediatedValue = state.s.mediationCoefficients[peakIndex];
    path.push(peakIndex);

    // Estimate tension reduction
    const reduction = peak.magnitude * mediatedValue * 0.5;
    totalReduction += reduction;
    currentTension -= reduction;

    if (currentTension <= 0) break;
  }

  const tensionReduction =
    tensionField.totalEnergy > 0 ? totalReduction / tensionField.totalEnergy : 0;
  return { tensionReduction: Math.min(1, tensionReduction), path };
}

/**
 * Checks if mediation preserves system integrity
 */
function checkIntegrityPreservation(state: StateSpaceRHS, _path: number[]): boolean {
  // Integrity is preserved if:
  // 1. R variance is not completely suppressed
  const rVariance = state.r.variance;
  if (rVariance < 0.01) return false;

  // 2. H coherence structure is maintained
  const coherenceStrength = state.h.holdingStrength;
  if (coherenceStrength < 0.1) return false;

  // 3. S clearance is not total
  if (state.s.clearanceLevel > 0.99) return false;

  return true;
}

// ============================================================================
// B07 - G3_DownwardClosure Test
// ============================================================================

/**
 * Tests G3: Downward Closure - Z_t constrains P(X/Y)_{t+1}
 */
export function testG3DownwardClosure(state: StateSpaceRHS): G3DownwardClosure {
  const closure = evaluateClosureCondition(state);

  // Convert closure evidence to G3 result
  const constraintStrength = closure.influenceStrength;

  // Compute p-value proxy (based on evidence strength)
  const pValue = computePValueProxy(closure.evidence);

  const passes = closure.isSatisfied && pValue < 0.05;

  return {
    type: 'G3_DOWNWARD_CLOSURE',
    passes,
    constraintStrength,
    pValue
  };
}

/**
 * Computes p-value proxy from closure evidence
 */
function computePValueProxy(
  evidence: Array<{ method: string; value: number; confidence: number }>
): number {
  if (evidence.length === 0) return 1;

  // Combine evidence using Fisher's method approximation
  let chiSquare = 0;
  for (const e of evidence) {
    // Convert value to pseudo p-value
    const pseudoP = Math.max(0.001, 1 - e.value * e.confidence);
    chiSquare += -2 * Math.log(pseudoP);
  }

  // Chi-square to p-value approximation (df = 2 * evidence.length)
  const df = 2 * evidence.length;
  const pValue = 1 - gammaCDF(chiSquare / 2, df / 2);

  return Math.max(0, Math.min(1, pValue));
}

/**
 * Gamma CDF approximation for chi-square conversion
 */
function gammaCDF(x: number, k: number): number {
  // Simple approximation using incomplete gamma function
  if (x <= 0) return 0;
  if (x > 100) return 1;

  // Series expansion
  let sum = 0;
  let term = 1 / k;
  sum += term;

  for (let n = 1; n < 100; n++) {
    term *= x / (k + n);
    sum += term;
    if (term < 1e-10) break;
  }

  return Math.pow(x, k) * Math.exp(-x) * sum;
}

// ============================================================================
// B08 - G_Status Classification
// ============================================================================

/**
 * Classifies overall generativity status: Generative iff G1∧G2∧G3
 */
export function classifyGStatus(
  g1: G1Irreducibility,
  g2: G2Mediation,
  g3: G3DownwardClosure
): GStatus {
  return {
    type: 'G_STATUS',
    g1,
    g2,
    g3,
    isGenerative: g1.passes && g2.passes && g3.passes
  };
}

/**
 * Full generativity test pipeline
 */
export function runFullGenerativityTest(
  x: number[],
  y: number[],
  z: number[],
  state: StateSpaceRHS
): GStatus {
  const g1 = testG1Irreducibility(x, y, z);
  const g2 = testG2Mediation(state);
  const g3 = testG3DownwardClosure(state);

  return classifyGStatus(g1, g2, g3);
}

/**
 * Quick generativity check (less thorough but faster)
 */
export function quickGenerativityCheck(state: StateSpaceRHS): {
  isGenerative: boolean;
  score: number;
} {
  // Quick G1: check if R-H-S are sufficiently distinct
  const rNorm = Math.sqrt(state.r.perturbation.reduce((s, v) => s + v * v, 0));
  const hNorm = Math.sqrt(state.h.structure.reduce((s, v) => s + v * v, 0));
  const sNorm = Math.sqrt(state.s.mediationCoefficients.reduce((s, v) => s + v * v, 0));

  const rhCorr = correlation(state.r.perturbation, state.h.structure);
  const hsCorr = correlation(state.h.structure, state.s.mediationCoefficients);
  const rsCorr = correlation(state.r.perturbation, state.s.mediationCoefficients);

  const distinctiveness = 1 - (Math.abs(rhCorr) + Math.abs(hsCorr) + Math.abs(rsCorr)) / 3;

  // Quick G2: check tension level
  const tensionProxy = Math.abs(rhCorr);
  const mediationProxy = state.s.clearanceLevel * (1 - Math.abs(rsCorr));

  // Quick G3: check history length (proxy for downward influence)
  const historyScore = Math.min(1, state.history.length / 20);

  const score = (distinctiveness + mediationProxy + historyScore) / 3;
  const isGenerative = score > 0.5 && distinctiveness > 0.3 && historyScore > 0.2;

  return { isGenerative, score };
}

// ============================================================================
// Utility Exports
// ============================================================================

export {
  computeResidual,
  correlation,
  testAggregation,
  testLinearCombination,
  testMonotoneTransform,
  computePhiProxy
};
