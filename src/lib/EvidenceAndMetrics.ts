/**
 * Evidence Tier System and Metrics/Proxies Implementation
 *
 * C-module: Evidence tier classification (E0-E3)
 * D-module: Signals and metrics for monitoring RHS dynamics
 */

import type {
  EvidenceTier,
  EAssignment,
  EvidenceRecord,
  KLDefinition,
  KLEstimator,
  KLWindow,
  KLThresholds,
  KLFailures,
  KLFailureType,
  FreeEnergyDefinition,
  FreeEnergyEstimator,
  FreeEnergyWindow,
  FreeEnergyThresholds,
  FreeEnergyFailures,
  FreeEnergyFailureType,
  TEDefinition,
  TEEstimator,
  TEWindow,
  TEThresholds,
  TEFailures,
  GrangerDefinition,
  GrangerEstimator,
  GrangerWindow,
  GrangerThresholds,
  GrangerFailures,
  PhiDefinition,
  PhiEstimator,
  PhiWindow,
  PhiThresholds,
  PhiFailures,
  ModeCondition,
  ModeType,
  NearMissFlag,
  EvidenceGapFlag,
  StateSpaceRHS,
  GStatus,
} from '../types/kernel';

// ============================================================================
// C. Evidence Tier System (C01-C05)
// ============================================================================

/**
 * C05 - Assigns evidence tier based on method and evidence record
 */
export function assignEvidenceTier(
  evidenceRecord: EvidenceRecord[],
  method: string
): EAssignment {
  // Count evidence types
  const typeCounts: Record<string, number> = {
    measurement: 0,
    simulation: 0,
    theory: 0,
    experiment: 0,
  };

  for (const record of evidenceRecord) {
    typeCounts[record.type]++;
  }

  // Determine tier based on evidence types
  let tier: EvidenceTier = 'E0';
  let confidence = 0.5;

  if (typeCounts.experiment > 0) {
    tier = 'E3';
    confidence = Math.min(1, 0.7 + typeCounts.experiment * 0.1);
  } else if (typeCounts.simulation > 0) {
    tier = 'E2';
    confidence = Math.min(1, 0.5 + typeCounts.simulation * 0.1);
  } else if (typeCounts.measurement > 0) {
    tier = 'E1';
    confidence = Math.min(1, 0.3 + typeCounts.measurement * 0.1);
  } else if (typeCounts.theory > 0) {
    tier = 'E0';
    confidence = Math.min(1, 0.2 + typeCounts.theory * 0.05);
  }

  return {
    type: 'E_ASSIGNMENT',
    tier,
    method,
    evidenceRecord,
    confidence,
  };
}

/**
 * Creates an evidence record entry
 */
export function createEvidenceRecord(
  source: string,
  type: EvidenceRecord['type'],
  value: number
): EvidenceRecord {
  return {
    source,
    type,
    value: Math.max(0, Math.min(1, value)),
    timestamp: Date.now(),
  };
}

/**
 * Upgrades evidence tier with new evidence
 */
export function upgradeEvidenceTier(
  current: EAssignment,
  newEvidence: EvidenceRecord[]
): EAssignment {
  const combinedEvidence = [...current.evidenceRecord, ...newEvidence];
  return assignEvidenceTier(combinedEvidence, current.method);
}

// ============================================================================
// D01-D05 - KL Divergence Pipeline
// ============================================================================

/**
 * D01 - Computes KL divergence D_KL(R||H) as tension proxy
 */
export function computeKLDivergence(r: number[], h: number[]): KLDefinition {
  // Normalize to probability distributions
  const pR = normalizeToProbability(r);
  const pH = normalizeToProbability(h);

  // Compute KL divergence: D_KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
  let kl = 0;
  for (let i = 0; i < pR.length; i++) {
    if (pR[i] > 1e-10 && pH[i] > 1e-10) {
      kl += pR[i] * Math.log(pR[i] / pH[i]);
    } else if (pR[i] > 1e-10) {
      kl += pR[i] * Math.log(pR[i] / 1e-10); // Smoothing
    }
  }

  return {
    type: 'KL_DEF',
    value: Math.max(0, kl),
    distributionR: pR,
    distributionH: pH,
  };
}

/**
 * Normalizes a vector to a probability distribution
 */
function normalizeToProbability(v: number[]): number[] {
  // Shift to positive
  const minVal = Math.min(...v);
  const shifted = v.map((x) => x - minVal + 1e-10);

  // Normalize
  const sum = shifted.reduce((a, b) => a + b, 0);
  return shifted.map((x) => x / sum);
}

/**
 * D02 - Estimates KL divergence from samples
 */
export function createKLEstimator(
  rSamples: number[][],
  hSamples: number[][]
): KLEstimator {
  if (rSamples.length === 0 || hSamples.length === 0) {
    return {
      type: 'KL_ESTIMATOR',
      estimate: 0,
      sampleSize: 0,
      confidenceInterval: [0, 0],
    };
  }

  // Compute KL for each pair and average
  const klValues: number[] = [];
  const minLen = Math.min(rSamples.length, hSamples.length);

  for (let i = 0; i < minLen; i++) {
    const kl = computeKLDivergence(rSamples[i], hSamples[i]);
    klValues.push(kl.value);
  }

  const mean = klValues.reduce((a, b) => a + b, 0) / klValues.length;
  const std = Math.sqrt(
    klValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / klValues.length
  );

  // 95% confidence interval
  const margin = 1.96 * std / Math.sqrt(klValues.length);

  return {
    type: 'KL_ESTIMATOR',
    estimate: mean,
    sampleSize: minLen,
    confidenceInterval: [Math.max(0, mean - margin), mean + margin],
  };
}

/**
 * D03 - Creates a rolling window for KL estimation
 */
export function createKLWindow(size: number = 20): KLWindow {
  return {
    size,
    position: 0,
    values: [],
  };
}

/**
 * Updates KL window with new value
 */
export function updateKLWindow(window: KLWindow, value: number): KLWindow {
  const newValues = [...window.values, value];
  if (newValues.length > window.size) {
    newValues.shift();
  }

  return {
    ...window,
    position: window.position + 1,
    values: newValues,
  };
}

/**
 * D04 - Default KL thresholds
 */
export function createKLThresholds(low: number = 0.1, high: number = 2.0): KLThresholds {
  return { low, high };
}

/**
 * D05 - Detects KL-based failures
 */
export function detectKLFailures(
  window: KLWindow,
  thresholds: KLThresholds,
  sustainedDuration: number = 5
): KLFailures {
  if (window.values.length < sustainedDuration) {
    return { type: 'KL_FAILURES', failureType: 'none', duration: 0, severity: 0 };
  }

  const recent = window.values.slice(-sustainedDuration);

  // Check for sustained high
  const allHigh = recent.every((v) => v > thresholds.high);
  if (allHigh) {
    const avgValue = recent.reduce((a, b) => a + b, 0) / recent.length;
    return {
      type: 'KL_FAILURES',
      failureType: 'sustained_high',
      duration: sustainedDuration,
      severity: Math.min(1, (avgValue - thresholds.high) / thresholds.high),
    };
  }

  // Check for sustained zero
  const allLow = recent.every((v) => v < thresholds.low);
  if (allLow) {
    const avgValue = recent.reduce((a, b) => a + b, 0) / recent.length;
    return {
      type: 'KL_FAILURES',
      failureType: 'sustained_zero',
      duration: sustainedDuration,
      severity: Math.min(1, (thresholds.low - avgValue) / thresholds.low),
    };
  }

  return { type: 'KL_FAILURES', failureType: 'none', duration: 0, severity: 0 };
}

// ============================================================================
// D06-D10 - Free Energy Pipeline
// ============================================================================

/**
 * D06 - Computes variational free energy
 */
export function computeFreeEnergy(
  observations: number[],
  predictions: number[],
  priorVariance: number = 1.0
): FreeEnergyDefinition {
  // Surprisal: -log P(o|model) ≈ prediction error
  const residuals = observations.map((o, i) => o - predictions[i]);
  const surprisal = residuals.reduce((sum, r) => sum + r * r, 0) / (2 * priorVariance);

  // Complexity: KL divergence from prior (approximated as variance ratio)
  const posteriorVariance = residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length;
  const complexity =
    0.5 * (posteriorVariance / priorVariance - 1 + Math.log(priorVariance / posteriorVariance));

  const value = surprisal + complexity;

  return {
    type: 'FREE_E_DEF',
    value: Math.max(0, value),
    surprisal,
    complexity: Math.max(0, complexity),
  };
}

/**
 * D07 - Free energy estimator from model residuals
 */
export function createFreeEnergyEstimator(residuals: number[]): FreeEnergyEstimator {
  const meanSquaredError = residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length;

  return {
    type: 'FREE_E_ESTIMATOR',
    estimate: meanSquaredError,
    residuals,
  };
}

/**
 * D08 - Creates free energy tracking window
 */
export function createFreeEnergyWindow(epochSize: number = 100): FreeEnergyWindow {
  return {
    epochSize,
    values: [],
  };
}

/**
 * Updates free energy window
 */
export function updateFreeEnergyWindow(
  window: FreeEnergyWindow,
  value: number
): FreeEnergyWindow {
  const newValues = [...window.values, value];
  // Keep last 10 epochs worth of data
  if (newValues.length > window.epochSize * 10) {
    newValues.splice(0, window.epochSize);
  }

  return {
    ...window,
    values: newValues,
  };
}

/**
 * D09 - Default free energy thresholds
 */
export function createFreeEnergyThresholds(
  explorationLower: number = 0.1,
  explorationUpper: number = 2.0,
  collapseThreshold: number = 0.01
): FreeEnergyThresholds {
  return { explorationLower, explorationUpper, collapseThreshold };
}

/**
 * D10 - Detects free energy failures
 */
export function detectFreeEnergyFailures(
  window: FreeEnergyWindow,
  thresholds: FreeEnergyThresholds
): FreeEnergyFailures {
  if (window.values.length < 10) {
    return { type: 'FREE_E_FAILURES', failureType: 'none', detectedAtEpoch: -1 };
  }

  const recent = window.values.slice(-10);
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const trend = recent[recent.length - 1] - recent[0];

  // Collapse: very low free energy, declining trend
  if (mean < thresholds.collapseThreshold || trend < -thresholds.explorationLower) {
    return {
      type: 'FREE_E_FAILURES',
      failureType: 'collapse',
      detectedAtEpoch: Math.floor(window.values.length / window.epochSize),
    };
  }

  // Over-constraint: very high free energy, can't reduce
  if (mean > thresholds.explorationUpper && trend > 0) {
    return {
      type: 'FREE_E_FAILURES',
      failureType: 'over_constraint',
      detectedAtEpoch: Math.floor(window.values.length / window.epochSize),
    };
  }

  return { type: 'FREE_E_FAILURES', failureType: 'none', detectedAtEpoch: -1 };
}

// ============================================================================
// D11-D15 - Transfer Entropy Pipeline
// ============================================================================

/**
 * D11 - Computes transfer entropy TE_{S→R}
 */
export function computeTransferEntropy(
  sSeries: number[][],
  rSeries: number[][],
  lag: number = 1
): TEDefinition {
  if (sSeries.length < lag + 2 || rSeries.length < lag + 2) {
    return { type: 'TE_DEF', value: 0, source: 'S', target: 'R' };
  }

  // Simplified TE estimation using correlation
  // TE(S→R) ≈ I(R_{t+1}; S_t | R_t)
  // We approximate this using partial correlation

  const n = Math.min(sSeries.length, rSeries.length) - lag;
  let sumS = 0, sumR = 0, sumR1 = 0;
  let sumSS = 0, sumRR = 0, sumR1R1 = 0;
  let sumSR = 0, sumSR1 = 0, sumRR1 = 0;

  for (let t = 0; t < n; t++) {
    const s = sSeries[t].reduce((a, b) => a + b, 0) / sSeries[t].length;
    const r = rSeries[t].reduce((a, b) => a + b, 0) / rSeries[t].length;
    const r1 = rSeries[t + lag].reduce((a, b) => a + b, 0) / rSeries[t + lag].length;

    sumS += s;
    sumR += r;
    sumR1 += r1;
    sumSS += s * s;
    sumRR += r * r;
    sumR1R1 += r1 * r1;
    sumSR += s * r;
    sumSR1 += s * r1;
    sumRR1 += r * r1;
  }

  // Compute correlations
  const corrSR1 = (n * sumSR1 - sumS * sumR1) /
    Math.sqrt((n * sumSS - sumS * sumS) * (n * sumR1R1 - sumR1 * sumR1) + 1e-10);
  const corrRR1 = (n * sumRR1 - sumR * sumR1) /
    Math.sqrt((n * sumRR - sumR * sumR) * (n * sumR1R1 - sumR1 * sumR1) + 1e-10);
  const corrSR = (n * sumSR - sumS * sumR) /
    Math.sqrt((n * sumSS - sumS * sumS) * (n * sumRR - sumR * sumR) + 1e-10);

  // Partial correlation: ρ(S,R1|R)
  const partialCorr = (corrSR1 - corrSR * corrRR1) /
    Math.sqrt((1 - corrSR * corrSR) * (1 - corrRR1 * corrRR1) + 1e-10);

  // Convert to information (TE ≈ -0.5 * log(1 - ρ²))
  const te = Math.max(0, -0.5 * Math.log(1 - partialCorr * partialCorr + 1e-10));

  return { type: 'TE_DEF', value: te, source: 'S', target: 'R' };
}

/**
 * D12 - Transfer entropy estimator with history embedding
 */
export function createTEEstimator(
  sSeries: number[][],
  rSeries: number[][],
  lagOrder: number = 3
): TEEstimator {
  // Compute TE for different lags and average
  let totalTE = 0;
  const historyEmbedding: number[] = [];

  for (let lag = 1; lag <= lagOrder; lag++) {
    const te = computeTransferEntropy(sSeries, rSeries, lag);
    totalTE += te.value;
    historyEmbedding.push(te.value);
  }

  return {
    type: 'TE_ESTIMATOR',
    estimate: totalTE / lagOrder,
    historyEmbedding,
    lagOrder,
  };
}

/**
 * D13 - Creates TE window parameters
 */
export function createTEWindow(lagOrder: number = 3, windowLength: number = 50): TEWindow {
  return { lagOrder, windowLength };
}

/**
 * D14 - Default TE thresholds
 */
export function createTEThresholds(minTEForClosure: number = 0.05): TEThresholds {
  return { minTEForClosure };
}

/**
 * D15 - Detects TE failures (epiphenomenal S)
 */
export function detectTEFailures(
  teEstimator: TEEstimator,
  thresholds: TEThresholds
): TEFailures {
  const isEpiphenomenal = teEstimator.estimate < thresholds.minTEForClosure;

  return {
    type: 'TE_FAILURES',
    isEpiphenomenal,
    teValue: teEstimator.estimate,
  };
}

// ============================================================================
// D16-D20 - Granger Causality Pipeline
// ============================================================================

/**
 * D16 - Computes Granger causality S→R
 */
export function computeGrangerCausality(
  sSeries: number[],
  rSeries: number[],
  maxLag: number = 5
): GrangerDefinition {
  if (sSeries.length < maxLag + 2) {
    return {
      type: 'GRANGER_DEF',
      fStatistic: 0,
      pValue: 1,
      direction: 'S_to_R',
    };
  }

  // Fit restricted model: R_t = f(R_{t-1}, ..., R_{t-k})
  const restrictedSSE = fitAR(rSeries, maxLag);

  // Fit unrestricted model: R_t = f(R_{t-1}, ..., R_{t-k}, S_{t-1}, ..., S_{t-k})
  const unrestrictedSSE = fitVAR(rSeries, sSeries, maxLag);

  // F-statistic
  const n = rSeries.length - maxLag;
  const fStatistic =
    ((restrictedSSE - unrestrictedSSE) / maxLag) /
    (unrestrictedSSE / (n - 2 * maxLag - 1) + 1e-10);

  // P-value approximation using F-distribution
  const pValue = computeFPValue(Math.max(0, fStatistic), maxLag, n - 2 * maxLag - 1);

  return {
    type: 'GRANGER_DEF',
    fStatistic: Math.max(0, fStatistic),
    pValue,
    direction: 'S_to_R',
  };
}

/**
 * Fits an AR model and returns SSE
 */
function fitAR(series: number[], lag: number): number {
  const n = series.length - lag;
  let sse = 0;

  for (let t = lag; t < series.length; t++) {
    // Simple average predictor
    let pred = 0;
    for (let l = 1; l <= lag; l++) {
      pred += series[t - l] / lag;
    }
    sse += Math.pow(series[t] - pred, 2);
  }

  return sse;
}

/**
 * Fits a VAR model and returns SSE
 */
function fitVAR(rSeries: number[], sSeries: number[], lag: number): number {
  const n = Math.min(rSeries.length, sSeries.length) - lag;
  let sse = 0;

  for (let t = lag; t < Math.min(rSeries.length, sSeries.length); t++) {
    // Predictor using both R and S history
    let pred = 0;
    for (let l = 1; l <= lag; l++) {
      pred += (rSeries[t - l] + sSeries[t - l]) / (2 * lag);
    }
    sse += Math.pow(rSeries[t] - pred, 2);
  }

  return sse;
}

/**
 * Computes F-distribution p-value (approximation)
 */
function computeFPValue(f: number, df1: number, df2: number): number {
  if (f <= 0 || df1 <= 0 || df2 <= 0) return 1;

  // Beta function approximation
  const x = df2 / (df2 + df1 * f);
  const a = df2 / 2;
  const b = df1 / 2;

  // Incomplete beta function approximation
  return incompleteBeta(x, a, b);
}

/**
 * Incomplete beta function approximation
 */
function incompleteBeta(x: number, a: number, b: number): number {
  if (x === 0) return 0;
  if (x === 1) return 1;

  // Continued fraction approximation
  const bt = Math.exp(
    a * Math.log(x) + b * Math.log(1 - x) -
    Math.log(a) - logBeta(a, b)
  );

  if (x < (a + 1) / (a + b + 2)) {
    return bt * betaCF(x, a, b) / a;
  } else {
    return 1 - bt * betaCF(1 - x, b, a) / b;
  }
}

/**
 * Log of beta function
 */
function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/**
 * Log gamma function (Lanczos approximation)
 */
function logGamma(x: number): number {
  const g = 7;
  const coefficients = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  }

  x -= 1;
  let a = coefficients[0];
  const t = x + g + 0.5;

  for (let i = 1; i < g + 2; i++) {
    a += coefficients[i] / (x + i);
  }

  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Beta continued fraction
 */
function betaCF(x: number, a: number, b: number): number {
  const maxIterations = 100;
  const eps = 1e-10;

  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - qab * x / qap;

  if (Math.abs(d) < eps) d = eps;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIterations; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < eps) d = eps;
    c = 1 + aa / c;
    if (Math.abs(c) < eps) c = eps;
    d = 1 / d;
    h *= d * c;

    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < eps) d = eps;
    c = 1 + aa / c;
    if (Math.abs(c) < eps) c = eps;
    d = 1 / d;
    const del = d * c;
    h *= del;

    if (Math.abs(del - 1) < eps) break;
  }

  return h;
}

/**
 * D17 - Creates Granger estimator
 */
export function createGrangerEstimator(
  rSeries: number[],
  sSeries: number[],
  maxLag: number = 5
): GrangerEstimator {
  const result = computeGrangerCausality(sSeries, rSeries, maxLag);

  return {
    type: 'GRANGER_ESTIMATOR',
    varCoefficients: [], // Simplified - would need full VAR implementation
    fStatistic: result.fStatistic,
    isSignificant: result.pValue < 0.05,
  };
}

/**
 * D18 - Creates Granger window
 */
export function createGrangerWindow(horizon: number = 50, samplingRate: number = 1): GrangerWindow {
  return { horizon, samplingRate };
}

/**
 * D19 - Default Granger thresholds
 */
export function createGrangerThresholds(
  significanceLevel: number = 0.05,
  ciWidth: number = 0.95
): GrangerThresholds {
  return { significanceLevel, ciWidth };
}

/**
 * D20 - Detects Granger failures
 */
export function detectGrangerFailures(
  estimator: GrangerEstimator,
  thresholds: GrangerThresholds
): GrangerFailures {
  // Non-significant F-statistic means no Granger causality
  const pValue = computeFPValue(estimator.fStatistic, 5, 40); // Approximate df

  return {
    type: 'GRANGER_FAILURES',
    isNonCausal: pValue > thresholds.significanceLevel,
    pValue,
  };
}

// ============================================================================
// D21-D25 - Phi (Irreducibility) Pipeline
// ============================================================================

/**
 * D21 - Computes Phi (integrated information) proxy
 */
export function computePhi(state: StateSpaceRHS): PhiDefinition {
  // Compute excess entropy as Phi proxy
  // E = I(past; future) = H(future) - H(future | past)

  const history = state.history;
  if (history.length < 10) {
    return { type: 'PHI_DEF', value: 0, excessEntropy: 0 };
  }

  const midpoint = Math.floor(history.length / 2);
  const past = history.slice(0, midpoint);
  const future = history.slice(midpoint);

  // Entropy of future states
  const futureEntropy = computeStateEntropy(future);

  // Conditional entropy of future given past
  const conditionalEntropy = computeConditionalEntropy(future, past);

  const excessEntropy = Math.max(0, futureEntropy - conditionalEntropy);

  // Phi is related to how much information is lost by partitioning
  const phi = excessEntropy * computePartitionLoss(state);

  return {
    type: 'PHI_DEF',
    value: Math.min(1, phi),
    excessEntropy,
  };
}

/**
 * Computes entropy of state snapshots
 */
function computeStateEntropy(snapshots: Array<{ r: number[]; h: number[]; s: number[] }>): number {
  if (snapshots.length < 2) return 0;

  // Discretize states and compute empirical entropy
  const binSize = 0.1;
  const counts: Map<string, number> = new Map();

  for (const snap of snapshots) {
    const key = snap.r.map((v) => Math.floor(v / binSize)).join(',');
    counts.set(key, (counts.get(key) || 0) + 1);
  }

  let entropy = 0;
  for (const count of counts.values()) {
    const p = count / snapshots.length;
    if (p > 0) {
      entropy -= p * Math.log(p);
    }
  }

  return entropy;
}

/**
 * Computes conditional entropy
 */
function computeConditionalEntropy(
  future: Array<{ r: number[]; h: number[]; s: number[] }>,
  past: Array<{ r: number[]; h: number[]; s: number[] }>
): number {
  // Simplified: H(future|past) ≈ H(future) - MI(past, future)
  const futureEntropy = computeStateEntropy(future);

  // Mutual information approximation
  const mi = computeMutualInformation(past, future);

  return Math.max(0, futureEntropy - mi);
}

/**
 * Computes mutual information between two state sequences
 */
function computeMutualInformation(
  seq1: Array<{ r: number[]; h: number[]; s: number[] }>,
  seq2: Array<{ r: number[]; h: number[]; s: number[] }>
): number {
  // Use correlation-based proxy
  const n = Math.min(seq1.length, seq2.length);
  if (n < 2) return 0;

  let correlation = 0;
  for (let i = 0; i < n; i++) {
    const v1 = seq1[i].r.reduce((a, b) => a + b, 0) / seq1[i].r.length;
    const v2 = seq2[i].r.reduce((a, b) => a + b, 0) / seq2[i].r.length;
    correlation += v1 * v2;
  }
  correlation /= n;

  // MI ≈ -0.5 * log(1 - ρ²)
  return Math.max(0, -0.5 * Math.log(1 - correlation * correlation + 1e-10));
}

/**
 * Computes information loss from partitioning the system
 */
function computePartitionLoss(state: StateSpaceRHS): number {
  // How much does partitioning R-H-S reduce system information?
  const rEntropy = computeVectorEntropy(state.r.perturbation);
  const hEntropy = computeVectorEntropy(state.h.structure);
  const sEntropy = computeVectorEntropy(state.s.mediationCoefficients);

  const combinedEntropy = computeVectorEntropy([
    ...state.r.perturbation,
    ...state.h.structure,
    ...state.s.mediationCoefficients,
  ]);

  const sumOfParts = rEntropy + hEntropy + sEntropy;
  const synergy = Math.max(0, combinedEntropy - sumOfParts);

  return Math.min(1, synergy / (combinedEntropy + 1e-10));
}

/**
 * Computes entropy of a single vector
 */
function computeVectorEntropy(v: number[]): number {
  const prob = normalizeToProbability(v);
  let entropy = 0;
  for (const p of prob) {
    if (p > 0) {
      entropy -= p * Math.log(p);
    }
  }
  return entropy;
}

/**
 * D22 - Creates Phi estimator
 */
export function createPhiEstimator(state: StateSpaceRHS): PhiEstimator {
  const phi = computePhi(state);

  // Simple partition: split state in half
  const partition = [
    Array.from({ length: Math.floor(state.dimension / 2) }, (_, i) => i),
    Array.from({ length: Math.ceil(state.dimension / 2) }, (_, i) => i + Math.floor(state.dimension / 2)),
  ];

  return {
    type: 'PHI_ESTIMATOR',
    estimate: phi.value,
    partition,
    sharedExclusions: Math.floor(phi.value * state.dimension),
  };
}

/**
 * D23 - Creates Phi window
 */
export function createPhiWindow(blockSize: number = 20): PhiWindow {
  return { blockSize };
}

/**
 * D24 - Default Phi thresholds
 */
export function createPhiThresholds(minPhiForG1: number = 0.1): PhiThresholds {
  return { minPhiForG1 };
}

/**
 * D25 - Detects Phi failures
 */
export function detectPhiFailures(
  estimator: PhiEstimator,
  thresholds: PhiThresholds
): PhiFailures {
  return {
    type: 'PHI_FAILURES',
    isReducible: estimator.estimate < thresholds.minPhiForG1,
    phiValue: estimator.estimate,
  };
}

// ============================================================================
// D26-D31 - Mode Conditions
// ============================================================================

/**
 * Detects mode conditions from state and metrics
 */
export function detectModeConditions(
  state: StateSpaceRHS,
  klFailures: KLFailures,
  freeEnergyFailures: FreeEnergyFailures,
  teFailures: TEFailures
): ModeCondition[] {
  const conditions: ModeCondition[] = [];

  // D26 - Mode Collapse
  if (klFailures.failureType === 'sustained_zero') {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'collapse',
      isActive: true,
      description: 'Tension→0 via averaging (loss of R variance)',
      severity: klFailures.severity,
    });
  }

  // D27 - Mode Conflict
  if (klFailures.failureType === 'sustained_high') {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'conflict',
      isActive: true,
      description: 'Tension high and unmediated (H rejected)',
      severity: klFailures.severity,
    });
  }

  // D28 - Mode Epiphenomenal
  if (teFailures.isEpiphenomenal) {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'epiphenomenal',
      isActive: true,
      description: 'TE≈0; S has no downward influence',
      severity: 1 - teFailures.teValue,
    });
  }

  // D29 - Mode Ossification
  if (freeEnergyFailures.failureType === 'collapse' && state.h.holdingStrength > 0.9) {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'ossification',
      isActive: true,
      description: 'H dominates; R suppressed; novelty stalls',
      severity: state.h.holdingStrength,
    });
  }

  // D30 - Mode Explosion
  if (freeEnergyFailures.failureType === 'over_constraint' && state.r.variance > 2) {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'explosion',
      isActive: true,
      description: 'R dominates; coherence cannot form',
      severity: Math.min(1, state.r.variance / 5),
    });
  }

  // D31 - Mode ModelCollapse
  if (detectModelCollapse(state)) {
    conditions.push({
      type: 'MODE_CONDITION',
      mode: 'model_collapse',
      isActive: true,
      description: 'Self-training reduces variance (model collapse)',
      severity: 0.8,
    });
  }

  return conditions;
}

/**
 * Detects model collapse from state history
 */
function detectModelCollapse(state: StateSpaceRHS): boolean {
  if (state.history.length < 20) return false;

  // Check if variance is monotonically decreasing
  const recentVariances: number[] = [];
  for (let i = state.history.length - 20; i < state.history.length; i++) {
    const variance = state.history[i].r.reduce((s, v) => s + v * v, 0) / state.history[i].r.length;
    recentVariances.push(variance);
  }

  // Check for consistent decrease
  let decreaseCount = 0;
  for (let i = 1; i < recentVariances.length; i++) {
    if (recentVariances[i] < recentVariances[i - 1]) {
      decreaseCount++;
    }
  }

  return decreaseCount > 15; // More than 75% decreasing
}

// ============================================================================
// D32-D33 - Flags
// ============================================================================

/**
 * D32 - Creates near-miss flag
 */
export function createNearMissFlag(gStatus: GStatus): NearMissFlag {
  const failedTests: ('G1' | 'G2' | 'G3')[] = [];

  if (!gStatus.g1.passes) failedTests.push('G1');
  if (!gStatus.g2.passes) failedTests.push('G2');
  if (!gStatus.g3.passes) failedTests.push('G3');

  // Near-miss: passes some but not all
  const isFlagged = failedTests.length > 0 && failedTests.length < 3;

  return {
    type: 'NEAR_MISS_FLAG',
    isFlagged,
    failedTests,
    reviewNotes: isFlagged
      ? `Looks triadic but fails ${failedTests.join(', ')}; tag for review`
      : '',
  };
}

/**
 * D33 - Creates evidence gap flag
 */
export function createEvidenceGapFlag(eAssignment: EAssignment): EvidenceGapFlag {
  const needsUpgrade = eAssignment.tier === 'E0' || eAssignment.tier === 'E1';

  let upgradePath = '';
  if (eAssignment.tier === 'E0') {
    upgradePath = 'Need operational proxies with measurement → E1';
  } else if (eAssignment.tier === 'E1') {
    upgradePath = 'Need computational simulation validation → E2';
  } else if (eAssignment.tier === 'E2') {
    upgradePath = 'Need experimental verification → E3';
  }

  return {
    type: 'EVIDENCE_GAP_FLAG',
    isFlagged: needsUpgrade,
    currentTier: eAssignment.tier,
    upgradePath,
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export {
  normalizeToProbability,
  computeVectorEntropy,
  computeStateEntropy,
  computeMutualInformation,
};
