/**
 * Variational Information Bottleneck (VIB) Implementation
 *
 * The Variational Information Bottleneck provides tractable bounds on the
 * Information Bottleneck objective using variational inference:
 *
 * L_VIB = E_z~q(z|x)[log q(y|z)] - β·KL(q(z|x) || r(z))
 *
 * where:
 * - q(z|x) is the encoder (approximate posterior)
 * - q(y|z) is the decoder (classifier)
 * - r(z) is the prior (typically N(0,I))
 *
 * This module provides:
 * - Variational lower bound computation
 * - Multiple prior distributions
 * - Rate-distortion analysis
 * - Non-parametric MI estimators (MINE, NWJ, InfoNCE)
 *
 * References:
 * - Alemi et al. (2017) "Deep Variational Information Bottleneck"
 * - Belghazi et al. (2018) "Mutual Information Neural Estimation"
 * - Poole et al. (2019) "On Variational Bounds of Mutual Information"
 *
 * @module losses/variational_ib
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface VIBConfig {
  /** Beta parameter for compression-prediction trade-off */
  beta: number;
  /** Dimension of latent representation Z */
  latentDim: number;
  /** Prior type: 'gaussian' | 'vampprior' | 'mog' */
  priorType?: 'gaussian' | 'vampprior' | 'mog';
  /** Number of components for mixture of Gaussians prior */
  mogComponents?: number;
  /** Use free bits (minimum KL per dimension) */
  freeBits?: number;
  /** Epsilon for numerical stability */
  epsilon?: number;
}

export interface VIBMetrics {
  /** Reconstruction/prediction loss E[log q(y|z)] */
  reconstructionLoss: number;
  /** KL divergence KL(q(z|x) || r(z)) */
  klDivergence: number;
  /** Combined VIB loss */
  vibLoss: number;
  /** Rate (bits per sample) */
  rate: number;
  /** Distortion (prediction error) */
  distortion: number;
  /** Current beta */
  beta: number;
}

export interface GaussianParams {
  /** Mean vector */
  mu: number[];
  /** Log variance (for stability) */
  logVar: number[];
}

export interface MIEstimate {
  /** Estimated mutual information */
  mi: number;
  /** Estimation method used */
  method: string;
  /** Confidence bounds if available */
  confidence?: { lower: number; upper: number };
}

// ============================================================================
// Gaussian Utilities
// ============================================================================

/**
 * Sample from Gaussian using reparameterization trick
 *
 * z = μ + σ·ε where ε ~ N(0,I)
 */
export function reparameterize(mu: number[], logVar: number[]): number[] {
  const z: number[] = [];
  for (let i = 0; i < mu.length; i++) {
    const std = Math.exp(0.5 * logVar[i]);
    const eps = gaussianSample();
    z.push(mu[i] + std * eps);
  }
  return z;
}

/**
 * Sample from standard normal using Box-Muller transform
 */
function gaussianSample(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Compute KL divergence between Gaussian q(z|x) and standard normal prior
 *
 * KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
 */
export function gaussianKL(mu: number[], logVar: number[]): number {
  let kl = 0;
  for (let i = 0; i < mu.length; i++) {
    const muSq = mu[i] * mu[i];
    const var_ = Math.exp(logVar[i]);
    kl += -0.5 * (1 + logVar[i] - muSq - var_);
  }
  return kl;
}

/**
 * Compute KL divergence between two Gaussians
 *
 * KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))
 */
export function gaussianKLGeneral(
  mu1: number[],
  logVar1: number[],
  mu2: number[],
  logVar2: number[]
): number {
  let kl = 0;
  for (let i = 0; i < mu1.length; i++) {
    const var1 = Math.exp(logVar1[i]);
    const var2 = Math.exp(logVar2[i]);
    const muDiff = mu2[i] - mu1[i];

    kl += 0.5 * (logVar2[i] - logVar1[i] + var1 / var2 + (muDiff * muDiff) / var2 - 1);
  }
  return kl;
}

/**
 * Log probability under Gaussian
 */
function gaussianLogProb(z: number[], mu: number[], logVar: number[]): number {
  const dim = z.length;
  let logProb = -0.5 * dim * Math.log(2 * Math.PI);

  for (let i = 0; i < dim; i++) {
    const var_ = Math.exp(logVar[i]);
    const diff = z[i] - mu[i];
    logProb -= 0.5 * (logVar[i] + (diff * diff) / var_);
  }

  return logProb;
}

// ============================================================================
// Prior Distributions
// ============================================================================

/**
 * Standard Gaussian prior: N(0, I)
 */
export function standardGaussianLogProb(z: number[]): number {
  const dim = z.length;
  let logProb = -0.5 * dim * Math.log(2 * Math.PI);

  for (let i = 0; i < dim; i++) {
    logProb -= 0.5 * z[i] * z[i];
  }

  return logProb;
}

/**
 * Mixture of Gaussians (MoG) prior
 *
 * p(z) = (1/K) Σ_k N(z; μ_k, I)
 */
export function mogLogProb(z: number[], componentMeans: number[][]): number {
  const K = componentMeans.length;
  if (K === 0) return standardGaussianLogProb(z);

  const logProbs: number[] = [];
  const logK = Math.log(K);

  for (const mu of componentMeans) {
    // Unit variance for simplicity
    const logVar = new Array(z.length).fill(0);
    logProbs.push(gaussianLogProb(z, mu, logVar) - logK);
  }

  // Log-sum-exp for numerical stability
  const maxLogProb = Math.max(...logProbs);
  let sumExp = 0;
  for (const lp of logProbs) {
    sumExp += Math.exp(lp - maxLogProb);
  }

  return maxLogProb + Math.log(sumExp);
}

// ============================================================================
// VIB Loss Computation
// ============================================================================

/**
 * Compute Variational Information Bottleneck loss
 *
 * L_VIB = -E[log q(y|z)] + β·KL(q(z|x) || r(z))
 *
 * @param encoderParams Gaussian parameters from encoder {mu, logVar}
 * @param decoderLogits Output logits from decoder [batchSize, numClasses]
 * @param targets True class indices [batchSize]
 * @param config VIB configuration
 * @returns VIB metrics
 */
export function computeVIBLoss(
  encoderParams: GaussianParams[],
  decoderLogits: number[][],
  targets: number[],
  config: VIBConfig
): VIBMetrics {
  const { beta, freeBits = 0, epsilon = 1e-10 } = config;
  const batchSize = encoderParams.length;

  if (batchSize === 0) {
    return {
      reconstructionLoss: 0,
      klDivergence: 0,
      vibLoss: 0,
      rate: 0,
      distortion: 0,
      beta
    };
  }

  // Compute reconstruction loss: -E[log q(y|z)]
  let reconstructionLoss = 0;
  for (let i = 0; i < batchSize; i++) {
    const logits = decoderLogits[i];
    const target = targets[i];

    // Softmax and cross-entropy
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const logProb = logits[target] - maxLogit - Math.log(sumExp + epsilon);

    reconstructionLoss -= logProb;
  }
  reconstructionLoss /= batchSize;

  // Compute KL divergence: KL(q(z|x) || r(z))
  let klDivergence = 0;
  for (const params of encoderParams) {
    const kl = gaussianKL(params.mu, params.logVar);

    // Free bits: clamp KL per dimension to minimum value
    const dim = params.mu.length;
    if (freeBits > 0) {
      const klPerDim = kl / dim;
      klDivergence += Math.max(freeBits, klPerDim) * dim;
    } else {
      klDivergence += kl;
    }
  }
  klDivergence /= batchSize;

  // Combined VIB loss
  const vibLoss = reconstructionLoss + beta * klDivergence;

  // Rate in nats (convert to bits by dividing by ln(2))
  const rate = klDivergence;

  // Distortion (prediction error)
  const distortion = reconstructionLoss;

  return {
    reconstructionLoss,
    klDivergence,
    vibLoss,
    rate,
    distortion,
    beta
  };
}

// ============================================================================
// Mutual Information Estimation
// ============================================================================

/**
 * MINE (Mutual Information Neural Estimation) lower bound
 *
 * I(X;Y) ≥ E_{p(x,y)}[T(x,y)] - log(E_{p(x)p(y)}[exp(T(x,y))])
 *
 * where T is a "statistics network" (here approximated by dot product)
 *
 * @param joint Joint samples (x_i, y_i)
 * @param marginalX Marginal samples x_i
 * @param marginalY Marginal samples y_i (shuffled)
 * @returns MINE MI estimate
 */
export function estimateMINE(
  joint: { x: number[]; y: number[] }[],
  marginalX: number[][],
  marginalY: number[][]
): MIEstimate {
  if (joint.length === 0) {
    return { mi: 0, method: 'MINE' };
  }

  // Simple statistics function: dot product
  const T = (x: number[], y: number[]): number => {
    let sum = 0;
    for (let i = 0; i < Math.min(x.length, y.length); i++) {
      sum += x[i] * y[i];
    }
    return sum / Math.max(x.length, 1);
  };

  // E[T(x,y)] over joint distribution
  let jointExpectation = 0;
  for (const { x, y } of joint) {
    jointExpectation += T(x, y);
  }
  jointExpectation /= joint.length;

  // E[exp(T(x,y))] over marginals
  let marginalExpectation = 0;
  const n = Math.min(marginalX.length, marginalY.length);
  for (let i = 0; i < n; i++) {
    marginalExpectation += Math.exp(T(marginalX[i], marginalY[i]));
  }
  marginalExpectation /= n;

  // MINE bound
  const mi = jointExpectation - Math.log(marginalExpectation + 1e-10);

  return {
    mi: Math.max(0, mi),
    method: 'MINE',
    confidence: {
      lower: mi - 0.5, // Approximate confidence interval
      upper: mi + 0.5
    }
  };
}

/**
 * InfoNCE (Noise Contrastive Estimation) lower bound
 *
 * I(X;Y) ≥ log(K) - L_NCE
 *
 * where K is the number of negative samples
 *
 * @param anchor Anchor samples X
 * @param positive Positive samples Y (paired with X)
 * @param negatives Negative samples (unpaired)
 * @param temperature Temperature for softmax
 * @returns InfoNCE MI estimate
 */
export function estimateInfoNCE(
  anchor: number[][],
  positive: number[][],
  negatives: number[][],
  temperature: number = 0.1
): MIEstimate {
  const batchSize = anchor.length;
  const numNegatives = negatives.length;

  if (batchSize === 0 || numNegatives === 0) {
    return { mi: 0, method: 'InfoNCE' };
  }

  // Similarity function (cosine similarity)
  const similarity = (a: number[], b: number[]): number => {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA * normB) + 1e-10);
  };

  let totalLoss = 0;

  for (let i = 0; i < batchSize; i++) {
    const posScore = similarity(anchor[i], positive[i]) / temperature;

    // Negative scores
    let negLogSumExp = posScore;
    for (const neg of negatives) {
      const negScore = similarity(anchor[i], neg) / temperature;
      negLogSumExp = logAddExp(negLogSumExp, negScore);
    }

    // NCE loss for this sample
    totalLoss += negLogSumExp - posScore;
  }

  const avgLoss = totalLoss / batchSize;
  const mi = Math.log(numNegatives + 1) - avgLoss;

  return {
    mi: Math.max(0, mi),
    method: 'InfoNCE',
    confidence: {
      lower: mi - Math.sqrt(1 / batchSize),
      upper: mi + Math.sqrt(1 / batchSize)
    }
  };
}

function logAddExp(a: number, b: number): number {
  const max = Math.max(a, b);
  return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
}

/**
 * NWJ (Nguyen-Wainwright-Jordan) f-divergence bound
 *
 * I(X;Y) ≥ E_joint[T] - E_marginal[exp(T-1)]
 *
 * Tighter than MINE for finite samples
 */
export function estimateNWJ(
  joint: { x: number[]; y: number[] }[],
  marginalX: number[][],
  marginalY: number[][]
): MIEstimate {
  if (joint.length === 0) {
    return { mi: 0, method: 'NWJ' };
  }

  const T = (x: number[], y: number[]): number => {
    let sum = 0;
    for (let i = 0; i < Math.min(x.length, y.length); i++) {
      sum += x[i] * y[i];
    }
    return sum / Math.max(x.length, 1);
  };

  let jointExpectation = 0;
  for (const { x, y } of joint) {
    jointExpectation += T(x, y);
  }
  jointExpectation /= joint.length;

  let marginalExpectation = 0;
  const n = Math.min(marginalX.length, marginalY.length);
  for (let i = 0; i < n; i++) {
    marginalExpectation += Math.exp(T(marginalX[i], marginalY[i]) - 1);
  }
  marginalExpectation /= n;

  const mi = jointExpectation - marginalExpectation;

  return {
    mi: Math.max(0, mi),
    method: 'NWJ',
    confidence: {
      lower: mi - 0.3,
      upper: mi + 0.3
    }
  };
}

// ============================================================================
// Rate-Distortion Analysis
// ============================================================================

export interface RateDistortionPoint {
  rate: number;
  distortion: number;
  beta: number;
}

/**
 * Compute rate-distortion curve by sweeping beta
 *
 * @param computeLoss Function that computes VIB loss for given beta
 * @param betaRange Range of beta values to sweep
 * @returns Rate-distortion points
 */
export function computeRateDistortionCurve(
  computeLoss: (beta: number) => VIBMetrics,
  betaRange: { start: number; end: number; steps: number }
): RateDistortionPoint[] {
  const { start, end, steps } = betaRange;
  const points: RateDistortionPoint[] = [];

  for (let i = 0; i <= steps; i++) {
    // Log-space sweep for better coverage
    const t = i / steps;
    const beta = start * Math.pow(end / start, t);

    const metrics = computeLoss(beta);
    points.push({
      rate: metrics.rate,
      distortion: metrics.distortion,
      beta
    });
  }

  return points;
}

/**
 * Find optimal beta for target rate
 *
 * @param computeLoss Function that computes VIB loss
 * @param targetRate Target rate in nats
 * @param tolerance Tolerance for binary search
 * @returns Optimal beta value
 */
export function findOptimalBeta(
  computeLoss: (beta: number) => VIBMetrics,
  targetRate: number,
  tolerance: number = 0.01
): number {
  let betaLow = 1e-4;
  let betaHigh = 100;

  // Binary search for beta that achieves target rate
  for (let iter = 0; iter < 50; iter++) {
    const betaMid = Math.sqrt(betaLow * betaHigh);
    const metrics = computeLoss(betaMid);

    if (Math.abs(metrics.rate - targetRate) < tolerance) {
      return betaMid;
    }

    if (metrics.rate > targetRate) {
      // Rate too high, increase beta to compress more
      betaLow = betaMid;
    } else {
      // Rate too low, decrease beta
      betaHigh = betaMid;
    }
  }

  return Math.sqrt(betaLow * betaHigh);
}

// ============================================================================
// Beta Scheduling with Warm-up
// ============================================================================

export interface BetaScheduleConfig {
  /** Initial beta value */
  betaInit: number;
  /** Final beta value */
  betaFinal: number;
  /** Warm-up epochs (constant at betaInit) */
  warmupEpochs: number;
  /** Annealing epochs */
  annealingEpochs: number;
  /** Schedule type: 'linear' | 'exponential' | 'cosine' | 'cyclical' */
  scheduleType: 'linear' | 'exponential' | 'cosine' | 'cyclical';
  /** Number of cycles for cyclical annealing */
  numCycles?: number;
}

/**
 * Get beta value with warm-up and annealing schedule
 */
export function getBetaWithWarmup(epoch: number, config: BetaScheduleConfig): number {
  const {
    betaInit,
    betaFinal,
    warmupEpochs,
    annealingEpochs,
    scheduleType,
    numCycles = 4
  } = config;

  // Warm-up phase
  if (epoch < warmupEpochs) {
    return betaInit;
  }

  // Post-warmup progress
  const annealingProgress = Math.min(1, (epoch - warmupEpochs) / annealingEpochs);

  switch (scheduleType) {
    case 'linear':
      return betaInit + (betaFinal - betaInit) * annealingProgress;

    case 'exponential':
      if (betaInit <= 0) return betaFinal * annealingProgress;
      return betaInit * Math.pow(betaFinal / betaInit, annealingProgress);

    case 'cosine':
      return betaFinal + ((betaInit - betaFinal) * (1 + Math.cos(Math.PI * annealingProgress))) / 2;

    case 'cyclical': {
      // Cyclical annealing: beta oscillates between betaInit and betaFinal
      const cycleProgress = (annealingProgress * numCycles) % 1;
      return betaInit + ((betaFinal - betaInit) * (1 - Math.cos(Math.PI * cycleProgress))) / 2;
    }

    default:
      return betaInit;
  }
}

// ============================================================================
// VIB Diagnostics
// ============================================================================

export interface VIBDiagnostics {
  /** Average KL per dimension */
  klPerDimension: number;
  /** Variance collapse indicator (many dims with near-zero KL) */
  varianceCollapse: number;
  /** Information utilization (effective dimensions used) */
  effectiveDimensions: number;
  /** Posterior collapse warning */
  posteriorCollapse: boolean;
  /** Recommendations */
  recommendations: string[];
}

/**
 * Diagnose VIB training issues
 */
export function diagnoseVIB(
  encoderParams: GaussianParams[],
  metrics: VIBMetrics,
  _config: VIBConfig
): VIBDiagnostics {
  const recommendations: string[] = [];

  if (encoderParams.length === 0) {
    return {
      klPerDimension: 0,
      varianceCollapse: 0,
      effectiveDimensions: 0,
      posteriorCollapse: false,
      recommendations: ['No encoder parameters provided']
    };
  }

  const latentDim = encoderParams[0].mu.length;
  const klPerDimension = metrics.klDivergence / latentDim;

  // Check for variance collapse per dimension
  let collapsedDims = 0;
  for (const params of encoderParams) {
    for (const lv of params.logVar) {
      // Near-zero variance (close to prior)
      if (Math.abs(lv) < 0.1) {
        collapsedDims++;
      }
    }
  }
  const varianceCollapse = collapsedDims / (encoderParams.length * latentDim);

  // Effective dimensions: count dims with significant KL
  let effectiveDims = 0;
  const dimKL = new Array(latentDim).fill(0);
  for (const params of encoderParams) {
    for (let i = 0; i < latentDim; i++) {
      const muSq = params.mu[i] * params.mu[i];
      const var_ = Math.exp(params.logVar[i]);
      dimKL[i] += -0.5 * (1 + params.logVar[i] - muSq - var_);
    }
  }
  for (let i = 0; i < latentDim; i++) {
    if (dimKL[i] / encoderParams.length > 0.1) {
      effectiveDims++;
    }
  }

  // Posterior collapse: KL too small
  const posteriorCollapse = metrics.klDivergence < 0.1;

  // Generate recommendations
  if (posteriorCollapse) {
    recommendations.push('Posterior collapse detected: reduce beta or use free bits');
  }

  if (varianceCollapse > 0.5) {
    recommendations.push('Many latent dimensions collapsed: consider reducing latent dimension');
  }

  if (effectiveDims < latentDim / 4) {
    recommendations.push(
      `Only ${effectiveDims}/${latentDim} dimensions used: consider smaller latent space`
    );
  }

  if (metrics.klDivergence > 100) {
    recommendations.push('Very high KL: increase beta or add KL regularization');
  }

  return {
    klPerDimension,
    varianceCollapse,
    effectiveDimensions: effectiveDims,
    posteriorCollapse,
    recommendations
  };
}

// ============================================================================
// Export utilities
// ============================================================================

export const VIB_DEFAULTS: VIBConfig = {
  beta: 1.0,
  latentDim: 64,
  priorType: 'gaussian',
  freeBits: 0,
  epsilon: 1e-10
};

export const VIB_BETA_SCHEDULE_DEFAULTS: BetaScheduleConfig = {
  betaInit: 0.01,
  betaFinal: 1.0,
  warmupEpochs: 5,
  annealingEpochs: 45,
  scheduleType: 'cosine',
  numCycles: 4
};
