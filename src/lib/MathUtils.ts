/**
 * Advanced Mathematical Utilities for Neural Networks
 *
 * This module provides rigorous mathematical implementations for:
 * - Advanced weight initialization (Xavier/He)
 * - Improved activation functions (LeakyReLU, ELU, GELU)
 * - Learning rate scheduling (Cosine Annealing, Exponential Decay)
 * - Numerical stability utilities
 * - Advanced sampling methods
 */

// ============================================================================
// Weight Initialization Methods
// ============================================================================

/**
 * Xavier/Glorot initialization for symmetric activation functions (tanh, sigmoid)
 *
 * Mathematical basis: Var(W) = 2 / (fan_in + fan_out)
 *
 * Reference: Glorot & Bengio (2010) - "Understanding the difficulty of training
 * deep feedforward neural networks"
 *
 * @param fanIn - Number of input units
 * @param fanOut - Number of output units
 * @param rng - Random number generator function returning values in [0,1)
 * @returns Random value from Xavier distribution
 */
export function xavierInit(fanIn: number, fanOut: number, rng: () => number): number {
  // Variance = 2 / (fan_in + fan_out)
  const variance = 2.0 / (fanIn + fanOut);
  const stddev = Math.sqrt(variance);

  // Box-Muller transform for normal distribution
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const normal = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

  return normal * stddev;
}

/**
 * He initialization for ReLU-based activation functions
 *
 * Mathematical basis: Var(W) = 2 / fan_in
 *
 * Reference: He et al. (2015) - "Delving Deep into Rectifiers: Surpassing
 * Human-Level Performance on ImageNet Classification"
 *
 * @param fanIn - Number of input units
 * @param rng - Random number generator function
 * @returns Random value from He distribution
 */
export function heInit(fanIn: number, rng: () => number): number {
  // Variance = 2 / fan_in (optimal for ReLU)
  const variance = 2.0 / fanIn;
  const stddev = Math.sqrt(variance);

  // Box-Muller transform
  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const normal = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

  return normal * stddev;
}

/**
 * LeCun initialization (used for SELU activation)
 *
 * Mathematical basis: Var(W) = 1 / fan_in
 *
 * @param fanIn - Number of input units
 * @param rng - Random number generator function
 * @returns Random value from LeCun distribution
 */
export function lecunInit(fanIn: number, rng: () => number): number {
  const variance = 1.0 / fanIn;
  const stddev = Math.sqrt(variance);

  let u = 0,
    v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const normal = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);

  return normal * stddev;
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Leaky ReLU activation function
 *
 * f(x) = max(αx, x) where α is a small positive constant (typically 0.01)
 *
 * Advantages over ReLU:
 * - Prevents dying ReLU problem
 * - Allows gradient flow for negative inputs
 *
 * @param x - Input value
 * @param alpha - Slope for negative values (default: 0.01)
 * @returns Activated value
 */
export function leakyRelu(x: number, alpha = 0.01): number {
  return x > 0 ? x : alpha * x;
}

/**
 * Derivative of Leaky ReLU
 *
 * f'(x) = 1 if x > 0, else α
 */
export function leakyReluDerivative(x: number, alpha = 0.01): number {
  return x > 0 ? 1 : alpha;
}

/**
 * Exponential Linear Unit (ELU) activation function
 *
 * f(x) = x if x > 0, else α(e^x - 1)
 *
 * Advantages:
 * - Smooth function everywhere
 * - Negative saturation reduces variance shift
 * - Self-normalizing properties
 *
 * Reference: Clevert et al. (2015) - "Fast and Accurate Deep Network Learning
 * by Exponential Linear Units"
 *
 * @param x - Input value
 * @param alpha - Controls the saturation point (default: 1.0)
 * @returns Activated value
 */
export function elu(x: number, alpha = 1.0): number {
  return x > 0 ? x : alpha * (Math.exp(x) - 1);
}

/**
 * Derivative of ELU
 *
 * f'(x) = 1 if x > 0, else α * e^x
 */
export function eluDerivative(x: number, alpha = 1.0): number {
  return x > 0 ? 1 : alpha * Math.exp(x);
}

/**
 * Gaussian Error Linear Unit (GELU) activation function
 *
 * f(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
 *
 * Approximation: f(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
 *
 * Used in BERT and GPT models
 *
 * Reference: Hendrycks & Gimpel (2016) - "Gaussian Error Linear Units"
 *
 * @param x - Input value
 * @returns Activated value
 */
export function gelu(x: number): number {
  // Fast approximation using tanh
  const sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
  const inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + Math.tanh(inner));
}

/**
 * Derivative of GELU (approximation)
 */
export function geluDerivative(x: number): number {
  const sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
  const inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
  const tanhVal = Math.tanh(inner);
  const sech2 = 1 - tanhVal * tanhVal;

  const dInner = sqrt2OverPi * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + tanhVal) + 0.5 * x * sech2 * dInner;
}

/**
 * Swish activation function (also called SiLU - Sigmoid Linear Unit)
 *
 * f(x) = x * σ(βx) where σ is sigmoid function
 *
 * @param x - Input value
 * @param beta - Scaling parameter (default: 1.0)
 * @returns Activated value
 */
export function swish(x: number, beta = 1.0): number {
  return x / (1 + Math.exp(-beta * x));
}

// ============================================================================
// Learning Rate Scheduling
// ============================================================================

/**
 * Cosine Annealing learning rate schedule
 *
 * η_t = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
 *
 * Reference: Loshchilov & Hutter (2016) - "SGDR: Stochastic Gradient Descent
 * with Warm Restarts"
 *
 * @param epoch - Current epoch (0-indexed)
 * @param totalEpochs - Total number of epochs
 * @param lrMax - Maximum learning rate
 * @param lrMin - Minimum learning rate (default: 0)
 * @returns Learning rate for current epoch
 */
export function cosineAnnealingLR(
  epoch: number,
  totalEpochs: number,
  lrMax: number,
  lrMin = 0
): number {
  if (totalEpochs <= 0) return lrMax;
  const progress = Math.min(epoch / totalEpochs, 1.0);
  return lrMin + (lrMax - lrMin) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

/**
 * Exponential decay learning rate schedule
 *
 * η_t = η_0 * γ^t
 *
 * @param epoch - Current epoch
 * @param lrInitial - Initial learning rate
 * @param decayRate - Decay rate (typically 0.9-0.99)
 * @returns Learning rate for current epoch
 */
export function exponentialDecayLR(epoch: number, lrInitial: number, decayRate: number): number {
  return lrInitial * Math.pow(decayRate, epoch);
}

/**
 * Step decay learning rate schedule
 *
 * η_t = η_0 * γ^⌊t/s⌋
 *
 * @param epoch - Current epoch
 * @param lrInitial - Initial learning rate
 * @param dropRate - Rate to drop LR by (typically 0.5)
 * @param epochsPerDrop - Number of epochs between drops
 * @returns Learning rate for current epoch
 */
export function stepDecayLR(
  epoch: number,
  lrInitial: number,
  dropRate: number,
  epochsPerDrop: number
): number {
  const drops = Math.floor(epoch / epochsPerDrop);
  return lrInitial * Math.pow(dropRate, drops);
}

/**
 * Warmup + Cosine Annealing schedule (used in transformers)
 *
 * Linear warmup for first warmup_epochs, then cosine decay
 *
 * @param epoch - Current epoch
 * @param totalEpochs - Total number of epochs
 * @param lrMax - Maximum learning rate
 * @param warmupEpochs - Number of warmup epochs
 * @returns Learning rate for current epoch
 */
export function warmupCosineAnnealingLR(
  epoch: number,
  totalEpochs: number,
  lrMax: number,
  warmupEpochs: number
): number {
  if (epoch < warmupEpochs) {
    // Linear warmup
    return (lrMax * epoch) / warmupEpochs;
  }
  // Cosine annealing after warmup
  const effectiveEpoch = epoch - warmupEpochs;
  const effectiveTotal = totalEpochs - warmupEpochs;
  return cosineAnnealingLR(effectiveEpoch, effectiveTotal, lrMax, 0);
}

// ============================================================================
// Numerical Stability Utilities
// ============================================================================

/**
 * Numerically stable log-sum-exp
 *
 * log(Σ exp(x_i)) = m + log(Σ exp(x_i - m)) where m = max(x_i)
 *
 * Prevents overflow in softmax calculations
 *
 * @param values - Array of values
 * @returns log(sum(exp(values)))
 */
export function logSumExp(values: number[]): number {
  if (values.length === 0) return -Infinity;

  const maxVal = Math.max(...values);
  if (!isFinite(maxVal)) return maxVal;

  let sum = 0;
  for (const v of values) {
    sum += Math.exp(v - maxVal);
  }

  return maxVal + Math.log(sum);
}

/**
 * Numerically stable softmax with temperature
 *
 * @param logits - Input logits
 * @param temperature - Temperature parameter (default: 1.0)
 * @returns Probability distribution
 */
export function stableSoftmax(logits: number[], temperature = 1.0): number[] {
  if (logits.length === 0) return [];

  const T = Math.max(temperature, 1e-8); // Prevent division by zero
  const scaled = logits.map((x) => x / T);
  const normalization = logSumExp(scaled);
  return scaled.map((x) => Math.exp(x - normalization));
}

/**
 * Numerically stable log-softmax
 *
 * log(softmax(x_i)) = x_i - log(Σ exp(x_j))
 *
 * More stable than log(softmax(x))
 *
 * @param logits - Input logits
 * @returns Log probabilities
 */
export function logSoftmax(logits: number[]): number[] {
  const lse = logSumExp(logits);
  return logits.map((x) => x - lse);
}

/**
 * Clip value to prevent numerical overflow/underflow
 *
 * @param value - Value to clip
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clipped value
 */
export function clip(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Safe division with epsilon to prevent divide-by-zero
 *
 * @param numerator - Numerator
 * @param denominator - Denominator
 * @param eps - Small constant to add to denominator
 * @returns numerator / (denominator + eps)
 */
export function safeDivide(numerator: number, denominator: number, eps = 1e-10): number {
  return numerator / (denominator + eps);
}

// ============================================================================
// Layer Normalization
// ============================================================================

/**
 * Layer Normalization
 *
 * y = γ * (x - μ) / √(σ² + ε) + β
 *
 * Reference: Ba et al. (2016) - "Layer Normalization"
 *
 * @param x - Input vector
 * @param gamma - Scale parameter (learned)
 * @param beta - Shift parameter (learned)
 * @param eps - Small constant for numerical stability
 * @returns Normalized vector
 */
export function layerNorm(x: number[], gamma: number[], beta: number[], eps = 1e-5): number[] {
  const n = x.length;

  // Compute mean
  const mean = x.reduce((sum, val) => sum + val, 0) / n;

  // Compute variance
  const variance = x.reduce((sum, val) => sum + (val - mean) ** 2, 0) / n;

  // Normalize and scale
  const std = Math.sqrt(variance + eps);
  return x.map((val, i) => gamma[i] * ((val - mean) / std) + beta[i]);
}

/**
 * Compute gradients for Layer Normalization
 *
 * @param dOut - Gradient from next layer
 * @param x - Original input
 * @param gamma - Scale parameter
 * @param eps - Small constant
 * @returns Gradients for input, gamma, and beta
 */
export function layerNormBackward(
  dOut: number[],
  x: number[],
  gamma: number[],
  eps = 1e-5
): { dx: number[]; dGamma: number[]; dBeta: number[] } {
  const n = x.length;

  const mean = x.reduce((sum, val) => sum + val, 0) / n;
  const variance = x.reduce((sum, val) => sum + (val - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance + eps);

  const xHat = x.map((val) => (val - mean) / std);

  // Gradient w.r.t. gamma and beta
  const dGamma = dOut.map((d, i) => d * xHat[i]);
  const dBeta = [...dOut];

  // Gradient w.r.t. x (complex due to normalization)
  const dXhat = dOut.map((d, i) => d * gamma[i]);
  const dVar = (dXhat.reduce((sum, d, i) => sum + d * (x[i] - mean), 0) * -0.5) / Math.pow(std, 3);
  const dMean =
    dXhat.reduce((sum, d) => sum - d / std, 0) +
    (dVar * x.reduce((sum, val) => sum - 2 * (val - mean), 0)) / n;

  const dx = dXhat.map((d, i) => d / std + (dVar * 2 * (x[i] - mean)) / n + dMean / n);

  return { dx, dGamma, dBeta };
}

// ============================================================================
// Advanced Sampling Methods
// ============================================================================

export type BeamCandidate = {
  tokens: number[];
  score: number;
  probabilities: number[];
};

/**
 * Beam Search decoding
 *
 * Maintains top-k most likely sequences at each step
 *
 * @param initialContext - Starting context tokens
 * @param beamWidth - Number of beams to maintain
 * @param maxLength - Maximum sequence length
 * @param forwardFn - Function to compute next-token probabilities
 * @param eosToken - End-of-sequence token index
 * @returns Best sequence found
 */
export function beamSearch(
  initialContext: number[],
  beamWidth: number,
  maxLength: number,
  forwardFn: (context: number[]) => number[],
  eosToken: number
): BeamCandidate {
  // Initialize beam with starting context
  let beams: BeamCandidate[] = [
    {
      tokens: [...initialContext],
      score: 0,
      probabilities: []
    }
  ];

  for (let step = 0; step < maxLength; step++) {
    const candidates: BeamCandidate[] = [];

    // Expand each beam
    for (const beam of beams) {
      // Don't expand if already ended
      if (beam.tokens[beam.tokens.length - 1] === eosToken) {
        candidates.push(beam);
        continue;
      }

      // Get probabilities for next token
      const probs = forwardFn(beam.tokens);

      // Get top-k tokens
      const topK = getTopKIndices(probs, beamWidth);

      // Create new candidates
      for (const tokenIdx of topK) {
        const logProb = Math.log(probs[tokenIdx] + 1e-10);
        candidates.push({
          tokens: [...beam.tokens, tokenIdx],
          score: beam.score + logProb,
          probabilities: [...beam.probabilities, probs[tokenIdx]]
        });
      }
    }

    // Keep top beamWidth candidates
    candidates.sort((a, b) => b.score - a.score);
    beams = candidates.slice(0, beamWidth);

    // Check if all beams have ended
    if (beams.every((b) => b.tokens[b.tokens.length - 1] === eosToken)) {
      break;
    }
  }

  // Return best beam
  return beams[0];
}

/**
 * Get indices of top-k values in array
 *
 * @param arr - Input array
 * @param k - Number of top values to return
 * @returns Indices of top-k values
 */
export function getTopKIndices(arr: number[], k: number): number[] {
  const indices = arr.map((_, i) => i);
  indices.sort((a, b) => arr[b] - arr[a]);
  return indices.slice(0, Math.min(k, arr.length));
}

/**
 * Nucleus (top-p) sampling with improved numerical stability
 *
 * @param probs - Probability distribution
 * @param p - Cumulative probability threshold
 * @param rng - Random number generator
 * @returns Sampled token index
 */
export function nucleusSampling(probs: number[], p: number, rng: () => number): number {
  const indices = probs.map((_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);

  let cumProb = 0;
  const nucleus: number[] = [];

  for (const idx of indices) {
    nucleus.push(idx);
    cumProb += probs[idx];
    if (cumProb >= p) break;
  }

  // Renormalize probabilities within nucleus
  const nucleusProbs = nucleus.map((i) => probs[i]);
  const sum = nucleusProbs.reduce((a, b) => a + b, 0);
  const normalized = nucleusProbs.map((p) => p / sum);

  // Sample from renormalized distribution
  let r = rng();
  for (let i = 0; i < nucleus.length; i++) {
    r -= normalized[i];
    if (r <= 0) return nucleus[i];
  }

  return nucleus[nucleus.length - 1];
}
