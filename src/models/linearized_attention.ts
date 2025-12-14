/**
 * Linearized Attention via Kernel Features
 *
 * Replaces standard softmax attention (O(n²)) with kernel-based attention (O(n)).
 * Uses feature maps φ(x) such that K(q,k) = φ(q)^T φ(k) approximates exp(q·k/√d).
 *
 * Supported kernel approximations:
 * - Random Fourier Features (RFF): Approximates RBF kernel
 * - Positive Random Features (FAVOR+): Non-negative features for attention
 * - Chebyshev Polynomial Features: Polynomial approximation
 * - ELU Features: Simple elu(x) + 1 mapping (Katharopoulos et al., 2020)
 *
 * References:
 * - Katharopoulos et al. (2020) "Transformers are RNNs"
 * - Choromanski et al. (2021) "Rethinking Attention with Performers"
 * - Peng et al. (2021) "Random Feature Attention"
 *
 * @module linearized_attention
 */

import type { Matrix } from './attention';

// ============================================================================
// Types and Interfaces
// ============================================================================

export type KernelType = 'rff' | 'favor' | 'chebyshev' | 'elu';

export interface LinearAttentionConfig {
  /** Kernel type for feature mapping. Default: 'elu' */
  kernelType: KernelType;
  /** Number of random features for RFF/FAVOR+. Default: 256 */
  numFeatures?: number;
  /** Number of Chebyshev polynomial terms. Default: 8 */
  chebyshevDegree?: number;
  /** Random seed for reproducibility. Default: 42 */
  seed?: number;
  /** Epsilon for numerical stability. Default: 1e-6 */
  epsilon?: number;
  /** Enable causal masking for autoregressive decoding. Default: false */
  causal?: boolean;
}

export interface LinearAttentionResult {
  /** Output matrix [seq_len, value_dim] */
  output: Matrix;
  /** Approximate attention weights (optional, computed only if requested) */
  attentionApprox?: Matrix;
  /** Computational statistics */
  stats: {
    /** Time complexity indicator */
    complexity: string;
    /** Feature dimension used */
    featureDim: number;
    /** Kernel type used */
    kernelType: KernelType;
  };
}

// ============================================================================
// Random Number Generation (Seeded)
// ============================================================================

class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  /** Returns a random number in [0, 1) */
  next(): number {
    // Linear congruential generator (Numerical Recipes)
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  /** Box-Muller transform for Gaussian samples */
  gaussian(): number {
    const u1 = this.next();
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }

  /** Generate Gaussian random matrix */
  gaussianMatrix(rows: number, cols: number): Matrix {
    const result: Matrix = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        row.push(this.gaussian());
      }
      result.push(row);
    }
    return result;
  }
}

// ============================================================================
// Feature Map Functions
// ============================================================================

/**
 * ELU-based feature map: φ(x) = elu(x) + 1
 * Simple, fast, and doesn't require random projections.
 * Reference: Katharopoulos et al. (2020) "Transformers are RNNs"
 */
function eluFeatureMap(x: number[], epsilon: number): number[] {
  return x.map((v) => (v > 0 ? v + 1 : Math.exp(v) + epsilon));
}

/**
 * Random Fourier Features (RFF) for RBF kernel approximation.
 * K(x, y) ≈ φ(x)^T φ(y) where φ(x) = [cos(ωx), sin(ωx)] / √D
 *
 * @param x Input vector
 * @param omega Random projection matrix [numFeatures, inputDim]
 * @returns Feature vector [2 * numFeatures]
 */
function rffFeatureMap(x: number[], omega: Matrix): number[] {
  const features: number[] = [];
  const scale = 1 / Math.sqrt(omega.length);

  for (const row of omega) {
    // Compute dot product x · ω
    let dot = 0;
    for (let i = 0; i < x.length && i < row.length; i++) {
      dot += x[i] * row[i];
    }
    features.push(Math.cos(dot) * scale);
    features.push(Math.sin(dot) * scale);
  }

  return features;
}

/**
 * FAVOR+ (Fast Attention Via positive Orthogonal Random features)
 * Uses positive random features for non-negative attention approximation.
 *
 * φ(x) = exp(ωx - ||x||²/2) for softmax approximation
 *
 * Reference: Choromanski et al. (2021) "Rethinking Attention with Performers"
 */
function favorFeatureMap(x: number[], omega: Matrix, epsilon: number): number[] {
  // Compute ||x||² / 2
  let normSq = 0;
  for (const v of x) {
    normSq += v * v;
  }
  const normHalf = normSq / 2;

  const features: number[] = [];
  const scale = 1 / Math.sqrt(omega.length);

  for (const row of omega) {
    // Compute dot product x · ω
    let dot = 0;
    for (let i = 0; i < x.length && i < row.length; i++) {
      dot += x[i] * row[i];
    }
    // Positive feature: exp(ωx - ||x||²/2)
    const feature = Math.exp(dot - normHalf);
    features.push(Math.max(feature, epsilon) * scale);
  }

  return features;
}

/**
 * Chebyshev Polynomial Features
 * Uses Chebyshev polynomials T_n(x) for polynomial kernel approximation.
 *
 * T_0(x) = 1, T_1(x) = x, T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
 */
function chebyshevFeatureMap(x: number[], degree: number): number[] {
  const features: number[] = [];

  for (const v of x) {
    // Clamp to [-1, 1] for Chebyshev stability
    const clamped = Math.max(-1, Math.min(1, v));

    // Generate Chebyshev polynomials
    let T_prev = 1; // T_0
    let T_curr = clamped; // T_1

    features.push(T_prev); // T_0
    if (degree > 0) {
      features.push(T_curr); // T_1
    }

    for (let n = 2; n <= degree; n++) {
      const T_next = 2 * clamped * T_curr - T_prev;
      features.push(T_next);
      T_prev = T_curr;
      T_curr = T_next;
    }
  }

  return features;
}

// ============================================================================
// Core Linear Attention Implementation
// ============================================================================

/**
 * Apply feature map to entire matrix.
 */
function applyFeatureMap(
  matrix: Matrix,
  config: LinearAttentionConfig,
  omega: Matrix | null
): Matrix {
  const epsilon = config.epsilon ?? 1e-6;

  return matrix.map((row) => {
    switch (config.kernelType) {
      case 'elu':
        return eluFeatureMap(row, epsilon);
      case 'rff':
        if (!omega) throw new Error('RFF requires omega matrix');
        return rffFeatureMap(row, omega);
      case 'favor':
        if (!omega) throw new Error('FAVOR+ requires omega matrix');
        return favorFeatureMap(row, omega, epsilon);
      case 'chebyshev':
        return chebyshevFeatureMap(row, config.chebyshevDegree ?? 8);
      default:
        return eluFeatureMap(row, epsilon);
    }
  });
}

/**
 * Compute outer product sum: Σ_i φ(k_i) ⊗ v_i
 * Results in a matrix of shape [feature_dim, value_dim]
 */
function computeKVProduct(keys: Matrix, values: Matrix): Matrix {
  const featureDim = keys[0].length;
  const valueDim = values[0].length;
  const result: Matrix = Array.from({ length: featureDim }, () => new Array(valueDim).fill(0));

  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const v = values[i];
    for (let f = 0; f < featureDim; f++) {
      for (let d = 0; d < valueDim; d++) {
        result[f][d] += k[f] * v[d];
      }
    }
  }

  return result;
}

/**
 * Compute normalizer: Σ_i φ(k_i)
 */
function computeNormalizer(keys: Matrix): number[] {
  const featureDim = keys[0].length;
  const normalizer = new Array(featureDim).fill(0);

  for (const k of keys) {
    for (let f = 0; f < featureDim; f++) {
      normalizer[f] += k[f];
    }
  }

  return normalizer;
}

/**
 * Non-causal linear attention (full context).
 *
 * Attention(Q, K, V) = (φ(Q) * (φ(K)^T V)) / (φ(Q) * Σφ(K))
 *
 * Complexity: O(n * d * D) where D is feature dimension
 */
function linearAttentionNonCausal(
  queries: Matrix,
  keys: Matrix,
  values: Matrix,
  epsilon: number
): Matrix {
  // Precompute KV product: [feature_dim, value_dim]
  const kvProduct = computeKVProduct(keys, values);

  // Precompute normalizer: [feature_dim]
  const normalizer = computeNormalizer(keys);

  // Compute output for each query
  const output: Matrix = [];

  for (const q of queries) {
    const outRow: number[] = new Array(values[0].length).fill(0);

    // Compute denominator: q · normalizer
    let denom = 0;
    for (let f = 0; f < q.length; f++) {
      denom += q[f] * normalizer[f];
    }
    denom = Math.max(denom, epsilon);

    // Compute numerator: q · KV^T for each value dimension
    for (let d = 0; d < values[0].length; d++) {
      let numer = 0;
      for (let f = 0; f < q.length; f++) {
        numer += q[f] * kvProduct[f][d];
      }
      outRow[d] = numer / denom;
    }

    output.push(outRow);
  }

  return output;
}

/**
 * Causal linear attention with cumulative sums.
 *
 * For autoregressive decoding, we need:
 * Attention(q_i, K_{1:i}, V_{1:i}) = (φ(q_i) * Σ_{j≤i} φ(k_j) ⊗ v_j) / (φ(q_i) * Σ_{j≤i} φ(k_j))
 *
 * This can be computed in O(n) using cumulative sums.
 */
function linearAttentionCausal(
  queries: Matrix,
  keys: Matrix,
  values: Matrix,
  epsilon: number
): Matrix {
  const seqLen = queries.length;
  const featureDim = keys[0].length;
  const valueDim = values[0].length;

  // Cumulative KV product
  const cumKV: Matrix = Array.from({ length: featureDim }, () => new Array(valueDim).fill(0));

  // Cumulative normalizer
  const cumNorm: number[] = new Array(featureDim).fill(0);

  const output: Matrix = [];

  for (let i = 0; i < seqLen; i++) {
    const k = keys[i];
    const v = values[i];
    const q = queries[i];

    // Update cumulative KV product
    for (let f = 0; f < featureDim; f++) {
      for (let d = 0; d < valueDim; d++) {
        cumKV[f][d] += k[f] * v[d];
      }
      cumNorm[f] += k[f];
    }

    // Compute output for this position
    const outRow: number[] = new Array(valueDim).fill(0);

    // Denominator
    let denom = 0;
    for (let f = 0; f < featureDim; f++) {
      denom += q[f] * cumNorm[f];
    }
    denom = Math.max(denom, epsilon);

    // Numerator
    for (let d = 0; d < valueDim; d++) {
      let numer = 0;
      for (let f = 0; f < featureDim; f++) {
        numer += q[f] * cumKV[f][d];
      }
      outRow[d] = numer / denom;
    }

    output.push(outRow);
  }

  return output;
}

// ============================================================================
// Main API
// ============================================================================

/**
 * Linearized Attention via Kernel Features
 *
 * Replaces O(n²) softmax attention with O(n) kernel-based attention.
 *
 * @param queries Query matrix [seq_len, dim]
 * @param keys Key matrix [seq_len, dim]
 * @param values Value matrix [seq_len, value_dim]
 * @param config Configuration options
 * @returns LinearAttentionResult with output and statistics
 *
 * @example
 * ```typescript
 * const result = linearizedAttention(queries, keys, values, {
 *   kernelType: 'elu',
 *   causal: true
 * });
 * console.log('Output shape:', result.output.length, result.output[0].length);
 * console.log('Complexity:', result.stats.complexity);
 * ```
 */
export function linearizedAttention(
  queries: Matrix,
  keys: Matrix,
  values: Matrix,
  config: LinearAttentionConfig
): LinearAttentionResult {
  const {
    kernelType = 'elu',
    numFeatures = 256,
    chebyshevDegree = 8,
    seed = 42,
    epsilon = 1e-6,
    causal = false
  } = config;

  // Validate inputs
  if (queries.length === 0 || keys.length === 0 || values.length === 0) {
    throw new Error('Input matrices cannot be empty');
  }
  if (queries.length !== keys.length || keys.length !== values.length) {
    throw new Error('Sequence lengths must match');
  }

  const inputDim = queries[0].length;

  // Generate random projection matrix for RFF/FAVOR+
  let omega: Matrix | null = null;
  if (kernelType === 'rff' || kernelType === 'favor') {
    const rng = new SeededRandom(seed);
    omega = rng.gaussianMatrix(numFeatures, inputDim);
  }

  // Apply feature maps
  const phi_q = applyFeatureMap(queries, { ...config, kernelType }, omega);
  const phi_k = applyFeatureMap(keys, { ...config, kernelType }, omega);

  // Compute attention
  const output = causal
    ? linearAttentionCausal(phi_q, phi_k, values, epsilon)
    : linearAttentionNonCausal(phi_q, phi_k, values, epsilon);

  // Determine feature dimension
  let featureDim: number;
  switch (kernelType) {
    case 'elu':
      featureDim = inputDim;
      break;
    case 'rff':
      featureDim = numFeatures * 2;
      break;
    case 'favor':
      featureDim = numFeatures;
      break;
    case 'chebyshev':
      featureDim = inputDim * (chebyshevDegree + 1);
      break;
    default:
      featureDim = inputDim;
  }

  return {
    output,
    stats: {
      complexity: `O(n × d × ${featureDim})`,
      featureDim,
      kernelType
    }
  };
}

/**
 * Multi-head Linearized Attention
 *
 * Applies linearized attention across multiple heads with GQA support.
 */
export interface MultiHeadLinearAttentionConfig extends LinearAttentionConfig {
  /** Number of attention heads */
  numHeads: number;
  /** Number of KV heads for GQA. Default: same as numHeads */
  numKVHeads?: number;
  /** Model dimension (must be divisible by numHeads) */
  modelDim: number;
}

export function multiHeadLinearizedAttention(
  queries: Matrix,
  keys: Matrix,
  values: Matrix,
  config: MultiHeadLinearAttentionConfig
): LinearAttentionResult {
  const { numHeads, numKVHeads = numHeads, modelDim, ...attentionConfig } = config;

  if (modelDim % numHeads !== 0) {
    throw new Error('modelDim must be divisible by numHeads');
  }
  if (numHeads % numKVHeads !== 0) {
    throw new Error('numHeads must be divisible by numKVHeads');
  }

  const headDim = modelDim / numHeads;
  const kvGroupSize = numHeads / numKVHeads;

  // Split into heads
  const splitHeads = (matrix: Matrix, nHeads: number): Matrix[] => {
    const heads: Matrix[] = [];
    for (let h = 0; h < nHeads; h++) {
      heads.push(matrix.map((row) => row.slice(h * headDim, (h + 1) * headDim)));
    }
    return heads;
  };

  const queryHeads = splitHeads(queries, numHeads);
  const keyHeads = splitHeads(keys, numKVHeads);
  const valueHeads = splitHeads(values, numKVHeads);

  // Process each head
  const headOutputs: Matrix[] = [];
  for (let h = 0; h < numHeads; h++) {
    const kvIdx = Math.floor(h / kvGroupSize);
    const result = linearizedAttention(queryHeads[h], keyHeads[kvIdx], valueHeads[kvIdx], {
      ...attentionConfig,
      seed: (attentionConfig.seed ?? 42) + h // Different seed per head
    });
    headOutputs.push(result.output);
  }

  // Combine heads
  const output = headOutputs[0].map((_, rowIdx) =>
    headOutputs.map((head) => head[rowIdx]).reduce((acc, row) => acc.concat(row), [] as number[])
  );

  return {
    output,
    stats: {
      complexity: `O(n × d × ${config.numFeatures ?? queryHeads[0][0].length} × ${numHeads})`,
      featureDim: config.numFeatures ?? queryHeads[0][0].length,
      kernelType: config.kernelType
    }
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compare linearized attention output to standard attention.
 * Useful for validating approximation quality.
 */
export function computeApproximationError(
  linearOutput: Matrix,
  standardOutput: Matrix
): { mse: number; maxError: number; relativeError: number } {
  let mse = 0;
  let maxError = 0;
  let sumStandard = 0;

  for (let i = 0; i < linearOutput.length; i++) {
    for (let j = 0; j < linearOutput[i].length; j++) {
      const diff = linearOutput[i][j] - standardOutput[i][j];
      mse += diff * diff;
      maxError = Math.max(maxError, Math.abs(diff));
      sumStandard += Math.abs(standardOutput[i][j]);
    }
  }

  const n = linearOutput.length * linearOutput[0].length;
  mse /= n;

  return {
    mse,
    maxError,
    relativeError: sumStandard > 0 ? Math.sqrt(mse * n) / sumStandard : 0
  };
}

/**
 * Estimate memory savings from linearized attention.
 */
export function estimateMemorySavings(
  seqLength: number,
  modelDim: number,
  numFeatures: number
): { standardMemory: number; linearMemory: number; savingsPercent: number } {
  // Standard attention: O(n²) for attention matrix
  const standardMemory = seqLength * seqLength * 4; // 4 bytes per float32

  // Linear attention: O(n × D) for feature maps + O(D × d) for KV product
  const linearMemory = seqLength * numFeatures * 4 + numFeatures * modelDim * 4;

  const savingsPercent = ((standardMemory - linearMemory) / standardMemory) * 100;

  return {
    standardMemory,
    linearMemory,
    savingsPercent: Math.max(0, savingsPercent)
  };
}
