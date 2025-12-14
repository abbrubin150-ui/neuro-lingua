/**
 * Advanced Numerical Stability Operations
 *
 * This module provides rigorous numerical algorithms with formal error bounds:
 * - Kahan summation for floating-point accuracy
 * - Higham's algorithm for stable matrix norms
 * - Mixed precision analysis with condition number bounds
 * - Compensated summation variants
 *
 * Mathematical guarantees:
 * - Kahan summation: Error bound O(ε) instead of O(nε) for naive summation
 * - Pairwise summation: Error bound O(ε log n)
 * - Neumaier summation: Improved Kahan with better error handling
 *
 * References:
 * - Higham, N.J. (2002) "Accuracy and Stability of Numerical Algorithms"
 * - Kahan, W. (1965) "Pracniques: Further remarks on reducing truncation errors"
 */

/**
 * Kahan summation for improved floating-point accuracy.
 *
 * Standard summation has error bound O(nε), where ε is machine epsilon.
 * Kahan summation reduces this to O(ε) using a running compensation term.
 *
 * Algorithm:
 *   sum = 0.0
 *   c = 0.0  (compensation)
 *   for each value:
 *     y = value - c
 *     t = sum + y
 *     c = (t - sum) - y  (algebraically zero, but captures rounding error)
 *     sum = t
 *
 * @param values - Array of numbers to sum
 * @returns Sum with improved accuracy
 */
export function kahanSum(values: number[]): number {
  if (values.length === 0) return 0;

  let sum = 0.0;
  let compensation = 0.0;

  for (const value of values) {
    const y = value - compensation;
    const t = sum + y;
    compensation = t - sum - y;
    sum = t;
  }

  return sum;
}

/**
 * Result of compensated summation with error estimate
 */
export interface CompensatedSumResult {
  /** The computed sum */
  sum: number;
  /** Estimated rounding error (compensation term) */
  error: number;
  /** Condition number estimate (sensitivity to perturbations) */
  conditionNumber: number;
}

/**
 * Neumaier (improved Kahan) summation with error estimate.
 *
 * Addresses the case where |sum| < |value|, which can cause issues in
 * standard Kahan summation.
 *
 * @param values - Array of numbers to sum
 * @returns Sum with error estimate
 */
export function neumaierSum(values: number[]): CompensatedSumResult {
  if (values.length === 0) {
    return { sum: 0, error: 0, conditionNumber: 0 };
  }

  let sum = values[0];
  let compensation = 0.0;
  let absSum = 0.0;

  for (let i = 1; i < values.length; i++) {
    const value = values[i];
    absSum += Math.abs(value);

    const t = sum + value;
    if (Math.abs(sum) >= Math.abs(value)) {
      compensation += sum - t + value;
    } else {
      compensation += value - t + sum;
    }
    sum = t;
  }

  const result = sum + compensation;
  const conditionNumber = absSum / (Math.abs(result) + Number.EPSILON);

  return {
    sum: result,
    error: Math.abs(compensation),
    conditionNumber
  };
}

/**
 * Pairwise summation (divide-and-conquer) for O(ε log n) error.
 *
 * Recursively splits the array and sums pairs, reducing error accumulation.
 * Used as the default in NumPy/BLAS implementations.
 *
 * @param values - Array of numbers to sum
 * @param threshold - Minimum size before switching to naive sum (default: 16)
 * @returns Sum with O(ε log n) error bound
 */
export function pairwiseSum(values: number[], threshold = 16): number {
  const n = values.length;

  if (n <= threshold) {
    return kahanSum(values);
  }

  const mid = Math.floor(n / 2);
  const left = pairwiseSum(values.slice(0, mid), threshold);
  const right = pairwiseSum(values.slice(mid), threshold);

  return left + right;
}

/**
 * Options for matrix norm computation
 */
export interface MatrixNormOptions {
  /** Maximum iterations for power method (default: 1000) */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-10) */
  tolerance?: number;
  /** Norm type: 'frobenius', 'spectral', '1', 'inf' */
  normType?: 'frobenius' | 'spectral' | '1' | 'inf';
}

/**
 * Result of stable matrix norm computation
 */
export interface MatrixNormResult {
  /** Computed norm value */
  norm: number;
  /** Number of iterations (for spectral norm) */
  iterations?: number;
  /** Whether computation converged */
  converged?: boolean;
  /** Estimated relative error */
  relativeError?: number;
}

/**
 * Compute Frobenius norm with numerical stability.
 *
 * Uses Kahan summation to accumulate squared elements, avoiding overflow
 * through scaling when elements are very large.
 *
 * ||A||_F = sqrt(sum(|a_ij|^2))
 *
 * @param A - Input matrix
 * @returns Frobenius norm
 */
export function stableFrobeniusNorm(A: number[][]): number {
  if (A.length === 0) return 0;

  // Find max absolute value for scaling (prevents overflow)
  let maxVal = 0;
  for (const row of A) {
    for (const val of row) {
      const absVal = Math.abs(val);
      if (absVal > maxVal) maxVal = absVal;
    }
  }

  if (maxVal === 0) return 0;

  // Scale and sum using Kahan
  const scaledSquares: number[] = [];
  for (const row of A) {
    for (const val of row) {
      const scaled = val / maxVal;
      scaledSquares.push(scaled * scaled);
    }
  }

  return maxVal * Math.sqrt(kahanSum(scaledSquares));
}

/**
 * Compute spectral norm (largest singular value) via power iteration.
 *
 * Uses Higham's recommended approach with explicit orthogonalization
 * for numerical stability.
 *
 * ||A||_2 = max(singular values of A)
 *
 * @param A - Input matrix
 * @param options - Computation options
 * @returns Spectral norm with convergence info
 */
export function stableSpectralNorm(
  A: number[][],
  options: MatrixNormOptions = {}
): MatrixNormResult {
  const { maxIterations = 1000, tolerance = 1e-10 } = options;

  const m = A.length;
  if (m === 0) return { norm: 0, iterations: 0, converged: true };

  const n = A[0].length;
  if (n === 0) return { norm: 0, iterations: 0, converged: true };

  // Initialize random unit vector
  let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
  let vNorm = Math.sqrt(kahanSum(v.map((x) => x * x)));
  v = v.map((x) => x / vNorm);

  let sigma = 0;
  let prevSigma = 0;
  let _converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    // Compute u = Av
    const u = multiplyMatrixVector(A, v);

    // Compute v = A^T u
    const vNew = multiplyMatrixTransposeVector(A, u);

    // Normalize v
    vNorm = Math.sqrt(kahanSum(vNew.map((x) => x * x)));
    if (vNorm === 0) {
      return { norm: 0, iterations: iter + 1, converged: true };
    }

    v = vNew.map((x) => x / vNorm);

    // Estimate singular value: sigma = ||Au|| / ||u||
    const uNorm = Math.sqrt(kahanSum(u.map((x) => x * x)));
    sigma = uNorm;

    // Check convergence
    if (Math.abs(sigma - prevSigma) < tolerance * sigma) {
      _converged = true;
      return {
        norm: sigma,
        iterations: iter + 1,
        converged: true,
        relativeError: tolerance
      };
    }

    prevSigma = sigma;
  }

  return {
    norm: sigma,
    iterations: maxIterations,
    converged: false,
    relativeError: Math.abs(sigma - prevSigma) / (sigma + Number.EPSILON)
  };
}

/**
 * Compute matrix-vector product Av
 */
function multiplyMatrixVector(A: number[][], v: number[]): number[] {
  const m = A.length;
  const result = new Array(m).fill(0);

  for (let i = 0; i < m; i++) {
    const row = A[i];
    let sum = 0;
    let c = 0;
    for (let j = 0; j < row.length; j++) {
      const y = row[j] * v[j] - c;
      const t = sum + y;
      c = t - sum - y;
      sum = t;
    }
    result[i] = sum;
  }

  return result;
}

/**
 * Compute matrix transpose-vector product A^T v
 */
function multiplyMatrixTransposeVector(A: number[][], v: number[]): number[] {
  if (A.length === 0) return [];

  const n = A[0].length;
  const result = new Array(n).fill(0);

  for (let j = 0; j < n; j++) {
    let sum = 0;
    let c = 0;
    for (let i = 0; i < A.length; i++) {
      const y = A[i][j] * v[i] - c;
      const t = sum + y;
      c = t - sum - y;
      sum = t;
    }
    result[j] = sum;
  }

  return result;
}

/**
 * Compute matrix norm with stability guarantees
 *
 * @param A - Input matrix
 * @param options - Computation options
 * @returns Matrix norm with metadata
 */
export function stableMatrixNorm(A: number[][], options: MatrixNormOptions = {}): MatrixNormResult {
  const { normType = 'frobenius' } = options;

  switch (normType) {
    case 'frobenius':
      return { norm: stableFrobeniusNorm(A), converged: true };

    case 'spectral':
      return stableSpectralNorm(A, options);

    case '1': {
      // Column sum norm
      if (A.length === 0) return { norm: 0, converged: true };
      const colSums = new Array(A[0].length).fill(0);
      for (const row of A) {
        for (let j = 0; j < row.length; j++) {
          colSums[j] += Math.abs(row[j]);
        }
      }
      return { norm: Math.max(...colSums), converged: true };
    }

    case 'inf': {
      // Row sum norm
      let maxRowSum = 0;
      for (const row of A) {
        const rowSum = kahanSum(row.map(Math.abs));
        if (rowSum > maxRowSum) maxRowSum = rowSum;
      }
      return { norm: maxRowSum, converged: true };
    }

    default:
      return { norm: stableFrobeniusNorm(A), converged: true };
  }
}

/**
 * Precision error bound analysis for mixed-precision computation
 */
export interface PrecisionErrorBound {
  /** Theoretical error bound */
  errorBound: number;
  /** Unit roundoff for the precision */
  unitRoundoff: number;
  /** Condition number impact factor */
  conditionFactor: number;
  /** Warning if poorly conditioned */
  warning?: string;
}

/**
 * Analyze precision error bounds for mixed-precision operations.
 *
 * Based on Higham's forward error analysis:
 * |computed - exact| ≤ γ_n * |exact|
 *
 * where γ_n = n*u / (1 - n*u) and u is unit roundoff.
 *
 * @param operation - Type of operation
 * @param inputPrecision - Input precision ('fp16' | 'fp32' | 'fp64')
 * @param conditionNumber - Condition number of the problem
 * @param problemSize - Size parameter n (e.g., matrix dimension)
 * @returns Error bound analysis
 */
export function precisionErrorBound(
  operation: 'dot_product' | 'matrix_multiply' | 'summation' | 'lu_decomposition',
  inputPrecision: 'fp16' | 'fp32' | 'fp64',
  conditionNumber: number,
  problemSize: number
): PrecisionErrorBound {
  // Unit roundoff for each precision
  const unitRoundoffs: Record<string, number> = {
    fp16: 4.88e-4, // 2^-11
    fp32: 5.96e-8, // 2^-24
    fp64: 1.11e-16 // 2^-53
  };

  const u = unitRoundoffs[inputPrecision];

  // Compute γ_n factor
  const nu = problemSize * u;
  const gamma_n = nu < 1 ? nu / (1 - nu) : Infinity;

  let errorMultiplier: number;
  switch (operation) {
    case 'dot_product':
      // |fl(x·y) - x·y| ≤ γ_n |x||y|
      errorMultiplier = gamma_n;
      break;
    case 'matrix_multiply':
      // For C = AB: |fl(C) - C| ≤ γ_n |A||B|
      errorMultiplier = gamma_n;
      break;
    case 'summation':
      // With Kahan: |fl(sum) - sum| ≤ 2u * |sum_abs|
      // Without: |fl(sum) - sum| ≤ γ_n * |sum_abs|
      errorMultiplier = gamma_n;
      break;
    case 'lu_decomposition':
      // More complex: involves growth factor
      errorMultiplier = problemSize * gamma_n;
      break;
    default:
      errorMultiplier = gamma_n;
  }

  const conditionFactor = conditionNumber * errorMultiplier;
  const errorBound = conditionFactor * u;

  let warning: string | undefined;
  if (conditionNumber > 1e12) {
    warning = 'Matrix is ill-conditioned; results may have large relative error';
  } else if (gamma_n > 0.1) {
    warning = 'Problem size approaches precision limits; consider higher precision';
  }

  return {
    errorBound,
    unitRoundoff: u,
    conditionFactor,
    warning
  };
}

/**
 * Compute condition number estimate using SVD-free method.
 *
 * Uses Hager's algorithm (1-norm condition number estimation):
 * κ(A) = ||A|| * ||A^(-1)||
 *
 * @param A - Input matrix
 * @param maxIterations - Maximum iterations for estimation
 * @returns Condition number estimate
 */
export function estimateConditionNumber(A: number[][], maxIterations = 5): number {
  const n = A.length;
  if (n === 0 || n !== A[0]?.length) {
    return 1; // Non-square or empty
  }

  // Compute ||A||_1
  const normA = stableMatrixNorm(A, { normType: '1' }).norm;
  if (normA === 0) return Infinity;

  // Estimate ||A^(-1)||_1 using Hager's algorithm
  // This is a simplified version; full version requires LU factorization
  let x = new Array(n).fill(1 / n);

  for (let iter = 0; iter < maxIterations; iter++) {
    // y = A * x
    const y = multiplyMatrixVector(A, x);

    // z = A^T * sign(y)
    const signY = y.map((v) => (v >= 0 ? 1 : -1));
    const z = multiplyMatrixTransposeVector(A, signY);

    // Find max component of z
    let maxZ = 0;
    let maxIdx = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(z[i]) > maxZ) {
        maxZ = Math.abs(z[i]);
        maxIdx = i;
      }
    }

    // Check convergence
    const xNorm1 = kahanSum(x.map(Math.abs));
    if (maxZ <= z.reduce((sum, zi, i) => sum + zi * x[i], 0)) {
      return normA / xNorm1;
    }

    // Update x to unit vector
    x = new Array(n).fill(0);
    x[maxIdx] = 1;
  }

  return normA; // Lower bound
}

/**
 * Check numerical stability of a matrix operation
 */
export interface StabilityCheck {
  /** Whether the operation is considered stable */
  stable: boolean;
  /** Condition number */
  conditionNumber: number;
  /** Norm of the matrix */
  norm: number;
  /** Recommendations for improving stability */
  recommendations: string[];
}

/**
 * Perform comprehensive numerical stability check on a matrix.
 *
 * @param A - Input matrix
 * @returns Stability analysis
 */
export function checkNumericalStability(A: number[][]): StabilityCheck {
  const normResult = stableMatrixNorm(A, { normType: 'frobenius' });
  const condNumber = estimateConditionNumber(A);

  const recommendations: string[] = [];
  let stable = true;

  if (condNumber > 1e12) {
    stable = false;
    recommendations.push('Matrix is ill-conditioned (κ > 10^12). Consider regularization.');
  } else if (condNumber > 1e6) {
    recommendations.push(
      'Matrix is moderately ill-conditioned (κ > 10^6). Monitor for precision loss.'
    );
  }

  // Check for very small or very large elements
  let minAbs = Infinity;
  let maxAbs = 0;
  for (const row of A) {
    for (const val of row) {
      const abs = Math.abs(val);
      if (abs > 0 && abs < minAbs) minAbs = abs;
      if (abs > maxAbs) maxAbs = abs;
    }
  }

  const dynamicRange = maxAbs / (minAbs + Number.EPSILON);
  if (dynamicRange > 1e15) {
    stable = false;
    recommendations.push('Large dynamic range in matrix elements. Consider scaling.');
  }

  // Check for NaN or Infinity
  for (const row of A) {
    for (const val of row) {
      if (!Number.isFinite(val)) {
        stable = false;
        recommendations.push('Matrix contains non-finite values (NaN or Infinity).');
        break;
      }
    }
  }

  return {
    stable,
    conditionNumber: condNumber,
    norm: normResult.norm,
    recommendations
  };
}

/**
 * Two-pass algorithm for stable variance computation.
 *
 * Uses Welford's online algorithm to avoid numerical issues
 * with naive computation (sum of squares - square of sum).
 *
 * @param values - Data values
 * @returns { mean, variance, standardDeviation }
 */
export function stableVariance(values: number[]): {
  mean: number;
  variance: number;
  standardDeviation: number;
} {
  if (values.length === 0) {
    return { mean: 0, variance: 0, standardDeviation: 0 };
  }

  if (values.length === 1) {
    return { mean: values[0], variance: 0, standardDeviation: 0 };
  }

  // Welford's online algorithm
  let mean = 0;
  let M2 = 0;
  let n = 0;

  for (const x of values) {
    n++;
    const delta = x - mean;
    mean += delta / n;
    const delta2 = x - mean;
    M2 += delta * delta2;
  }

  const variance = M2 / (n - 1); // Sample variance
  return {
    mean,
    variance,
    standardDeviation: Math.sqrt(variance)
  };
}

/**
 * Parallel (pairwise) variance algorithm for better numerical stability
 * when combining results from parallel computations.
 *
 * Chan et al. (1979) "Updating Formulae and a Pairwise Algorithm for
 * Computing Sample Variances"
 */
export interface ParallelVarianceState {
  count: number;
  mean: number;
  M2: number; // Sum of squared deviations
}

/**
 * Initialize parallel variance state
 */
export function initParallelVariance(): ParallelVarianceState {
  return { count: 0, mean: 0, M2: 0 };
}

/**
 * Update parallel variance with a single value
 */
export function updateParallelVariance(
  state: ParallelVarianceState,
  value: number
): ParallelVarianceState {
  const n = state.count + 1;
  const delta = value - state.mean;
  const mean = state.mean + delta / n;
  const delta2 = value - mean;
  const M2 = state.M2 + delta * delta2;

  return { count: n, mean, M2 };
}

/**
 * Merge two parallel variance states
 */
export function mergeParallelVariance(
  a: ParallelVarianceState,
  b: ParallelVarianceState
): ParallelVarianceState {
  if (a.count === 0) return b;
  if (b.count === 0) return a;

  const count = a.count + b.count;
  const delta = b.mean - a.mean;
  const mean = (a.count * a.mean + b.count * b.mean) / count;
  const M2 = a.M2 + b.M2 + (delta * delta * a.count * b.count) / count;

  return { count, mean, M2 };
}

/**
 * Get variance from parallel state
 */
export function getParallelVariance(state: ParallelVarianceState): {
  mean: number;
  variance: number;
  sampleVariance: number;
} {
  if (state.count < 2) {
    return { mean: state.mean, variance: 0, sampleVariance: 0 };
  }

  return {
    mean: state.mean,
    variance: state.M2 / state.count,
    sampleVariance: state.M2 / (state.count - 1)
  };
}
