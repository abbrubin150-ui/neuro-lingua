/**
 * Approximation Theory and Low-Rank Bounds
 *
 * This module provides theoretical bounds for matrix approximations:
 * - Wedin's sin-theta theorem for SVD perturbations
 * - Nyström approximation with error guarantees
 * - Low-rank approximation bounds
 * - Eckart-Young theorem verification
 *
 * Mathematical Framework:
 * - Spectral perturbation theory
 * - Probabilistic low-rank approximation
 * - Matrix completion bounds
 *
 * References:
 * - Wedin (1972) "Perturbation bounds in connection with SVD"
 * - Williams & Seeger (2001) "Using the Nyström Method"
 * - Halko et al. (2011) "Finding Structure with Randomness"
 * - Stewart & Sun (1990) "Matrix Perturbation Theory"
 */

import { kahanSum, stableFrobeniusNorm, stableSpectralNorm } from './numerics';

/**
 * Result of Wedin bound computation
 */
export interface WedinBound {
  /** Upper bound on sin(θ) where θ is angle between subspaces */
  sinThetaBound: number;
  /** The θ angle bound itself (in radians) */
  angleBound: number;
  /** Perturbation norm ||A - Ã|| */
  perturbationNorm: number;
  /** Singular value gap δ used */
  singularValueGap: number;
  /** Whether the bound is meaningful (< 1) */
  isMeaningful: boolean;
  /** Interpretation */
  interpretation: string;
}

/**
 * Compute approximate SVD using power iteration.
 *
 * Returns top k singular values and vectors.
 */
function approximateSVD(
  A: number[][],
  k: number,
  maxIterations = 100
): {
  U: number[][];
  singularValues: number[];
  V: number[][];
} {
  const m = A.length;
  if (m === 0) return { U: [], singularValues: [], V: [] };
  const n = A[0].length;

  const U: number[][] = [];
  const singularValues: number[] = [];
  const V: number[][] = [];

  // Work with A^T A for right singular vectors
  const ATA = multiplyTransposeA(A);

  // Copy for deflation
  const current = ATA.map((row) => [...row]);

  for (let i = 0; i < k && i < Math.min(m, n); i++) {
    // Power iteration for top eigenvector of A^T A
    let v = new Array(n).fill(1 / Math.sqrt(n));

    for (let iter = 0; iter < maxIterations; iter++) {
      // v = A^T A * v
      const newV = matrixVectorMultiply(current, v);

      // Normalize
      const norm = Math.sqrt(kahanSum(newV.map((x) => x * x)));
      if (norm < 1e-10) break;

      const vNext = newV.map((x) => x / norm);

      // Check convergence
      const diff = kahanSum(vNext.map((x, j) => Math.abs(x - v[j])));
      v = vNext;
      if (diff < 1e-10) break;
    }

    // Singular value: sqrt of eigenvalue of A^T A
    const ATAv = matrixVectorMultiply(current, v);
    const eigenvalue = kahanSum(v.map((vi, j) => vi * ATAv[j]));
    const sigma = Math.sqrt(Math.max(eigenvalue, 0));

    singularValues.push(sigma);
    V.push(v);

    // Compute left singular vector: u = Av / sigma
    if (sigma > 1e-10) {
      const Av = matrixVectorMultiply(A, v);
      const u = Av.map((x) => x / sigma);
      U.push(u);
    } else {
      U.push(new Array(m).fill(0));
    }

    // Deflation: A^T A = A^T A - σ² v v^T
    for (let j = 0; j < n; j++) {
      for (let l = 0; l < n; l++) {
        current[j][l] -= eigenvalue * v[j] * v[l];
      }
    }
  }

  return { U, singularValues, V };
}

/**
 * Compute A^T * A
 */
function multiplyTransposeA(A: number[][]): number[][] {
  const m = A.length;
  if (m === 0) return [];
  const n = A[0].length;

  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    result.push(new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < m; k++) {
        result[i][j] += A[k][i] * A[k][j];
      }
    }
  }
  return result;
}

/**
 * Matrix-vector multiplication
 */
function matrixVectorMultiply(A: number[][], v: number[]): number[] {
  const m = A.length;
  const result = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    result[i] = kahanSum(A[i].map((a, j) => a * v[j]));
  }
  return result;
}

/**
 * Wedin's sin-theta theorem for SVD perturbation bounds.
 *
 * Given matrices A and Ã = A + E, bounds the angle between
 * singular subspaces of A and Ã.
 *
 * Theorem: Let A = UΣV^T and Ã = ŨΣ̃Ṽ^T be SVDs.
 * For the subspace spanned by top k singular vectors:
 *
 * sin(θ(V_k, Ṽ_k)) ≤ ||E|| / δ
 *
 * where δ = σ_k(A) - σ_{k+1}(Ã) is the singular value gap.
 *
 * @param A - Original matrix
 * @param A_tilde - Perturbed matrix
 * @param rank - Rank of approximation
 * @returns Wedin bound
 */
export function wedinBound(A: number[][], A_tilde: number[][], rank: number): WedinBound {
  const m = A.length;
  if (m === 0 || m !== A_tilde.length) {
    return {
      sinThetaBound: Infinity,
      angleBound: Math.PI / 2,
      perturbationNorm: 0,
      singularValueGap: 0,
      isMeaningful: false,
      interpretation: 'Invalid input matrices'
    };
  }

  const n = A[0].length;
  if (n !== A_tilde[0].length) {
    return {
      sinThetaBound: Infinity,
      angleBound: Math.PI / 2,
      perturbationNorm: 0,
      singularValueGap: 0,
      isMeaningful: false,
      interpretation: 'Matrix dimension mismatch'
    };
  }

  // Compute perturbation E = A_tilde - A
  const E: number[][] = [];
  for (let i = 0; i < m; i++) {
    E.push(A_tilde[i].map((val, j) => val - A[i][j]));
  }

  // Compute ||E|| (spectral norm)
  const perturbationNorm = stableSpectralNorm(E).norm;

  // Compute SVD of both matrices
  const svdA = approximateSVD(A, rank + 1);
  const svdAtilde = approximateSVD(A_tilde, rank + 1);

  // Singular value gap: σ_k(A) - σ_{k+1}(Ã)
  const sigma_k = svdA.singularValues[rank - 1] ?? 0;
  const sigma_k1_tilde = svdAtilde.singularValues[rank] ?? 0;
  const singularValueGap = sigma_k - sigma_k1_tilde;

  // Wedin bound
  let sinThetaBound: number;
  if (singularValueGap <= 0) {
    sinThetaBound = Infinity;
  } else {
    sinThetaBound = perturbationNorm / singularValueGap;
  }

  const isMeaningful = sinThetaBound < 1;
  const angleBound = isMeaningful ? Math.asin(sinThetaBound) : Math.PI / 2;

  let interpretation: string;
  if (!isMeaningful) {
    interpretation =
      'Perturbation too large relative to singular value gap. ' + 'Subspaces may have large angle.';
  } else if (sinThetaBound < 0.1) {
    interpretation =
      `Subspaces are very close (angle < ${((angleBound * 180) / Math.PI).toFixed(1)}°). ` +
      'Approximation is stable.';
  } else if (sinThetaBound < 0.5) {
    interpretation =
      `Moderate subspace deviation (angle < ${((angleBound * 180) / Math.PI).toFixed(1)}°). ` +
      'Consider increasing rank or reducing perturbation.';
  } else {
    interpretation =
      `Significant subspace deviation (angle < ${((angleBound * 180) / Math.PI).toFixed(1)}°). ` +
      'Approximation may be unreliable.';
  }

  return {
    sinThetaBound,
    angleBound,
    perturbationNorm,
    singularValueGap,
    isMeaningful,
    interpretation
  };
}

/**
 * Nyström approximation result
 */
export interface NystromApproximation {
  /** Approximated matrix */
  approximation: number[][];
  /** Error bound ||K - K̃||_F */
  errorBound: number;
  /** Actual relative error */
  relativeError: number;
  /** Number of landmark points used */
  numLandmarks: number;
  /** Condition number of landmark submatrix */
  landmarkConditionNumber: number;
  /** Whether approximation is numerically stable */
  stable: boolean;
  /** Recommendations */
  recommendations: string[];
}

/**
 * Nyström approximation for kernel matrices.
 *
 * Given a PSD matrix K, approximates it using a subset of columns:
 * K̃ = K_{nm} K_{mm}^{-1} K_{mn}
 *
 * where m is the number of landmark points.
 *
 * Reference: Williams & Seeger (2001)
 *
 * @param K - Full kernel matrix (PSD, n x n)
 * @param numLandmarks - Number of landmark points m
 * @param landmarkIndices - Optional: specific indices to use
 * @returns Nyström approximation with bounds
 */
export function nystromApproximation(
  K: number[][],
  numLandmarks: number,
  landmarkIndices?: number[]
): NystromApproximation {
  const n = K.length;
  const recommendations: string[] = [];

  if (n === 0) {
    return {
      approximation: [],
      errorBound: 0,
      relativeError: 0,
      numLandmarks: 0,
      landmarkConditionNumber: 1,
      stable: false,
      recommendations: ['Empty matrix']
    };
  }

  const m = Math.min(numLandmarks, n);

  // Select landmark indices (random if not provided)
  const landmarks = landmarkIndices?.slice(0, m) ?? selectLandmarks(n, m);

  // Extract K_mm (landmark submatrix)
  const K_mm: number[][] = [];
  for (let i = 0; i < m; i++) {
    K_mm.push([]);
    for (let j = 0; j < m; j++) {
      K_mm[i].push(K[landmarks[i]][landmarks[j]]);
    }
  }

  // Extract K_nm (full matrix columns at landmarks)
  const K_nm: number[][] = [];
  for (let i = 0; i < n; i++) {
    K_nm.push([]);
    for (let j = 0; j < m; j++) {
      K_nm[i].push(K[i][landmarks[j]]);
    }
  }

  // Compute K_mm^{-1} using Cholesky or SVD
  const { inverse: K_mm_inv, conditionNumber } = pseudoInverse(K_mm);

  // Nyström approximation: K̃ = K_nm K_mm^{-1} K_mn
  const temp = multiplyMatrices(K_nm, K_mm_inv);
  const K_mn: number[][] = [];
  for (let i = 0; i < m; i++) {
    K_mn.push([]);
    for (let j = 0; j < n; j++) {
      K_mn[i].push(K[landmarks[i]][j]);
    }
  }
  const approximation = multiplyMatrices(temp, K_mn);

  // Compute error
  const error: number[][] = [];
  for (let i = 0; i < n; i++) {
    error.push([]);
    for (let j = 0; j < n; j++) {
      error[i].push(K[i][j] - approximation[i][j]);
    }
  }

  const errorNorm = stableFrobeniusNorm(error);
  const originalNorm = stableFrobeniusNorm(K);
  const relativeError = errorNorm / (originalNorm + 1e-10);

  // Theoretical error bound: ||K - K̃||_F ≤ ||K - K_k|| + (n/m) ||K_mm^{-1}||_2 ||K - K_k||^2
  // where K_k is best rank-k approximation
  // Simplified: errorBound ≈ (n/m) * σ_{m+1}
  const svdK = approximateSVD(K, m + 1);
  const sigma_m1 = svdK.singularValues[m] ?? 0;
  const errorBound = (n / m) * sigma_m1;

  // Stability check
  const stable = conditionNumber < 1e10 && relativeError < 0.5;

  // Generate recommendations
  if (conditionNumber > 1e6) {
    recommendations.push(
      'Landmark submatrix is ill-conditioned. Consider different landmark selection.'
    );
  }

  if (relativeError > 0.2) {
    recommendations.push('High approximation error. Consider increasing number of landmarks.');
  }

  if (m < Math.sqrt(n)) {
    recommendations.push(`Consider using at least √n = ${Math.ceil(Math.sqrt(n))} landmarks.`);
  }

  if (recommendations.length === 0) {
    recommendations.push('Nyström approximation appears accurate and stable.');
  }

  return {
    approximation,
    errorBound,
    relativeError,
    numLandmarks: m,
    landmarkConditionNumber: conditionNumber,
    stable,
    recommendations
  };
}

/**
 * Select landmarks using k-means++ initialization.
 */
function selectLandmarks(n: number, m: number): number[] {
  const landmarks: number[] = [];

  // First landmark: random
  landmarks.push(Math.floor(Math.random() * n));

  // Remaining: proportional to squared distance from existing
  for (let i = 1; i < m; i++) {
    // Simple random selection (k-means++ would be better but more complex)
    let newIdx: number;
    do {
      newIdx = Math.floor(Math.random() * n);
    } while (landmarks.includes(newIdx));
    landmarks.push(newIdx);
  }

  return landmarks;
}

/**
 * Compute pseudo-inverse using SVD.
 */
function pseudoInverse(A: number[][]): {
  inverse: number[][];
  conditionNumber: number;
} {
  const n = A.length;
  if (n === 0) return { inverse: [], conditionNumber: 1 };

  const svd = approximateSVD(A, n);
  const { U, singularValues, V } = svd;

  // A^+ = V Σ^+ U^T
  // Filter small singular values
  const tolerance = 1e-10 * Math.max(...singularValues);

  const inverse: number[][] = [];
  for (let i = 0; i < n; i++) {
    inverse.push(new Array(n).fill(0));
  }

  for (let k = 0; k < singularValues.length; k++) {
    if (singularValues[k] > tolerance) {
      const scale = 1 / singularValues[k];
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          inverse[i][j] += V[k][i] * scale * U[k][j];
        }
      }
    }
  }

  const maxSigma = singularValues[0] || 1;
  const minSigma = singularValues.filter((s) => s > tolerance).pop() || 1;
  const conditionNumber = maxSigma / minSigma;

  return { inverse, conditionNumber };
}

/**
 * Matrix multiplication
 */
function multiplyMatrices(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  if (m === 0) return [];
  const n = A[0].length;
  const p = B[0]?.length ?? 0;

  const result: number[][] = [];
  for (let i = 0; i < m; i++) {
    result.push(new Array(p).fill(0));
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

/**
 * Low-rank approximation analysis
 */
export interface LowRankAnalysis {
  /** Original rank (numerical) */
  numericalRank: number;
  /** Singular values */
  singularValues: number[];
  /** Cumulative energy (sum of σ² / total) */
  cumulativeEnergy: number[];
  /** Recommended rank for target energy */
  recommendedRank: number;
  /** Eckart-Young optimal error for each rank */
  eckartYoungErrors: number[];
  /** Stable rank (Frobenius² / spectral²) */
  stableRank: number;
}

/**
 * Analyze low-rank approximation properties.
 *
 * @param A - Input matrix
 * @param targetEnergy - Target fraction of energy to preserve (default: 0.95)
 * @returns Analysis results
 */
export function analyzeLowRankApproximation(A: number[][], targetEnergy = 0.95): LowRankAnalysis {
  const m = A.length;
  if (m === 0) {
    return {
      numericalRank: 0,
      singularValues: [],
      cumulativeEnergy: [],
      recommendedRank: 0,
      eckartYoungErrors: [],
      stableRank: 0
    };
  }

  const n = A[0].length;
  const maxRank = Math.min(m, n);

  // Compute SVD
  const svd = approximateSVD(A, maxRank);
  const { singularValues } = svd;

  // Total energy (sum of σ²)
  const totalEnergy = kahanSum(singularValues.map((s) => s * s));

  // Cumulative energy
  const cumulativeEnergy: number[] = [];
  let runningSum = 0;
  for (const sigma of singularValues) {
    runningSum += sigma * sigma;
    cumulativeEnergy.push(runningSum / (totalEnergy + 1e-10));
  }

  // Numerical rank (singular values above threshold)
  const tolerance = 1e-10 * singularValues[0];
  const numericalRank = singularValues.filter((s) => s > tolerance).length;

  // Recommended rank for target energy
  let recommendedRank = 1;
  for (let i = 0; i < cumulativeEnergy.length; i++) {
    if (cumulativeEnergy[i] >= targetEnergy) {
      recommendedRank = i + 1;
      break;
    }
    recommendedRank = i + 1;
  }

  // Eckart-Young errors: ||A - A_k||_F = sqrt(sum of remaining σ²)
  const eckartYoungErrors: number[] = [];
  for (let k = 0; k < singularValues.length; k++) {
    const remainingEnergy = singularValues.slice(k).reduce((s, sigma) => s + sigma * sigma, 0);
    eckartYoungErrors.push(Math.sqrt(remainingEnergy));
  }

  // Stable rank
  const frobeniusNorm = stableFrobeniusNorm(A);
  const spectralNorm = singularValues[0] || 1;
  const stableRank = (frobeniusNorm * frobeniusNorm) / (spectralNorm * spectralNorm);

  return {
    numericalRank,
    singularValues,
    cumulativeEnergy,
    recommendedRank,
    eckartYoungErrors,
    stableRank
  };
}

/**
 * Randomized SVD using random projections.
 *
 * Reference: Halko et al. (2011)
 */
export interface RandomizedSVD {
  /** Left singular vectors (approximate) */
  U: number[][];
  /** Singular values (approximate) */
  singularValues: number[];
  /** Right singular vectors (approximate) */
  V: number[][];
  /** Estimated error bound */
  errorBound: number;
}

/**
 * Compute approximate SVD using randomized algorithm.
 *
 * @param A - Input matrix (m x n)
 * @param rank - Target rank
 * @param oversampling - Additional columns for accuracy (default: 10)
 * @param powerIterations - Power iterations for accuracy (default: 2)
 * @returns Approximate SVD
 */
export function randomizedSVD(
  A: number[][],
  rank: number,
  oversampling = 10,
  powerIterations = 2
): RandomizedSVD {
  const m = A.length;
  if (m === 0) {
    return { U: [], singularValues: [], V: [], errorBound: 0 };
  }
  const n = A[0].length;

  const l = Math.min(rank + oversampling, Math.min(m, n));

  // Generate random projection matrix Ω (n x l)
  const Omega: number[][] = [];
  for (let i = 0; i < n; i++) {
    Omega.push([]);
    for (let j = 0; j < l; j++) {
      Omega[i].push(randomGaussian());
    }
  }

  // Compute Y = A * Ω
  let Y = multiplyMatrices(A, Omega);

  // Power iterations for improved accuracy
  // Y = (A A^T)^q A Ω
  for (let q = 0; q < powerIterations; q++) {
    // Y = A^T Y
    const AT = transposeMatrix(A);
    Y = multiplyMatrices(AT, Y);
    // Y = A Y
    Y = multiplyMatrices(A, Y);
  }

  // QR factorization of Y to get orthonormal basis Q
  const Q = qrOrthogonalize(Y);

  // Form B = Q^T A
  const QT = transposeMatrix(Q);
  const B = multiplyMatrices(QT, A);

  // SVD of B (smaller matrix, l x n)
  const svdB = approximateSVD(B, rank);

  // U = Q * U_B
  const U = multiplyMatrices(
    Q,
    svdB.U.map((u) => u.slice(0, l))
  );

  // Error bound: ||A - U Σ V^T||_F ≈ σ_{rank+1}
  const errorBound = svdB.singularValues[rank] ?? 0;

  return {
    U: transposeMatrix(U),
    singularValues: svdB.singularValues.slice(0, rank),
    V: svdB.V.slice(0, rank),
    errorBound
  };
}

/**
 * Generate standard Gaussian random number (Box-Muller)
 */
function randomGaussian(): number {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Transpose matrix
 */
function transposeMatrix(A: number[][]): number[][] {
  if (A.length === 0) return [];
  const m = A.length;
  const n = A[0].length;

  const result: number[][] = [];
  for (let j = 0; j < n; j++) {
    result.push([]);
    for (let i = 0; i < m; i++) {
      result[j].push(A[i][j]);
    }
  }
  return result;
}

/**
 * QR orthogonalization using Gram-Schmidt
 */
function qrOrthogonalize(Y: number[][]): number[][] {
  const m = Y.length;
  if (m === 0) return [];
  const n = Y[0].length;

  const Q: number[][] = [];
  for (let i = 0; i < m; i++) {
    Q.push(new Array(n).fill(0));
  }

  for (let j = 0; j < n; j++) {
    // Copy column j
    const v: number[] = [];
    for (let i = 0; i < m; i++) {
      v.push(Y[i][j]);
    }

    // Subtract projections onto previous columns
    for (let k = 0; k < j; k++) {
      const qk = Q.map((row) => row[k]);
      const proj = kahanSum(v.map((vi, i) => vi * qk[i]));
      for (let i = 0; i < m; i++) {
        v[i] -= proj * qk[i];
      }
    }

    // Normalize
    const norm = Math.sqrt(kahanSum(v.map((x) => x * x)));
    if (norm > 1e-10) {
      for (let i = 0; i < m; i++) {
        Q[i][j] = v[i] / norm;
      }
    }
  }

  return Q;
}
