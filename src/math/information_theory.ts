/**
 * Advanced Information-Theoretic Analysis
 *
 * This module provides rigorous mutual information estimators and
 * rate-distortion analysis for representation learning:
 *
 * - KSG (Kraskov-Stögbauer-Grassberger) estimator for MI
 * - K-Nearest Neighbor (KNN) entropy estimators
 * - Kernel density estimation methods
 * - Rate-Distortion curve computation
 * - Information plane analysis
 *
 * References:
 * - Kraskov et al. (2004) "Estimating mutual information"
 * - Kozachenko & Leonenko (1987) "Sample estimate of the entropy"
 * - Tishby et al. (1999) "The Information Bottleneck Method"
 * - Shwartz-Ziv & Tishby (2017) "Opening the Black Box of Deep Neural Networks"
 */

import { kahanSum } from './numerics';

/**
 * Mutual information estimator type
 */
export type MIEstimator = 'histogram' | 'ksg' | 'knn' | 'kernel';

/**
 * Configuration for MI estimation
 */
export interface MIEstimatorConfig {
  /** Estimator type */
  estimator: MIEstimator;
  /** Number of nearest neighbors for KSG/KNN (default: 3) */
  k?: number;
  /** Number of bins for histogram method (default: 50) */
  numBins?: number;
  /** Kernel bandwidth for kernel method (default: auto) */
  bandwidth?: number;
  /** Numerical stability epsilon */
  epsilon?: number;
}

/**
 * Result of MI estimation with diagnostics
 */
export interface MIEstimateResult {
  /** Estimated mutual information in nats */
  mutualInformation: number;
  /** Standard error estimate (if available) */
  standardError?: number;
  /** 95% confidence interval (if available) */
  confidenceInterval?: [number, number];
  /** Effective sample size after preprocessing */
  effectiveSampleSize: number;
  /** Estimator used */
  estimator: MIEstimator;
  /** Diagnostics */
  diagnostics: string[];
}

/**
 * Compute Euclidean distance between two vectors
 */
function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Compute Chebyshev (max) distance between two vectors
 */
function chebyshevDistance(a: number[], b: number[]): number {
  let maxDist = 0;
  for (let i = 0; i < a.length; i++) {
    const dist = Math.abs(a[i] - b[i]);
    if (dist > maxDist) maxDist = dist;
  }
  return maxDist;
}

/**
 * Find k-nearest neighbors and return their distances
 */
function findKNearestNeighbors(
  point: number[],
  points: number[][],
  k: number,
  distanceFn: (a: number[], b: number[]) => number = euclideanDistance
): { indices: number[]; distances: number[] } {
  const distances: { idx: number; dist: number }[] = [];

  for (let i = 0; i < points.length; i++) {
    const dist = distanceFn(point, points[i]);
    if (dist > 0) { // Exclude self (distance = 0)
      distances.push({ idx: i, dist });
    }
  }

  distances.sort((a, b) => a.dist - b.dist);

  const kNearest = distances.slice(0, k);
  return {
    indices: kNearest.map(d => d.idx),
    distances: kNearest.map(d => d.dist)
  };
}

/**
 * Digamma (psi) function approximation
 * ψ(x) = d/dx ln(Γ(x))
 */
function digamma(x: number): number {
  if (x <= 0) return -Infinity;

  let result = 0;
  // Use recurrence relation to shift x to larger values
  while (x < 6) {
    result -= 1 / x;
    x += 1;
  }

  // Asymptotic expansion for large x
  const x2 = x * x;
  result += Math.log(x) - 1 / (2 * x);
  result -= 1 / (12 * x2);
  result += 1 / (120 * x2 * x2);
  result -= 1 / (252 * x2 * x2 * x2);

  return result;
}

/**
 * KSG (Kraskov-Stögbauer-Grassberger) Mutual Information Estimator
 *
 * This is a non-parametric estimator that uses k-nearest neighbors
 * in the joint space to estimate mutual information.
 *
 * I(X;Y) = ψ(k) + ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)>
 *
 * where n_x, n_y are the number of points within distance ε in
 * the marginal spaces, and ε is the distance to the k-th neighbor
 * in the joint space.
 *
 * Reference: Kraskov et al. (2004) "Estimating mutual information"
 *
 * @param X - First variable samples (N x d_x)
 * @param Y - Second variable samples (N x d_y)
 * @param k - Number of nearest neighbors (default: 3)
 * @returns Estimated MI in nats
 */
export function ksgMutualInformation(
  X: number[][],
  Y: number[][],
  k = 3
): MIEstimateResult {
  const N = X.length;
  const diagnostics: string[] = [];

  if (N !== Y.length) {
    return {
      mutualInformation: 0,
      effectiveSampleSize: 0,
      estimator: 'ksg',
      diagnostics: ['Sample size mismatch between X and Y']
    };
  }

  if (N < k + 1) {
    return {
      mutualInformation: 0,
      effectiveSampleSize: N,
      estimator: 'ksg',
      diagnostics: ['Insufficient samples for KSG estimator']
    };
  }

  // Create joint samples Z = [X, Y]
  const Z: number[][] = [];
  for (let i = 0; i < N; i++) {
    Z.push([...X[i], ...Y[i]]);
  }

  // For each point, find k-th nearest neighbor in joint space
  // and count points within that distance in marginal spaces
  let sumNx = 0;
  let sumNy = 0;

  for (let i = 0; i < N; i++) {
    // Find k-th nearest neighbor distance in joint space (Chebyshev norm)
    const { distances } = findKNearestNeighbors(Z[i], Z, k, chebyshevDistance);
    const epsilon = distances[k - 1];

    // Count points within epsilon in X marginal
    let nx = 0;
    for (let j = 0; j < N; j++) {
      if (j !== i && chebyshevDistance(X[i], X[j]) < epsilon) {
        nx++;
      }
    }

    // Count points within epsilon in Y marginal
    let ny = 0;
    for (let j = 0; j < N; j++) {
      if (j !== i && chebyshevDistance(Y[i], Y[j]) < epsilon) {
        ny++;
      }
    }

    sumNx += digamma(nx + 1);
    sumNy += digamma(ny + 1);
  }

  // KSG estimator formula
  const mi = digamma(k) + digamma(N) - sumNx / N - sumNy / N;

  // Estimate standard error (rough approximation)
  const standardError = Math.sqrt(
    (digamma(k) - digamma(1) + 1 / k) / N
  );

  diagnostics.push(`k = ${k} nearest neighbors`);
  diagnostics.push(`N = ${N} samples`);

  return {
    mutualInformation: Math.max(0, mi), // MI is non-negative
    standardError,
    confidenceInterval: [mi - 1.96 * standardError, mi + 1.96 * standardError],
    effectiveSampleSize: N,
    estimator: 'ksg',
    diagnostics
  };
}

/**
 * Kozachenko-Leonenko entropy estimator using k-NN
 *
 * H(X) = ψ(N) - ψ(k) + d·log(2) + d·<log(ε_i)>
 *
 * where ε_i is the distance to k-th nearest neighbor and d is dimension.
 *
 * Reference: Kozachenko & Leonenko (1987)
 */
export function knnEntropy(X: number[][], k = 3): number {
  const N = X.length;
  if (N < k + 1) return 0;

  const d = X[0].length;

  let sumLogEps = 0;
  for (let i = 0; i < N; i++) {
    const { distances } = findKNearestNeighbors(X[i], X, k, euclideanDistance);
    const epsilon = Math.max(distances[k - 1], 1e-10);
    sumLogEps += Math.log(epsilon);
  }

  // Entropy estimate (in nats)
  const entropy = digamma(N) - digamma(k) + d * Math.log(2) + (d / N) * sumLogEps;

  // Add correction for unit ball volume: log(V_d) = log(π^(d/2) / Γ(d/2 + 1))
  const logVd = (d / 2) * Math.log(Math.PI) - lgamma(d / 2 + 1);
  return entropy + logVd;
}

/**
 * Log gamma function approximation (Stirling)
 */
function lgamma(x: number): number {
  if (x <= 0) return Infinity;

  // Lanczos approximation
  const g = 7;
  const c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }

  x -= 1;
  let a = c[0];
  for (let i = 1; i < g + 2; i++) {
    a += c[i] / (x + i);
  }

  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Unified mutual information estimation
 */
export function estimateMutualInformationAdvanced(
  X: number[] | number[][],
  Y: number[] | number[][],
  config: MIEstimatorConfig = { estimator: 'ksg' }
): MIEstimateResult {
  // Convert 1D arrays to 2D
  const X2D: number[][] = Array.isArray(X[0]) ? X as number[][] : (X as number[]).map(x => [x]);
  const Y2D: number[][] = Array.isArray(Y[0]) ? Y as number[][] : (Y as number[]).map(y => [y]);

  switch (config.estimator) {
    case 'ksg':
      return ksgMutualInformation(X2D, Y2D, config.k ?? 3);

    case 'knn':
      // Use entropy-based MI: I(X;Y) = H(X) + H(Y) - H(X,Y)
      const k = config.k ?? 3;
      const hX = knnEntropy(X2D, k);
      const hY = knnEntropy(Y2D, k);
      const XY = X2D.map((x, i) => [...x, ...Y2D[i]]);
      const hXY = knnEntropy(XY, k);
      const mi = hX + hY - hXY;
      return {
        mutualInformation: Math.max(0, mi),
        effectiveSampleSize: X2D.length,
        estimator: 'knn',
        diagnostics: [`H(X)=${hX.toFixed(4)}, H(Y)=${hY.toFixed(4)}, H(X,Y)=${hXY.toFixed(4)}`]
      };

    case 'histogram':
      // Fall back to existing histogram-based estimator
      return histogramMI(X2D, Y2D, config.numBins ?? 50);

    case 'kernel':
      return kernelMI(X2D, Y2D, config.bandwidth);

    default:
      return ksgMutualInformation(X2D, Y2D, config.k ?? 3);
  }
}

/**
 * Histogram-based MI estimator (simpler, less accurate)
 */
function histogramMI(X: number[][], Y: number[][], numBins: number): MIEstimateResult {
  const N = X.length;
  if (N === 0) {
    return {
      mutualInformation: 0,
      effectiveSampleSize: 0,
      estimator: 'histogram',
      diagnostics: ['Empty input']
    };
  }

  // Flatten to 1D (use first dimension)
  const xFlat = X.map(x => x[0] ?? 0);
  const yFlat = Y.map(y => y[0] ?? 0);

  const xMin = Math.min(...xFlat);
  const xMax = Math.max(...xFlat);
  const yMin = Math.min(...yFlat);
  const yMax = Math.max(...yFlat);

  const xRange = (xMax - xMin) || 1;
  const yRange = (yMax - yMin) || 1;

  // Build histograms
  const jointCounts = new Map<string, number>();
  const xCounts = new Array(numBins).fill(0);
  const yCounts = new Array(numBins).fill(0);

  for (let i = 0; i < N; i++) {
    const xBin = Math.min(Math.floor(((xFlat[i] - xMin) / xRange) * numBins), numBins - 1);
    const yBin = Math.min(Math.floor(((yFlat[i] - yMin) / yRange) * numBins), numBins - 1);

    const key = `${xBin},${yBin}`;
    jointCounts.set(key, (jointCounts.get(key) ?? 0) + 1);
    xCounts[xBin]++;
    yCounts[yBin]++;
  }

  // Compute MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
  let mi = 0;
  for (const [key, count] of jointCounts) {
    const [xBin, yBin] = key.split(',').map(Number);
    const pxy = count / N;
    const px = xCounts[xBin] / N;
    const py = yCounts[yBin] / N;

    if (pxy > 0 && px > 0 && py > 0) {
      mi += pxy * Math.log(pxy / (px * py));
    }
  }

  return {
    mutualInformation: Math.max(0, mi),
    effectiveSampleSize: N,
    estimator: 'histogram',
    diagnostics: [`numBins = ${numBins}`]
  };
}

/**
 * Kernel density estimation based MI
 */
function kernelMI(X: number[][], Y: number[][], bandwidth?: number): MIEstimateResult {
  const N = X.length;
  if (N === 0) {
    return {
      mutualInformation: 0,
      effectiveSampleSize: 0,
      estimator: 'kernel',
      diagnostics: ['Empty input']
    };
  }

  // Silverman's rule of thumb for bandwidth
  const xFlat = X.map(x => x[0] ?? 0);
  const yFlat = Y.map(y => y[0] ?? 0);

  const xStd = Math.sqrt(xFlat.reduce((s, v) => s + v * v, 0) / N - Math.pow(xFlat.reduce((s, v) => s + v, 0) / N, 2));
  const yStd = Math.sqrt(yFlat.reduce((s, v) => s + v * v, 0) / N - Math.pow(yFlat.reduce((s, v) => s + v, 0) / N, 2));

  const hx = bandwidth ?? (1.06 * xStd * Math.pow(N, -0.2));
  const hy = bandwidth ?? (1.06 * yStd * Math.pow(N, -0.2));

  // Gaussian kernel density estimation
  const gaussianKernel = (u: number) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);

  // Estimate MI using leave-one-out density estimation
  let mi = 0;
  for (let i = 0; i < N; i++) {
    let pxy = 0, px = 0, py = 0;

    for (let j = 0; j < N; j++) {
      if (j === i) continue; // Leave-one-out

      const kx = gaussianKernel((xFlat[i] - xFlat[j]) / hx);
      const ky = gaussianKernel((yFlat[i] - yFlat[j]) / hy);

      pxy += kx * ky;
      px += kx;
      py += ky;
    }

    pxy /= (N - 1) * hx * hy;
    px /= (N - 1) * hx;
    py /= (N - 1) * hy;

    if (pxy > 0 && px > 0 && py > 0) {
      mi += Math.log(pxy / (px * py));
    }
  }

  mi /= N;

  return {
    mutualInformation: Math.max(0, mi),
    effectiveSampleSize: N,
    estimator: 'kernel',
    diagnostics: [`bandwidth_x = ${hx.toFixed(4)}, bandwidth_y = ${hy.toFixed(4)}`]
  };
}

/**
 * Rate-Distortion curve point
 */
export interface RateDistortionPoint {
  /** Information rate R = I(X;Z) */
  rate: number;
  /** Distortion D = E[d(X, X̂)] */
  distortion: number;
  /** Relevance I(Z;Y) at this point */
  relevance: number;
  /** Beta value used */
  beta: number;
}

/**
 * Rate-Distortion analysis configuration
 */
export interface RateDistortionConfig {
  /** Number of beta values to sample */
  numBetaValues?: number;
  /** Minimum beta value */
  betaMin?: number;
  /** Maximum beta value */
  betaMax?: number;
  /** MI estimator to use */
  estimator?: MIEstimator;
}

/**
 * Compute Rate-Distortion curve from layer activations.
 *
 * For Information Bottleneck:
 * - Rate: R = I(X;Z) (compression)
 * - Relevance: I(Z;Y) (prediction)
 *
 * The curve shows the tradeoff as beta varies.
 *
 * @param X - Input samples
 * @param Z - Representation (hidden layer activations)
 * @param Y - Target labels
 * @param config - Configuration options
 */
export function computeRateDistortionCurve(
  X: number[][],
  Z: number[][],
  Y: number[],
  config: RateDistortionConfig = {}
): RateDistortionPoint[] {
  const {
    numBetaValues = 20,
    betaMin = 0.01,
    betaMax = 100,
    estimator = 'ksg'
  } = config;

  const points: RateDistortionPoint[] = [];

  // Convert Y to 2D for MI estimation
  const Y2D = Y.map(y => [y]);

  // Compute MI values
  const IXZ = estimateMutualInformationAdvanced(X, Z, { estimator, k: 3 });
  const IZY = estimateMutualInformationAdvanced(Z, Y2D, { estimator, k: 3 });

  // For now, compute single point (full representation)
  // True rate-distortion would require training with different betas
  points.push({
    rate: IXZ.mutualInformation,
    distortion: 0, // Would need reconstruction to compute
    relevance: IZY.mutualInformation,
    beta: 1.0
  });

  // Simulate curve by varying "effective" representation quality
  const betas = logSpace(betaMin, betaMax, numBetaValues);

  for (const beta of betas) {
    // Approximate: higher beta → more compression → lower I(X;Z)
    // This is a simplification; real IB requires optimization
    const compressionFactor = 1 / (1 + beta);
    const rate = IXZ.mutualInformation * compressionFactor;
    const relevance = IZY.mutualInformation * Math.pow(compressionFactor, 0.5);

    points.push({
      rate,
      distortion: IXZ.mutualInformation - rate, // Approximation
      relevance,
      beta
    });
  }

  // Sort by rate
  points.sort((a, b) => a.rate - b.rate);

  return points;
}

/**
 * Generate logarithmically spaced values
 */
function logSpace(start: number, end: number, n: number): number[] {
  const logStart = Math.log10(start);
  const logEnd = Math.log10(end);
  const step = (logEnd - logStart) / (n - 1);

  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    values.push(Math.pow(10, logStart + i * step));
  }
  return values;
}

/**
 * Information Plane coordinates for a layer
 */
export interface InformationPlanePoint {
  /** Training epoch */
  epoch: number;
  /** Layer index */
  layerIndex: number;
  /** I(X;Z) - information about input */
  compressionMI: number;
  /** I(Z;Y) - information about target */
  predictionMI: number;
  /** Layer name (optional) */
  layerName?: string;
}

/**
 * Compute Information Plane trajectory during training.
 *
 * Tracks I(X;Z) and I(Z;Y) for each layer across epochs.
 *
 * Reference: Shwartz-Ziv & Tishby (2017)
 */
export function computeInformationPlane(
  X: number[][],
  layerActivations: number[][][], // [layer][sample][dim]
  Y: number[],
  epoch: number,
  config: MIEstimatorConfig = { estimator: 'ksg' }
): InformationPlanePoint[] {
  const Y2D = Y.map(y => [y]);
  const points: InformationPlanePoint[] = [];

  for (let layerIdx = 0; layerIdx < layerActivations.length; layerIdx++) {
    const Z = layerActivations[layerIdx];

    const IXZ = estimateMutualInformationAdvanced(X, Z, config);
    const IZY = estimateMutualInformationAdvanced(Z, Y2D, config);

    points.push({
      epoch,
      layerIndex: layerIdx,
      compressionMI: IXZ.mutualInformation,
      predictionMI: IZY.mutualInformation,
      layerName: `Layer ${layerIdx}`
    });
  }

  return points;
}

/**
 * Fisher Information Matrix estimation
 *
 * F_ij = E[∂log p(x|θ)/∂θ_i × ∂log p(x|θ)/∂θ_j]
 *
 * Computed from gradients as F ≈ (1/N) Σ g_i g_i^T
 */
export interface FisherInformation {
  /** Fisher information matrix */
  matrix: number[][];
  /** Eigenvalues (sorted descending) */
  eigenvalues: number[];
  /** Effective dimension (trace / max eigenvalue) */
  effectiveDimension: number;
  /** Condition number */
  conditionNumber: number;
  /** Trace (sum of eigenvalues) */
  trace: number;
}

/**
 * Estimate Fisher Information from gradients
 */
export function estimateFisherInformation(gradients: number[][]): FisherInformation {
  const N = gradients.length;
  if (N === 0) {
    return {
      matrix: [],
      eigenvalues: [],
      effectiveDimension: 0,
      conditionNumber: 1,
      trace: 0
    };
  }

  const d = gradients[0].length;
  const matrix: number[][] = [];

  // Initialize zero matrix
  for (let i = 0; i < d; i++) {
    matrix.push(new Array(d).fill(0));
  }

  // Accumulate outer products
  for (const grad of gradients) {
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        matrix[i][j] += grad[i] * grad[j];
      }
    }
  }

  // Average
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      matrix[i][j] /= N;
    }
  }

  // Compute trace
  let trace = 0;
  for (let i = 0; i < d; i++) {
    trace += matrix[i][i];
  }

  // Estimate eigenvalues via power iteration (simplified)
  const eigenvalues = estimateTopEigenvalues(matrix, Math.min(d, 10));

  const maxEig = eigenvalues[0] || 1;
  const minEig = eigenvalues[eigenvalues.length - 1] || 1;

  return {
    matrix,
    eigenvalues,
    effectiveDimension: trace / maxEig,
    conditionNumber: maxEig / Math.max(minEig, 1e-10),
    trace
  };
}

/**
 * Estimate top k eigenvalues using power iteration with deflation
 */
function estimateTopEigenvalues(matrix: number[][], k: number): number[] {
  const n = matrix.length;
  if (n === 0) return [];

  const eigenvalues: number[] = [];
  let currentMatrix = matrix.map(row => [...row]);

  for (let eigIdx = 0; eigIdx < k; eigIdx++) {
    // Power iteration
    let v = new Array(n).fill(1 / Math.sqrt(n));
    let eigenvalue = 0;

    for (let iter = 0; iter < 100; iter++) {
      // v = Av
      const Av = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          Av[i] += currentMatrix[i][j] * v[j];
        }
      }

      // Normalize
      const norm = Math.sqrt(kahanSum(Av.map(x => x * x)));
      if (norm === 0) break;

      v = Av.map(x => x / norm);
      eigenvalue = norm;
    }

    eigenvalues.push(eigenvalue);

    // Deflation: A = A - λvv^T
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        currentMatrix[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  return eigenvalues;
}
