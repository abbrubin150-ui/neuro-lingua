/**
 * Spectral Graph Analysis for Attention Patterns
 *
 * This module provides graph-theoretic analysis of attention mechanisms:
 * - Laplacian spectrum analysis
 * - Algebraic connectivity (Fiedler value)
 * - Community structure detection
 * - Expander graph properties
 *
 * Mathematical Framework:
 * - Graph Laplacian: L = D - A (unnormalized) or L = I - D^(-1/2)AD^(-1/2) (normalized)
 * - Spectral clustering via Fiedler vector
 * - Cheeger inequality for expansion
 *
 * References:
 * - Chung, F. (1997) "Spectral Graph Theory"
 * - Von Luxburg (2007) "A Tutorial on Spectral Clustering"
 * - Hoory et al. (2006) "Expander Graphs and their Applications"
 */

import { kahanSum } from './numerics';

/**
 * Attention graph analysis result
 */
export interface AttentionGraphAnalysis {
  /** Second smallest eigenvalue of Laplacian (Fiedler value) */
  algebraicConnectivity: number;
  /** Fiedler vector (eigenvector for algebraic connectivity) */
  fiedlerVector: number[];
  /** Spectral gap (λ₂ - λ₁) */
  spectralGap: number;
  /** All Laplacian eigenvalues */
  eigenvalues: number[];
  /** Detected community structure */
  communityStructure: number[][];
  /** Number of connected components */
  numComponents: number;
  /** Graph density */
  density: number;
  /** Average attention weight */
  averageAttention: number;
}

/**
 * Expander graph properties
 */
export interface ExpanderProperties {
  /** Expansion ratio (edge boundary / vertex set size) */
  expansionRatio: number;
  /** Whether the graph is a good expander */
  isExpander: boolean;
  /** Cheeger constant (edge expansion) */
  cheegerConstant: number;
  /** Upper bound on mixing time */
  mixingTimeBound: number;
  /** Recommendations for improving expansion */
  recommendations: string[];
}

/**
 * Sparse attention pattern representation
 */
export interface SparseAttentionPattern {
  /** Number of positions */
  numPositions: number;
  /** Sparsity pattern: which positions attend to which */
  connections: Map<number, Set<number>>;
  /** Pattern type (local, global, strided, etc.) */
  patternType: string;
}

/**
 * Compute graph Laplacian from attention weights.
 *
 * The attention matrix A is treated as an adjacency matrix.
 * Laplacian L = D - A where D is degree matrix.
 *
 * @param attention - Attention weight matrix [n x n]
 * @param normalized - Use normalized Laplacian (default: true)
 * @returns Laplacian matrix
 */
export function computeLaplacian(attention: number[][], normalized = true): number[][] {
  const n = attention.length;
  if (n === 0) return [];

  // Compute degree matrix D (diagonal with row sums)
  const degrees = attention.map((row) => kahanSum(row));

  // Initialize Laplacian
  const L: number[][] = [];
  for (let i = 0; i < n; i++) {
    L.push(new Array(n).fill(0));
  }

  if (normalized) {
    // Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          L[i][j] = degrees[i] > 0 ? 1 : 0;
        } else {
          const denom = Math.sqrt(degrees[i] * degrees[j]);
          L[i][j] = denom > 0 ? -attention[i][j] / denom : 0;
        }
      }
    }
  } else {
    // Unnormalized Laplacian: L = D - A
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          L[i][j] = degrees[i];
        } else {
          L[i][j] = -attention[i][j];
        }
      }
    }
  }

  return L;
}

/**
 * Compute eigenvalues and eigenvectors of a symmetric matrix using
 * power iteration with deflation.
 *
 * @param matrix - Symmetric matrix
 * @param numEigenvalues - Number of eigenvalues to compute
 * @param maxIterations - Maximum iterations per eigenvalue
 * @returns Eigenvalues and eigenvectors
 */
function computeEigendecomposition(
  matrix: number[][],
  numEigenvalues: number,
  maxIterations = 1000
): { eigenvalues: number[]; eigenvectors: number[][] } {
  const n = matrix.length;
  if (n === 0) return { eigenvalues: [], eigenvectors: [] };

  const eigenvalues: number[] = [];
  const eigenvectors: number[][] = [];

  // Work with a copy to allow deflation
  const M: number[][] = matrix.map((row) => [...row]);

  for (let k = 0; k < numEigenvalues && k < n; k++) {
    // Initialize random vector
    let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(kahanSum(v.map((x) => x * x)));
    v = v.map((x) => x / norm);

    let eigenvalue = 0;

    // Power iteration
    for (let iter = 0; iter < maxIterations; iter++) {
      // Multiply: w = M * v
      const w = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          w[i] += M[i][j] * v[j];
        }
      }

      // Compute Rayleigh quotient
      const vMv = kahanSum(v.map((vi, i) => vi * w[i]));
      const vv = kahanSum(v.map((x) => x * x));
      eigenvalue = vMv / vv;

      // Normalize
      norm = Math.sqrt(kahanSum(w.map((x) => x * x)));
      if (norm === 0) break;

      const vNew = w.map((x) => x / norm);

      // Check convergence
      const diff = kahanSum(vNew.map((x, i) => Math.abs(x - v[i])));
      v = vNew;

      if (diff < 1e-10) break;
    }

    eigenvalues.push(eigenvalue);
    eigenvectors.push(v);

    // Deflation: M = M - λ * v * v^T
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        M[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  return { eigenvalues, eigenvectors };
}

/**
 * Analyze attention graph using spectral methods.
 *
 * @param attention - Attention weight matrix [heads][n][n] or [n][n]
 * @returns Graph analysis results
 */
export function analyzeAttentionGraph(
  attention: number[][] | number[][][]
): AttentionGraphAnalysis {
  // Handle multi-head attention by averaging
  let avgAttention: number[][];

  if (attention.length > 0 && Array.isArray(attention[0][0])) {
    // Multi-head: average across heads
    const heads = attention as number[][][];
    const n = heads[0].length;
    avgAttention = [];
    for (let i = 0; i < n; i++) {
      avgAttention.push(new Array(n).fill(0));
      for (let j = 0; j < n; j++) {
        for (const head of heads) {
          avgAttention[i][j] += head[i][j];
        }
        avgAttention[i][j] /= heads.length;
      }
    }
  } else {
    avgAttention = attention as number[][];
  }

  const n = avgAttention.length;
  if (n === 0) {
    return {
      algebraicConnectivity: 0,
      fiedlerVector: [],
      spectralGap: 0,
      eigenvalues: [],
      communityStructure: [],
      numComponents: 0,
      density: 0,
      averageAttention: 0
    };
  }

  // Compute Laplacian
  const L = computeLaplacian(avgAttention, true);

  // Compute eigenvalues (for Laplacian, we want smallest)
  // Use negative of L to get smallest eigenvalues via power iteration
  const negL = L.map((row) => row.map((x) => -x));
  const { eigenvalues: negEigs, eigenvectors } = computeEigendecomposition(negL, Math.min(n, 10));

  // Convert back to positive eigenvalues and sort
  const eigenvalues = negEigs.map((e) => -e).sort((a, b) => a - b);

  // Algebraic connectivity is the second smallest eigenvalue
  const algebraicConnectivity = eigenvalues.length >= 2 ? eigenvalues[1] : 0;

  // Fiedler vector is the eigenvector corresponding to λ₂
  const fiedlerVector = eigenvectors.length >= 2 ? eigenvectors[1] : new Array(n).fill(0);

  // Spectral gap
  const spectralGap = eigenvalues.length >= 2 ? eigenvalues[1] - eigenvalues[0] : 0;

  // Count connected components (number of zero eigenvalues)
  const tolerance = 1e-6;
  const numComponents = eigenvalues.filter((e) => Math.abs(e) < tolerance).length;

  // Detect community structure using Fiedler vector
  const communityStructure = detectCommunities(fiedlerVector);

  // Graph density
  let edgeCount = 0;
  let totalWeight = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && avgAttention[i][j] > 0.01) {
        edgeCount++;
        totalWeight += avgAttention[i][j];
      }
    }
  }
  const density = edgeCount / (n * (n - 1));
  const averageAttention = edgeCount > 0 ? totalWeight / edgeCount : 0;

  return {
    algebraicConnectivity,
    fiedlerVector,
    spectralGap,
    eigenvalues,
    communityStructure,
    numComponents,
    density,
    averageAttention
  };
}

/**
 * Detect communities using Fiedler vector partitioning.
 *
 * The sign of Fiedler vector components indicates community membership.
 *
 * @param fiedlerVector - Eigenvector for algebraic connectivity
 * @returns Community assignments (list of index groups)
 */
function detectCommunities(fiedlerVector: number[]): number[][] {
  if (fiedlerVector.length === 0) return [];

  // Simple two-way partition by sign
  const positive: number[] = [];
  const negative: number[] = [];

  for (let i = 0; i < fiedlerVector.length; i++) {
    if (fiedlerVector[i] >= 0) {
      positive.push(i);
    } else {
      negative.push(i);
    }
  }

  return [positive, negative].filter((c) => c.length > 0);
}

/**
 * Check expander graph properties of attention pattern.
 *
 * Good expanders have:
 * - Large spectral gap (fast mixing)
 * - Large Cheeger constant (good expansion)
 * - Low diameter (short paths)
 *
 * @param attention - Attention weight matrix
 * @param threshold - Threshold for considering an edge present
 * @returns Expander properties
 */
export function checkExpanderProperties(
  attention: number[][],
  threshold = 0.01
): ExpanderProperties {
  const n = attention.length;
  const recommendations: string[] = [];

  if (n === 0) {
    return {
      expansionRatio: 0,
      isExpander: false,
      cheegerConstant: 0,
      mixingTimeBound: Infinity,
      recommendations: ['Empty attention matrix']
    };
  }

  // Compute degrees
  const degrees = attention.map((row) => row.filter((w) => w > threshold).length);
  const avgDegree = kahanSum(degrees) / n;

  // Analyze spectral properties
  const L = computeLaplacian(attention, true);
  const negL = L.map((row) => row.map((x) => -x));
  const { eigenvalues: negEigs } = computeEigendecomposition(negL, Math.min(n, 5));
  const eigenvalues = negEigs.map((e) => -e).sort((a, b) => a - b);

  // Algebraic connectivity λ₂
  const lambda2 = eigenvalues.length >= 2 ? eigenvalues[1] : 0;

  // Cheeger inequality: h ≥ λ₂/2 and h ≤ √(2λ₂)
  // where h is the Cheeger constant (edge expansion)
  const cheegerLower = lambda2 / 2;
  const cheegerUpper = Math.sqrt(2 * lambda2);
  const cheegerConstant = (cheegerLower + cheegerUpper) / 2;

  // Expansion ratio: edge boundary / vertex set
  // Approximate by computing for random vertex subsets
  let totalExpansion = 0;
  const numTrials = Math.min(10, n);

  for (let trial = 0; trial < numTrials; trial++) {
    // Take random subset of size n/2
    const subsetSize = Math.floor(n / 2);
    const subset = new Set<number>();
    while (subset.size < subsetSize) {
      subset.add(Math.floor(Math.random() * n));
    }

    // Count edges crossing the boundary
    let boundaryEdges = 0;
    for (const i of subset) {
      for (let j = 0; j < n; j++) {
        if (!subset.has(j) && attention[i][j] > threshold) {
          boundaryEdges++;
        }
      }
    }

    totalExpansion += boundaryEdges / subsetSize;
  }

  const expansionRatio = totalExpansion / numTrials;

  // Is it a good expander?
  // Ramanujan bound: spectral gap ≥ 2√(d-1) for d-regular graphs
  const ramanujanBound = 2 * Math.sqrt(Math.max(avgDegree - 1, 0));
  const isExpander = lambda2 >= ramanujanBound * 0.5 && expansionRatio >= 0.1;

  // Mixing time bound: O(log(n) / λ₂)
  const mixingTimeBound = lambda2 > 0 ? Math.ceil(Math.log(n) / lambda2) : Infinity;

  // Recommendations
  if (lambda2 < 0.1) {
    recommendations.push(
      'Low algebraic connectivity indicates potential bottlenecks in attention flow.'
    );
  }

  if (expansionRatio < 0.1) {
    recommendations.push('Low expansion ratio may cause slow information propagation.');
  }

  if (avgDegree < n * 0.1) {
    recommendations.push(
      'Sparse attention pattern. Consider adding global tokens or more connections.'
    );
  }

  if (mixingTimeBound > n) {
    recommendations.push('High mixing time suggests information takes many layers to propagate.');
  }

  if (recommendations.length === 0) {
    recommendations.push('Attention pattern has good expansion properties.');
  }

  return {
    expansionRatio,
    isExpander,
    cheegerConstant,
    mixingTimeBound,
    recommendations
  };
}

/**
 * Analyze sparse attention pattern efficiency.
 *
 * @param pattern - Sparse attention pattern
 * @returns Analysis results
 */
export function analyzeSparseAttentionPattern(pattern: SparseAttentionPattern): {
  density: number;
  reachability: number;
  diameter: number;
  clusteringCoefficient: number;
  isSymmetric: boolean;
  recommendations: string[];
} {
  const { numPositions, connections } = pattern;
  const recommendations: string[] = [];

  if (numPositions === 0) {
    return {
      density: 0,
      reachability: 0,
      diameter: 0,
      clusteringCoefficient: 0,
      isSymmetric: true,
      recommendations: ['Empty pattern']
    };
  }

  // Count edges and compute density
  let edgeCount = 0;
  for (const neighbors of connections.values()) {
    edgeCount += neighbors.size;
  }
  const density = edgeCount / (numPositions * numPositions);

  // Check symmetry
  let isSymmetric = true;
  for (const [i, neighbors] of connections) {
    for (const j of neighbors) {
      if (!connections.get(j)?.has(i)) {
        isSymmetric = false;
        break;
      }
    }
    if (!isSymmetric) break;
  }

  // Compute reachability using BFS from each node
  let totalReachable = 0;
  let maxDist = 0;

  for (let start = 0; start < numPositions; start++) {
    const distances = new Map<number, number>();
    const queue: number[] = [start];
    distances.set(start, 0);

    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentDist = distances.get(current)!;

      const neighbors = connections.get(current) ?? new Set();
      for (const neighbor of neighbors) {
        if (!distances.has(neighbor)) {
          distances.set(neighbor, currentDist + 1);
          queue.push(neighbor);
          maxDist = Math.max(maxDist, currentDist + 1);
        }
      }
    }

    totalReachable += distances.size;
  }

  const reachability = totalReachable / (numPositions * numPositions);
  const diameter = maxDist;

  // Compute clustering coefficient
  let totalClustering = 0;
  for (let i = 0; i < numPositions; i++) {
    const neighbors = connections.get(i) ?? new Set();
    const neighborList = Array.from(neighbors);
    const k = neighborList.length;

    if (k < 2) continue;

    // Count edges between neighbors
    let neighborEdges = 0;
    for (let a = 0; a < k; a++) {
      for (let b = a + 1; b < k; b++) {
        if (connections.get(neighborList[a])?.has(neighborList[b])) {
          neighborEdges++;
        }
      }
    }

    totalClustering += (2 * neighborEdges) / (k * (k - 1));
  }

  const clusteringCoefficient = totalClustering / numPositions;

  // Generate recommendations
  if (reachability < 1) {
    recommendations.push(
      'Not all positions are reachable. Add global tokens or increase local window.'
    );
  }

  if (diameter > Math.log2(numPositions) * 2) {
    recommendations.push(
      'High diameter indicates slow information flow. Consider shortcuts or global attention.'
    );
  }

  if (clusteringCoefficient < 0.1) {
    recommendations.push('Low clustering. Consider adding local attention or strided patterns.');
  }

  if (density > 0.5) {
    recommendations.push('High density reduces efficiency gains from sparsity.');
  }

  if (recommendations.length === 0) {
    recommendations.push('Pattern appears well-balanced for sparse attention.');
  }

  return {
    density,
    reachability,
    diameter,
    clusteringCoefficient,
    isSymmetric,
    recommendations
  };
}

/**
 * Generate common sparse attention patterns.
 */
export function generateSparsePattern(
  numPositions: number,
  patternType: 'local' | 'strided' | 'global' | 'bigbird',
  windowSize = 3,
  numGlobalTokens = 1
): SparseAttentionPattern {
  const connections = new Map<number, Set<number>>();

  for (let i = 0; i < numPositions; i++) {
    connections.set(i, new Set());
  }

  switch (patternType) {
    case 'local':
      // Local sliding window attention
      for (let i = 0; i < numPositions; i++) {
        for (
          let j = Math.max(0, i - windowSize);
          j <= Math.min(numPositions - 1, i + windowSize);
          j++
        ) {
          connections.get(i)!.add(j);
        }
      }
      break;

    case 'strided':
      // Strided attention (every k-th position)
      for (let i = 0; i < numPositions; i++) {
        // Local window
        for (
          let j = Math.max(0, i - windowSize);
          j <= Math.min(numPositions - 1, i + windowSize);
          j++
        ) {
          connections.get(i)!.add(j);
        }
        // Strided
        for (let j = 0; j < numPositions; j += windowSize * 2) {
          connections.get(i)!.add(j);
        }
      }
      break;

    case 'global':
      // Global tokens attend to all
      for (let i = 0; i < numPositions; i++) {
        // Local attention
        for (
          let j = Math.max(0, i - windowSize);
          j <= Math.min(numPositions - 1, i + windowSize);
          j++
        ) {
          connections.get(i)!.add(j);
        }
        // Global tokens
        for (let g = 0; g < numGlobalTokens; g++) {
          connections.get(i)!.add(g);
          connections.get(g)!.add(i);
        }
      }
      break;

    case 'bigbird': {
      // BigBird pattern: local + global + random
      const numRandom = Math.ceil(numPositions * 0.1);

      for (let i = 0; i < numPositions; i++) {
        // Local window
        for (
          let j = Math.max(0, i - windowSize);
          j <= Math.min(numPositions - 1, i + windowSize);
          j++
        ) {
          connections.get(i)!.add(j);
        }

        // Global tokens
        for (let g = 0; g < numGlobalTokens; g++) {
          connections.get(i)!.add(g);
          connections.get(g)!.add(i);
        }

        // Random connections
        for (let r = 0; r < numRandom; r++) {
          const randomIdx = Math.floor(Math.random() * numPositions);
          connections.get(i)!.add(randomIdx);
        }
      }
      break;
    }
  }

  return {
    numPositions,
    connections,
    patternType
  };
}
