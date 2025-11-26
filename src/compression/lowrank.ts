/**
 * Low-Rank Approximation Module
 *
 * Uses Singular Value Decomposition (SVD) to approximate weight matrices
 * as products of smaller matrices:
 *
 * W (m×n) ≈ U (m×k) × Σ (k×k) × V^T (k×n)
 *
 * Where k < min(m,n) is the rank.
 *
 * Benefits:
 * - Reduces parameters: mn → k(m+n)
 * - Compression ratio: mn / k(m+n)
 * - Example: 128×64 matrix with k=16 → 8192 params → 3072 params (2.67x)
 *
 * Reference:
 * - Denil et al. (2013) "Predicting Parameters in Deep Learning"
 */

export interface SVDResult {
  U: number[][];
  S: number[];
  V: number[][];
  rank: number;
}

export interface LowRankWeights {
  U: number[][];
  SV: number[][]; // S × V^T combined
  rank: number;
  originalShape: [number, number];
  compressionRatio: number;
}

export interface LowRankModel {
  embedding: LowRankWeights;
  wHidden: LowRankWeights;
  wOutput: LowRankWeights;
  totalCompressionRatio: number;
  approximationError: number;
}

/**
 * Compute matrix transpose
 */
export function transpose(matrix: number[][]): number[][] {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;

  const result: number[][] = Array(cols)
    .fill(0)
    .map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }

  return result;
}

/**
 * Matrix multiplication
 */
export function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B[0]?.length ?? 0;
  const p = B.length;

  if (A[0].length !== p) {
    throw new Error(`Matrix dimensions incompatible: ${A[0].length} !== ${p}`);
  }

  const result: number[][] = Array(m)
    .fill(0)
    .map(() => Array(n).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}

/**
 * Frobenius norm of matrix
 */
export function frobeniusNorm(matrix: number[][]): number {
  let sum = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      sum += matrix[i][j] * matrix[i][j];
    }
  }
  return Math.sqrt(sum);
}

/**
 * Simplified SVD using power iteration
 * For production, consider using numeric.js or ml-matrix libraries
 *
 * This is a basic implementation for educational purposes.
 * Returns top k singular vectors.
 */
export function simplifiedSVD(matrix: number[][], k: number): SVDResult {
  const m = matrix.length;
  const n = matrix[0]?.length ?? 0;
  const maxRank = Math.min(m, n, k);

  // For simplicity, we'll use a basic power iteration approach
  // In production, use a proper SVD library

  const U: number[][] = [];
  const S: number[] = [];
  const V: number[][] = [];

  // Compute A^T A for right singular vectors
  const AtA = matmul(transpose(matrix), matrix);

  // Find top k eigenvectors of A^T A
  for (let i = 0; i < maxRank; i++) {
    // Initialize random vector
    let v = Array(n)
      .fill(0)
      .map(() => Math.random() - 0.5);

    // Power iteration (simplified)
    for (let iter = 0; iter < 10; iter++) {
      // Multiply by AtA
      const newV: number[] = Array(n).fill(0);
      for (let row = 0; row < n; row++) {
        for (let col = 0; col < n; col++) {
          newV[row] += AtA[row][col] * v[col];
        }
      }

      // Normalize
      const norm = Math.sqrt(newV.reduce((sum, x) => sum + x * x, 0));
      if (norm > 1e-10) {
        v = newV.map((x) => x / norm);
      }
    }

    V.push(v);

    // Compute corresponding singular value and left vector
    const Av: number[] = Array(m).fill(0);
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        Av[row] += matrix[row][col] * v[col];
      }
    }

    const sigma = Math.sqrt(Av.reduce((sum, x) => sum + x * x, 0));
    S.push(sigma);

    const u = Av.map((x) => (sigma > 1e-10 ? x / sigma : 0));
    U.push(u);

    // Deflate matrix for next iteration
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        matrix[row][col] -= sigma * u[row] * v[col];
      }
    }
  }

  return {
    U: transpose(U), // m×k
    S, // k singular values
    V: transpose(V), // n×k (transposed)
    rank: maxRank
  };
}

/**
 * Approximate matrix using top k singular values
 */
export function lowRankApproximation(matrix: number[][], rank: number): LowRankWeights {
  const originalShape: [number, number] = [matrix.length, matrix[0]?.length ?? 0];

  // Make a copy to avoid modifying original during SVD
  const matrixCopy = matrix.map((row) => [...row]);

  // Compute SVD
  const svd = simplifiedSVD(matrixCopy, rank);

  // U is m×k, we keep it
  const U = svd.U;

  // Compute S × V^T as a single matrix (k×n)
  const SV: number[][] = Array(rank)
    .fill(0)
    .map(() => Array(originalShape[1]).fill(0));

  for (let i = 0; i < rank; i++) {
    for (let j = 0; j < originalShape[1]; j++) {
      SV[i][j] = svd.S[i] * svd.V[j][i];
    }
  }

  // Calculate compression ratio
  const originalParams = originalShape[0] * originalShape[1];
  const compressedParams = rank * (originalShape[0] + originalShape[1]);
  const compressionRatio = originalParams / compressedParams;

  return {
    U,
    SV,
    rank,
    originalShape,
    compressionRatio
  };
}

/**
 * Reconstruct original matrix from low-rank form
 */
export function reconstructFromLowRank(lowRank: LowRankWeights): number[][] {
  return matmul(lowRank.U, lowRank.SV);
}

/**
 * Calculate approximation error (Frobenius norm)
 */
export function approximationError(original: number[][], approximation: number[][]): number {
  const diff: number[][] = original.map((row, i) => row.map((val, j) => val - approximation[i][j]));

  return frobeniusNorm(diff);
}

/**
 * Determine optimal rank for target compression ratio
 */
export function findOptimalRank(m: number, n: number, targetCompressionRatio: number): number {
  // mn / (k(m+n)) = targetRatio
  // k = mn / (targetRatio * (m+n))

  const k = Math.floor((m * n) / (targetCompressionRatio * (m + n)));
  return Math.max(1, Math.min(k, Math.min(m, n)));
}

/**
 * Apply low-rank approximation to all model weights
 * Returns compression statistics and approximation errors
 */
export function compressModelLowRank(
  embedding: number[][],
  wHidden: number[][],
  wOutput: number[][],
  rank: number
): {
  compressed: LowRankModel;
  errors: { embedding: number; wHidden: number; wOutput: number };
} {
  // Compress each weight matrix
  const embeddingLR = lowRankApproximation(embedding, rank);
  const wHiddenLR = lowRankApproximation(wHidden, rank);
  const wOutputLR = lowRankApproximation(wOutput, rank);

  // Calculate errors
  const embeddingReconstructed = reconstructFromLowRank(embeddingLR);
  const wHiddenReconstructed = reconstructFromLowRank(wHiddenLR);
  const wOutputReconstructed = reconstructFromLowRank(wOutputLR);

  const errors = {
    embedding: approximationError(embedding, embeddingReconstructed),
    wHidden: approximationError(wHidden, wHiddenReconstructed),
    wOutput: approximationError(wOutput, wOutputReconstructed)
  };

  // Calculate total compression ratio
  const totalOriginal =
    embedding.length * embedding[0].length +
    wHidden.length * wHidden[0].length +
    wOutput.length * wOutput[0].length;

  const totalCompressed =
    rank * (embedding.length + embedding[0].length) +
    rank * (wHidden.length + wHidden[0].length) +
    rank * (wOutput.length + wOutput[0].length);

  const totalCompressionRatio = totalOriginal / totalCompressed;

  const avgApproximationError = (errors.embedding + errors.wHidden + errors.wOutput) / 3;

  return {
    compressed: {
      embedding: embeddingLR,
      wHidden: wHiddenLR,
      wOutput: wOutputLR,
      totalCompressionRatio,
      approximationError: avgApproximationError
    },
    errors
  };
}

/**
 * Serialize low-rank weights to JSON
 */
export function serializeLowRankWeights(weights: LowRankWeights): object {
  return {
    U: weights.U,
    SV: weights.SV,
    rank: weights.rank,
    originalShape: weights.originalShape,
    compressionRatio: weights.compressionRatio
  };
}

/**
 * Deserialize low-rank weights from JSON
 */
export function deserializeLowRankWeights(data: any): LowRankWeights {
  return {
    U: data.U,
    SV: data.SV,
    rank: data.rank,
    originalShape: data.originalShape,
    compressionRatio: data.compressionRatio
  };
}
