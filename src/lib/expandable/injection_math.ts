import type { CerebroBubble } from '../../types/injection';

export interface ResidualAnalysis {
  residuals: number[][];
  energies: number[];
  meanEnergy: number;
}

export function reshapeWeightMatrix(weight: Float32Array, dModel: number): number[][] {
  const rows = Math.floor(weight.length / dModel);
  const vectors: number[][] = [];

  for (let i = 0; i < rows; i += 1) {
    const row: number[] = [];
    for (let j = 0; j < dModel; j += 1) {
      row.push(weight[i * dModel + j] ?? 0);
    }
    vectors.push(row);
  }

  return vectors;
}

export function normaliseVector(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));
  if (norm === 0) return Array.from(vector);
  return vector.map((v) => v / norm);
}

export function gramSchmidt(vectors: number[][]): number[][] {
  const basis: number[][] = [];

  vectors.forEach((vector) => {
    let candidate = Array.from(vector);
    basis.forEach((b) => {
      const dot = candidate.reduce((acc, v, idx) => acc + v * b[idx], 0);
      candidate = candidate.map((v, idx) => v - dot * b[idx]);
    });

    const norm = Math.sqrt(candidate.reduce((acc, v) => acc + v * v, 0));
    if (norm > 1e-8) {
      basis.push(candidate.map((v) => v / norm));
    }
  });

  return basis;
}

export function projectOntoBasis(vector: number[], basis: number[][]): number[] {
  if (basis.length === 0) return Array.from(vector);

  let projection = new Array(vector.length).fill(0);
  basis.forEach((b) => {
    const dot = vector.reduce((acc, v, idx) => acc + v * b[idx], 0);
    projection = projection.map((v, idx) => v + dot * b[idx]);
  });
  return projection;
}

export function computeResidual(vector: number[], basis: number[][]): number[] {
  const projection = projectOntoBasis(vector, basis);
  return vector.map((v, idx) => v - projection[idx]);
}

export function analyseResiduals(bubbles: CerebroBubble[], basis: number[][]): ResidualAnalysis {
  const residuals: number[][] = [];
  const energies: number[] = [];

  bubbles.forEach((bubble) => {
    const residual = computeResidual(bubble.embedding, basis);
    const energy = residual.reduce((acc, v) => acc + v * v, 0);
    residuals.push(residual);
    energies.push(energy);
  });

  const meanEnergy = energies.length
    ? energies.reduce((acc, e) => acc + e, 0) / Math.max(1, energies.length)
    : 0;

  return { residuals, energies, meanEnergy };
}

export function weightedCovariance(bubbles: CerebroBubble[], dimension: number): number[][] {
  const mean = new Array(dimension).fill(0);
  const totalActivation = bubbles.reduce((acc, b) => acc + b.activation, 0) || 1;

  bubbles.forEach((bubble) => {
    bubble.embedding.forEach((v, idx) => {
      mean[idx] += v * bubble.activation;
    });
  });

  const weightedMean = mean.map((v) => v / totalActivation);
  const covariance = Array.from({ length: dimension }, () => new Array(dimension).fill(0));

  bubbles.forEach((bubble) => {
    const centered = bubble.embedding.map((v, idx) => v - weightedMean[idx]);
    centered.forEach((v_i, i) => {
      centered.forEach((v_j, j) => {
        covariance[i][j] += bubble.activation * v_i * v_j;
      });
    });
  });

  const normaliser = Math.max(totalActivation, 1);
  return covariance.map((row) => row.map((v) => v / normaliser));
}

export function orthogonalProjector(basis: number[][], dimension: number): number[][] {
  const projector = Array.from({ length: dimension }, (_, i) =>
    Array.from({ length: dimension }, (_, j) => (i === j ? 1 : 0))
  );

  basis.forEach((b) => {
    b.forEach((b_i, i) => {
      b.forEach((b_j, j) => {
        projector[i][j] -= b_i * b_j;
      });
    });
  });

  return projector;
}

export function multiplyMatrices(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0]?.length ?? 0;
  const inner = B.length;
  const result = Array.from({ length: rows }, () => new Array(cols).fill(0));

  for (let i = 0; i < rows; i += 1) {
    for (let k = 0; k < inner; k += 1) {
      for (let j = 0; j < cols; j += 1) {
        result[i][j] += (A[i]?.[k] ?? 0) * (B[k]?.[j] ?? 0);
      }
    }
  }

  return result;
}

export function projectCovariance(covariance: number[][], projector: number[][]): number[][] {
  // Σ^⊥ = P Σ P
  const temp = multiplyMatrices(projector, covariance);
  return multiplyMatrices(temp, projector);
}

export function powerIteration(matrix: number[][], iterations = 25): number[] {
  const dimension = matrix.length;
  let vector = Array.from({ length: dimension }, () => Math.random());
  vector = normaliseVector(vector);

  for (let iter = 0; iter < iterations; iter += 1) {
    const next = new Array(dimension).fill(0);
    for (let i = 0; i < dimension; i += 1) {
      for (let j = 0; j < dimension; j += 1) {
        next[i] += (matrix[i]?.[j] ?? 0) * vector[j];
      }
    }
    const normNext = normaliseVector(next);
    vector = normNext;
  }

  return vector;
}

export function topEigenvectors(matrix: number[][], k: number): number[][] {
  const dimension = matrix.length;
  const vectors: number[][] = [];
  const working = matrix.map((row) => Array.from(row));

  for (let i = 0; i < k; i += 1) {
    const eigVec = powerIteration(working);
    vectors.push(normaliseVector(eigVec));

    // Deflation
    const outer = eigVec.map((v_i) => eigVec.map((v_j) => v_i * v_j));
    for (let r = 0; r < dimension; r += 1) {
      for (let c = 0; c < dimension; c += 1) {
        working[r][c] -= outer[r][c];
      }
    }
  }

  return vectors;
}

export function trace(matrix: number[][]): number {
  return matrix.reduce((acc, row, idx) => acc + (row[idx] ?? 0), 0);
}

export function estimateEnergyGain(residualTrace: number, orthPenalty: number): number {
  return Math.max(0, residualTrace - orthPenalty * residualTrace * 0.5);
}

export function suggestKFromEnergy(
  meanResidual: number,
  residualTrace: number,
  hiddenSize: number
): number {
  if (residualTrace <= 0 || meanResidual <= 0) return 0;
  const scaled = Math.ceil((residualTrace / meanResidual) ** 0.5);
  return Math.max(1, Math.min(hiddenSize, scaled));
}
