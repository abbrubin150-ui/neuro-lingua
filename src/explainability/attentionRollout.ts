/**
 * Attention rollout for transformer-style attention stacks.
 */

export type AttentionHead = number[][]; // shape: seq x seq
export type AttentionLayer = AttentionHead[]; // collection of heads

export interface AttentionRolloutOptions {
  /** Aggregation strategy across attention heads. */
  headAggregation?: 'mean' | 'max';
  /** Whether to inject the identity connection before multiplying layers. */
  addIdentity?: boolean;
  /** Numerical stability epsilon added before normalisation. */
  epsilon?: number;
}

function ensureSquare(matrix: number[][]): void {
  const size = matrix.length;
  for (const row of matrix) {
    if (row.length !== size) {
      throw new Error('Attention matrices must be square.');
    }
  }
}

function aggregateHeads(layer: AttentionLayer, strategy: 'mean' | 'max'): number[][] {
  if (layer.length === 0) {
    throw new Error('Attention layer must contain at least one head.');
  }
  const size = layer[0].length;
  const aggregated = Array.from({ length: size }, () => new Array(size).fill(0));
  for (const head of layer) {
    ensureSquare(head);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (strategy === 'mean') {
          aggregated[i][j] += head[i][j];
        } else {
          aggregated[i][j] = Math.max(aggregated[i][j], head[i][j]);
        }
      }
    }
  }
  if (strategy === 'mean') {
    const inv = 1 / layer.length;
    for (const row of aggregated) {
      for (let j = 0; j < row.length; j++) row[j] *= inv;
    }
  }
  return aggregated;
}

function normalize(matrix: number[][], epsilon: number): void {
  for (const row of matrix) {
    let sum = epsilon;
    for (const value of row) sum += value;
    if (sum === 0) continue;
    for (let j = 0; j < row.length; j++) row[j] /= sum;
  }
}

function identity(size: number): number[][] {
  return Array.from({ length: size }, (_, i) => {
    const row = new Array(size).fill(0);
    row[i] = 1;
    return row;
  });
}

function multiply(a: number[][], b: number[][]): number[][] {
  const size = a.length;
  const result = Array.from({ length: size }, () => new Array(size).fill(0));
  for (let i = 0; i < size; i++) {
    for (let k = 0; k < size; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < size; j++) {
        result[i][j] += aik * b[k][j];
      }
    }
  }
  return result;
}

/**
 * Perform attention rollout across stacked layers.
 */
export function attentionRollout(
  layers: AttentionLayer[],
  options: AttentionRolloutOptions = {}
): number[][] {
  if (layers.length === 0) throw new Error('No attention layers provided.');
  const strategy = options.headAggregation ?? 'mean';
  const epsilon = options.epsilon ?? 1e-6;
  const aggregatedLayers = layers.map((layer) => aggregateHeads(layer, strategy));
  const size = aggregatedLayers[0].length;
  for (const matrix of aggregatedLayers) {
    if (matrix.length !== size) {
      throw new Error('All attention layers must share the same sequence length.');
    }
    normalize(matrix, epsilon);
  }
  const rolloutStart = options.addIdentity
    ? multiply(identity(size), aggregatedLayers[0])
    : aggregatedLayers[0];
  let rollout = rolloutStart;
  for (let i = 1; i < aggregatedLayers.length; i++) {
    rollout = multiply(rollout, aggregatedLayers[i]);
  }
  return rollout;
}
