/**
 * Lightweight attention modules for experimenting with recurrent and transformer
 * architectures. The implementation emphasises readability and numerical
 * stability so that the building blocks can be mixed with classic RNN cells or
 * newer attention-driven layers.
 */

import { stableSoftmax } from '../lib/MathUtils';

export type Matrix = number[][];

export interface AttentionWeights {
  query: Matrix;
  key: Matrix;
  value: Matrix;
}

export interface ScaledDotProductAttentionOptions {
  /** Optional temperature. Defaults to sqrt(keyDim). */
  temperature?: number;
  /** Enables causal masking for autoregressive decoding. */
  causal?: boolean;
}

function matmul(a: Matrix, b: Matrix): Matrix {
  if (a[0].length !== b.length) throw new Error('Matrix dimensions do not align.');
  const result: Matrix = Array.from({ length: a.length }, () => new Array(b[0].length).fill(0));
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < b.length; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < b[0].length; j++) {
        result[i][j] += aik * b[k][j];
      }
    }
  }
  return result;
}

function transpose(matrix: Matrix): Matrix {
  return matrix[0].map((_, i) => matrix.map((row) => row[i]));
}

function applyMask(matrix: Matrix, mask: Matrix): Matrix {
  return matrix.map((row, i) => row.map((value, j) => (mask[i]?.[j] ? value : Number.NEGATIVE_INFINITY)));
}

export function scaledDotProductAttention(
  queries: Matrix,
  keys: Matrix,
  values: Matrix,
  options: ScaledDotProductAttentionOptions = {}
): { output: Matrix; attention: Matrix } {
  const { temperature, causal = false } = options;
  const dk = keys[0].length;
  const scale = 1 / Math.sqrt(temperature ?? dk);

  let scores = matmul(queries, transpose(keys)).map((row) => row.map((v) => v * scale));

  if (causal) {
    const mask: Matrix = scores.map((row, i) => row.map((_, j) => (j <= i ? 1 : 0)));
    scores = applyMask(scores, mask);
  }

  const attention = scores.map((row) => stableSoftmax(row));
  const output = matmul(attention, values);
  return { output, attention };
}

export interface MultiHeadAttentionConfig {
  heads: number;
  modelDim: number;
  keyDim: number;
  valueDim: number;
  dropout?: number;
}

export class MultiHeadAttention {
  private readonly headDim: number;
  private readonly projectionScale: number;
  private readonly dropout: number;

  constructor(private readonly config: MultiHeadAttentionConfig) {
    if (config.modelDim % config.heads !== 0) {
      throw new Error('modelDim must be divisible by number of heads.');
    }
    this.headDim = config.modelDim / config.heads;
    this.projectionScale = 1 / Math.sqrt(this.headDim);
    this.dropout = config.dropout ?? 0;
  }

  project(input: Matrix, weights: AttentionWeights): AttentionWeights {
    const project = (matrix: Matrix) => matmul(input, matrix);
    return {
      query: project(weights.query),
      key: project(weights.key),
      value: project(weights.value)
    };
  }

  splitHeads(matrix: Matrix): Matrix[] {
    const heads: Matrix[] = [];
    for (let h = 0; h < this.config.heads; h++) {
      heads.push(matrix.map((row) => row.slice(h * this.headDim, (h + 1) * this.headDim)));
    }
    return heads;
  }

  combineHeads(heads: Matrix[]): Matrix {
    return heads[0].map((_, rowIndex) =>
      heads
        .map((head) => head[rowIndex])
        .reduce((acc, row) => acc.concat(row), [] as number[])
    );
  }

  private applyDropout(matrix: Matrix): Matrix {
    if (this.dropout <= 0) return matrix;
    return matrix.map((row) =>
      row.map((value) => (Math.random() < this.dropout ? 0 : value / (1 - this.dropout)))
    );
  }

  forward(inputs: Matrix, weights: AttentionWeights, options: ScaledDotProductAttentionOptions = {}) {
    const projections = this.project(inputs, weights);
    const queries = this.splitHeads(projections.query);
    const keys = this.splitHeads(projections.key);
    const values = this.splitHeads(projections.value);

    const headOutputs: Matrix[] = [];
    for (let h = 0; h < this.config.heads; h++) {
      const { output } = scaledDotProductAttention(queries[h], keys[h], values[h], {
        ...options,
        temperature: this.config.keyDim
      });
      headOutputs.push(this.applyDropout(output));
    }

    return this.combineHeads(headOutputs).map((row) => row.map((v) => v * this.projectionScale));
  }
}
