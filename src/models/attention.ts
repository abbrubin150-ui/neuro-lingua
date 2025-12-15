/**
 * Lightweight attention modules for experimenting with recurrent and transformer
 * architectures. The implementation emphasises readability and numerical
 * stability so that the building blocks can be mixed with classic RNN cells or
 * newer attention-driven layers.
 */

import { stableSoftmax } from '../lib/MathUtils';
import { GPUNeuralOps } from '../backend/gpu_neural_ops';

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

export interface MultiHeadForwardOptions extends ScaledDotProductAttentionOptions {
  /**
   * Absolute positions for each token in the sequence. When provided, rotary
   * positional embeddings (RoPE) are applied to queries/keys for long-context
   * extrapolation (v4 default).
   */
  positions?: number[];
  /** RoPE base. Defaults to 500000 per v4 spec. */
  ropeBase?: number;
}

function cpuMatmul(a: Matrix, b: Matrix): Matrix {
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
  return matrix.map((row, i) =>
    row.map((value, j) => (mask[i]?.[j] ? value : Number.NEGATIVE_INFINITY))
  );
}

function applyRoPE(matrix: Matrix, positions: number[], base: number): Matrix {
  // Rotary embeddings expect even dimensions so we rotate pairs (even, odd)
  const dim = matrix[0]?.length ?? 0;
  if (dim === 0 || dim % 2 !== 0 || positions.length === 0) {
    return matrix;
  }

  const halfDim = dim / 2;
  const theta = new Float64Array(halfDim);
  for (let i = 0; i < halfDim; i++) {
    theta[i] = Math.pow(base, (-2 * i) / dim);
  }

  return matrix.map((row, rowIdx) => {
    const pos = positions[rowIdx] ?? rowIdx;
    const rotated = [...row];
    for (let i = 0; i < halfDim; i++) {
      const angle = pos * theta[i];
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const even = row[2 * i];
      const odd = row[2 * i + 1];
      rotated[2 * i] = even * cos - odd * sin;
      rotated[2 * i + 1] = even * sin + odd * cos;
    }
    return rotated;
  });
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

  let scores = cpuMatmul(queries, transpose(keys)).map((row) => row.map((v) => v * scale));

  if (causal) {
    const mask: Matrix = scores.map((row, i) => row.map((_, j) => (j <= i ? 1 : 0)));
    scores = applyMask(scores, mask);
  }

  const attention = scores.map((row) => stableSoftmax(row));
  const output = cpuMatmul(attention, values);
  return { output, attention };
}

export interface MultiHeadAttentionConfig {
  heads: number;
  modelDim: number;
  keyDim: number;
  valueDim: number;
  dropout?: number;
  /**
   * Number of key-value heads for Grouped-Query Attention (GQA).
   * When numKVHeads < heads, multiple query heads share the same K/V heads.
   * This reduces KV cache memory by (heads / numKVHeads) factor.
   *
   * Examples:
   * - numKVHeads = heads: Standard Multi-Head Attention (MHA)
   * - numKVHeads = 1: Multi-Query Attention (MQA)
   * - numKVHeads = heads/4: Grouped-Query Attention (GQA) with 4:1 ratio
   *
   * Reference: Ainslie et al. (2023) "GQA: Training Generalized Multi-Query
   * Transformer Models from Multi-Head Checkpoints"
   */
  numKVHeads?: number;
  /** Optional GPU accelerator. */
  gpuOps?: GPUNeuralOps | null;
}

export class MultiHeadAttention {
  private readonly headDim: number;
  private readonly projectionScale: number;
  private readonly dropout: number;
  private readonly numKVHeads: number;
  private readonly kvGroupSize: number;
  private gpuOps: GPUNeuralOps | null;

  constructor(private readonly config: MultiHeadAttentionConfig) {
    if (config.modelDim % config.heads !== 0) {
      throw new Error('modelDim must be divisible by number of heads.');
    }
    this.headDim = config.modelDim / config.heads;
    this.projectionScale = 1 / Math.sqrt(this.headDim);
    this.dropout = config.dropout ?? 0;

    // GQA configuration
    this.numKVHeads = config.numKVHeads ?? config.heads;
    if (config.heads % this.numKVHeads !== 0) {
      throw new Error(
        `Number of query heads (${config.heads}) must be divisible by ` +
          `number of KV heads (${this.numKVHeads}).`
      );
    }
    this.kvGroupSize = config.heads / this.numKVHeads;
    this.gpuOps = config.gpuOps ?? null;
  }

  /**
   * Get the number of KV heads (for GQA weight initialization).
   */
  getNumKVHeads(): number {
    return this.numKVHeads;
  }

  /**
   * Get the KV dimension (numKVHeads * headDim).
   */
  getKVDim(): number {
    return this.numKVHeads * this.headDim;
  }

  setGPUOps(gpuOps: GPUNeuralOps | null) {
    this.gpuOps = gpuOps;
  }

  private async matmul(a: Matrix, b: Matrix): Promise<Matrix> {
    if (this.gpuOps && this.gpuOps.isEnabled()) {
      return this.gpuOps.matrixMultiply(a, b);
    }
    return cpuMatmul(a, b);
  }

  private async project(input: Matrix, weights: AttentionWeights): Promise<AttentionWeights> {
    const project = (matrix: Matrix) => this.matmul(input, matrix);
    const [query, key, value] = await Promise.all([
      project(weights.query),
      project(weights.key),
      project(weights.value)
    ]);
    return { query, key, value };
  }

  /**
   * Split matrix into query heads (numHeads).
   */
  splitHeads(matrix: Matrix): Matrix[] {
    const heads: Matrix[] = [];
    for (let h = 0; h < this.config.heads; h++) {
      heads.push(matrix.map((row) => row.slice(h * this.headDim, (h + 1) * this.headDim)));
    }
    return heads;
  }

  /**
   * Split matrix into KV heads (numKVHeads, may be fewer than query heads).
   */
  splitKVHeads(matrix: Matrix): Matrix[] {
    const heads: Matrix[] = [];
    for (let h = 0; h < this.numKVHeads; h++) {
      heads.push(matrix.map((row) => row.slice(h * this.headDim, (h + 1) * this.headDim)));
    }
    return heads;
  }

  /**
   * Repeat KV heads to match the number of query heads.
   * Each KV head is repeated kvGroupSize times.
   * This is the key operation for GQA: multiple Q heads share the same K/V.
   */
  repeatKVHeads(kvHeads: Matrix[]): Matrix[] {
    if (this.kvGroupSize === 1) {
      // Standard MHA: no repetition needed
      return kvHeads;
    }

    const repeatedHeads: Matrix[] = [];
    for (let kvIdx = 0; kvIdx < this.numKVHeads; kvIdx++) {
      const kvHead = kvHeads[kvIdx];
      // Repeat this KV head for each query head in its group
      for (let g = 0; g < this.kvGroupSize; g++) {
        // Deep copy to avoid reference issues
        repeatedHeads.push(kvHead.map((row) => [...row]));
      }
    }
    return repeatedHeads;
  }

  combineHeads(heads: Matrix[]): Matrix {
    return heads[0].map((_, rowIndex) =>
      heads.map((head) => head[rowIndex]).reduce((acc, row) => acc.concat(row), [] as number[])
    );
  }

  private applyDropout(matrix: Matrix): Matrix {
    if (this.dropout <= 0) return matrix;
    return matrix.map((row) =>
      row.map((value) => (Math.random() < this.dropout ? 0 : value / (1 - this.dropout)))
    );
  }

  private async scaledDotProductAttention(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    options: ScaledDotProductAttentionOptions = {}
  ): Promise<{ output: Matrix; attention: Matrix }> {
    const { temperature, causal = false } = options;
    const dk = keys[0].length;
    const scale = 1 / Math.sqrt(temperature ?? dk);

    let scores = (await this.matmul(queries, transpose(keys))).map((row) => row.map((v) => v * scale));

    if (causal) {
      const mask: Matrix = scores.map((row, i) => row.map((_, j) => (j <= i ? 1 : 0)));
      scores = applyMask(scores, mask);
    }

    const attention = scores.map((row) => stableSoftmax(row));
    const output = await this.matmul(attention, values);
    return { output, attention };
  }

  async forward(
    inputs: Matrix,
    weights: AttentionWeights,
    options: MultiHeadForwardOptions = {}
  ): Promise<Matrix> {
    const { positions, ropeBase = 500000, ...attentionOpts } = options;
    const projections = await this.project(inputs, weights);

    // Split queries into numHeads heads
    const queries = this.splitHeads(
      positions ? applyRoPE(projections.query, positions, ropeBase) : projections.query
    );

    // Split keys and values into numKVHeads heads (GQA: may be fewer than query heads)
    const kvKeys = this.splitKVHeads(
      positions ? applyRoPE(projections.key, positions, ropeBase) : projections.key
    );
    const kvValues = this.splitKVHeads(projections.value);

    // Repeat KV heads to match query heads count (GQA core operation)
    const keys = this.repeatKVHeads(kvKeys);
    const values = this.repeatKVHeads(kvValues);

    const headOutputs: Matrix[] = [];
    for (let h = 0; h < this.config.heads; h++) {
      const { output } = await this.scaledDotProductAttention(queries[h], keys[h], values[h], {
        ...attentionOpts,
        temperature: this.config.keyDim
      });
      headOutputs.push(this.applyDropout(output));
    }

    return this.combineHeads(headOutputs).map((row) => row.map((v) => v * this.projectionScale));
  }
}
