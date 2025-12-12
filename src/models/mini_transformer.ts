import { MultiHeadAttention, AttentionWeights, Matrix, MultiHeadForwardOptions } from './attention';
import { applyDropConnect } from './regularizers';
import { rmsNorm, RMSNormState } from '../lib/RMSNorm';

export interface FeedForwardConfig {
  hiddenDim: number;
  activation?: (x: number) => number;
}

export interface MiniTransformerConfig {
  modelDim: number;
  heads: number;
  ff: FeedForwardConfig;
  attentionDropout?: number;
  dropConnectRate?: number;
  attentionRms?: RMSNormState;
  ffnRms?: RMSNormState;
  /**
   * @deprecated Use attentionRms/ffnRms for explicit pre-attention and pre-FFN norms
   */
  rmsState?: RMSNormState;
  ropeBase?: number;
  /**
   * Number of key-value heads for Grouped-Query Attention (GQA).
   * Defaults to `heads` (standard MHA). Set lower for memory efficiency.
   * Example: heads=8, numKVHeads=2 gives 4× KV cache reduction.
   */
  numKVHeads?: number;
}

export class MiniTransformerBlock {
  private readonly attention: MultiHeadAttention;
  private readonly activation: (x: number) => number;
  private readonly attentionGamma: number[];
  private readonly ffnGamma: number[];
  private readonly attentionEpsilon: number;
  private readonly ffnEpsilon: number;
  private readonly ropeBase: number;

  constructor(private readonly config: MiniTransformerConfig) {
    this.attention = new MultiHeadAttention({
      heads: config.heads,
      modelDim: config.modelDim,
      keyDim: config.modelDim / config.heads,
      valueDim: config.modelDim / config.heads,
      dropout: config.attentionDropout,
      numKVHeads: config.numKVHeads // GQA support
    });
    this.activation = config.ff.activation ?? ((x) => Math.tanh(x));
    const fallback = config.rmsState ?? { gamma: new Array(config.modelDim).fill(1), epsilon: 1e-6 };
    this.attentionGamma = config.attentionRms?.gamma ?? fallback.gamma;
    this.ffnGamma = config.ffnRms?.gamma ?? fallback.gamma;
    this.attentionEpsilon = config.attentionRms?.epsilon ?? fallback.epsilon;
    this.ffnEpsilon = config.ffnRms?.epsilon ?? fallback.epsilon;
    this.ropeBase = config.ropeBase ?? 500000;
  }

  private applyRMS(row: number[]): number[] {
    return rmsNorm(row, this.attentionGamma, this.attentionEpsilon);
  }

  private applyFFNRMS(row: number[]): number[] {
    return rmsNorm(row, this.ffnGamma, this.ffnEpsilon);
  }

  private feedForward(inputs: Matrix, weights1: Matrix, weights2: Matrix): Matrix {
    // SwiGLU: (XW) * swish(XV)
    const projected = inputs.map((row) =>
      weights1[0].map((_, j) => row.reduce((sum, value, idx) => sum + value * weights1[idx][j], 0))
    );

    const hiddenDim = projected[0]?.length ?? 0;
    const half = hiddenDim / 2;
    const swish = (x: number) => x / (1 + Math.exp(-x));

    const gated = projected.map((row) => {
      const values: number[] = [];
      for (let i = 0; i < half; i++) {
        const main = row[i];
        const gate = row[half + i];
        values.push(main * swish(gate));
      }
      return values;
    });

    const output = gated.map((row) =>
      weights2[0].map((_, j) => row.reduce((sum, value, idx) => sum + value * weights2[idx][j], 0))
    );
    return output;
  }

  forward(
    inputs: Matrix,
    attentionWeights: AttentionWeights,
    ffWeights1: Matrix,
    ffWeights2: Matrix,
    options: MultiHeadForwardOptions = {}
  ): Matrix {
    const normedForAttention = inputs.map((row) => this.applyRMS(row));
    const dropconnectedAttention: AttentionWeights = {
      query: applyDropConnect(attentionWeights.query, {
        rate: this.config.dropConnectRate ?? 0
      }),
      key: applyDropConnect(attentionWeights.key, { rate: this.config.dropConnectRate ?? 0 }),
      value: applyDropConnect(attentionWeights.value, { rate: this.config.dropConnectRate ?? 0 })
    };

    const attentionOutput = this.attention.forward(normedForAttention, dropconnectedAttention, {
      causal: false,
      ropeBase: this.ropeBase,
      ...options
    });
    const residualAttention = inputs.map((row, idx) =>
      row.map((value, col) => value + attentionOutput[idx][col])
    );

    const renormed = residualAttention.map((row) => this.applyFFNRMS(row));
    const feedForwardOutput = this.feedForward(renormed, ffWeights1, ffWeights2);

    return renormed.map((row, idx) => row.map((value, col) => value + feedForwardOutput[idx][col]));
  }
}
