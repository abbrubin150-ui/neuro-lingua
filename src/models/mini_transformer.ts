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
  rmsState?: RMSNormState;
  ropeBase?: number;
}

export class MiniTransformerBlock {
  private readonly attention: MultiHeadAttention;
  private readonly activation: (x: number) => number;
  private readonly gamma: number[];
  private readonly epsilon: number;
  private readonly ropeBase: number;

  constructor(private readonly config: MiniTransformerConfig) {
    this.attention = new MultiHeadAttention({
      heads: config.heads,
      modelDim: config.modelDim,
      keyDim: config.modelDim / config.heads,
      valueDim: config.modelDim / config.heads,
      dropout: config.attentionDropout
    });
    this.activation = config.ff.activation ?? ((x) => Math.tanh(x));
    this.gamma = config.rmsState?.gamma ?? new Array(config.modelDim).fill(1);
    this.epsilon = config.rmsState?.epsilon ?? 1e-6;
    this.ropeBase = config.ropeBase ?? 500000;
  }

  private applyRMS(row: number[]): number[] {
    return rmsNorm(row, this.gamma, this.epsilon);
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

    const renormed = residualAttention.map((row) => this.applyRMS(row));
    const feedForwardOutput = this.feedForward(renormed, ffWeights1, ffWeights2);

    return renormed.map((row, idx) => row.map((value, col) => value + feedForwardOutput[idx][col]));
  }
}
