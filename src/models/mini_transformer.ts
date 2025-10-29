import { MultiHeadAttention, AttentionWeights, Matrix } from './attention';
import { applyDropConnect, batchRenormalize, BatchRenormState } from './regularizers';

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
  renormState: BatchRenormState;
}

export class MiniTransformerBlock {
  private readonly attention: MultiHeadAttention;
  private readonly activation: (x: number) => number;

  constructor(private readonly config: MiniTransformerConfig) {
    this.attention = new MultiHeadAttention({
      heads: config.heads,
      modelDim: config.modelDim,
      keyDim: config.modelDim / config.heads,
      valueDim: config.modelDim / config.heads,
      dropout: config.attentionDropout
    });
    this.activation = config.ff.activation ?? ((x) => Math.tanh(x));
  }

  private feedForward(inputs: Matrix, weights1: Matrix, weights2: Matrix): Matrix {
    const hidden = inputs.map((row) =>
      weights1[0].map((_, j) =>
        row.reduce((sum, value, idx) => sum + value * weights1[idx][j], 0)
      )
    );
    const activated = hidden.map((row) => row.map((v) => this.activation(v)));
    const output = activated.map((row) =>
      weights2[0].map((_, j) =>
        row.reduce((sum, value, idx) => sum + value * weights2[idx][j], 0)
      )
    );
    return output;
  }

  forward(
    inputs: Matrix,
    attentionWeights: AttentionWeights,
    ffWeights1: Matrix,
    ffWeights2: Matrix
  ): Matrix {
    const dropconnectedAttention: AttentionWeights = {
      query: applyDropConnect(attentionWeights.query, {
        rate: this.config.dropConnectRate ?? 0
      }),
      key: applyDropConnect(attentionWeights.key, { rate: this.config.dropConnectRate ?? 0 }),
      value: applyDropConnect(attentionWeights.value, { rate: this.config.dropConnectRate ?? 0 })
    };

    const attentionOutput = this.attention.forward(inputs, dropconnectedAttention, { causal: false });
    const residualAttention = inputs.map((row, idx) =>
      row.map((value, col) => value + attentionOutput[idx][col])
    );

    const { normalized: renormed } = batchRenormalize(residualAttention, this.config.renormState);
    const feedForwardOutput = this.feedForward(renormed, ffWeights1, ffWeights2);

    return renormed.map((row, idx) =>
      row.map((value, col) => value + feedForwardOutput[idx][col])
    );
  }
}
