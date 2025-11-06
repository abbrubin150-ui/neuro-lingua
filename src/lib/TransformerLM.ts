/**
 * TransformerLM - Transformer-based Neural Language Model
 *
 * A character-level language model using transformer architecture:
 * - Multi-head self-attention mechanism
 * - Position-aware embeddings
 * - Feed-forward layers with residual connections
 * - Layer normalization
 * - Compatible with ProNeuralLM interface for seamless UI integration
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { ProNeuralLM, type Optimizer, type TokenizerConfig } from './ProNeuralLM';
import { MiniTransformerBlock, type MiniTransformerConfig } from '../models/mini_transformer';
import type { BatchRenormState } from '../models/regularizers';
import { stableSoftmax } from './MathUtils';

export type TransformerConfig = {
  numLayers?: number;
  numHeads?: number;
  ffHiddenDim?: number;
  attentionDropout?: number;
  dropConnectRate?: number;
};

const DEFAULT_TRANSFORMER_CONFIG: Required<TransformerConfig> = {
  numLayers: 2,
  numHeads: 4,
  ffHiddenDim: 128,
  attentionDropout: 0.1,
  dropConnectRate: 0.1
};

/**
 * TransformerLM - Full transformer language model
 */
export class TransformerLM extends ProNeuralLM {
  private transformerConfig: Required<TransformerConfig>;
  private transformerLayers: MiniTransformerBlock[] = [];
  private positionEmbeddings: number[][] = [];
  private maxSeqLength = 128;

  // Transformer-specific weights (stored in addition to base embeddings)
  private attentionWeights: {
    query: number[][][];
    key: number[][][];
    value: number[][][];
  }[] = [];

  private ffWeights1: number[][][] = [];
  private ffWeights2: number[][][] = [];

  // Batch renorm state for each layer
  private renormStates: BatchRenormState[] = [];

  constructor(
    vocab: string[],
    hiddenSize = 64,
    lr = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'adam',
    momentum = 0.9,
    dropout = 0.1,
    seed = 1337,
    tokenizerConfig: TokenizerConfig = { mode: 'unicode' },
    transformerConfig: TransformerConfig = {}
  ) {
    // Initialize base ProNeuralLM (provides embedding layer and basic infrastructure)
    super(vocab, hiddenSize, lr, contextSize, optimizer, momentum, dropout, seed, tokenizerConfig);

    this.transformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...transformerConfig };
    this.initializeTransformerLayers();
    this.initializePositionEmbeddings();
  }

  /**
   * Initialize transformer layers
   */
  private initializeTransformerLayers(): void {
    const { numLayers, numHeads, ffHiddenDim, attentionDropout, dropConnectRate } =
      this.transformerConfig;
    const modelDim = this.getHiddenSize();

    this.transformerLayers = [];
    this.attentionWeights = [];
    this.ffWeights1 = [];
    this.ffWeights2 = [];
    this.renormStates = [];

    for (let i = 0; i < numLayers; i++) {
      const renormState: BatchRenormState = {
        runningMean: new Array(modelDim).fill(0),
        runningVar: new Array(modelDim).fill(1),
        momentum: 0.99,
        epsilon: 1e-5,
        rMax: 3.0,
        dMax: 5.0
      };

      const config: MiniTransformerConfig = {
        modelDim,
        heads: numHeads,
        ff: {
          hiddenDim: ffHiddenDim,
          activation: (x: number) => Math.max(0, x) // ReLU
        },
        attentionDropout,
        dropConnectRate,
        renormState
      };

      this.transformerLayers.push(new MiniTransformerBlock(config));
      this.renormStates.push(renormState);

      // Initialize attention weights (Q, K, V)
      const headDim = modelDim / numHeads;
      this.attentionWeights.push({
        query: this.initWeightMatrix([numHeads, modelDim, headDim]),
        key: this.initWeightMatrix([numHeads, modelDim, headDim]),
        value: this.initWeightMatrix([numHeads, modelDim, headDim])
      });

      // Initialize feedforward weights
      this.ffWeights1.push(this.initWeightMatrix([modelDim, ffHiddenDim]));
      this.ffWeights2.push(this.initWeightMatrix([ffHiddenDim, modelDim]));
    }
  }

  /**
   * Initialize position embeddings
   */
  private initializePositionEmbeddings(): void {
    const modelDim = this.getHiddenSize();
    this.positionEmbeddings = [];

    for (let pos = 0; pos < this.maxSeqLength; pos++) {
      const embedding = new Array(modelDim);
      for (let i = 0; i < modelDim; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / modelDim);
        embedding[i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
      this.positionEmbeddings.push(embedding);
    }
  }

  /**
   * Initialize weight matrix with He initialization
   */
  private initWeightMatrix(shape: number[]): any {
    if (shape.length === 2) {
      const [rows, cols] = shape;
      const scale = Math.sqrt(2.0 / rows);
      return Array(rows)
        .fill(0)
        .map(() =>
          Array(cols)
            .fill(0)
            .map(() => (Math.random() - 0.5) * 2 * scale)
        );
    } else if (shape.length === 3) {
      const [depth, rows, cols] = shape;
      return Array(depth)
        .fill(0)
        .map(() => this.initWeightMatrix([rows, cols]));
    }
    return [];
  }

  /**
   * Add position embeddings to input embeddings
   */
  private addPositionEmbeddings(embeddings: number[][]): number[][] {
    return embeddings.map((emb, pos) => {
      if (pos >= this.maxSeqLength) return emb;
      const posEmb = this.positionEmbeddings[pos];
      return emb.map((val, i) => val + posEmb[i]);
    });
  }

  /**
   * Transformer forward pass with multi-head self-attention
   * @private
   */
  private transformerForward(inputEmbeddings: number[][]): {
    output: number[][];
    intermediates: Array<{ attentionOut: number[][]; residual1: number[][]; ffOut: number[][] }>;
  } {
    // Add position embeddings
    let hidden = this.addPositionEmbeddings(inputEmbeddings);

    const intermediates: Array<{
      attentionOut: number[][];
      residual1: number[][];
      ffOut: number[][];
    }> = [];

    const modelDim = this.getHiddenSize();

    // Pass through each transformer layer
    for (let layerIdx = 0; layerIdx < this.transformerConfig.numLayers; layerIdx++) {
      const layer = this.transformerLayers[layerIdx];

      // The attention weights need to be in the format:
      // { query: Matrix, key: Matrix, value: Matrix }
      // Each matrix should be [modelDim x (modelDim)]
      // We'll use a simple combined weight matrix for all heads
      const attentionWeightsMatrix = {
        query: this.initWeightMatrix([modelDim, modelDim]),
        key: this.initWeightMatrix([modelDim, modelDim]),
        value: this.initWeightMatrix([modelDim, modelDim])
      };

      // Forward through transformer block
      const output = layer.forward(
        hidden,
        attentionWeightsMatrix as any,
        this.ffWeights1[layerIdx],
        this.ffWeights2[layerIdx]
      );

      // Store intermediates for backprop
      intermediates.push({
        attentionOut: output,
        residual1: hidden,
        ffOut: output
      });

      hidden = output;
    }

    return { output: hidden, intermediates };
  }

  /**
   * Get architecture type for display
   */
  getArchitectureType(): string {
    return 'Transformer';
  }

  /**
   * Get transformer-specific info
   */
  getTransformerInfo(): {
    numLayers: number;
    numHeads: number;
    ffHiddenDim: number;
  } {
    return {
      numLayers: this.transformerConfig.numLayers,
      numHeads: this.transformerConfig.numHeads,
      ffHiddenDim: this.transformerConfig.ffHiddenDim
    };
  }

  /**
   * Override train method to use transformer forward/backward passes
   */
  async train(
    text: string,
    epochs = 10
  ): Promise<{
    readonly loss: number;
    readonly accuracy: number;
    readonly history: { loss: number; accuracy: number; timestamp: number }[];
  }> {
    // Create training sequences (reuse parent method via any cast)
    const createTrainingSequences = (this as any).createTrainingSequences.bind(this);
    const shuffleInPlace = (this as any).shuffleInPlace.bind(this);
    const trainingHistory = (this as any).trainingHistory as Array<{
      loss: number;
      accuracy: number;
      timestamp: number;
    }>;

    const seqs = createTrainingSequences(text);
    if (seqs.length === 0) return { loss: 0, accuracy: 0, history: trainingHistory };

    let totalLoss = 0;
    let correct = 0;
    let count = 0;

    for (let e = 0; e < epochs; e++) {
      shuffleInPlace(seqs);
      let epochLoss = 0;
      let epochCorrect = 0;
      for (const [ctx, tgt] of seqs) {
        const cache = await this.transformerForwardPass(ctx, true);
        const loss = -Math.log(cache.probs[tgt] + 1e-8);
        epochLoss += loss;
        totalLoss += loss;
        const pred = cache.probs.indexOf(Math.max(...cache.probs));
        if (pred === tgt) {
          epochCorrect++;
          correct++;
        }
        count++;
        await this.transformerBackwardPass(ctx, tgt, cache);
      }
      const avgLoss = epochLoss / seqs.length;
      const accuracy = epochCorrect / seqs.length;
      trainingHistory.push({ loss: avgLoss, accuracy, timestamp: Date.now() });
    }

    const payload = {
      loss: totalLoss / Math.max(1, count),
      accuracy: correct / Math.max(1, count),
      history: trainingHistory
    } as const;
    (this as any).lastUpdatedAt = Date.now();
    return payload;
  }

  /**
   * Transformer-specific forward pass
   * @private
   */
  private async transformerForwardPass(
    inputs: number[],
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _train = false
  ): Promise<{
    h: number[];
    logits: number[];
    probs: number[];
    avgEmb: number[];
    dropMask: number[] | null;
    preAct: number[];
    transformerHidden: number[][];
    transformerIntermed: Array<{
      attentionOut: number[][];
      residual1: number[][];
      ffOut: number[][];
    }>;
  }> {
    const V = super.getVocabSize();
    const H = this.getHiddenSize();

    // Get embeddings for each input token
    const embeddings: number[][] = inputs.map((i) => (this as any).embedding[i]);

    // Pass through transformer layers
    const { output: transformerOutput, intermediates } = this.transformerForward(embeddings);

    // Pool transformer output (mean pooling over sequence)
    const avgEmb = this.transformerAverageVectors(embeddings);
    const h = this.transformerAverageVectors(transformerOutput);

    // Output projection (reuse base class output weights)
    const wOutput = (this as any).wOutput as number[][];
    const bOutput = (this as any).bOutput as number[];

    const logits = new Array(V).fill(0);
    for (let j = 0; j < V; j++) {
      let sum = bOutput[j];
      for (let i = 0; i < H; i++) {
        sum += wOutput[i][j] * h[i];
      }
      logits[j] = sum;
    }

    const probs = stableSoftmax(logits);

    return {
      h,
      logits,
      probs,
      avgEmb,
      dropMask: null,
      preAct: h,
      transformerHidden: transformerOutput,
      transformerIntermed: intermediates
    };
  }

  /**
   * Transformer-specific backward pass
   * @private
   */
  private async transformerBackwardPass(
    inputs: number[],
    target: number,
    cache: {
      h: number[];
      probs: number[];
      avgEmb: number[];
      dropMask: number[] | null;
      preAct: number[];
      transformerHidden: number[][];
      transformerIntermed: Array<{
        attentionOut: number[][];
        residual1: number[][];
        ffOut: number[][];
      }>;
    }
  ): Promise<void> {
    const V = super.getVocabSize();
    const H = this.getHiddenSize();
    const { h, probs } = cache;

    // Gradient of loss w.r.t. logits (softmax + cross-entropy)
    const dLogits = probs.map((p, i) => p - (i === target ? 1 : 0));

    // Gradient w.r.t. output weights
    const wOutput = (this as any).wOutput as number[][];
    const dWout = this.createZerosMat(H, V);
    const dBout = new Array(V).fill(0);

    for (let i = 0; i < H; i++) {
      for (let j = 0; j < V; j++) {
        dWout[i][j] = h[i] * dLogits[j];
      }
    }
    for (let j = 0; j < V; j++) {
      dBout[j] = dLogits[j];
    }

    // Gradient w.r.t. hidden state (pooled transformer output)
    const dHidden = new Array(H).fill(0);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < V; j++) {
        dHidden[i] += wOutput[i][j] * dLogits[j];
      }
    }

    // Update attention weights with simplified gradient descent
    // This is an approximation - full backprop through attention would be more accurate
    const lr = (this as any).learningRate as number;
    for (let layerIdx = 0; layerIdx < this.transformerConfig.numLayers; layerIdx++) {
      // Update attention weights with small gradient step
      // Using finite differences approximation for simplicity
      for (let h = 0; h < this.transformerConfig.numHeads; h++) {
        const headDim = H / this.transformerConfig.numHeads;

        for (let i = 0; i < H; i++) {
          for (let j = 0; j < headDim; j++) {
            // Simplified gradient estimate based on hidden state gradients
            const gradScale = dHidden[i] * 0.01; // Small scale factor
            this.attentionWeights[layerIdx].query[h][i][j] -= lr * gradScale * 0.1;
            this.attentionWeights[layerIdx].key[h][i][j] -= lr * gradScale * 0.1;
            this.attentionWeights[layerIdx].value[h][i][j] -= lr * gradScale * 0.1;
          }
        }
      }

      // Update feedforward weights
      for (let i = 0; i < this.ffWeights1[layerIdx].length; i++) {
        for (let j = 0; j < this.ffWeights1[layerIdx][i].length; j++) {
          const grad = dHidden[i % H] * 0.01;
          this.ffWeights1[layerIdx][i][j] -= lr * grad * 0.1;
        }
      }
    }

    // Update output projection using base class optimizer
    await this.applyGradients(dWout, dBout, inputs, dHidden);
  }

  /**
   * Apply gradients using the base class optimizer
   * @private
   */
  private async applyGradients(
    dWout: number[][],
    dBout: number[],
    inputs: number[],
    dHidden: number[]
  ): Promise<void> {
    const lr = (this as any).learningRate as number;
    const wOutput = (this as any).wOutput as number[][];
    const bOutput = (this as any).bOutput as number[];

    // Simple SGD update for output weights
    for (let i = 0; i < wOutput.length; i++) {
      for (let j = 0; j < wOutput[i].length; j++) {
        wOutput[i][j] -= lr * dWout[i][j];
      }
    }
    for (let j = 0; j < bOutput.length; j++) {
      bOutput[j] -= lr * dBout[j];
    }

    // Update embeddings
    const embedding = (this as any).embedding as number[][];
    for (const idx of inputs) {
      for (let i = 0; i < embedding[idx].length; i++) {
        embedding[idx][i] -= lr * dHidden[i] * 0.01;
      }
    }
  }

  /**
   * Helper methods
   */
  private transformerAverageVectors(vectors: number[][]): number[] {
    const H = this.getHiddenSize();
    const y = new Array(H).fill(0);
    const n = vectors.length || 1;
    for (const v of vectors) {
      for (let i = 0; i < H; i++) y[i] += v[i];
    }
    for (let i = 0; i < H; i++) y[i] /= n;
    return y;
  }

  private createZerosMat(r: number, c: number): number[][] {
    return new Array(r).fill(0).map(() => new Array(c).fill(0));
  }

  /**
   * Get hidden size (model dimension)
   */
  private getHiddenSize(): number {
    return (this as any).hiddenSize;
  }
}
