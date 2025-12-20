/**
 * TransformerLM - Transformer-based Neural Language Model
 *
 * A character-level language model using transformer architecture:
 * - Multi-head self-attention mechanism
 * - Position-aware embeddings
 * - Feed-forward layers with residual connections
 * - RMS normalization (pre-norm residual layout)
 * - Compatible with ProNeuralLM interface for seamless UI integration
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { ProNeuralLM, type Optimizer, type TokenizerConfig } from './ProNeuralLM';
import { MiniTransformerBlock, type MiniTransformerConfig } from '../models/mini_transformer';
import type { AttentionWeights, Matrix } from '../models/attention';
import { stableSoftmax } from './MathUtils';
import { type RMSNormState } from './RMSNorm';
import { GPUNeuralOps } from '../backend/gpu_neural_ops';

export type TransformerConfig = {
  numLayers?: number;
  numHeads?: number;
  ffHiddenDim?: number;
  attentionDropout?: number;
  dropConnectRate?: number;
  normType?: 'rms' | 'layer';
  /**
   * Number of key-value heads for Grouped-Query Attention (GQA).
   * When numKVHeads < numHeads, multiple query heads share the same K/V heads.
   * This reduces KV cache memory by (numHeads / numKVHeads) factor.
   *
   * Examples:
   * - numKVHeads = numHeads: Standard Multi-Head Attention (MHA)
   * - numKVHeads = 1: Multi-Query Attention (MQA)
   * - numKVHeads = numHeads/4: GQA with 4:1 ratio (used in Llama-3.2)
   *
   * Default: equals numHeads (standard MHA for backward compatibility)
   */
  numKVHeads?: number;
};

const DEFAULT_TRANSFORMER_CONFIG: Required<TransformerConfig> = {
  numLayers: 2,
  numHeads: 4,
  ffHiddenDim: 128,
  attentionDropout: 0.1,
  dropConnectRate: 0.1,
  normType: 'rms',
  numKVHeads: 4 // Default: same as numHeads (standard MHA)
};

type BaseModelJson = ReturnType<ProNeuralLM['toJSON']>;
type TransformerSerializedModel = BaseModelJson & {
  architecture: 'transformer';
  transformer: {
    config: Required<TransformerConfig>;
    attentionWeights: AttentionWeights[];
    ffWeights1: Matrix[];
    ffWeights2: Matrix[];
    renormStates: TransformerRMSStates[] | RMSNormState[];
  };
};

export type TransformerRMSStates = {
  attention: RMSNormState;
  ffn: RMSNormState;
};

type TransformerRng = { next(): number; getState(): number };
function createTransformerRng(seed: number, state?: number): TransformerRng {
  const baseSeed = seed >>> 0;
  let t = (state !== undefined ? state : baseSeed) >>> 0;
  return {
    next() {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    },
    getState() {
      return t >>> 0;
    }
  };
}

/**
 * TransformerLM - Full transformer language model
 */
export class TransformerLM extends ProNeuralLM {
  private transformerConfig: Required<TransformerConfig>;
  private transformerLayers: MiniTransformerBlock[] = [];
  private positionEmbeddings: number[][] = [];
  private maxSeqLength = 128;

  // Transformer-specific weights (stored in addition to base embeddings)
  private attentionWeights: AttentionWeights[] = [];

  private ffWeights1: Matrix[] = [];
  private ffWeights2: Matrix[] = [];

  // RMSNorm state (gamma/epsilon) for each layer
  private renormStates: TransformerRMSStates[] = [];

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
   * Ensure RMSNorm states have separate attention/FFN entries and correct dimensions.
   */
  private normalizeRenormState(
    state: TransformerRMSStates | RMSNormState | undefined,
    modelDim: number
  ): TransformerRMSStates {
    const defaultState = {
      gamma: new Array(modelDim).fill(1),
      epsilon: 1e-6
    } satisfies RMSNormState;
    const clone = (s: RMSNormState): RMSNormState => ({ gamma: [...s.gamma], epsilon: s.epsilon });
    if (!state) return { attention: clone(defaultState), ffn: clone(defaultState) };

    // Backward compatibility: previously a single RMSNormState was used for both sublayers
    if ('gamma' in state && !('attention' in state)) {
      return { attention: clone(state), ffn: clone(state) };
    }

    const attention = (state as TransformerRMSStates).attention ?? defaultState;
    const ffn = (state as TransformerRMSStates).ffn ?? defaultState;
    return {
      attention: clone(attention),
      ffn: clone(ffn)
    };
  }

  /**
   * Initialize transformer layers
   */
  private initializeTransformerLayers(existing?: {
    attention?: AttentionWeights[];
    ff1?: Matrix[];
    ff2?: Matrix[];
    renormStates?: TransformerRMSStates[] | RMSNormState[];
  }): void {
    const { numLayers, numHeads, ffHiddenDim, attentionDropout, dropConnectRate, numKVHeads } =
      this.transformerConfig;
    const modelDim = this.getHiddenSize();
    const normalizedHeads = this.normalizeHeadCount(modelDim, numHeads);
    if (normalizedHeads !== this.transformerConfig.numHeads) {
      this.transformerConfig = { ...this.transformerConfig, numHeads: normalizedHeads };
    }

    // Normalize numKVHeads to be valid
    const normalizedKVHeads = this.normalizeKVHeadCount(
      normalizedHeads,
      numKVHeads ?? normalizedHeads
    );
    if (normalizedKVHeads !== this.transformerConfig.numKVHeads) {
      this.transformerConfig = { ...this.transformerConfig, numKVHeads: normalizedKVHeads };
    }

    // Calculate KV dimension for GQA
    const headDim = modelDim / normalizedHeads;
    const kvDim = normalizedKVHeads * headDim;

    this.transformerLayers = [];
    this.attentionWeights = [];
    this.ffWeights1 = [];
    this.ffWeights2 = [];
    this.renormStates = [];

    for (let i = 0; i < numLayers; i++) {
      const existingState = existing?.renormStates?.[i];
      const normalizedState: TransformerRMSStates = this.normalizeRenormState(
        existingState,
        modelDim
      );

      const config: MiniTransformerConfig = {
        modelDim,
        heads: this.transformerConfig.numHeads,
        ff: {
          hiddenDim: ffHiddenDim,
          activation: (x: number) => Math.max(0, x) // Unused in SwiGLU but kept for compatibility
        },
        attentionDropout,
        dropConnectRate,
        attentionRms: normalizedState.attention,
        ffnRms: normalizedState.ffn,
        ropeBase: 500000,
        numKVHeads: normalizedKVHeads, // GQA support
        gpuOps: this.getGPUOpsInstance()
      };

      this.transformerLayers.push(new MiniTransformerBlock(config));
      this.renormStates.push(normalizedState);

      // GQA: K/V weights are smaller when numKVHeads < numHeads
      // Query: [modelDim x modelDim], Key/Value: [modelDim x kvDim]
      this.attentionWeights.push(
        existing?.attention?.[i] ?? {
          query: this.initWeightMatrix([modelDim, modelDim]),
          key: this.initWeightMatrix([modelDim, kvDim]),
          value: this.initWeightMatrix([modelDim, kvDim])
        }
      );

      this.ffWeights1.push(
        existing?.ff1?.[i] ?? this.initWeightMatrix([modelDim, ffHiddenDim * 2])
      );
      this.ffWeights2.push(existing?.ff2?.[i] ?? this.initWeightMatrix([ffHiddenDim, modelDim]));
    }
  }

  /**
   * Normalize numKVHeads to be a valid divisor of numHeads.
   */
  private normalizeKVHeadCount(numHeads: number, requestedKVHeads: number): number {
    if (requestedKVHeads <= 0) return 1;
    if (requestedKVHeads >= numHeads) return numHeads;
    if (numHeads % requestedKVHeads === 0) return requestedKVHeads;
    // Find largest valid divisor <= requested
    for (let candidate = requestedKVHeads; candidate >= 1; candidate--) {
      if (numHeads % candidate === 0) return candidate;
    }
    return 1;
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
  private async transformerForward(inputEmbeddings: number[][]): Promise<{
    output: number[][];
    intermediates: Array<{ attentionOut: number[][]; residual1: number[][]; ffOut: number[][] }>;
  }> {
    let hidden = inputEmbeddings;
    const positions = hidden.map((_, idx) => idx);

    const intermediates: Array<{
      attentionOut: number[][];
      residual1: number[][];
      ffOut: number[][];
    }> = [];

    // Pass through each transformer layer
    for (let layerIdx = 0; layerIdx < this.transformerConfig.numLayers; layerIdx++) {
      const layer = this.transformerLayers[layerIdx];

      // Forward through transformer block
      const output = await layer.forward(
        hidden,
        this.attentionWeights[layerIdx],
        this.ffWeights1[layerIdx],
        this.ffWeights2[layerIdx],
        { positions }
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
    numKVHeads: number;
  } {
    return {
      numLayers: this.transformerConfig.numLayers,
      numHeads: this.transformerConfig.numHeads,
      ffHiddenDim: this.transformerConfig.ffHiddenDim,
      numKVHeads: this.transformerConfig.numKVHeads
    };
  }

  /**
   * Propagate GPU accelerator to transformer layers.
   */
  setGPUOps(gpuOps: GPUNeuralOps | null) {
    super.setGPUOps(gpuOps);
    for (const layer of this.transformerLayers) {
      layer.setGPUOps(gpuOps);
    }
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

  async generate(
    seedText: string,
    maxLen = 25,
    temperature = 0.9,
    topK = 0,
    topP = 0
  ): Promise<string> {
    const tokenize = (this as any).tokenize.bind(this);
    const toIndex = (this as any).toIndex.bind(this);
    const contextSize = (this as any).contextSize as number;
    const bosToken = (this as any).bos as string;
    const bosIdx = toIndex(bosToken);
    const eosToken = (this as any).eos as string;
    const sampleFromLogits = (this as any).sampleFromLogits.bind(this);
    const idxToWord = (this as any).idxToWord as Map<number, string>;

    const ctx: number[] = new Array(contextSize).fill(bosIdx);
    for (const tok of tokenize(seedText)) ctx.push(toIndex(tok));

    const out: string[] = [];
    while (out.length < maxLen) {
      const window = ctx.slice(-contextSize);
      const { logits } = await this.transformerForwardPass(window, false);
      const { index: idx } = sampleFromLogits(logits, temperature, topK, topP);
      const token = idxToWord.get(idx) ?? eosToken;
      if (token === eosToken) break;
      out.push(token);
      ctx.push(idx);
    }
    return out.join(' ');
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
    const { output: transformerOutput, intermediates } = await this.transformerForward(embeddings);

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
        // wOutput is [hiddenSize x vocabSize], so index as [i][j]
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
    // wOutput is [H x V], so dWout should also be [H x V]
    const wOutput = (this as any).wOutput as number[][];
    const dWout = this.createZerosMat(H, V);
    const dBout = new Array(V).fill(0);

    for (let i = 0; i < H; i++) {
      for (let j = 0; j < V; j++) {
        // dL/dW[i][j] = h[i] * dL/dLogits[j]
        dWout[i][j] = h[i] * dLogits[j];
      }
    }
    for (let j = 0; j < V; j++) {
      dBout[j] = dLogits[j];
    }

    // Gradient w.r.t. hidden state (pooled transformer output)
    // dL/dh[i] = sum_j(W[i][j] * dL/dLogits[j])
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
      const attention = this.attentionWeights[layerIdx];
      for (let i = 0; i < attention.query.length; i++) {
        for (let j = 0; j < attention.query[i].length; j++) {
          const gradScale = dHidden[i % H] * 0.01;
          attention.query[i][j] -= lr * gradScale * 0.1;
          attention.key[i][j] -= lr * gradScale * 0.1;
          attention.value[i][j] -= lr * gradScale * 0.1;
        }
      }

      // Update feedforward weights
      for (let i = 0; i < this.ffWeights1[layerIdx].length; i++) {
        for (let j = 0; j < this.ffWeights1[layerIdx][i].length; j++) {
          const grad = dHidden[i % H] * 0.01;
          this.ffWeights1[layerIdx][i][j] -= lr * grad * 0.1;
        }
      }
      for (let i = 0; i < this.ffWeights2[layerIdx].length; i++) {
        for (let j = 0; j < this.ffWeights2[layerIdx][i].length; j++) {
          const grad = dHidden[j % H] * 0.01;
          this.ffWeights2[layerIdx][i][j] -= lr * grad * 0.1;
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
    const optimizer = (this as any).optimizer as Optimizer;
    const wOutput = (this as any).wOutput as number[][];
    const bOutput = (this as any).bOutput as number[];
    const embedding = (this as any).embedding as number[][];

    if (optimizer === 'sophia') {
      // Use Sophia optimizer for output weights
      const sophiaOptimizer = (this as any).sophiaOptimizer;
      if (sophiaOptimizer) {
        sophiaOptimizer.updateMatrix(wOutput, dWout, 'wOutput');
        sophiaOptimizer.updateVector(bOutput, dBout, 'bOutput');
        // Update embeddings
        for (const idx of inputs) {
          const dEmb = dHidden.map((v) => v * 0.01);
          sophiaOptimizer.updateRow(embedding, idx, dEmb, 'embedding');
        }
        sophiaOptimizer.step();
      }
    } else {
      // Simple SGD update for output weights (default)
      for (let i = 0; i < wOutput.length; i++) {
        for (let j = 0; j < wOutput[i].length; j++) {
          wOutput[i][j] -= lr * dWout[i][j];
        }
      }
      for (let j = 0; j < bOutput.length; j++) {
        bOutput[j] -= lr * dBout[j];
      }

      // Update embeddings
      for (const idx of inputs) {
        for (let i = 0; i < embedding[idx].length; i++) {
          embedding[idx][i] -= lr * dHidden[i] * 0.01;
        }
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
   * Expand hidden layer (modelDim) by adding k neurons.
   * Used by Cerebro neuron injection system.
   * Extends base class expansion with transformer-specific weights.
   * @param k Number of neurons to add
   * @param useHeInit Use He initialization (for ReLU networks)
   */
  expandHiddenLayer(k: number, useHeInit = true): void {
    if (k <= 0) return;

    const oldModelDim = this.getHiddenSize();
    const newModelDim = oldModelDim + k;
    const { ffHiddenDim } = this.transformerConfig;

    // Call parent to expand base weights (wHidden, wOutput, etc.)
    // Note: This will also update hiddenSize
    super.expandHiddenLayer(k, useHeInit);

    // Calculate initialization scale
    const scale = useHeInit ? Math.sqrt(2.0 / oldModelDim) : Math.sqrt(1.0 / oldModelDim);

    // Expand token embeddings: add k dimensions to each token
    // In transformer, embeddings are used directly for attention so must match modelDim
    const embedding = (this as any).embedding as number[][];
    for (let i = 0; i < embedding.length; i++) {
      for (let j = 0; j < k; j++) {
        embedding[i].push((Math.random() - 0.5) * 2 * scale);
      }
    }

    // Expand position embeddings: add k dimensions to each position
    for (let pos = 0; pos < this.positionEmbeddings.length; pos++) {
      const posEmb = this.positionEmbeddings[pos];
      for (let i = 0; i < k; i++) {
        // Use sinusoidal encoding for new dimensions
        const dimIdx = oldModelDim + i;
        const angle = pos / Math.pow(10000, (2 * Math.floor(dimIdx / 2)) / newModelDim);
        posEmb.push(dimIdx % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
      }
    }

    // Expand each transformer layer's weights
    for (let layerIdx = 0; layerIdx < this.transformerConfig.numLayers; layerIdx++) {
      const attention = this.attentionWeights[layerIdx];
      const ff1 = this.ffWeights1[layerIdx];
      const ff2 = this.ffWeights2[layerIdx];
      const renormState = this.renormStates[layerIdx];

      // Expand attention weights: query, key, value are [modelDim x modelDim]
      // Add k new rows to each matrix
      for (let i = 0; i < k; i++) {
        const newQueryRow: number[] = [];
        const newKeyRow: number[] = [];
        const newValueRow: number[] = [];
        for (let j = 0; j < oldModelDim; j++) {
          newQueryRow.push((Math.random() - 0.5) * 2 * scale);
          newKeyRow.push((Math.random() - 0.5) * 2 * scale);
          newValueRow.push((Math.random() - 0.5) * 2 * scale);
        }
        attention.query.push(newQueryRow);
        attention.key.push(newKeyRow);
        attention.value.push(newValueRow);
      }
      // Add k new columns to each existing row
      for (let i = 0; i < oldModelDim; i++) {
        for (let j = 0; j < k; j++) {
          attention.query[i].push((Math.random() - 0.5) * 2 * scale);
          attention.key[i].push((Math.random() - 0.5) * 2 * scale);
          attention.value[i].push((Math.random() - 0.5) * 2 * scale);
        }
      }
      // Also add columns to the new rows
      for (let i = oldModelDim; i < newModelDim; i++) {
        for (let j = 0; j < k; j++) {
          attention.query[i].push((Math.random() - 0.5) * 2 * scale);
          attention.key[i].push((Math.random() - 0.5) * 2 * scale);
          attention.value[i].push((Math.random() - 0.5) * 2 * scale);
        }
      }

      // Expand ffWeights1: [modelDim x (ffHiddenDim*2)] -> add k rows
      for (let i = 0; i < k; i++) {
        const newRow: number[] = [];
        for (let j = 0; j < ffHiddenDim * 2; j++) {
          newRow.push((Math.random() - 0.5) * 2 * scale);
        }
        ff1.push(newRow);
      }

      // Expand ffWeights2: [ffHiddenDim x modelDim] -> add k columns to each row
      for (let i = 0; i < ffHiddenDim; i++) {
        for (let j = 0; j < k; j++) {
          ff2[i].push((Math.random() - 0.5) * 2 * scale);
        }
      }

      // Expand renorm state gamma vectors
      for (let i = 0; i < k; i++) {
        renormState.attention.gamma.push(1);
        renormState.ffn.gamma.push(1);
      }
    }

    // Reinitialize transformer layers with new dimensions
    this.reinitializeTransformerLayers();
  }

  /**
   * Reinitialize transformer layer objects after dimension change.
   * Preserves existing weights while updating internal structures.
   */
  private reinitializeTransformerLayers(): void {
    const modelDim = this.getHiddenSize();
    const { numLayers, ffHiddenDim, attentionDropout, dropConnectRate, numKVHeads } =
      this.transformerConfig;
    const normalizedHeads = this.normalizeHeadCount(modelDim, this.transformerConfig.numHeads);
    if (normalizedHeads !== this.transformerConfig.numHeads) {
      this.transformerConfig = { ...this.transformerConfig, numHeads: normalizedHeads };
    }

    // Normalize numKVHeads
    const normalizedKVHeads = this.normalizeKVHeadCount(normalizedHeads, numKVHeads);
    if (normalizedKVHeads !== this.transformerConfig.numKVHeads) {
      this.transformerConfig = { ...this.transformerConfig, numKVHeads: normalizedKVHeads };
    }

    // Recreate transformer blocks with new dimensions but existing weights
    this.transformerLayers = [];
    for (let i = 0; i < numLayers; i++) {
      const config: MiniTransformerConfig = {
        modelDim,
        heads: this.transformerConfig.numHeads,
        ff: {
          hiddenDim: ffHiddenDim,
          activation: (x: number) => Math.max(0, x)
        },
        attentionDropout,
        dropConnectRate,
        attentionRms: this.renormStates[i].attention,
        ffnRms: this.renormStates[i].ffn,
        ropeBase: 500000,
        numKVHeads: normalizedKVHeads, // GQA support
        gpuOps: this.getGPUOpsInstance()
      };
      this.transformerLayers.push(new MiniTransformerBlock(config));
    }
  }

  /**
   * Get transformer-specific weights for adapter serialization.
   */
  getTransformerWeights(): {
    attentionWeights: AttentionWeights[];
    ffWeights1: Matrix[];
    ffWeights2: Matrix[];
    positionEmbeddings: number[][];
    renormStates: TransformerRMSStates[];
  } {
    return {
      attentionWeights: this.attentionWeights,
      ffWeights1: this.ffWeights1,
      ffWeights2: this.ffWeights2,
      positionEmbeddings: this.positionEmbeddings,
      renormStates: this.renormStates
    };
  }

  /**
   * Set transformer-specific weights from adapter (for rollback).
   */
  setTransformerWeights(weights: {
    attentionWeights: AttentionWeights[];
    ffWeights1: Matrix[];
    ffWeights2: Matrix[];
    positionEmbeddings: number[][];
    renormStates: TransformerRMSStates[] | RMSNormState[];
  }): void {
    this.attentionWeights = weights.attentionWeights;
    this.ffWeights1 = weights.ffWeights1;
    this.ffWeights2 = weights.ffWeights2;
    this.positionEmbeddings = weights.positionEmbeddings;
    this.renormStates = weights.renormStates.map((state) =>
      this.normalizeRenormState(state as TransformerRMSStates | RMSNormState, this.getHiddenSize())
    );
    this.reinitializeTransformerLayers();
  }

  toJSON() {
    const base = super.toJSON() as BaseModelJson;
    return {
      ...base,
      architecture: 'transformer' as const,
      transformer: {
        config: this.transformerConfig,
        attentionWeights: this.attentionWeights,
        ffWeights1: this.ffWeights1,
        ffWeights2: this.ffWeights2,
        renormStates: this.renormStates
      }
    } satisfies TransformerSerializedModel;
  }

  static loadFromLocalStorage(key: string): TransformerLM | null {
    try {
      if (typeof localStorage === 'undefined') return null;
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const data = JSON.parse(raw) as TransformerSerializedModel;
      if (data.architecture !== 'transformer' || !data.transformer) return null;
      return TransformerLM.fromSerialized(data);
    } catch (error) {
      console.warn('Failed to load transformer model', error);
      return null;
    }
  }

  private static fromSerialized(data: TransformerSerializedModel): TransformerLM {
    const model = new TransformerLM(
      data.vocab,
      data.hiddenSize,
      data.learningRate ?? 0.08,
      data.contextSize ?? 3,
      (data.optimizer as Optimizer) ?? 'momentum',
      data.momentum ?? 0.9,
      data.dropout ?? 0,
      data.rngSeed ?? 1337,
      data.tokenizerConfig,
      data.transformer.config
    );

    (model as any).embedding = data.embedding;
    (model as any).wHidden = data.wHidden;
    (model as any).wOutput = data.wOutput;
    (model as any).bHidden = data.bHidden;
    (model as any).bOutput = data.bOutput;
    if (typeof data.adamT === 'number') (model as any).adamT = data.adamT;
    if (data.mEmbedding) (model as any).mEmbedding = data.mEmbedding;
    if (data.mWHidden) (model as any).mWHidden = data.mWHidden;
    if (data.mWOutput) (model as any).mWOutput = data.mWOutput;
    if (data.mBHidden) (model as any).mBHidden = data.mBHidden;
    if (data.mBOutput) (model as any).mBOutput = data.mBOutput;
    if (data.aEmbedding) (model as any).aEmbedding = { m: data.aEmbedding.m, v: data.aEmbedding.v };
    if (data.aWHidden) (model as any).aWHidden = { m: data.aWHidden.m, v: data.aWHidden.v };
    if (data.aWOutput) (model as any).aWOutput = { m: data.aWOutput.m, v: data.aWOutput.v };
    if (data.aBHidden) (model as any).aBHidden = { m: data.aBHidden.m, v: data.aBHidden.v };
    if (data.aBOutput) (model as any).aBOutput = { m: data.aBOutput.m, v: data.aBOutput.v };
    (model as any).wordToIdx = new Map(data.wordToIdx);
    (model as any).idxToWord = new Map(data.idxToWord);
    (model as any).trainingHistory = data.trainingHistory || [];
    if (typeof data.lastUpdatedAt === 'number') {
      (model as any).lastUpdatedAt = data.lastUpdatedAt;
    }

    if (typeof data.rngSeed === 'number') {
      (model as any).rngSeed = data.rngSeed >>> 0;
    }
    const rngState =
      typeof (data as BaseModelJson).rngState === 'number'
        ? ((data as BaseModelJson).rngState as number) >>> 0
        : undefined;
    (model as any).rng = createTransformerRng((model as any).rngSeed, rngState);
    (model as any).rngState = (model as any).rng.getState();

    model.transformerConfig = data.transformer.config;
    model.initializeTransformerLayers({
      attention: data.transformer.attentionWeights,
      ff1: data.transformer.ffWeights1,
      ff2: data.transformer.ffWeights2,
      renormStates: data.transformer.renormStates
    });

    return model;
  }

  private normalizeHeadCount(modelDim: number, requestedHeads: number): number {
    if (requestedHeads <= 0) return 1;
    if (modelDim % requestedHeads === 0) return requestedHeads;
    for (let candidate = requestedHeads; candidate >= 1; candidate--) {
      if (modelDim % candidate === 0) return candidate;
    }
    return 1;
  }
}
