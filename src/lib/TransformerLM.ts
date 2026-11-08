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
import type { LossMaskConfig } from '../types/project';
import {
  blockBackward,
  blockForward,
  createBlockGrads,
  zeros,
  type BlockCache,
  type BlockConfig,
  type BlockGrads,
  type BlockParams
} from '../models/transformer_backprop';
import {
  globalNorm,
  ParamOptimizer,
  scaleTensors,
  type ParamOptimizerKind
} from '../training/ParamOptimizer';

const TRANSFORMER_ROPE_BASE = 500000;
/** Global-norm gradient clipping threshold for one training step. */
const TRANSFORMER_GRAD_CLIP = 1.0;

type StackPass = {
  h: number[];
  logits: number[];
  probs: number[];
  avgEmb: number[];
  positions: number[];
  caches: BlockCache[];
  output: number[][];
  dropMask: number[] | null;
};

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

  // Optimizer state for the full transformer stack (not serialized; rebuilt lazily)
  private paramOptimizer: ParamOptimizer | null = null;

  // Seeded RNG for transformer weight initialization (reproducible for a given seed)
  private initRng: TransformerRng | null = null;

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
    this.paramOptimizer?.reset();
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
        ropeBase: TRANSFORMER_ROPE_BASE,
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

  private initRandom(): number {
    if (!this.initRng) {
      const seed = ((this as any).rngSeed as number | undefined) ?? 1337;
      this.initRng = createTransformerRng((seed ^ 0x9e3779b9) >>> 0);
    }
    return this.initRng.next();
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
            .map(() => (this.initRandom() - 0.5) * 2 * scale)
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
   * Get architecture type for display
   */
  getArchitectureType(): string {
    return 'Transformer';
  }

  /**
   * Trainable parameter count of the transformer model: token embeddings, every transformer
   * block (Q/K/V, SwiGLU FFN, two RMSNorm gains) and the output projection. The base MLP's
   * hidden layer is not used by TransformerLM and is therefore not counted.
   */
  getParametersCount(): number {
    const V = this.getVocabSize();
    const D = this.getHiddenSize();
    let total = V * D + D * V + V;
    for (let l = 0; l < this.transformerConfig.numLayers; l++) {
      const att = this.attentionWeights[l];
      const size = (m: number[][]) => m.length * (m[0]?.length ?? 0);
      total += size(att.query) + size(att.key) + size(att.value);
      total += size(this.ffWeights1[l]) + size(this.ffWeights2[l]);
      total += this.renormStates[l].attention.gamma.length + this.renormStates[l].ffn.gamma.length;
    }
    return total;
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
    epochs = 10,
    lossMaskConfig?: LossMaskConfig
  ): Promise<{
    readonly loss: number;
    readonly accuracy: number;
    readonly history: { loss: number; accuracy: number; timestamp: number }[];
  }> {
    // Create training sequences (reuse parent method via any cast)
    const createTrainingSequences = (this as any).createTrainingSequences.bind(this);
    const buildTrainingLossMask = (this as any).buildTrainingLossMask.bind(this);
    const shuffleInPlace = (this as any).shuffleInPlace.bind(this);
    const trainingHistory = (this as any).trainingHistory as Array<{
      loss: number;
      accuracy: number;
      timestamp: number;
    }>;

    const seqs = createTrainingSequences(text);
    if (seqs.length === 0) return { loss: 0, accuracy: 0, history: trainingHistory };

    // Build loss mask if config provided
    const lossMask = lossMaskConfig
      ? buildTrainingLossMask(text, lossMaskConfig)
      : new Array(seqs.length).fill(1);

    // Pair sequences with their mask values for shuffling
    const seqsWithMask: [number[], number, number][] = seqs.map(
      (seq: [number[], number], i: number) => [seq[0], seq[1], lossMask[i] ?? 1]
    );

    let totalLoss = 0;
    let correct = 0;
    let count = 0;
    let maskedCount = 0;

    for (let e = 0; e < epochs; e++) {
      shuffleInPlace(seqsWithMask);
      let epochLoss = 0;
      let epochCorrect = 0;
      let epochMaskedCount = 0;

      for (const [ctx, tgt, maskValue] of seqsWithMask) {
        const pass = this.stackForward(ctx, maskValue === 1);
        const { probs } = pass;
        let pred = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[pred]) pred = i;

        // Only compute loss and backprop for unmasked positions
        if (maskValue === 1) {
          const loss = -Math.log(probs[tgt] + 1e-8);
          epochLoss += loss;
          totalLoss += loss;
          epochMaskedCount++;
          maskedCount++;
          this.stackBackwardAndUpdate(ctx, tgt, pass);
        }

        // Still track accuracy for all positions
        if (pred === tgt) {
          epochCorrect++;
          correct++;
        }
        count++;
      }

      const avgLoss = epochMaskedCount > 0 ? epochLoss / epochMaskedCount : 0;
      const accuracy = epochCorrect / seqs.length;
      trainingHistory.push({ loss: avgLoss, accuracy, timestamp: Date.now() });
    }

    const payload = {
      loss: maskedCount > 0 ? totalLoss / maskedCount : 0,
      accuracy: correct / Math.max(1, count),
      history: trainingHistory
    } as const;
    (this as any).lastUpdatedAt = Date.now();
    return payload;
  }

  /**
   * Forward pass used by every inference path inherited from ProNeuralLM (generate with all
   * sampling modes and penalties, getLogitsForContext / distillation, evaluation).
   * Replaces the base feed-forward MLP so that TransformerLM always runs the transformer stack.
   */
  protected async forward(inputs: number[], _train = false) {
    const pass = this.stackForward(inputs, false);
    return {
      h: pass.h,
      logits: pass.logits,
      probs: pass.probs,
      avgEmb: pass.avgEmb,
      dropMask: null as number[] | null,
      preAct: pass.h
    };
  }

  /**
   * Inspection helper kept for diagnostics/tests: returns the pooled state and the final
   * hidden sequence of the transformer stack.
   * @private
   */
  private async transformerForwardPass(inputs: number[], _train = false) {
    const pass = this.stackForward(inputs, false);
    return {
      h: pass.h,
      logits: pass.logits,
      probs: pass.probs,
      avgEmb: pass.avgEmb,
      dropMask: null as number[] | null,
      preAct: pass.h,
      transformerHidden: pass.output
    };
  }

  private blockParams(layerIdx: number): BlockParams {
    const norm = this.renormStates[layerIdx];
    return {
      attention: this.attentionWeights[layerIdx],
      ff1: this.ffWeights1[layerIdx],
      ff2: this.ffWeights2[layerIdx],
      attnGamma: norm.attention.gamma,
      ffnGamma: norm.ffn.gamma,
      attnEpsilon: norm.attention.epsilon,
      ffnEpsilon: norm.ffn.epsilon
    };
  }

  private blockConfig(): BlockConfig {
    return {
      numHeads: this.transformerConfig.numHeads,
      numKVHeads: this.transformerConfig.numKVHeads,
      ropeBase: TRANSFORMER_ROPE_BASE
    };
  }

  /**
   * Exact forward pass through the transformer stack (RoPE positions 0..T-1), mean pooling,
   * optional inverted dropout on the pooled state (training only) and the output projection.
   */
  private stackForward(inputs: number[], train: boolean): StackPass {
    const embedding = (this as any).embedding as number[][];
    const wOutput = (this as any).wOutput as number[][];
    const bOutput = (this as any).bOutput as number[];
    const dropout = (this as any).dropout as number;
    const positions = inputs.map((_, i) => i);
    const config = this.blockConfig();

    const X0 = inputs.map((i) => embedding[i].slice());
    const caches: BlockCache[] = [];
    let hidden = X0;
    for (let l = 0; l < this.transformerConfig.numLayers; l++) {
      const { out, cache } = blockForward(hidden, this.blockParams(l), config, positions);
      caches.push(cache);
      hidden = out;
    }

    const pooled = this.transformerAverageVectors(hidden);
    let dropMask: number[] | null = null;
    let h = pooled;
    if (train && dropout > 0) {
      const scale = 1 / (1 - dropout);
      const nextRandom = (this as any).nextRandom.bind(this) as () => number;
      dropMask = pooled.map(() => (nextRandom() > dropout ? scale : 0));
      h = pooled.map((v, i) => v * dropMask![i]);
    }

    const V = bOutput.length;
    const logits = new Array(V);
    for (let j = 0; j < V; j++) {
      let sum = bOutput[j];
      for (let i = 0; i < h.length; i++) sum += wOutput[i][j] * h[i];
      logits[j] = sum;
    }

    return {
      h,
      logits,
      probs: stableSoftmax(logits),
      avgEmb: this.transformerAverageVectors(X0),
      positions,
      caches,
      output: hidden,
      dropMask
    };
  }

  private getParamOptimizer(): ParamOptimizer {
    const optimizer = (this as any).optimizer as Optimizer;
    const momentum = (this as any).momentum as number;
    // Second-order modes (newton/bfgs/sophia) are not implemented for the transformer stack;
    // they fall back to Adam, the most robust first-order method.
    const kind: ParamOptimizerKind =
      optimizer === 'momentum' ? 'momentum' : optimizer === 'lion' ? 'lion' : 'adam';
    if (!this.paramOptimizer) {
      this.paramOptimizer = new ParamOptimizer(kind, momentum);
    } else {
      this.paramOptimizer.setKind(kind, momentum);
    }
    return this.paramOptimizer;
  }

  /**
   * Full backpropagation (softmax cross-entropy → output projection → pooling → every
   * transformer block → token embeddings), global-norm gradient clipping and one optimizer step.
   */
  private stackBackwardAndUpdate(inputs: number[], target: number, pass: StackPass): void {
    const wOutput = (this as any).wOutput as number[][];
    const bOutput = (this as any).bOutput as number[];
    const embedding = (this as any).embedding as number[][];
    const lr = (this as any).learningRate as number;
    const H = this.getHiddenSize();
    const V = bOutput.length;
    const config = this.blockConfig();

    const dLogits = pass.probs.map((p, i) => p - (i === target ? 1 : 0));
    const dWout = zeros(H, V);
    const dh = new Array(H).fill(0);
    for (let i = 0; i < H; i++) {
      const row = wOutput[i];
      const gRow = dWout[i];
      let s = 0;
      for (let j = 0; j < V; j++) {
        gRow[j] = pass.h[i] * dLogits[j];
        s += row[j] * dLogits[j];
      }
      dh[i] = pass.dropMask ? s * pass.dropMask[i] : s;
    }
    const dBout = dLogits.slice();

    // Mean pooling distributes the gradient evenly across positions.
    const T = inputs.length;
    let dHidden: number[][] = pass.output.map(() => dh.map((v) => v / T));
    const layerGrads: BlockGrads[] = new Array(this.transformerConfig.numLayers);
    for (let l = this.transformerConfig.numLayers - 1; l >= 0; l--) {
      const params = this.blockParams(l);
      const grads = createBlockGrads(params);
      dHidden = blockBackward(dHidden, pass.caches[l], params, config, pass.positions, grads);
      layerGrads[l] = grads;
    }

    // Sparse embedding gradient (tokens may repeat inside the context window).
    const dEmb: number[][] = [];
    const touched = new Set<number>();
    inputs.forEach((tok, t) => {
      if (!dEmb[tok]) dEmb[tok] = new Array(H).fill(0);
      const row = dEmb[tok];
      for (let i = 0; i < H; i++) row[i] += dHidden[t][i];
      touched.add(tok);
    });

    const allGrads: Array<number[] | number[][]> = [dWout, dBout];
    for (const g of layerGrads) {
      allGrads.push(g.query, g.key, g.value, g.ff1, g.ff2, g.attnGamma, g.ffnGamma);
    }
    for (const tok of touched) allGrads.push(dEmb[tok]);
    const norm = globalNorm(allGrads);
    if (norm > TRANSFORMER_GRAD_CLIP) scaleTensors(allGrads, TRANSFORMER_GRAD_CLIP / norm);

    const opt = this.getParamOptimizer();
    opt.step();
    opt.update('wOutput', wOutput, dWout, lr);
    opt.update('bOutput', bOutput, dBout, lr);
    for (let l = 0; l < layerGrads.length; l++) {
      const p = this.blockParams(l);
      const g = layerGrads[l];
      opt.update(`L${l}.query`, p.attention.query, g.query, lr);
      opt.update(`L${l}.key`, p.attention.key, g.key, lr);
      opt.update(`L${l}.value`, p.attention.value, g.value, lr);
      opt.update(`L${l}.ff1`, p.ff1, g.ff1, lr);
      opt.update(`L${l}.ff2`, p.ff2, g.ff2, lr);
      opt.update(`L${l}.attnGamma`, p.attnGamma, g.attnGamma, lr);
      opt.update(`L${l}.ffnGamma`, p.ffnGamma, g.ffnGamma, lr);
    }
    opt.update('embedding', embedding, dEmb, lr, touched);
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
        embedding[i].push((this.initRandom() - 0.5) * 2 * scale);
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
          newQueryRow.push((this.initRandom() - 0.5) * 2 * scale);
          newKeyRow.push((this.initRandom() - 0.5) * 2 * scale);
          newValueRow.push((this.initRandom() - 0.5) * 2 * scale);
        }
        attention.query.push(newQueryRow);
        attention.key.push(newKeyRow);
        attention.value.push(newValueRow);
      }
      // Add k new columns to each existing row
      for (let i = 0; i < oldModelDim; i++) {
        for (let j = 0; j < k; j++) {
          attention.query[i].push((this.initRandom() - 0.5) * 2 * scale);
          attention.key[i].push((this.initRandom() - 0.5) * 2 * scale);
          attention.value[i].push((this.initRandom() - 0.5) * 2 * scale);
        }
      }
      // Also add columns to the new rows
      for (let i = oldModelDim; i < newModelDim; i++) {
        for (let j = 0; j < k; j++) {
          attention.query[i].push((this.initRandom() - 0.5) * 2 * scale);
          attention.key[i].push((this.initRandom() - 0.5) * 2 * scale);
          attention.value[i].push((this.initRandom() - 0.5) * 2 * scale);
        }
      }

      // Expand ffWeights1: [modelDim x (ffHiddenDim*2)] -> add k rows
      for (let i = 0; i < k; i++) {
        const newRow: number[] = [];
        for (let j = 0; j < ffHiddenDim * 2; j++) {
          newRow.push((this.initRandom() - 0.5) * 2 * scale);
        }
        ff1.push(newRow);
      }

      // Expand ffWeights2: [ffHiddenDim x modelDim] -> add k columns to each row
      for (let i = 0; i < ffHiddenDim; i++) {
        for (let j = 0; j < k; j++) {
          ff2[i].push((this.initRandom() - 0.5) * 2 * scale);
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
        ropeBase: TRANSFORMER_ROPE_BASE,
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
    this.paramOptimizer?.reset();
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

  /**
   * Rebuild a TransformerLM from the object produced by toJSON().
   */
  static fromJSON(data: any): TransformerLM {
    if (data?.architecture !== 'transformer' || !data.transformer) {
      throw new Error('Serialized data is not a transformer model');
    }
    return TransformerLM.fromSerialized(data as TransformerSerializedModel);
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
