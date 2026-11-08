/**
 * TransformerLM — decoder-only transformer language model trained with exact backpropagation.
 *
 * Objective 'causal' (default for new models):
 *   x_t   = E[token_t]
 *   h     = Block_L(…Block_1(x))     pre-norm blocks (see models/transformer_backprop.ts):
 *                                    RMSNorm → causal MHA/GQA with per-head RoPE → residual →
 *                                    RMSNorm → SwiGLU FFN → residual
 *   n_t   = RMSNorm(h_t; γ_final)
 *   p(token_{t+1} | token_≤t) = softmax(Woutᵀ·n_t + b)
 *   Training minimises the next-token cross-entropy at every position of every context window
 *   (one window = `contextSize` tokens); inference (generation, evaluation, distillation) reads
 *   the last position.
 *
 * Objective 'pooled' (legacy): models saved before the causal revision used bidirectional
 * attention, full-row RoPE (base 500000) and mean pooling over the window with a single
 * next-token loss. Loading such a model keeps that objective so its predictions are unchanged.
 *
 * Regularisation (training only): attention-probability dropout, DropConnect on Q/K/V and
 * dropout on the read-out vector, all with exact gradients through the sampled masks.
 * Optimisation: Adam / Lion / momentum-SGD over every parameter, mini-batches of
 * `batchSize` windows, global-norm gradient clipping. Computation is CPU-only.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import {
  ProNeuralLM,
  type ForwardCache,
  type Optimizer,
  type TokenizerConfig
} from './ProNeuralLM';
import type { AttentionWeights, Matrix } from '../models/attention';
import type { RMSNormState } from './RMSNorm';
import type { GPUNeuralOps } from '../backend/gpu_neural_ops';
import type { LossMaskConfig } from '../types/project';
import { parseStoredModel } from './modelPackageCodec';
import {
  blockBackward,
  blockForward,
  createBlockGrads,
  rmsBackward,
  rmsForward,
  zeros,
  type BlockCache,
  type BlockConfig,
  type BlockDropout,
  type BlockGrads,
  type BlockMasks,
  type BlockParams,
  hadamard,
  sampleWeightMasks
} from '../models/transformer_backprop';
import { globalNorm, ParamOptimizer, type ParamOptimizerKind } from '../training/ParamOptimizer';

/** RoPE base for causal models (short in-browser windows) and for legacy pooled models. */
const ROPE_BASE_CAUSAL = 10000;
const ROPE_BASE_LEGACY = 500000;
const NORM_EPSILON = 1e-6;
/** Global-norm gradient clipping threshold for one optimisation step. */
const TRANSFORMER_GRAD_CLIP = 1.0;

export type TransformerObjective = 'causal' | 'pooled';

export type TransformerConfig = {
  numLayers?: number;
  numHeads?: number;
  ffHiddenDim?: number;
  /** Dropout probability on attention weights (training only). */
  attentionDropout?: number;
  /** DropConnect probability on individual Q/K/V weights (training only). */
  dropConnectRate?: number;
  /** Only RMSNorm is implemented; the field is kept for configuration compatibility. */
  normType?: 'rms' | 'layer';
  /**
   * Number of key-value heads for Grouped-Query Attention (GQA).
   * When numKVHeads < numHeads, multiple query heads share the same K/V heads.
   *
   * Examples:
   * - numKVHeads = numHeads: Standard Multi-Head Attention (MHA)
   * - numKVHeads = 1: Multi-Query Attention (MQA)
   * - numKVHeads = numHeads/4: GQA with 4:1 ratio (used in Llama-3.2)
   */
  numKVHeads?: number;
  /** Training objective / architecture revision (see the file header). Default 'causal'. */
  objective?: TransformerObjective;
  /** RoPE base. Default 10000 for 'causal', 500000 for legacy 'pooled' models. */
  ropeBase?: number;
  /**
   * Number of context windows whose gradients are averaged before one optimizer step
   * (DropConnect masks are shared within a mini-batch). 1 reproduces per-window updates.
   */
  batchSize?: number;
  /**
   * Causal objective only: train on every `trainingStride`-th context window. With every
   * position of a window trained, a stride equal to the context size visits each target once per
   * epoch (non-overlapping chunks, as in GPT-style training) instead of contextSize times.
   * 1 keeps every overlapping window (default).
   */
  trainingStride?: number;
  /** Decoupled (AdamW-style) weight decay for weight matrices and embeddings; 0 disables it. */
  weightDecay?: number;
};

/** Per-optimizer-step learning-rate schedule: linear warm-up, then cosine decay. */
export type LearningRateSchedule = {
  warmupSteps: number;
  totalSteps: number;
  /** Final learning rate as a fraction of the base rate (default 0.1). */
  minFactor?: number;
};

const DEFAULT_TRANSFORMER_CONFIG: Required<TransformerConfig> = {
  numLayers: 2,
  numHeads: 4,
  ffHiddenDim: 128,
  attentionDropout: 0.1,
  dropConnectRate: 0.1,
  normType: 'rms',
  numKVHeads: 4,
  objective: 'causal',
  ropeBase: ROPE_BASE_CAUSAL,
  batchSize: 8,
  trainingStride: 1,
  weightDecay: 0
};

type BaseModelJson = ReturnType<ProNeuralLM['toJSON']>;
type TransformerSerializedModel = BaseModelJson & {
  architecture: 'transformer';
  transformer: {
    config: TransformerConfig;
    attentionWeights: AttentionWeights[];
    ffWeights1: Matrix[];
    ffWeights2: Matrix[];
    renormStates: TransformerRMSStates[] | RMSNormState[];
    finalNorm?: RMSNormState;
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

/** Activations of one forward pass through the stack. */
type StackPass = {
  positions: number[];
  X0: Matrix;
  caches: BlockCache[];
  /** Output of the last block, one row per position. */
  hidden: Matrix;
  /** Causal objective: RMS of each hidden row and the normalised rows (before dropout). */
  finalRms: number[] | null;
  normed: Matrix | null;
  /** Positions read out by the output projection (pooled objective: [-1]). */
  scored: number[];
  /** Vectors fed to the output projection (after dropout), one per scored entry. */
  readouts: number[][];
  masks: (number[] | null)[];
  logits: number[][];
  probs: number[][];
};

type WeightMasks = Pick<BlockMasks, 'query' | 'key' | 'value'>;

/** DropConnect state of one layer for the current mini-batch: masks and W̃ = W ⊙ M. */
type LayerDropConnect = { masks: WeightMasks; masked: AttentionWeights };

type GradBuffers = {
  wOutput: Matrix;
  bOutput: number[];
  finalGamma: number[] | null;
  layers: BlockGrads[];
  embedding: Map<number, number[]>;
};

// ---------------------------------------------------------------------------
// Index re-mapping helpers used when the model width grows (neuron injection)
// ---------------------------------------------------------------------------

/** New index of every old column when `newSpan − oldSpan` slots are appended to each block. */
// ---------------------------------------------------------------------------
// Output-layer kernels: the vocabulary projection dominates training time. Written as plain
// module-level loops (V8 optimises them far better than closures) that stream each row of the
// hidden × vocabulary matrix once per window for all scored positions, instead of once per
// position. Every output element is accumulated in the same order as a per-position loop, so the
// results are bit-identical.
// ---------------------------------------------------------------------------

/** logits = bias + h·W for one readout row (W is hidden × vocabulary). */
function projectOutput(h: number[], W: number[][], bias: number[]): number[] {
  const V = bias.length;
  // A typed accumulator keeps the inner loop monomorphic whatever the arrays' element kinds.
  const out = new Float64Array(V);
  for (let j = 0; j < V; j++) out[j] = bias[j];
  for (let i = 0; i < h.length; i++) {
    const hi = h[i];
    if (hi === 0) continue;
    const row = W[i];
    for (let j = 0; j < V; j++) out[j] += row[j] * hi;
  }
  return Array.from(out);
}

/**
 * Softmax of one row of logits with exactly the arithmetic of MathUtils' `stableSoftmax` at
 * temperature 1 (p = exp(x − (max + log Σ exp(x − max)))), but O(1) validation instead of three
 * extra passes: non-finite logits or an unstable span still throw.
 */
function outputSoftmax(logits: number[]): number[] {
  const n = logits.length;
  let max = -Infinity;
  let min = Infinity;
  for (let j = 0; j < n; j++) {
    const v = logits[j];
    if (v > max) max = v;
    if (v < min) min = v;
  }
  let sum = 0;
  for (let j = 0; j < n; j++) sum += Math.exp(logits[j] - max);
  const lse = max + Math.log(sum);
  // A NaN logit makes the sum NaN; ±Infinity makes max or min non-finite.
  if (!Number.isFinite(lse) || !Number.isFinite(min)) {
    throw new Error('Softmax logits contain non-finite values');
  }
  if (max - min > 1e6) {
    throw new Error(`Softmax logits span too wide for stable normalization: ${max - min}`);
  }
  const probs = new Array<number>(n);
  for (let j = 0; j < n; j++) probs[j] = Math.exp(logits[j] - lse);
  return probs;
}

/**
 * Backward of the output layer for softmax cross-entropy, each scored position weighted by
 * `weight` (targets < 0 are skipped): adds Σ r ⊗ dLogits to gW and Σ dLogits to gBias and
 * returns W·dLogits per position (null for skipped ones).
 */
function outputBackward(
  readouts: number[][],
  probs: number[][],
  targets: number[],
  weight: number,
  W: number[][],
  gW: number[][],
  gBias: number[]
): (number[] | null)[] {
  const V = gBias.length;
  const H = W.length;
  const active: number[] = [];
  const dLogits: Float64Array[] = [];
  targets.forEach((target, k) => {
    if (target < 0) return;
    const p = probs[k];
    const d = new Float64Array(V);
    for (let j = 0; j < V; j++) d[j] = (p[j] - (j === target ? 1 : 0)) * weight;
    active.push(k);
    dLogits.push(d);
  });
  const dr = active.map(() => new Array<number>(H));
  for (let i = 0; i < H; i++) {
    const row = W[i];
    const gRow = gW[i];
    for (let a = 0; a < active.length; a++) {
      const ri = readouts[active[a]][i];
      const d = dLogits[a];
      let s = 0;
      for (let j = 0; j < V; j++) {
        gRow[j] += ri * d[j];
        s += row[j] * d[j];
      }
      dr[a][i] = s;
    }
  }
  for (const d of dLogits) for (let j = 0; j < V; j++) gBias[j] += d[j];
  const out: (number[] | null)[] = targets.map(() => null);
  active.forEach((k, a) => (out[k] = dr[a]));
  return out;
}

function blockwiseIndexMap(oldWidth: number, oldSpan: number, newSpan: number): number[] {
  return Array.from(
    { length: oldWidth },
    (_, j) => Math.floor(j / oldSpan) * newSpan + (j % oldSpan)
  );
}

function identityMap(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** Place v[from] at out[map[from]]; unmapped slots are filled with fill(). */
function remapVector(v: number[], map: number[], newLength: number, fill: () => number): number[] {
  const out = new Array<number>(newLength);
  const mapped = new Array<boolean>(newLength).fill(false);
  map.forEach((to, from) => {
    if (to >= 0 && to < newLength) {
      out[to] = v[from];
      mapped[to] = true;
    }
  });
  for (let i = 0; i < newLength; i++) if (!mapped[i]) out[i] = fill();
  return out;
}

function remapRows(M: Matrix, map: number[], newRows: number, newRow: () => number[]): Matrix {
  const out: Matrix = new Array(newRows);
  map.forEach((to, from) => {
    if (to >= 0 && to < newRows) out[to] = M[from];
  });
  for (let i = 0; i < newRows; i++) if (!out[i]) out[i] = newRow();
  return out;
}

function remapCols(M: Matrix, map: number[], newCols: number, fill: () => number): Matrix {
  return M.map((row) => remapVector(row, map, newCols, fill));
}

/**
 * TransformerLM - Full transformer language model
 */
export class TransformerLM extends ProNeuralLM {
  private transformerConfig: Required<TransformerConfig>;
  /**
   * Sinusoidal position table kept for the Cerebro adapter API. Positions are encoded with RoPE
   * inside attention; this table is not used by the forward pass.
   */
  private positionEmbeddings: number[][] = [];
  private maxSeqLength = 128;

  // Transformer-specific weights (stored in addition to base embeddings)
  private attentionWeights: AttentionWeights[] = [];
  private ffWeights1: Matrix[] = [];
  private ffWeights2: Matrix[] = [];

  // RMSNorm state (gamma/epsilon) for each layer, plus the final norm of the causal objective
  private renormStates: TransformerRMSStates[] = [];
  private finalNorm: RMSNormState = { gamma: [], epsilon: NORM_EPSILON };

  // Optimizer state and gradient accumulators (not serialized; rebuilt lazily)
  private paramOptimizer: ParamOptimizer | null = null;
  private gradBuffers: GradBuffers | null = null;
  /** Optional per-step learning-rate schedule (not serialized; set per training run). */
  private lrSchedule: LearningRateSchedule | null = null;
  private scheduleStep = 0;

  /** DropConnect masks and masked Q/K/V weights of the current mini-batch (one per layer). */
  private batchDropConnect: LayerDropConnect[] | null = null;

  /**
   * Causal objective: also train the predictions made from shorter prefixes inside each window
   * (positions 0..T−2). When false only the full-context prediction at T−1 is trained.
   */
  private causalLossOnAllPositions = true;

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
    super(vocab, hiddenSize, lr, contextSize, optimizer, momentum, dropout, seed, tokenizerConfig);

    const objective = transformerConfig.objective ?? DEFAULT_TRANSFORMER_CONFIG.objective;
    this.transformerConfig = {
      ...DEFAULT_TRANSFORMER_CONFIG,
      ...transformerConfig,
      objective,
      ropeBase:
        transformerConfig.ropeBase ?? (objective === 'pooled' ? ROPE_BASE_LEGACY : ROPE_BASE_CAUSAL)
    };
    this.initializeTransformerLayers();
    this.initializePositionEmbeddings();
  }

  // ---------------------------------------------------------------------------
  // Initialisation
  // ---------------------------------------------------------------------------

  /**
   * Ensure RMSNorm states have separate attention/FFN entries and correct dimensions.
   */
  private normalizeRenormState(
    state: TransformerRMSStates | RMSNormState | undefined,
    modelDim: number
  ): TransformerRMSStates {
    const defaultState = {
      gamma: new Array(modelDim).fill(1),
      epsilon: NORM_EPSILON
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
   * Initialize (or restore) every transformer parameter.
   */
  private initializeTransformerLayers(existing?: {
    attention?: AttentionWeights[];
    ff1?: Matrix[];
    ff2?: Matrix[];
    renormStates?: TransformerRMSStates[] | RMSNormState[];
    finalNorm?: RMSNormState;
  }): void {
    const { numLayers, numHeads, ffHiddenDim, numKVHeads } = this.transformerConfig;
    const modelDim = this.getHiddenSize();
    const normalizedHeads = this.normalizeHeadCount(modelDim, numHeads);
    const normalizedKVHeads = this.normalizeKVHeadCount(
      normalizedHeads,
      numKVHeads ?? normalizedHeads
    );
    this.transformerConfig = {
      ...this.transformerConfig,
      numHeads: normalizedHeads,
      numKVHeads: normalizedKVHeads
    };

    // GQA: K/V weights are [modelDim x kvDim]
    const headDim = modelDim / normalizedHeads;
    const kvDim = normalizedKVHeads * headDim;

    this.paramOptimizer?.reset();
    this.gradBuffers = null;
    this.attentionWeights = [];
    this.ffWeights1 = [];
    this.ffWeights2 = [];
    this.renormStates = [];

    for (let i = 0; i < numLayers; i++) {
      this.renormStates.push(this.normalizeRenormState(existing?.renormStates?.[i], modelDim));
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

    this.finalNorm =
      existing?.finalNorm && existing.finalNorm.gamma.length === modelDim
        ? { gamma: [...existing.finalNorm.gamma], epsilon: existing.finalNorm.epsilon }
        : { gamma: new Array(modelDim).fill(1), epsilon: NORM_EPSILON };
  }

  /**
   * Normalize numKVHeads to be a valid divisor of numHeads.
   */
  private normalizeKVHeadCount(numHeads: number, requestedKVHeads: number): number {
    if (requestedKVHeads <= 0) return 1;
    if (requestedKVHeads >= numHeads) return numHeads;
    if (numHeads % requestedKVHeads === 0) return requestedKVHeads;
    for (let candidate = requestedKVHeads; candidate >= 1; candidate--) {
      if (numHeads % candidate === 0) return candidate;
    }
    return 1;
  }

  private normalizeHeadCount(modelDim: number, requestedHeads: number): number {
    if (requestedHeads <= 0) return 1;
    if (modelDim % requestedHeads === 0) return requestedHeads;
    for (let candidate = requestedHeads; candidate >= 1; candidate--) {
      if (modelDim % candidate === 0) return candidate;
    }
    return 1;
  }

  private sinusoid(pos: number, dimIdx: number, modelDim: number): number {
    const angle = pos / Math.pow(10000, (2 * Math.floor(dimIdx / 2)) / modelDim);
    return dimIdx % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  }

  private initializePositionEmbeddings(): void {
    const modelDim = this.getHiddenSize();
    this.positionEmbeddings = [];
    for (let pos = 0; pos < this.maxSeqLength; pos++) {
      const embedding = new Array(modelDim);
      for (let i = 0; i < modelDim; i++) embedding[i] = this.sinusoid(pos, i, modelDim);
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
   * He-uniform initialisation: U(−√(6/fanIn)·½, …) ≈ variance 2/fanIn·⅓ (kept from the original
   * implementation so that seeds keep producing the same networks).
   */
  private initWeightMatrix(shape: [number, number]): Matrix {
    const [rows, cols] = shape;
    const scale = Math.sqrt(2.0 / rows);
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (this.initRandom() - 0.5) * 2 * scale)
    );
  }

  // ---------------------------------------------------------------------------
  // Introspection
  // ---------------------------------------------------------------------------

  getArchitectureType(): string {
    return 'Transformer';
  }

  /**
   * Trainable parameter count of the transformer model: token embeddings, every transformer
   * block (Q/K/V, SwiGLU FFN, two RMSNorm gains), the final RMSNorm (causal objective) and the
   * output projection. The base MLP's hidden layer is not used by TransformerLM.
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
    if (this.isCausal()) total += this.finalNorm.gamma.length;
    return total;
  }

  getTransformerInfo(): {
    numLayers: number;
    numHeads: number;
    ffHiddenDim: number;
    numKVHeads: number;
    objective: TransformerObjective;
    batchSize: number;
  } {
    return {
      numLayers: this.transformerConfig.numLayers,
      numHeads: this.transformerConfig.numHeads,
      ffHiddenDim: this.transformerConfig.ffHiddenDim,
      numKVHeads: this.transformerConfig.numKVHeads,
      objective: this.transformerConfig.objective,
      batchSize: this.transformerConfig.batchSize
    };
  }

  /**
   * The transformer stack runs on the CPU (exact forward/backward); the GPU accelerator is only
   * stored for API compatibility.
   */
  setGPUOps(gpuOps: GPUNeuralOps | null) {
    super.setGPUOps(gpuOps);
  }

  private isCausal(): boolean {
    return this.transformerConfig.objective === 'causal';
  }

  // ---------------------------------------------------------------------------
  // Training
  // ---------------------------------------------------------------------------

  async train(
    text: string,
    epochs = 10,
    lossMaskConfig?: LossMaskConfig
  ): Promise<{
    readonly loss: number;
    readonly accuracy: number;
    readonly history: { loss: number; accuracy: number; timestamp: number }[];
  }> {
    const self = this as any;
    const trainingHistory = self.trainingHistory as Array<{
      loss: number;
      accuracy: number;
      timestamp: number;
    }>;

    const seqs = self.createTrainingSequences(text) as [number[], number][];
    if (seqs.length === 0) return { loss: 0, accuracy: 0, history: trainingHistory };

    const lossMask: number[] = lossMaskConfig
      ? self.buildTrainingLossMask(text, lossMaskConfig)
      : new Array(seqs.length).fill(1);
    const stride = this.isCausal()
      ? Math.max(1, Math.floor(this.transformerConfig.trainingStride ?? 1))
      : 1;
    const allItems = seqs.map(([ctx, tgt], index) => ({ ctx, tgt, index }));
    // Chunked training: one window every `stride` targets. The chunk alignment is re-drawn every
    // epoch so that each target is, over the epochs, predicted from every position (and hence
    // every context length) of the window; a fixed alignment would train some targets only with
    // a short context.
    const epochItems = () => {
      if (stride === 1) return allItems.slice();
      const offset = Math.floor(self.nextRandom() * stride);
      return allItems.filter(
        ({ index }) => (index + offset) % stride === stride - 1 || index === allItems.length - 1
      );
    };
    const batchSize = Math.max(1, Math.floor(this.transformerConfig.batchSize));
    const causal = this.isCausal();

    let totalLoss = 0;
    let correct = 0;
    let count = 0;
    let maskedCount = 0;

    for (let e = 0; e < epochs; e++) {
      const items = epochItems();
      self.shuffleInPlace(items);
      let epochLoss = 0;
      let epochCorrect = 0;
      let epochMaskedCount = 0;
      let pending = 0;

      let processed = 0;
      let stopped = false;
      for (const { ctx, tgt, index } of items) {
        if (await this.trainingShouldStop()) {
          stopped = true;
          break;
        }
        this.reportTrainingProgress(++processed, items.length, e);
        const T = ctx.length;
        const lastMasked = (lossMask[index] ?? 1) === 1;

        // Scored entries: positions whose next token is a real (unmasked) target.
        let scored: number[];
        let targets: number[];
        if (causal) {
          scored = [];
          targets = [];
          if (this.causalLossOnAllPositions) {
            for (let t = 0; t < T - 1; t++) {
              // ctx[t + 1] is the target of training sequence (index + t + 1 − T);
              // negative indices are the BOS padding, which is never predicted.
              const seq = index + t + 1 - T;
              if (seq >= 0 && (lossMask[seq] ?? 1) === 1) {
                scored.push(t);
                targets.push(ctx[t + 1]);
              }
            }
          }
          scored.push(T - 1);
          targets.push(lastMasked ? tgt : -1);
        } else {
          scored = [-1];
          targets = [lastMasked ? tgt : -1];
        }

        const learn = targets.some((t) => t >= 0);
        if (learn && pending === 0) this.batchDropConnect = this.sampleBatchDropConnect();
        const pass = this.stackForward(ctx, learn, scored, learn);
        const lastProbs = pass.probs[pass.probs.length - 1];
        let pred = 0;
        for (let i = 1; i < lastProbs.length; i++) if (lastProbs[i] > lastProbs[pred]) pred = i;

        if (lastMasked) {
          const loss = -Math.log(lastProbs[tgt] + 1e-8);
          epochLoss += loss;
          totalLoss += loss;
          epochMaskedCount++;
          maskedCount++;
        }
        if (pred === tgt) {
          epochCorrect++;
          correct++;
        }
        count++;

        if (learn) {
          this.accumulateGradients(ctx, pass, targets);
          pending++;
          if (pending === batchSize) {
            this.applyAccumulatedGradients(pending);
            pending = 0;
          }
        }
      }
      if (pending > 0) this.applyAccumulatedGradients(pending);
      if (stopped) break; // an interrupted epoch is not recorded

      const avgLoss = epochMaskedCount > 0 ? epochLoss / epochMaskedCount : 0;
      // Over the windows of this epoch (fewer than the targets when trainingStride > 1).
      const accuracy = epochCorrect / Math.max(1, items.length);
      trainingHistory.push({ loss: avgLoss, accuracy, timestamp: Date.now() });
    }

    const payload = {
      loss: maskedCount > 0 ? totalLoss / maskedCount : 0,
      accuracy: correct / Math.max(1, count),
      history: trainingHistory
    } as const;
    self.lastUpdatedAt = Date.now();
    return payload;
  }

  /**
   * Forward pass used by every inference path inherited from ProNeuralLM (generate with all
   * sampling modes and penalties, getLogitsForContext / distillation, evaluate).
   */
  protected async forward(inputs: number[], _train = false): Promise<ForwardCache> {
    const pass = this.stackForward(inputs, false);
    const h = pass.readouts[pass.readouts.length - 1];
    return {
      h,
      logits: pass.logits[pass.logits.length - 1],
      probs: pass.probs[pass.probs.length - 1],
      avgEmb: this.meanRows(pass.X0),
      dropMask: null,
      preAct: h
    };
  }

  /**
   * Inspection helper kept for diagnostics/tests: next-token distribution and the final hidden
   * sequence of the transformer stack.
   * @private
   */
  private async transformerForwardPass(inputs: number[], _train = false) {
    const cache = await this.forward(inputs, false);
    const pass = this.stackForward(inputs, false);
    return { ...cache, transformerHidden: pass.hidden };
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
    const causal = this.isCausal();
    return {
      numHeads: this.transformerConfig.numHeads,
      numKVHeads: this.transformerConfig.numKVHeads,
      ropeBase: this.transformerConfig.ropeBase,
      causal,
      ropePerHead: causal
    };
  }

  private meanRows(rows: Matrix): number[] {
    const H = rows[0]?.length ?? this.getHiddenSize();
    const y = new Array(H).fill(0);
    for (const v of rows) for (let i = 0; i < H; i++) y[i] += v[i];
    const n = rows.length || 1;
    for (let i = 0; i < H; i++) y[i] /= n;
    return y;
  }

  /**
   * Exact forward pass. `train` enables the stochastic regularisers; `scored` lists the
   * positions whose next-token distribution is needed (causal objective; default: the last
   * position). The pooled objective always produces a single read-out.
   */
  /**
   * Sample one DropConnect mask per layer for the next mini-batch and pre-compute the masked
   * weights (weights are constant within a mini-batch, so W̃ = W ⊙ M is built once).
   */
  private sampleBatchDropConnect(): LayerDropConnect[] | null {
    const rate = this.transformerConfig.dropConnectRate;
    if (rate <= 0) return null;
    const random = () => (this as any).nextRandom() as number;
    return this.attentionWeights.map((att, l) => {
      const masks = sampleWeightMasks(this.blockParams(l), rate, random);
      return {
        masks,
        masked: {
          query: hadamard(att.query, masks.query),
          key: hadamard(att.key, masks.key),
          value: hadamard(att.value, masks.value)
        }
      };
    });
  }

  private stackForward(
    inputs: number[],
    train: boolean,
    scored?: number[],
    useBatchDropConnect = false
  ): StackPass {
    const embedding = (this as any).embedding as number[][];
    return this.stackForwardFrom(
      inputs.map((i) => embedding[i].slice()),
      train,
      scored,
      useBatchDropConnect
    );
  }

  /** Forward pass from explicit input rows X0 (token embeddings or interpolations of them). */
  private stackForwardFrom(
    X0: Matrix,
    train: boolean,
    scored?: number[],
    useBatchDropConnect = false
  ): StackPass {
    const self = this as any;
    const wOutput = self.wOutput as number[][];
    const bOutput = self.bOutput as number[];
    const dropout = self.dropout as number;
    const random = () => self.nextRandom() as number;
    const positions = X0.map((_, i) => i);
    const config = this.blockConfig();
    const T = X0.length;

    // Attention dropout is sampled per window; DropConnect uses the mini-batch masks (W̃ = W ⊙ M).
    const { attentionDropout } = this.transformerConfig;
    const blockDropout: BlockDropout | undefined =
      train && attentionDropout > 0
        ? { attention: attentionDropout, dropConnect: 0, random }
        : undefined;
    const dropConnect = train && useBatchDropConnect ? this.batchDropConnect : null;

    const caches: BlockCache[] = [];
    let hidden = X0;
    for (let l = 0; l < this.transformerConfig.numLayers; l++) {
      const params = this.blockParams(l);
      if (dropConnect) params.attention = dropConnect[l].masked;
      const { out, cache } = blockForward(hidden, params, config, positions, {
        dropout: blockDropout
      });
      caches.push(cache);
      hidden = out;
    }

    let finalRms: number[] | null = null;
    let normed: Matrix | null = null;
    let scoredPositions: number[];
    let rawReadouts: number[][];
    if (this.isCausal()) {
      const fn = rmsForward(hidden, this.finalNorm.gamma, this.finalNorm.epsilon);
      finalRms = fn.rms;
      normed = fn.N;
      scoredPositions = scored ?? [T - 1];
      rawReadouts = scoredPositions.map((p) => normed![p]);
    } else {
      scoredPositions = [-1];
      rawReadouts = [this.meanRows(hidden)];
    }

    const masks: (number[] | null)[] = [];
    const readouts = rawReadouts.map((r) => {
      if (!train || dropout <= 0) {
        masks.push(null);
        return r;
      }
      const keepScale = 1 / (1 - dropout);
      const mask = r.map(() => (random() > dropout ? keepScale : 0));
      masks.push(mask);
      return r.map((v, i) => v * mask[i]);
    });

    const logits = readouts.map((h) => projectOutput(h, wOutput, bOutput));

    return {
      positions,
      X0,
      caches,
      hidden,
      finalRms,
      normed,
      scored: scoredPositions,
      readouts,
      masks,
      logits,
      probs: logits.map((z) => outputSoftmax(z))
    };
  }

  private ensureGradBuffers(): GradBuffers {
    const self = this as any;
    const H = this.getHiddenSize();
    const V = (self.bOutput as number[]).length;
    const g = this.gradBuffers;
    if (
      g &&
      g.wOutput.length === H &&
      g.bOutput.length === V &&
      g.layers.length === this.transformerConfig.numLayers &&
      g.layers.every(
        (lg, l) =>
          lg.query.length === H &&
          lg.key[0]?.length === this.attentionWeights[l].key[0]?.length &&
          lg.ff2.length === this.ffWeights2[l].length
      )
    ) {
      return g;
    }
    this.gradBuffers = this.createGradBuffers();
    return this.gradBuffers;
  }

  private createGradBuffers(): GradBuffers {
    const self = this as any;
    const H = this.getHiddenSize();
    const V = (self.bOutput as number[]).length;
    return {
      wOutput: zeros((self.wOutput as number[][]).length, V),
      bOutput: new Array(V).fill(0),
      finalGamma: this.isCausal() ? new Array(H).fill(0) : null,
      layers: this.attentionWeights.map((_, l) => createBlockGrads(this.blockParams(l))),
      embedding: new Map()
    };
  }

  /**
   * Backpropagate the mean cross-entropy over the scored entries of one window (targets < 0 are
   * skipped) and add the gradients to the accumulators. Returns that mean loss.
   */
  private accumulateGradients(inputs: number[], pass: StackPass, targets: number[]): number {
    const H = this.getHiddenSize();
    const g = this.ensureGradBuffers();
    const { loss, dInputs } = this.backpropagate(pass, targets, g);
    inputs.forEach((tok, t) => {
      let row = g.embedding.get(tok);
      if (!row) {
        row = new Array(H).fill(0);
        g.embedding.set(tok, row);
      }
      for (let i = 0; i < H; i++) row[i] += dInputs[t][i];
    });
    return loss;
  }

  /**
   * Backpropagate the mean cross-entropy over the scored entries (targets < 0 are skipped),
   * adding parameter gradients to `g`; returns the loss and the gradient w.r.t. each input row.
   */
  private backpropagate(
    pass: StackPass,
    targets: number[],
    g: GradBuffers
  ): { loss: number; dInputs: Matrix } {
    const self = this as any;
    const wOutput = self.wOutput as number[][];
    const H = this.getHiddenSize();
    const T = pass.X0.length;
    const config = this.blockConfig();

    const active = targets.filter((t) => t >= 0).length;
    if (active === 0) return { loss: 0, dInputs: zeros(T, H) };
    const w = 1 / active;

    let loss = 0;
    targets.forEach((tgt, k) => {
      if (tgt >= 0) loss -= Math.log(pass.probs[k][tgt] + 1e-12) * w;
    });
    const outputGradients = outputBackward(
      pass.readouts,
      pass.probs,
      targets,
      w,
      wOutput,
      g.wOutput,
      g.bOutput
    );
    const dReadouts: (number[] | null)[] = outputGradients.map((dr, k) => {
      if (!dr) return null;
      const mask = pass.masks[k];
      return mask ? dr.map((v, i) => v * mask[i]) : dr;
    });

    let dHidden: Matrix;
    if (this.isCausal()) {
      const dNormed = zeros(T, H);
      dReadouts.forEach((dr, k) => {
        if (!dr) return;
        const row = dNormed[pass.scored[k]];
        for (let i = 0; i < H; i++) row[i] += dr[i];
      });
      dHidden = rmsBackward(
        dNormed,
        pass.hidden,
        pass.finalRms!,
        this.finalNorm.gamma,
        g.finalGamma!
      );
    } else {
      const dr = dReadouts[0]!;
      dHidden = pass.hidden.map(() => dr.map((v) => v / T));
    }

    for (let l = this.transformerConfig.numLayers - 1; l >= 0; l--) {
      dHidden = blockBackward(
        dHidden,
        pass.caches[l],
        this.blockParams(l),
        config,
        pass.positions,
        g.layers[l]
      );
    }
    return { loss, dInputs: dHidden };
  }

  // ---------------------------------------------------------------------------
  // Explainability
  // ---------------------------------------------------------------------------

  /**
   * log p(target | context) from explicit input rows (no dropout, no parameter updates) and its
   * exact gradient with respect to every row.
   */
  explainLogProb(
    contextEmbeddings: number[][],
    target: number
  ): { logProb: number; gradient: number[][] } {
    const pass = this.stackForwardFrom(
      contextEmbeddings.map((row) => row.slice()),
      false
    );
    const probs = pass.probs[pass.probs.length - 1];
    const targets = pass.scored.map((_, k) => (k === pass.scored.length - 1 ? target : -1));
    const { dInputs } = this.backpropagate(pass, targets, this.createGradBuffers());
    return {
      logProb: Math.log(Math.max(probs[target], 1e-300)),
      gradient: dInputs.map((row) => row.map((v) => -v)) // ∂ log p = −∂ CE
    };
  }

  /**
   * Attention probabilities of every layer and head for a context window
   * ([layer][head][query position][key position]).
   */
  getAttentionMaps(context: number[]): number[][][][] {
    const pass = this.stackForward(context, false);
    return pass.caches.map((cache) => cache.attn.map((head) => head.map((row) => row.slice())));
  }

  /**
   * Use a per-optimizer-step schedule relative to the model's learning rate (null restores a
   * constant rate). The step counter restarts whenever a schedule is set.
   */
  setLearningRateSchedule(schedule: LearningRateSchedule | null): void {
    this.lrSchedule = schedule;
    this.scheduleStep = 0;
  }

  /** Learning rate the next optimizer step will use. */
  getScheduledLearningRate(): number {
    return this.scheduledLearningRate(this.scheduleStep);
  }

  private scheduledLearningRate(step: number): number {
    const base = (this as any).learningRate as number;
    const schedule = this.lrSchedule;
    if (!schedule) return base;
    const warmup = Math.max(0, Math.floor(schedule.warmupSteps));
    const total = Math.max(warmup + 1, Math.floor(schedule.totalSteps));
    const minFactor = schedule.minFactor ?? 0.1;
    if (step < warmup) return (base * (step + 1)) / warmup;
    const progress = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
    return base * (minFactor + (1 - minFactor) * 0.5 * (1 + Math.cos(Math.PI * progress)));
  }

  private nextScheduledLearningRate(): number {
    const lr = this.scheduledLearningRate(this.scheduleStep);
    this.scheduleStep++;
    return lr;
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
   * DropConnect: gradients were accumulated w.r.t. W̃ = W ⊙ M, so ∂L/∂W = M ⊙ ∂L/∂W̃.
   * Converts the Q/K/V accumulators in place and ends the mini-batch's mask lifetime.
   */
  private finalizeDropConnectGradients(g: GradBuffers): void {
    if (!this.batchDropConnect) return;
    this.batchDropConnect.forEach(({ masks }, l) => {
      const lg = g.layers[l];
      for (const [grad, mask] of [
        [lg.query, masks.query],
        [lg.key, masks.key],
        [lg.value, masks.value]
      ] as const) {
        if (!mask) continue;
        for (let i = 0; i < grad.length; i++) {
          const row = grad[i];
          const mrow = mask[i];
          for (let j = 0; j < row.length; j++) row[j] *= mrow[j];
        }
      }
    });
    this.batchDropConnect = null;
  }

  /**
   * Average the accumulated gradients over `count` windows, clip their global norm, apply one
   * optimizer step to every parameter and reset the accumulators.
   */
  private applyAccumulatedGradients(count: number): void {
    const g = this.gradBuffers;
    if (!g || count <= 0) return;
    const self = this as any;
    const lr = this.nextScheduledLearningRate();
    const weightDecay = Math.max(0, this.transformerConfig.weightDecay ?? 0);

    this.finalizeDropConnectGradients(g);

    const tensors: Array<number[] | number[][]> = [g.wOutput, g.bOutput];
    if (g.finalGamma) tensors.push(g.finalGamma);
    for (const lg of g.layers) {
      tensors.push(lg.query, lg.key, lg.value, lg.ff1, lg.ff2, lg.attnGamma, lg.ffnGamma);
    }
    for (const row of g.embedding.values()) tensors.push(row);

    // One pass for the norm; averaging, clipping and the accumulator reset are fused into the
    // optimizer update below.
    const norm = globalNorm(tensors) / count;
    const gradScale =
      (1 / count) * (norm > TRANSFORMER_GRAD_CLIP ? TRANSFORMER_GRAD_CLIP / norm : 1);
    const opts = { gradScale, zeroGrad: true };
    // Weight decay applies to weight matrices and embeddings, not to norm gains or biases.
    const decayed = { ...opts, weightDecay };

    const opt = this.getParamOptimizer();
    opt.step();
    opt.update('wOutput', self.wOutput, g.wOutput, lr, undefined, decayed);
    opt.update('bOutput', self.bOutput, g.bOutput, lr, undefined, opts);
    if (g.finalGamma && this.isCausal()) {
      opt.update('finalGamma', this.finalNorm.gamma, g.finalGamma, lr, undefined, opts);
    }
    for (let l = 0; l < g.layers.length; l++) {
      const p = this.blockParams(l);
      const lg = g.layers[l];
      opt.update(`L${l}.query`, p.attention.query, lg.query, lr, undefined, decayed);
      opt.update(`L${l}.key`, p.attention.key, lg.key, lr, undefined, decayed);
      opt.update(`L${l}.value`, p.attention.value, lg.value, lr, undefined, decayed);
      opt.update(`L${l}.ff1`, p.ff1, lg.ff1, lr, undefined, decayed);
      opt.update(`L${l}.ff2`, p.ff2, lg.ff2, lr, undefined, decayed);
      opt.update(`L${l}.attnGamma`, p.attnGamma, lg.attnGamma, lr, undefined, opts);
      opt.update(`L${l}.ffnGamma`, p.ffnGamma, lg.ffnGamma, lr, undefined, opts);
    }
    const sparse: number[][] = [];
    for (const [tok, row] of g.embedding) sparse[tok] = row;
    opt.update('embedding', self.embedding, sparse, lr, g.embedding.keys(), decayed);
    g.embedding.clear();
  }

  // ---------------------------------------------------------------------------
  // Structural changes (Cerebro neuron injection)
  // ---------------------------------------------------------------------------

  /**
   * Widen the model from D to D + k.
   *
   * When k is a multiple of numHeads every head grows by k/numHeads dimensions appended at the
   * end of its slice, so the learned head structure is kept (the residual stream is re-indexed
   * consistently in every tensor). New output-projection rows and new FFN output columns start
   * at zero. Otherwise the new dimensions are appended and a valid head layout is re-derived
   * (learned head structure is not preserved). Optimizer state is reset in both cases.
   */
  expandHiddenLayer(k: number, useHeInit = true): void {
    if (k <= 0) return;
    const self = this as any;

    const oldD = this.getHiddenSize();
    const newD = oldD + k;
    const cfg = this.transformerConfig;
    const oldHeads = cfg.numHeads;
    const oldKV = cfg.numKVHeads;
    const oldHd = oldD / oldHeads;
    const oldKvDim = oldKV * oldHd;

    let newHeads = oldHeads;
    let newKV = oldKV;
    let residualMap: number[];
    let kvMap: number[];
    if (k % oldHeads === 0) {
      const newHd = newD / oldHeads;
      residualMap = blockwiseIndexMap(oldD, oldHd, newHd);
      kvMap = blockwiseIndexMap(oldKvDim, oldHd, newHd);
    } else {
      newHeads = this.normalizeHeadCount(newD, oldHeads);
      newKV = this.normalizeKVHeadCount(newHeads, oldKV);
      residualMap = identityMap(oldD);
      kvMap = identityMap(Math.min(oldKvDim, newKV * (newD / newHeads)));
    }
    const newKvDim = newKV * (newD / newHeads);

    const scale = useHeInit ? Math.sqrt(2.0 / newD) : Math.sqrt(1.0 / newD);
    const rand = () => (this.initRandom() - 0.5) * 2 * scale;
    const small = () => (this.initRandom() - 0.5) * 0.1;
    const zero = () => 0;
    const one = () => 1;
    const V = this.getVocabSize();

    // Base tensors: embeddings, output projection and the (unused) MLP hidden layer
    self.embedding = (self.embedding as Matrix).map((row) =>
      remapVector(row, residualMap, newD, small)
    );
    self.wOutput = remapRows(self.wOutput as Matrix, residualMap, newD, () => new Array(V).fill(0));
    self.wHidden = remapCols(
      remapRows(self.wHidden as Matrix, residualMap, newD, () => new Array(oldD).fill(0)),
      residualMap,
      newD,
      zero
    );
    self.bHidden = remapVector(self.bHidden as number[], residualMap, newD, zero);
    self.hiddenSize = newD;

    for (let l = 0; l < cfg.numLayers; l++) {
      const att = this.attentionWeights[l];
      const newRow = (width: number) => () => Array.from({ length: width }, rand);
      att.query = remapCols(
        remapRows(att.query, residualMap, newD, newRow(oldD)),
        residualMap,
        newD,
        rand
      );
      att.key = remapCols(
        remapRows(att.key, residualMap, newD, newRow(oldKvDim)),
        kvMap,
        newKvDim,
        rand
      );
      att.value = remapCols(
        remapRows(att.value, residualMap, newD, newRow(oldKvDim)),
        kvMap,
        newKvDim,
        rand
      );
      this.ffWeights1[l] = remapRows(
        this.ffWeights1[l],
        residualMap,
        newD,
        newRow(this.ffWeights1[l][0]?.length ?? 0)
      );
      this.ffWeights2[l] = remapCols(this.ffWeights2[l], residualMap, newD, zero);
      const norm = this.renormStates[l];
      norm.attention.gamma = remapVector(norm.attention.gamma, residualMap, newD, one);
      norm.ffn.gamma = remapVector(norm.ffn.gamma, residualMap, newD, one);
    }
    this.finalNorm.gamma = remapVector(this.finalNorm.gamma, residualMap, newD, one);

    // Position table (API compatibility): extend with sinusoids for the new dimensions
    for (let pos = 0; pos < this.positionEmbeddings.length; pos++) {
      const posEmb = this.positionEmbeddings[pos];
      for (let i = oldD; i < newD; i++) posEmb.push(this.sinusoid(pos, i, newD));
    }

    this.transformerConfig = { ...cfg, numHeads: newHeads, numKVHeads: newKV };
    this.paramOptimizer?.reset();
    this.gradBuffers = null;
    this.reinitializeOptimizerState();
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
    finalNorm: RMSNormState | null;
  } {
    return {
      attentionWeights: this.attentionWeights,
      ffWeights1: this.ffWeights1,
      ffWeights2: this.ffWeights2,
      positionEmbeddings: this.positionEmbeddings,
      renormStates: this.renormStates,
      finalNorm: this.isCausal() ? this.finalNorm : null
    };
  }

  /**
   * Set transformer-specific weights from adapter (for rollback). The KV-head count is derived
   * from the restored key matrices so that GQA layouts round-trip.
   */
  setTransformerWeights(weights: {
    attentionWeights: AttentionWeights[];
    ffWeights1: Matrix[];
    ffWeights2: Matrix[];
    positionEmbeddings: number[][];
    renormStates: TransformerRMSStates[] | RMSNormState[];
    finalNorm?: RMSNormState | null;
  }): void {
    const D = this.getHiddenSize();
    const heads = this.normalizeHeadCount(D, this.transformerConfig.numHeads);
    const kvDim = weights.attentionWeights[0]?.key[0]?.length ?? D;
    const headDim = D / heads;
    const kvHeads = Number.isInteger(kvDim / headDim) ? kvDim / headDim : heads;
    this.transformerConfig = { ...this.transformerConfig, numHeads: heads, numKVHeads: kvHeads };

    this.attentionWeights = weights.attentionWeights;
    this.ffWeights1 = weights.ffWeights1;
    this.ffWeights2 = weights.ffWeights2;
    this.positionEmbeddings = weights.positionEmbeddings;
    this.renormStates = weights.renormStates.map((state) =>
      this.normalizeRenormState(state as TransformerRMSStates | RMSNormState, D)
    );
    this.finalNorm =
      weights.finalNorm && weights.finalNorm.gamma.length === D
        ? { gamma: [...weights.finalNorm.gamma], epsilon: weights.finalNorm.epsilon }
        : { gamma: new Array(D).fill(1), epsilon: NORM_EPSILON };
    this.paramOptimizer?.reset();
    this.gradBuffers = null;
  }

  // ---------------------------------------------------------------------------
  // Serialization
  // ---------------------------------------------------------------------------

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
        renormStates: this.renormStates,
        ...(this.isCausal() ? { finalNorm: this.finalNorm } : {})
      }
    } satisfies TransformerSerializedModel;
  }

  static loadFromLocalStorage(key: string): TransformerLM | null {
    try {
      if (typeof localStorage === 'undefined') return null;
      const data = parseStoredModel(localStorage.getItem(key)) as TransformerSerializedModel | null;
      if (!data || data.architecture !== 'transformer' || !data.transformer) return null;
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
    const saved = data.transformer.config ?? {};
    // Models saved before the causal revision carry no objective: keep their exact behaviour.
    const objective: TransformerObjective = saved.objective ?? 'pooled';
    const config: TransformerConfig = {
      ...saved,
      objective,
      ropeBase: saved.ropeBase ?? (objective === 'pooled' ? ROPE_BASE_LEGACY : ROPE_BASE_CAUSAL)
    };

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
      config
    );
    TransformerLM.restoreSerializedState(model, data);
    model.initializeTransformerLayers({
      attention: data.transformer.attentionWeights,
      ff1: data.transformer.ffWeights1,
      ff2: data.transformer.ffWeights2,
      renormStates: data.transformer.renormStates,
      finalNorm: data.transformer.finalNorm
    });
    return model;
  }
}
