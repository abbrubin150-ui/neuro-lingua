import {
  applyUpdateVector,
  createSecondOrderState,
  dampedNewtonStep,
  flattenGradients,
  flattenStructure,
  quasiNewtonStep,
  updateCurvatureEMA,
  type SecondOrderConfig,
  type SecondOrderState
} from '../training/optimizer';
import { stableSoftmax } from './MathUtils';
import {
  applyRepetitionControls,
  mirostatV2Sample,
  type MirostatV2State,
  type RepetitionControls,
  sampleFromLogits as drawToken,
  typicalSample
} from '../generation/sampling';
import { GPUNeuralOps } from '../backend/gpu_neural_ops';
import { SophiaOptimizer, type SophiaConfig } from '../training/SophiaOptimizer';
import type { LossMaskConfig } from '../types/project';
import { findRegExpMatches, buildCharMaskFromMatches } from '../losses/lossMask';
import { parseStoredModel, storeModelCompactly } from './modelPackageCodec';
import {
  decodeSubword,
  encodeSubword,
  isSubwordModel,
  subwordVocabulary,
  type SubwordModel
} from '../tokenizer/subword';

export type EpochMetrics = { loss: number; accuracy: number };

export type EvaluationResult = {
  /** Mean cross-entropy in nats per scored token. */
  loss: number;
  perplexity: number;
  accuracy: number;
  /** Number of scored positions (including the final EOS). */
  tokens: number;
  /** Fraction of targets that were out-of-vocabulary. */
  oovRate: number;
};

/** Position of a model's random stream (see getRandomState). */
export type RandomState = { seed: number; state: number };

/** Sums over a range of prediction windows (see `evaluateSums` / `evaluationResult`). */
export type EvaluationSums = {
  /** Negative log-likelihood in nats. */
  nll: number;
  correct: number;
  oov: number;
  /** Windows scored in the range. */
  count: number;
  /** Windows in the whole text. */
  total: number;
};

/** Activations produced by a forward pass and consumed by backward(). */
export type ForwardCache = {
  h: number[];
  logits: number[];
  probs: number[];
  avgEmb: number[];
  dropMask: number[] | null;
  preAct: number[];
  /** Hidden pre-activation after normalizeHidden (equals preAct when no normalisation). */
  z?: number[];
};

export type Optimizer = 'momentum' | 'adam' | 'newton' | 'bfgs' | 'lion' | 'sophia';
export type TokenizerMode = 'unicode' | 'ascii' | 'custom' | 'character' | 'bpe';

export type TokenizerConfig = {
  mode: TokenizerMode;
  pattern?: string;
  /** Trained subword model (mode 'bpe'); see src/tokenizer/subword.ts. */
  bpe?: SubwordModel;
  /** Requested subword vocabulary size when the merges are (re)trained from a corpus. */
  bpeVocabSize?: number;
};

/** EMA decay of the Gauss-Newton curvature used by the 'newton' optimizer. */
const NEWTON_CURVATURE_DECAY = 0.95;
/** Lower bound of the Levenberg-Marquardt damping λ for the 'newton' optimizer. */
const NEWTON_MIN_DAMPING = 0.1;

const DEFAULT_TOKENIZER_CONFIG: TokenizerConfig = { mode: 'unicode' };

export const MODEL_VERSION = '3.2.4';
export const MODEL_COMPACT_VERSION = MODEL_VERSION.replace(/\./g, '');
export const MODEL_STORAGE_KEY = `neuro-lingua-pro-v${MODEL_COMPACT_VERSION}`;
export const TRANSFORMER_MODEL_STORAGE_KEY = `neuro-lingua-transformer-v${MODEL_COMPACT_VERSION}`;
export const MODEL_EXPORT_FILENAME = `neuro-lingua-v${MODEL_COMPACT_VERSION}.json`;

type Rng = {
  next(): number;
  getState(): number;
};

function makeRng(seed: number, state?: number): Rng {
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

export function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

function vectorDot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function l2norm(a: number[]) {
  return Math.sqrt(vectorDot(a, a));
}

function clipVectorInPlace(a: number[], maxNorm: number) {
  const n = l2norm(a);
  if (n > maxNorm && n > 0) {
    const s = maxNorm / n;
    for (let i = 0; i < a.length; i++) a[i] *= s;
  }
}

export class ProNeuralLM {
  private vocab: string[];
  private wordToIdx: Map<string, number>;
  private idxToWord: Map<number, string>;

  private embedding: number[][] = [];
  private wHidden: number[][] = [];
  private wOutput: number[][] = [];
  private bHidden: number[] = [];
  private bOutput: number[] = [];

  private hiddenSize: number;
  private learningRate: number;
  private contextSize: number;
  private rng: Rng;
  private rngSeed: number;
  private rngState: number;

  private dropout: number;
  private optimizer: Optimizer;

  private mEmbedding: number[][] = [];
  private mWHidden: number[][] = [];
  private mWOutput: number[][] = [];
  private mBHidden: number[] = [];
  private mBOutput: number[] = [];
  private momentum: number;

  private adamBeta1 = 0.9;
  private adamBeta2 = 0.999;
  private adamEps = 1e-8;
  private adamT = 0;

  // Lion optimizer hyperparameters (v4.0)
  private lionBeta1 = 0.9;
  private lionBeta2 = 0.99;
  private lionWeightDecay = 0.01;

  // Sophia optimizer instance (v4.2)
  private sophiaOptimizer: SophiaOptimizer | null = null;
  private aEmbedding: { m: number[][]; v: number[][] } = {
    m: [] as number[][],
    v: [] as number[][]
  };
  private aWHidden: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aWOutput: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aBHidden: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };
  private aBOutput: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };

  private trainingHistory: { loss: number; accuracy: number; timestamp: number }[] = [];

  private tokenizerConfig: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG;
  private lastUpdatedAt: number | null = null;

  private secondOrderConfig: SecondOrderConfig = {
    damping: 1e-4,
    epsilon: 1e-9,
    maxHistory: 7
  };
  private secondOrderState: SecondOrderState = createSecondOrderState();
  private progressCallback: ((done: number, total: number, epoch: number) => void) | null = null;
  private yieldIntervalMs = 0;
  private lastYieldAt = 0;
  private trainingSignal: AbortSignal | null = null;

  private bos = '<BOS>';
  private eos = '<EOS>';
  private unk = '<UNK>';

  private gpuOps: GPUNeuralOps | null = null;

  constructor(
    vocab: string[],
    hiddenSize = 64,
    learningRate = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'momentum',
    momentum = 0.9,
    dropout = 0.0,
    seed = 1337,
    tokenizerConfig: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG
  ) {
    this.vocab = vocab;
    this.hiddenSize = hiddenSize;
    this.learningRate = learningRate;
    this.contextSize = contextSize;
    this.optimizer = optimizer;
    this.momentum = momentum;
    this.dropout = clamp(dropout, 0, 0.5);
    this.rngSeed = seed >>> 0;
    this.rng = makeRng(this.rngSeed);
    this.rngState = this.rng.getState();
    this.setTokenizerConfig(tokenizerConfig);

    this.wordToIdx = new Map(vocab.map((w, i) => [w, i]));
    this.idxToWord = new Map(vocab.map((w, i) => [i, w]));
    this.initializeParameters();

    // Initialize Sophia optimizer if selected
    if (this.optimizer === 'sophia') {
      this.sophiaOptimizer = new SophiaOptimizer({ lr: this.learningRate });
    }
  }

  protected static normalizeTokenizerConfig(config?: TokenizerConfig): TokenizerConfig {
    if (!config) return { ...DEFAULT_TOKENIZER_CONFIG };
    if (config.mode === 'character') return { mode: 'character' };
    if (config.mode === 'bpe') {
      // Keep the same object so that the encoder cache (keyed by model identity) is reused.
      return isSubwordModel(config.bpe)
        ? { mode: 'bpe', bpe: config.bpe }
        : { ...DEFAULT_TOKENIZER_CONFIG };
    }
    if (config.mode === 'ascii') return { mode: 'ascii' };
    if (config.mode === 'custom') {
      if (config.pattern && config.pattern.length > 0) {
        return { mode: 'custom', pattern: config.pattern };
      }
      return { ...DEFAULT_TOKENIZER_CONFIG };
    }
    return { mode: 'unicode' };
  }

  private static tokenizerRegexFromConfig(config: TokenizerConfig): RegExp {
    try {
      if (config.mode === 'ascii') {
        return /[^a-z0-9\s'-]/g;
      }
      if (config.mode === 'custom' && config.pattern) {
        try {
          return new RegExp(config.pattern, 'gu');
        } catch {
          return new RegExp(config.pattern, 'g');
        }
      }
      return /[^\p{L}\d\s'-]/gu;
    } catch (err) {
      console.warn('Failed to compile tokenizer regex. Falling back to Unicode pattern.', err);
      return /[^\p{L}\d\s'-]/gu;
    }
  }

  /**
   * Tokenize `text` and return every token with its [start, end) offsets in the original string.
   *
   * - character mode: one token per code point; every run of whitespace becomes a single ' '
   *   token (leading/trailing whitespace is dropped) so that a character model can learn word
   *   and expression boundaries. Case is preserved.
   * - word modes (unicode / ascii / custom): lower-cased; characters matched by the mode's
   *   regex act as separators; tokens are the remaining non-whitespace runs.
   */
  static tokenizeWithOffsets(
    text: string,
    config?: TokenizerConfig
  ): { token: string; start: number; end: number }[] {
    const normalized = ProNeuralLM.normalizeTokenizerConfig(config);
    const out: { token: string; start: number; end: number }[] = [];

    if (normalized.mode === 'bpe' && normalized.bpe) {
      return encodeSubword(text, normalized.bpe);
    }

    if (normalized.mode === 'character') {
      let i = 0;
      let pendingSpace: { start: number; end: number } | null = null;
      for (const ch of text) {
        const start = i;
        i += ch.length;
        if (/\s/u.test(ch)) {
          if (out.length > 0) {
            pendingSpace = pendingSpace ? { start: pendingSpace.start, end: i } : { start, end: i };
          }
          continue;
        }
        if (pendingSpace) {
          out.push({ token: ' ', ...pendingSpace });
          pendingSpace = null;
        }
        out.push({ token: ch, start, end: i });
      }
      return out;
    }

    const regex = ProNeuralLM.tokenizerRegexFromConfig(normalized);
    const lower = text.toLowerCase();
    // Replace separators with the same number of spaces so offsets stay aligned with `text`.
    const cleaned = lower.replace(regex, (m) => ' '.repeat(m.length));
    const aligned = cleaned.length === text.length;
    const re = /\S+/gu;
    let match: RegExpExecArray | null;
    while ((match = re.exec(cleaned)) !== null) {
      const start = aligned ? match.index : Math.min(match.index, text.length);
      const end = aligned
        ? match.index + match[0].length
        : Math.min(start + match[0].length, text.length);
      out.push({ token: match[0], start, end });
    }
    return out;
  }

  static tokenizeText(text: string, config?: TokenizerConfig): string[] {
    return ProNeuralLM.tokenizeWithOffsets(text, config).map((t) => t.token);
  }

  /**
   * Join generated tokens back into text: characters are concatenated, subwords are decoded
   * (word markers become spaces) and words are space-separated.
   */
  protected joinTokens(tokens: string[]): string {
    // Generated tokens continue the prompt: keep the space a leading word marker stands for.
    if (this.tokenizerConfig.mode === 'bpe') return decodeSubword(tokens, { continuation: true });
    return this.tokenizerConfig.mode === 'character' ? tokens.join('') : tokens.join(' ');
  }

  /**
   * Model vocabulary for a corpus: the four special tokens followed by the tokens of `text`
   * (in order of first appearance) or, for a subword tokenizer, its full subword vocabulary
   * (so that any text made of known characters has no out-of-vocabulary tokens).
   */
  static buildVocabulary(text: string, config?: TokenizerConfig): string[] {
    const normalized = ProNeuralLM.normalizeTokenizerConfig(config);
    const specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
    const tokens =
      normalized.mode === 'bpe' && normalized.bpe
        ? subwordVocabulary(normalized.bpe)
        : ProNeuralLM.tokenizeText(text, normalized);
    return Array.from(new Set([...specials, ...tokens]));
  }

  setTokenizerConfig(config: TokenizerConfig) {
    const normalized = ProNeuralLM.normalizeTokenizerConfig(config);
    this.tokenizerConfig = normalized;
  }

  getTokenizerConfig(): TokenizerConfig {
    return { ...this.tokenizerConfig };
  }

  exportTokenizerConfig() {
    return this.getTokenizerConfig();
  }

  importTokenizerConfig(config: TokenizerConfig) {
    this.setTokenizerConfig(config);
  }

  getLastUpdatedAt() {
    if (this.lastUpdatedAt) return this.lastUpdatedAt;
    const last = this.trainingHistory[this.trainingHistory.length - 1];
    return last?.timestamp ?? null;
  }

  /**
   * Set GPU operations handler for hardware acceleration
   */
  setGPUOps(gpuOps: GPUNeuralOps | null) {
    this.gpuOps = gpuOps;
  }

  /**
   * Accessor for subclasses that need to share the GPU accelerator.
   */
  protected getGPUOpsInstance(): GPUNeuralOps | null {
    return this.gpuOps;
  }

  /**
   * Check if GPU acceleration is enabled and available
   */
  isGPUEnabled(): boolean {
    return this.gpuOps !== null && this.gpuOps.isEnabled();
  }

  private nextRandom() {
    const value = this.rng.next();
    this.rngState = this.rng.getState();
    return value;
  }

  private randn(scale = 0.1) {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.nextRandom();
    while (v === 0) v = this.nextRandom();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
  }

  private randomArray(size: number, scale: number): number[] {
    const a = new Array(size);
    for (let i = 0; i < size; i++) a[i] = this.randn(scale);
    return a;
  }

  private zerosVec(n: number) {
    return new Array(n).fill(0);
  }

  private zerosMat(r: number, c: number) {
    return new Array(r).fill(0).map(() => new Array(c).fill(0));
  }

  private initializeParameters() {
    const V = this.vocab.length;
    const H = this.hiddenSize;

    this.embedding = new Array(V);
    for (let i = 0; i < V; i++) this.embedding[i] = this.randomArray(H, 0.05);

    this.wHidden = new Array(H);
    for (let i = 0; i < H; i++) this.wHidden[i] = this.randomArray(H, 0.05);

    this.wOutput = new Array(H);
    for (let i = 0; i < H; i++) this.wOutput[i] = this.randomArray(V, 0.05);

    this.bHidden = new Array(H).fill(0);
    this.bOutput = new Array(V).fill(0);

    this.mEmbedding = this.zerosMat(V, H);
    this.mWHidden = this.zerosMat(H, H);
    this.mWOutput = this.zerosMat(H, V);
    this.mBHidden = this.zerosVec(H);
    this.mBOutput = this.zerosVec(V);

    this.aEmbedding = { m: this.zerosMat(V, H), v: this.zerosMat(V, H) };
    this.aWHidden = { m: this.zerosMat(H, H), v: this.zerosMat(H, H) };
    this.aWOutput = { m: this.zerosMat(H, V), v: this.zerosMat(H, V) };
    this.aBHidden = { m: this.zerosVec(H), v: this.zerosVec(H) };
    this.aBOutput = { m: this.zerosVec(V), v: this.zerosVec(V) };

    this.resetSecondOrderState();
  }

  private resetSecondOrderState() {
    const maxHistory = this.secondOrderConfig.maxHistory ?? 7;
    this.secondOrderState = createSecondOrderState(maxHistory);
  }

  // ---------------------------------------------------------------------------
  // Extension hooks (overridden by AdvancedNeuralLM). The base model is a ReLU MLP with no
  // normalisation, no global gradient clipping and a constant learning rate.
  // ---------------------------------------------------------------------------

  /** Hidden-layer activation φ(z). */
  protected activate(x: number): number {
    return x > 0 ? x : 0;
  }

  /** Derivative φ'(z) of the hidden-layer activation. */
  protected activateDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  /** Transform applied to the hidden pre-activation before φ (identity in the base model). */
  protected normalizeHidden(preAct: number[]): number[] {
    return preAct;
  }

  /**
   * Backpropagate through normalizeHidden: maps ∂L/∂z to ∂L/∂preAct. Implementations that own
   * parameters (e.g. LayerNorm γ/β) update them here unless `updateParams` is false.
   */
  protected normalizeHiddenBackward(
    dZ: number[],
    _preAct: number[],
    _updateParams = true
  ): number[] {
    return dZ;
  }

  /** Global L2 gradient-norm threshold for one step (≤ 0 disables global clipping). */
  protected getGradientClipNorm(): number {
    return 0;
  }

  /** Called at the start of every train() call. */
  protected onTrainStart(_epochs: number): void {}

  /** Called before each epoch inside train(). */
  protected onEpochStart(_epoch: number, _epochs: number): void {}

  /** Called after every optimisation step inside train(). */
  protected onTrainingStep(): void {}

  /** Called after each epoch inside train(). */
  protected onEpochEnd(_epoch: number, _epochs: number, _metrics: EpochMetrics): void {}

  /**
   * Configure the Sophia optimizer (β₁, β₂, ρ, weight decay, Hessian refresh interval).
   * Recreates its state; ignored unless the model was created with optimizer 'sophia'.
   */
  setSophiaConfig(config: Omit<SophiaConfig, 'lr'>): void {
    if (this.optimizer !== 'sophia') return;
    this.sophiaOptimizer = new SophiaOptimizer({ ...config, lr: this.learningRate });
  }

  /**
   * Report progress inside train(): called with (windows processed, windows in the epoch, epoch)
   * roughly every percent of an epoch. Pass null to remove it.
   */
  setTrainingProgressCallback(
    callback: ((done: number, total: number, epoch: number) => void) | null
  ): void {
    this.progressCallback = callback;
  }

  /**
   * Interactive use (the browser app): make train() hand control back to the event loop at least
   * every `intervalMs` milliseconds, so that the page stays responsive and can show progress, and
   * stop before the next training window once `signal` is aborted (an interrupted epoch is not
   * added to the training history). 0 disables yielding — the default, used by the CLI and tests.
   */
  setInteractiveTraining(intervalMs: number, signal: AbortSignal | null = null): void {
    this.yieldIntervalMs = Math.max(0, intervalMs);
    this.trainingSignal = signal;
    this.lastYieldAt = Date.now();
  }

  /**
   * Checked before every training window: yields to the event loop when the interval has elapsed
   * and returns true when training was asked to stop.
   */
  protected async trainingShouldStop(): Promise<boolean> {
    if (this.yieldIntervalMs > 0 && Date.now() - this.lastYieldAt >= this.yieldIntervalMs) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      this.lastYieldAt = Date.now();
    }
    return this.trainingSignal?.aborted === true;
  }

  protected reportTrainingProgress(done: number, total: number, epoch: number): void {
    const cb = this.progressCallback;
    if (!cb) return;
    const every = Math.max(1, Math.floor(total / 100));
    if (done === total || done % every === 0) cb(done, total, epoch);
  }

  /** Current (possibly scheduled) learning rate. */
  getLearningRate(): number {
    return this.learningRate;
  }

  protected setLearningRateInternal(lr: number): void {
    this.learningRate = lr;
  }

  private async matrixVectorMul(A: number[][], x: number[]): Promise<number[]> {
    // Use GPU if available and enabled
    if (this.gpuOps && this.gpuOps.isEnabled()) {
      try {
        return await this.gpuOps.matrixVectorMul(A, x);
      } catch (error) {
        console.warn('GPU matrixVectorMul failed, falling back to CPU:', error);
        // Fall through to CPU implementation
      }
    }

    // CPU implementation
    const y = new Array(A.length).fill(0);
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      const row = A[i];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      y[i] = s;
    }
    return y;
  }

  private async matrixVectorMulTranspose(A: number[][], x: number[]): Promise<number[]> {
    // Use GPU if available and enabled
    if (this.gpuOps && this.gpuOps.isEnabled()) {
      try {
        return await this.gpuOps.matrixVectorMulTranspose(A, x);
      } catch (error) {
        console.warn('GPU matrixVectorMulTranspose failed, falling back to CPU:', error);
        // Fall through to CPU implementation
      }
    }

    // CPU implementation
    if (A.length === 0) return [];
    const cols = A[0].length;
    const y = new Array(cols).fill(0);
    for (let i = 0; i < A.length; i++) {
      const row = A[i];
      const xi = x[i];
      for (let j = 0; j < cols; j++) y[j] += row[j] * xi;
    }
    return y;
  }

  private averageVectors(vectors: number[][]): number[] {
    const H = this.hiddenSize;
    const y = new Array(H).fill(0);
    const n = vectors.length || 1;
    for (const v of vectors) for (let i = 0; i < H; i++) y[i] += v[i];
    for (let i = 0; i < H; i++) y[i] /= n;
    return y;
  }

  protected async forward(inputs: number[], train = false): Promise<ForwardCache> {
    const emb = this.averageVectors(inputs.map((i) => this.embedding[i]));

    const hPreCore = await this.matrixVectorMul(this.wHidden, emb);
    const preAct = hPreCore.map((v, i) => v + this.bHidden[i]);
    const z = this.normalizeHidden(preAct);
    let h = z.map((v) => this.activate(v));

    let dropMask: number[] | null = null;
    if (train && this.dropout > 0) {
      const scale = 1 / (1 - this.dropout);
      dropMask = new Array(h.length)
        .fill(0)
        .map(() => (this.nextRandom() > this.dropout ? scale : 0));
      h = h.map((v, i) => v * dropMask![i]);
    }

    const logits = (await this.matrixVectorMulTranspose(this.wOutput, h)).map(
      (v, i) => v + this.bOutput[i]
    );
    const probs = stableSoftmax(logits);
    return { h, logits, probs, avgEmb: emb, dropMask, preAct, z };
  }

  private applyMomentumMatrix(W: number[][], G: number[][], M: number[][]) {
    const lr = this.learningRate;
    const mom = this.momentum;
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        M[i][j] = mom * M[i][j] + lr * G[i][j];
        W[i][j] -= M[i][j];
      }
    }
  }

  private applyMomentumVector(b: number[], g: number[], m: number[]) {
    const lr = this.learningRate;
    const mom = this.momentum;
    for (let i = 0; i < b.length; i++) {
      m[i] = mom * m[i] + lr * g[i];
      b[i] -= m[i];
    }
  }

  private applyAdamMatrix(
    W: number[][],
    G: number[][],
    M: number[][],
    V: number[][],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        const g = G[i][j];
        M[i][j] = b1 * M[i][j] + (1 - b1) * g;
        V[i][j] = b2 * V[i][j] + (1 - b2) * (g * g);
        const mhat = M[i][j] / b1t;
        const vhat = V[i][j] / b2t;
        W[i][j] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
      }
    }
  }

  private applyAdamVector(
    b: number[],
    g: number[],
    M: number[],
    V: number[],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    for (let i = 0; i < b.length; i++) {
      const gi = g[i];
      M[i] = b1 * M[i] + (1 - b1) * gi;
      V[i] = b2 * V[i] + (1 - b2) * (gi * gi);
      const mhat = M[i] / b1t;
      const vhat = V[i] / b2t;
      b[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }

  private applyAdamRow(
    W: number[][],
    rowIdx: number,
    gRow: number[],
    M: number[][],
    V: number[][],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    const row = W[rowIdx];
    const mRow = M[rowIdx];
    const vRow = V[rowIdx];
    for (let i = 0; i < row.length; i++) {
      const g = gRow[i];
      mRow[i] = b1 * mRow[i] + (1 - b1) * g;
      vRow[i] = b2 * vRow[i] + (1 - b2) * (g * g);
      const mhat = mRow[i] / b1t;
      const vhat = vRow[i] / b2t;
      row[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }

  /**
   * Lion optimizer update for matrix parameters
   * Lion uses sign(β₁m + (1-β₁)g) for updates, then updates momentum with β₂
   */
  private applyLionMatrix(W: number[][], G: number[][], M: number[][]) {
    const lr = this.learningRate;
    const b1 = this.lionBeta1;
    const b2 = this.lionBeta2;
    const wd = this.lionWeightDecay;

    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        const g = G[i][j];
        // Compute update direction: sign(β₁m + (1-β₁)g)
        const interpolated = b1 * M[i][j] + (1 - b1) * g;
        const update = Math.sign(interpolated);
        // Apply update with weight decay: θ -= η × sign + η × λ × θ
        W[i][j] -= lr * update + lr * wd * W[i][j];
        // Update momentum: m = β₂m + (1-β₂)g
        M[i][j] = b2 * M[i][j] + (1 - b2) * g;
      }
    }
  }

  /**
   * Lion optimizer update for vector parameters (biases)
   */
  private applyLionVector(b: number[], g: number[], m: number[]) {
    const lr = this.learningRate;
    const b1 = this.lionBeta1;
    const b2 = this.lionBeta2;
    const wd = this.lionWeightDecay;

    for (let i = 0; i < b.length; i++) {
      const gi = g[i];
      // Compute update direction: sign(β₁m + (1-β₁)g)
      const interpolated = b1 * m[i] + (1 - b1) * gi;
      const update = Math.sign(interpolated);
      // Apply update with weight decay
      b[i] -= lr * update + lr * wd * b[i];
      // Update momentum
      m[i] = b2 * m[i] + (1 - b2) * gi;
    }
  }

  /**
   * Lion optimizer update for a single row (for embedding updates)
   */
  private applyLionRow(W: number[][], rowIdx: number, gRow: number[], M: number[][]) {
    const lr = this.learningRate;
    const b1 = this.lionBeta1;
    const b2 = this.lionBeta2;
    const wd = this.lionWeightDecay;

    const row = W[rowIdx];
    const mRow = M[rowIdx];
    for (let i = 0; i < row.length; i++) {
      const g = gRow[i];
      // Compute update direction: sign(β₁m + (1-β₁)g)
      const interpolated = b1 * mRow[i] + (1 - b1) * g;
      const update = Math.sign(interpolated);
      // Apply update with weight decay
      row[i] -= lr * update + lr * wd * row[i];
      // Update momentum
      mRow[i] = b2 * mRow[i] + (1 - b2) * g;
    }
  }

  /**
   * Exact diagonal of the Gauss-Newton matrix G = Jᵀ·A·J of the cross-entropy for one sample,
   * where A = diag(p) − ppᵀ is the Hessian of the loss w.r.t. the logits and J = ∂logits/∂θ:
   *   wOutput[i][j]: h_i²·p_j(1−p_j)          bOutput[j]: p_j(1−p_j)
   *   wHidden[a][c]: s_a²·e_c²·c_a            bHidden[a]: s_a²·c_a
   * with s_a = φ'(z_a)·dropMask_a, e = mean context embedding and
   * c_a = Σ_k p_k·w_ak² − (Σ_k p_k·w_ak)². (A hidden normalisation such as LayerNorm couples the
   * units; its Jacobian is ignored here, which makes the hidden-layer entries approximate.)
   */
  private gaussNewtonDiagonal(
    h: number[],
    probs: number[],
    avgEmb: number[],
    z: number[],
    dropMask: number[] | null
  ): { wOutput: number[][]; bOutput: number[]; wHidden: number[][]; bHidden: number[] } {
    const pq = probs.map((p) => p * (1 - p));
    const wOutput = h.map((hi) => pq.map((a) => hi * hi * a));
    const wHidden: number[][] = [];
    const bHidden: number[] = [];
    for (let a = 0; a < this.wOutput.length; a++) {
      const row = this.wOutput[a];
      let m1 = 0;
      let m2 = 0;
      for (let k = 0; k < row.length; k++) {
        m1 += probs[k] * row[k];
        m2 += probs[k] * row[k] * row[k];
      }
      const sa = this.activateDerivative(z[a]) * (dropMask ? dropMask[a] : 1);
      const base = sa * sa * Math.max(0, m2 - m1 * m1);
      bHidden.push(base);
      wHidden.push(avgEmb.map((e) => base * e * e));
    }
    return { wOutput, bOutput: pq, wHidden, bHidden };
  }

  private async backward(inputs: number[], target: number, cache: Omit<ForwardCache, 'logits'>) {
    const { h, probs, avgEmb, dropMask, preAct } = cache;
    const z = cache.z ?? preAct;
    const V = this.vocab.length;
    const H = this.hiddenSize;

    const dLogits = probs.map((p, i) => p - (i === target ? 1 : 0));

    const dWout = this.zerosMat(H, V);
    const dBout = this.zerosVec(V);
    for (let i = 0; i < H; i++) {
      const hi = h[i];
      for (let j = 0; j < V; j++) dWout[i][j] += hi * dLogits[j];
    }
    for (let j = 0; j < V; j++) dBout[j] += dLogits[j];

    const rawHiddenGrad = await this.matrixVectorMul(this.wOutput, dLogits);
    const dZ = rawHiddenGrad.map((value, i) => {
      let s = value * this.activateDerivative(z[i]);
      if (dropMask) s *= dropMask[i];
      return s;
    });
    const dHidden = this.normalizeHiddenBackward(dZ, preAct);

    clipVectorInPlace(dHidden, 5);

    const dWh = this.zerosMat(H, H);
    const dBh = this.zerosVec(H);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < H; j++) dWh[i][j] += dHidden[i] * avgEmb[j];
      dBh[i] += dHidden[i];
    }

    const clipRow = (row: number[], m = 5) => {
      const n = l2norm(row);
      if (n > m) {
        const s = m / n;
        for (let k = 0; k < row.length; k++) row[k] *= s;
      }
    };
    for (let i = 0; i < dWout.length; i++) clipRow(dWout[i]);
    for (let i = 0; i < dWh.length; i++) clipRow(dWh[i]);

    const wHiddenSnap = this.wHidden.map((r) => r.slice());

    const scale = 1 / Math.max(1, inputs.length);
    let dEmb = await this.matrixVectorMulTranspose(wHiddenSnap, dHidden);
    if (scale !== 1) {
      dEmb = dEmb.map((value) => value * scale);
    }

    // The context embedding is the mean of the input rows, so a token that occurs k times in the
    // window receives k·dEmb.
    const tokenCounts = new Map<number, number>();
    for (const idx of inputs) tokenCounts.set(idx, (tokenCounts.get(idx) ?? 0) + 1);

    // Optional global-norm clipping over every gradient of this step.
    const maxGlobalNorm = this.getGradientClipNorm();
    if (maxGlobalNorm > 0) {
      let ss = 0;
      for (const row of dWout) for (const v of row) ss += v * v;
      for (const row of dWh) for (const v of row) ss += v * v;
      for (const v of dBout) ss += v * v;
      for (const v of dBh) ss += v * v;
      let embSq = 0;
      for (const v of dEmb) embSq += v * v;
      for (const count of tokenCounts.values()) ss += count * count * embSq;
      const norm = Math.sqrt(ss);
      if (norm > maxGlobalNorm) {
        const f = maxGlobalNorm / norm;
        for (const row of dWout) for (let j = 0; j < row.length; j++) row[j] *= f;
        for (const row of dWh) for (let j = 0; j < row.length; j++) row[j] *= f;
        for (let j = 0; j < dBout.length; j++) dBout[j] *= f;
        for (let j = 0; j < dBh.length; j++) dBh[j] *= f;
        dEmb = dEmb.map((v) => v * f);
      }
    }
    let adamB1t: number | null = null;
    let adamB2t: number | null = null;

    if (this.optimizer === 'newton' || this.optimizer === 'bfgs') {
      const paramStruct = {
        wOutput: this.wOutput,
        bOutput: this.bOutput,
        wHidden: this.wHidden,
        bHidden: this.bHidden
      };
      const gradStruct = {
        wOutput: dWout,
        bOutput: dBout,
        wHidden: dWh,
        bHidden: dBh
      };
      const { vector: paramVec, meta } = flattenStructure(paramStruct);
      const gradVec = flattenGradients(gradStruct, meta);
      if (this.optimizer === 'newton') {
        // Damped Newton with the exact Gauss-Newton diagonal of the softmax cross-entropy,
        // smoothed over steps: θ ← θ − η·g / (EMA(diag GGN) + λ).
        const ggn = this.gaussNewtonDiagonal(h, probs, avgEmb, z, dropMask);
        const curvature = updateCurvatureEMA(
          this.secondOrderState,
          flattenGradients(ggn, meta),
          NEWTON_CURVATURE_DECAY
        );
        const step = dampedNewtonStep(gradVec, curvature, {
          ...this.secondOrderConfig,
          damping: Math.max(this.secondOrderConfig.damping ?? 0, NEWTON_MIN_DAMPING),
          learningRate: this.learningRate
        });
        applyUpdateVector(paramStruct, step, meta);
      } else {
        const step = quasiNewtonStep(paramVec, gradVec, this.secondOrderState, {
          ...this.secondOrderConfig,
          learningRate: this.learningRate
        });
        applyUpdateVector(paramStruct, step, meta);
      }
    } else if (this.optimizer === 'adam') {
      this.adamT += 1;
      adamB1t = 1 - Math.pow(this.adamBeta1, this.adamT);
      adamB2t = 1 - Math.pow(this.adamBeta2, this.adamT);
      this.applyAdamMatrix(this.wOutput, dWout, this.aWOutput.m, this.aWOutput.v, adamB1t, adamB2t);
      this.applyAdamVector(this.bOutput, dBout, this.aBOutput.m, this.aBOutput.v, adamB1t, adamB2t);
      this.applyAdamMatrix(this.wHidden, dWh, this.aWHidden.m, this.aWHidden.v, adamB1t, adamB2t);
      this.applyAdamVector(this.bHidden, dBh, this.aBHidden.m, this.aBHidden.v, adamB1t, adamB2t);
    } else if (this.optimizer === 'lion') {
      // Lion optimizer (v4.0): uses sign of momentum for updates
      this.applyLionMatrix(this.wOutput, dWout, this.mWOutput);
      this.applyLionVector(this.bOutput, dBout, this.mBOutput);
      this.applyLionMatrix(this.wHidden, dWh, this.mWHidden);
      this.applyLionVector(this.bHidden, dBh, this.mBHidden);
    } else if (this.optimizer === 'sophia' && this.sophiaOptimizer) {
      // Sophia optimizer (v4.2): second-order with diagonal Hessian
      this.sophiaOptimizer.updateMatrix(this.wOutput, dWout, 'wOutput');
      this.sophiaOptimizer.updateVector(this.bOutput, dBout, 'bOutput');
      this.sophiaOptimizer.updateMatrix(this.wHidden, dWh, 'wHidden');
      this.sophiaOptimizer.updateVector(this.bHidden, dBh, 'bHidden');
      this.sophiaOptimizer.step();
    } else {
      this.applyMomentumMatrix(this.wOutput, dWout, this.mWOutput);
      this.applyMomentumVector(this.bOutput, dBout, this.mBOutput);
      this.applyMomentumMatrix(this.wHidden, dWh, this.mWHidden);
      this.applyMomentumVector(this.bHidden, dBh, this.mBHidden);
    }

    // Aggregate per unique token and apply exactly one optimizer step to each row (applying the
    // update once per occurrence would compound Adam/Lion/momentum state).
    const rowGrad = (count: number) => (count === 1 ? dEmb : dEmb.map((v) => v * count));

    for (const [idx, count] of tokenCounts) {
      const g = rowGrad(count);
      if (this.optimizer === 'adam' && adamB1t !== null && adamB2t !== null) {
        this.applyAdamRow(
          this.embedding,
          idx,
          g,
          this.aEmbedding.m,
          this.aEmbedding.v,
          adamB1t,
          adamB2t
        );
      } else if (this.optimizer === 'momentum') {
        for (let i = 0; i < H; i++) {
          this.mEmbedding[idx][i] =
            this.momentum * this.mEmbedding[idx][i] + this.learningRate * g[i];
          this.embedding[idx][i] -= this.mEmbedding[idx][i];
        }
      } else if (this.optimizer === 'lion') {
        this.applyLionRow(this.embedding, idx, g, this.mEmbedding);
      } else if (this.optimizer === 'sophia' && this.sophiaOptimizer) {
        this.sophiaOptimizer.updateRow(this.embedding, idx, g, 'embedding');
      } else if (this.optimizer === 'newton' || this.optimizer === 'bfgs') {
        for (let i = 0; i < H; i++) {
          this.embedding[idx][i] -= this.learningRate * g[i];
        }
      }
    }
  }

  private tokenize(text: string): string[] {
    return ProNeuralLM.tokenizeText(text, this.tokenizerConfig);
  }

  private toIndex(tok: string) {
    return this.wordToIdx.get(tok) ?? this.wordToIdx.get(this.unk)!;
  }

  private createTrainingSequences(text: string): [number[], number][] {
    const bosArr = Array(this.contextSize).fill(this.bos);
    const toks = [...bosArr, ...this.tokenize(text), this.eos];
    const seqs: [number[], number][] = [];
    for (let i = this.contextSize; i < toks.length; i++) {
      const ctx = toks.slice(i - this.contextSize, i).map((t) => this.toIndex(t));
      const tgt = this.toIndex(toks[i]);
      seqs.push([ctx, tgt]);
    }
    return seqs;
  }

  /**
   * Build a loss mask array for training sequences.
   * Returns an array where mask[i] = 1 means compute loss for sequence i.
   *
   * @param text - Training text
   * @param lossMaskConfig - Loss mask configuration
   * @returns Array of 0s and 1s, same length as training sequences
   */
  private buildTrainingLossMask(text: string, lossMaskConfig: LossMaskConfig): number[] {
    const mode = lossMaskConfig.mode;
    const answerTag = lossMaskConfig.answerTag || 'A:';
    const toks = this.tokenize(text);
    const numSeqs = toks.length + 1; // +1 for EOS

    if (mode === 'none') {
      // All positions contribute to loss
      return new Array(numSeqs).fill(1);
    }

    // Handle custom RegExp mode
    if (mode === 'customRegExp') {
      const pattern = lossMaskConfig.customPattern;
      const position = lossMaskConfig.regExpPosition || 'after';

      if (!pattern || pattern.trim() === '') {
        // No pattern provided - return all zeros
        return new Array(numSeqs).fill(0);
      }

      // Find matches in the text
      const matches = findRegExpMatches(text, pattern);

      if (matches.length === 0) {
        // No matches found - return all zeros to prevent training on malformed examples
        // Exception: 'exclude' mode with no matches means include everything
        return new Array(numSeqs).fill(position === 'exclude' ? 1 : 0);
      }

      // Build character-level mask
      const charMask = buildCharMaskFromMatches(text, matches, position);

      // Map character positions to token positions using each token's real span in `text`
      // (accumulating token lengths would ignore the separators the tokenizer drops).
      const mask = new Array(numSeqs).fill(0);
      const spans = ProNeuralLM.tokenizeWithOffsets(text, this.tokenizerConfig);
      for (let t = 0; t < spans.length; t++) {
        const { start, end } = spans[t];
        const tokenLen = end - start;
        let includedChars = 0;
        for (let c = start; c < end && c < charMask.length; c++) {
          includedChars += charMask[c];
        }
        // Token is included if the majority of its characters are included
        mask[t] = tokenLen > 0 && includedChars > tokenLen / 2 ? 1 : 0;
      }

      // EOS token at the end - include if the last character was included
      mask[numSeqs - 1] = charMask.length > 0 && charMask[charMask.length - 1] ? 1 : 0;

      return mask;
    }

    // Find delimiter in the original text (for afterEquals and afterAnswerTag modes)
    const delimiter = mode === 'afterEquals' ? '=' : answerTag;
    const delimPos = text.indexOf(delimiter);

    if (delimPos < 0) {
      // No delimiter found - return all zeros (no loss computed)
      // This prevents training on malformed examples
      return new Array(numSeqs).fill(0);
    }

    // Count tokens before the delimiter (in the original text, not including BOS)
    const textBeforeDelim = text.slice(0, delimPos + delimiter.length);
    const tokensBeforeDelim = this.tokenize(textBeforeDelim).length;

    // Training sequences start at position contextSize in the token array
    // Position i in toks corresponds to sequence index (i - contextSize)
    // We add contextSize BOS tokens at the start
    // So the first token of the actual text is at position contextSize in toks
    // And the delimiter ends at position (contextSize + tokensBeforeDelim)
    // We want to compute loss AFTER the delimiter

    const mask = new Array(numSeqs).fill(0);

    // Sequences correspond to positions contextSize...(contextSize + numSeqs - 1) in toks
    // Sequence i corresponds to predicting token at position (contextSize + i) in toks
    // We want loss only for tokens after the delimiter
    // The first token after delimiter is at position tokensBeforeDelim in the text tokens (0-indexed)
    // Which is at position (contextSize + tokensBeforeDelim) in the full toks array
    // Which corresponds to sequence index tokensBeforeDelim

    for (let i = tokensBeforeDelim; i < numSeqs; i++) {
      mask[i] = 1;
    }

    return mask;
  }

  private shuffleInPlace<T>(arr: T[]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.nextRandom() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  async train(text: string, epochs = 10, lossMaskConfig?: LossMaskConfig) {
    const seqs = this.createTrainingSequences(text);
    if (seqs.length === 0) return { loss: 0, accuracy: 0, history: this.trainingHistory };

    // Build loss mask if config provided
    const lossMask = lossMaskConfig
      ? this.buildTrainingLossMask(text, lossMaskConfig)
      : new Array(seqs.length).fill(1);

    // Pair sequences with their mask values for shuffling
    const seqsWithMask: [number[], number, number][] = seqs.map((seq, i) => [
      seq[0],
      seq[1],
      lossMask[i] ?? 1
    ]);

    let totalLoss = 0;
    let correct = 0;
    let count = 0;
    let maskedCount = 0; // Track how many were actually trained

    this.onTrainStart(epochs);
    for (let e = 0; e < epochs; e++) {
      this.onEpochStart(e, epochs);
      this.shuffleInPlace(seqsWithMask);
      let epochLoss = 0;
      let epochCorrect = 0;
      let epochMaskedCount = 0;

      let processed = 0;
      let stopped = false;
      for (const [ctx, tgt, maskValue] of seqsWithMask) {
        if (await this.trainingShouldStop()) {
          stopped = true;
          break;
        }
        this.reportTrainingProgress(++processed, seqsWithMask.length, e);
        const cache = await this.forward(ctx, true);
        const pred = cache.probs.indexOf(Math.max(...cache.probs));

        // Only compute loss and backprop for unmasked positions
        if (maskValue === 1) {
          const loss = -Math.log(cache.probs[tgt] + 1e-8);
          epochLoss += loss;
          totalLoss += loss;
          epochMaskedCount++;
          maskedCount++;
          await this.backward(ctx, tgt, cache);
          this.onTrainingStep();
        }

        // Still track accuracy for all positions
        if (pred === tgt) {
          epochCorrect++;
          correct++;
        }
        count++;
      }
      if (stopped) break; // an interrupted epoch is not recorded

      // Average loss over masked positions only
      const avgLoss = epochMaskedCount > 0 ? epochLoss / epochMaskedCount : 0;
      const accuracy = epochCorrect / seqs.length;
      this.trainingHistory.push({ loss: avgLoss, accuracy, timestamp: Date.now() });
      this.onEpochEnd(e, epochs, { loss: avgLoss, accuracy });
    }

    const payload = {
      loss: maskedCount > 0 ? totalLoss / maskedCount : 0,
      accuracy: correct / Math.max(1, count),
      history: this.trainingHistory
    } as const;
    this.lastUpdatedAt = Date.now();
    return payload;
  }

  private sampleFromLogits(
    logits: number[],
    temperature = 1.0,
    topK = 0,
    topP = 0,
    generatedTokens: number[] = [],
    frequencyPenalty = 0,
    presencePenalty = 0,
    typicalTau = 0,
    mirostatTau = 0,
    mirostatEta = 0.1,
    mirostatState?: MirostatV2State,
    context: readonly number[] = [],
    controls: RepetitionControls = {}
  ): { index: number; state?: MirostatV2State } {
    // Loop controls apply to every decoding mode below (greedy = temperature → 0 included).
    logits = applyRepetitionControls(logits, context, controls);
    if (mirostatTau > 0) {
      const { index, state } = mirostatV2Sample(logits, {
        targetEntropy: mirostatTau,
        learningRate: mirostatEta,
        temperature,
        rng: () => this.nextRandom(),
        state: mirostatState
      });
      return { index, state };
    }

    // If typical sampling is enabled (tau > 0), use it instead
    if (typicalTau > 0 && typicalTau < 1) {
      const typicalIndex = typicalSample(logits, typicalTau, {
        temperature,
        rng: () => this.nextRandom()
      });
      return { index: typicalIndex };
    }

    // Otherwise use standard sampling with penalties
    return {
      index: drawToken(logits, {
        temperature,
        topK,
        topP,
        frequencyPenalty,
        presencePenalty,
        generatedTokens,
        rng: () => this.nextRandom()
      })
    };
  }

  async generate(
    seedText: string,
    maxLen = 25,
    temperature = 0.9,
    topK = 0,
    topP = 0,
    frequencyPenalty = 0,
    presencePenalty = 0,
    typicalTau = 0,
    mirostatTau = 0,
    mirostatEta = 0.1,
    repetition: RepetitionControls = {}
  ): Promise<string> {
    const { text } = await this.generateDetailed(
      seedText,
      maxLen,
      temperature,
      topK,
      topP,
      frequencyPenalty,
      presencePenalty,
      typicalTau,
      mirostatTau,
      mirostatEta,
      repetition
    );
    return text;
  }

  /**
   * `generate` that also reports how many tokens were sampled and why it stopped ('eos' when the
   * model produced the end-of-sequence token, 'length' when `maxLen` was reached).
   *
   * `repetition` (repetition penalty / no-repeat n-grams) looks at the prompt and the generated
   * tokens; the frequency and presence penalties count generated tokens only.
   */
  async generateDetailed(
    seedText: string,
    maxLen = 25,
    temperature = 0.9,
    topK = 0,
    topP = 0,
    frequencyPenalty = 0,
    presencePenalty = 0,
    typicalTau = 0,
    mirostatTau = 0,
    mirostatEta = 0.1,
    repetition: RepetitionControls = {}
  ): Promise<{ text: string; tokens: number; finishReason: 'eos' | 'length' }> {
    const seedToks = this.tokenize(seedText).map((t) => this.toIndex(t));
    const ctx: number[] = new Array(this.contextSize).fill(this.toIndex(this.bos));
    for (const t of seedToks) ctx.push(t);

    const out: string[] = [];
    const generatedTokenIds: number[] = [];

    let mirostatState: MirostatV2State | undefined;
    let finishReason: 'eos' | 'length' = 'length';
    while (out.length < maxLen) {
      const window = ctx.slice(-this.contextSize);
      const { logits } = await this.forward(window, false);
      const { index: idx, state } = this.sampleFromLogits(
        logits,
        temperature,
        topK,
        topP,
        generatedTokenIds,
        frequencyPenalty,
        presencePenalty,
        typicalTau,
        mirostatTau,
        mirostatEta,
        mirostatState,
        ctx.slice(this.contextSize),
        repetition
      );
      mirostatState = state;
      const tok = this.idxToWord.get(idx)!;
      if (tok === this.eos) {
        finishReason = 'eos';
        break;
      }
      out.push(tok);
      ctx.push(idx);
      generatedTokenIds.push(idx); // Track for repetition penalty
    }
    return { text: this.joinTokens(out), tokens: out.length, finishReason };
  }

  /**
   * Evaluate next-token prediction on `text` without changing any weights (no dropout).
   *
   * Uses the same windows as training (BOS-padded contexts, EOS as the final target) and reports
   * the mean cross-entropy in nats/token, perplexity = exp(loss), top-1 accuracy, the number of
   * scored positions and the fraction of targets that are out-of-vocabulary (mapped to <UNK>).
   * Use it on held-out text to measure generalisation rather than memorisation.
   */
  async evaluate(text: string): Promise<EvaluationResult> {
    if (this.tokenize(text).length === 0) {
      return { loss: NaN, perplexity: NaN, accuracy: 0, tokens: 0, oovRate: 0 };
    }
    return ProNeuralLM.evaluationResult([await this.evaluateSums(text)]);
  }

  /**
   * The sums behind `evaluate` for the prediction windows [start, end) of `text`, so that an
   * evaluation can be split into ranges (e.g. across worker threads) and combined with
   * `evaluationResult`.
   */
  async evaluateSums(text: string, start = 0, end = Infinity): Promise<EvaluationSums> {
    const seqs = this.createTrainingSequences(text);
    const unkIdx = this.toIndex(this.unk);
    const sums: EvaluationSums = { nll: 0, correct: 0, oov: 0, count: 0, total: seqs.length };
    const last = Math.min(end, seqs.length);
    for (let s = Math.max(0, start); s < last; s++) {
      const [ctx, tgt] = seqs[s];
      const { probs } = await this.forward(ctx, false);
      sums.nll -= Math.log(Math.max(probs[tgt], 1e-12));
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      if (best === tgt) sums.correct++;
      if (tgt === unkIdx) sums.oov++;
      sums.count++;
    }
    return sums;
  }

  /** Combine the sums of consecutive ranges (in order) into an evaluation result. */
  static evaluationResult(parts: EvaluationSums[]): EvaluationResult {
    let nll = 0;
    let correct = 0;
    let oov = 0;
    let count = 0;
    for (const p of parts) {
      nll += p.nll;
      correct += p.correct;
      oov += p.oov;
      count += p.count;
    }
    if (count === 0) return { loss: NaN, perplexity: NaN, accuracy: 0, tokens: 0, oovRate: 0 };
    const loss = nll / count;
    return {
      loss,
      perplexity: Math.exp(loss),
      accuracy: correct / count,
      tokens: count,
      oovRate: oov / count
    };
  }

  getVocabSize() {
    return this.vocab.length;
  }

  getTokenIndex(token: string): number {
    const idx = this.wordToIdx.get(token);
    return idx !== undefined ? idx : -1;
  }

  getParametersCount() {
    const V = this.vocab.length;
    const H = this.hiddenSize;
    return V * H + H * H + H * V + H + V;
  }

  getTrainingHistory() {
    return this.trainingHistory;
  }

  getVocabSignature() {
    return this.vocab.join('\u241F');
  }

  /**
   * Returns a copy of the embedding matrix.
   * Each row corresponds to a token in the vocabulary.
   * @returns 2D array of shape [vocab_size, hidden_size]
   */
  getEmbeddings(): number[][] {
    return this.embedding.map((row) => [...row]);
  }

  /**
   * Returns a copy of the vocabulary array.
   * @returns Array of token strings
   */
  getVocab(): string[] {
    return [...this.vocab];
  }

  /**
   * Returns the hidden layer size.
   * @returns Hidden layer dimension
   */
  /** Restart the sampling / dropout random stream from `seed` (reproducible generation). */
  reseed(seed: number): void {
    this.rngSeed = seed >>> 0;
    this.rng = makeRng(this.rngSeed);
    this.rngState = this.rng.getState();
  }

  /** Position of the random stream (dropout, shuffling, sampling), e.g. for checkpoints. */
  getRandomState(): RandomState {
    return { seed: this.rngSeed, state: this.rng.getState() };
  }

  /** Continue the random stream from a position returned by getRandomState. */
  setRandomState(position: RandomState): void {
    this.rngSeed = position.seed >>> 0;
    this.rng = makeRng(this.rngSeed, position.state >>> 0);
    this.rngState = this.rng.getState();
  }

  getHiddenSize(): number {
    return this.hiddenSize;
  }

  /**
   * Returns the context window size.
   * @returns Context size (number of tokens in input window)
   */
  getContextSize(): number {
    return this.contextSize;
  }

  /**
   * Get all weight matrices and biases.
   * Used by Cerebro injection system for snapshots and rollback.
   */
  getWeights(): {
    embedding: number[][];
    wHidden: number[][];
    wOutput: number[][];
    bHidden: number[];
    bOutput: number[];
  } {
    return {
      embedding: this.embedding.map((row) => [...row]),
      wHidden: this.wHidden.map((row) => [...row]),
      wOutput: this.wOutput.map((row) => [...row]),
      bHidden: [...this.bHidden],
      bOutput: [...this.bOutput]
    };
  }

  /**
   * Set all weight matrices and biases.
   * Used by Cerebro injection system for rollback.
   */
  setWeights(weights: {
    embedding: number[][];
    wHidden: number[][];
    wOutput: number[][];
    bHidden: number[];
    bOutput: number[];
  }): void {
    this.embedding = weights.embedding.map((row) => [...row]);
    this.wHidden = weights.wHidden.map((row) => [...row]);
    this.wOutput = weights.wOutput.map((row) => [...row]);
    this.bHidden = [...weights.bHidden];
    this.bOutput = [...weights.bOutput];

    // Update hidden size if it changed
    this.hiddenSize = this.wHidden.length;

    // Reinitialize optimizer state for new dimensions
    this.reinitializeOptimizerState();
  }

  /**
   * Expand the model width by k (Cerebro neuron injection).
   *
   * In ProNeuralLM the embedding dimension equals the hidden size, so both grow together:
   *   embedding  V×H → V×(H+k)   new dimensions drawn like the initial embeddings
   *   wHidden    H×H → (H+k)×(H+k)  existing units get zero weights on the new embedding
   *                                 dimensions; new units read every dimension
   *   wOutput    H×V → (H+k)×V      new units start with zero outgoing weights
   * The expansion is function-preserving: predictions are identical right after injection, and
   * the new units start learning on the next step (their outgoing weights receive gradient).
   * Optimizer state is reset to the new shapes.
   */
  expandHiddenLayer(k: number, useHeInit = true): void {
    if (k <= 0) return;

    const H = this.hiddenSize;
    const newH = H + k;
    const V = this.vocab.length;
    const inScale = useHeInit ? Math.sqrt(2.0 / newH) : Math.sqrt(1.0 / newH);

    for (const row of this.embedding) {
      for (let j = 0; j < k; j++) row.push(this.randn(0.05));
    }
    for (const row of this.wHidden) {
      for (let j = 0; j < k; j++) row.push(0);
    }
    for (let i = 0; i < k; i++) {
      this.wHidden.push(this.randomArray(newH, inScale));
      this.bHidden.push(0);
      this.wOutput.push(new Array(V).fill(0));
    }

    this.hiddenSize = newH;
    this.reinitializeOptimizerState();
  }

  /**
   * Reset every optimizer buffer to zeros with the current parameter shapes
   * (after a structural change such as neuron injection or setWeights).
   */
  protected reinitializeOptimizerState(): void {
    const zerosLike = (m: number[][]) => m.map((row) => row.map(() => 0));
    const V = this.vocab.length;
    const H = this.hiddenSize;

    this.mEmbedding = zerosLike(this.embedding);
    this.mWHidden = zerosLike(this.wHidden);
    this.mWOutput = zerosLike(this.wOutput);
    this.mBHidden = new Array(H).fill(0);
    this.mBOutput = new Array(V).fill(0);

    this.aEmbedding = { m: zerosLike(this.embedding), v: zerosLike(this.embedding) };
    this.aWHidden = { m: zerosLike(this.wHidden), v: zerosLike(this.wHidden) };
    this.aWOutput = { m: zerosLike(this.wOutput), v: zerosLike(this.wOutput) };
    this.aBHidden = { m: new Array(H).fill(0), v: new Array(H).fill(0) };
    this.aBOutput = { m: new Array(V).fill(0), v: new Array(V).fill(0) };
    this.adamT = 0;
    this.resetSecondOrderState();
    this.sophiaOptimizer?.reset();
  }

  /**
   * Get logits (raw model outputs before softmax) for a given context.
   * Used for knowledge distillation to extract soft targets from teacher model.
   *
   * @param context - Array of token indices
   * @returns Promise<number[]> - Raw logit values for each vocabulary token
   */
  /**
   * Hidden representations Z (the vector fed to the output projection; no dropout) and the
   * next-token targets Y for up to `maxSamples` evenly spaced training windows of `text`.
   * Used for information-plane monitoring.
   */
  async collectRepresentations(
    text: string,
    maxSamples = 256
  ): Promise<{ representations: number[][]; labels: number[]; probabilities: number[][] }> {
    const seqs = this.createTrainingSequences(text);
    const step = Math.max(1, Math.ceil(seqs.length / Math.max(1, maxSamples)));
    const representations: number[][] = [];
    const labels: number[] = [];
    const probabilities: number[][] = [];
    for (let i = 0; i < seqs.length; i += step) {
      const [ctx, tgt] = seqs[i];
      const { h, probs } = await this.forward(ctx, false);
      representations.push(h.slice());
      labels.push(tgt);
      probabilities.push(probs.slice());
    }
    return { representations, labels, probabilities };
  }

  /** Number of training windows (next-token targets) that `text` produces. */
  countTrainingWindows(text: string): number {
    return this.createTrainingSequences(text).length;
  }

  /**
   * Token indices of the context window that would be used to predict the token following
   * `text` (the last `contextSize` tokens, left-padded with BOS).
   */
  getContextWindow(text: string): number[] {
    const ids = this.tokenize(text).map((t) => this.toIndex(t));
    const bos = this.toIndex(this.bos);
    const padded = [...new Array(this.contextSize).fill(bos), ...ids];
    return padded.slice(-this.contextSize);
  }

  /** Copies of the embedding rows of `indices`. */
  getEmbeddingRows(indices: number[]): number[][] {
    return indices.map((i) => this.embedding[i].slice());
  }

  /**
   * log p(target | context) computed from explicit context embedding rows (no dropout, no
   * parameter updates), together with its exact gradient with respect to those rows.
   * Used by the explainability tools (integrated gradients, Shapley values).
   */
  explainLogProb(
    contextEmbeddings: number[][],
    target: number
  ): { logProb: number; gradient: number[][] } {
    const H = this.hiddenSize;
    const n = Math.max(1, contextEmbeddings.length);
    const emb = new Array(H).fill(0);
    for (const row of contextEmbeddings) for (let i = 0; i < H; i++) emb[i] += row[i] / n;

    const preAct = this.wHidden.map(
      (row, a) => row.reduce((sum, w, c) => sum + w * emb[c], 0) + this.bHidden[a]
    );
    const z = this.normalizeHidden(preAct);
    const h = z.map((v) => this.activate(v));
    const V = this.vocab.length;
    const logits = new Array(V);
    for (let j = 0; j < V; j++) {
      let s = this.bOutput[j];
      for (let a = 0; a < H; a++) s += this.wOutput[a][j] * h[a];
      logits[j] = s;
    }
    const probs = stableSoftmax(logits);
    const logProb = Math.log(Math.max(probs[target], 1e-300));

    // ∂ log p_target / ∂ logits = onehot − p
    const dLogits = probs.map((p, j) => (j === target ? 1 : 0) - p);
    const dZ = this.wOutput.map(
      (row, a) => row.reduce((sum, w, j) => sum + w * dLogits[j], 0) * this.activateDerivative(z[a])
    );
    const dPre = this.normalizeHiddenBackward(dZ, preAct, false);
    const dEmb = new Array(H).fill(0);
    for (let a = 0; a < H; a++) {
      const row = this.wHidden[a];
      for (let c = 0; c < H; c++) dEmb[c] += row[c] * dPre[a];
    }
    return { logProb, gradient: contextEmbeddings.map(() => dEmb.map((v) => v / n)) };
  }

  async getLogitsForContext(context: number[]): Promise<number[]> {
    const { logits } = await this.forward(context, false);
    return logits;
  }

  toJSON() {
    this.rngState = this.rng.getState();
    return {
      version: MODEL_VERSION,
      vocab: this.vocab,
      hiddenSize: this.hiddenSize,
      learningRate: this.learningRate,
      contextSize: this.contextSize,
      optimizer: this.optimizer,
      momentum: this.momentum,
      dropout: this.dropout,
      rngSeed: this.rngSeed,
      rngState: this.rngState,
      embedding: this.embedding,
      wHidden: this.wHidden,
      wOutput: this.wOutput,
      bHidden: this.bHidden,
      bOutput: this.bOutput,
      adamT: this.adamT,
      mEmbedding: this.mEmbedding,
      mWHidden: this.mWHidden,
      mWOutput: this.mWOutput,
      mBHidden: this.mBHidden,
      mBOutput: this.mBOutput,
      aEmbedding: this.aEmbedding,
      aWHidden: this.aWHidden,
      aWOutput: this.aWOutput,
      aBHidden: this.aBHidden,
      aBOutput: this.aBOutput,
      wordToIdx: Array.from(this.wordToIdx.entries()),
      idxToWord: Array.from(this.idxToWord.entries()),
      trainingHistory: this.trainingHistory,
      tokenizerConfig: this.tokenizerConfig,
      lastUpdatedAt: this.lastUpdatedAt
    } as const;
  }

  /**
   * Save to localStorage. Models whose plain JSON exceeds the storage quota (e.g. a default
   * transformer, ~23 MB) are stored as a compact package instead (float32, then float16).
   * Returns the encoding used, or null when the model could not be stored.
   */
  saveToLocalStorage(key: string): 'json' | 'float32' | 'float16' | null {
    try {
      const encoding = storeModelCompactly(
        (value) => localStorage.setItem(key, value),
        this.toJSON() as unknown as Record<string, unknown>
      );
      if (!encoding) console.warn('Failed to save model: it does not fit in localStorage');
      return encoding;
    } catch (e) {
      console.warn('Failed to save model', e);
      return null;
    }
  }

  /**
   * Rebuild a model from the object produced by toJSON().
   */
  static fromJSON(d: any): ProNeuralLM {
    const m = new ProNeuralLM(
      d.vocab,
      d.hiddenSize,
      d.learningRate ?? 0.08,
      d.contextSize ?? 3,
      (d.optimizer as Optimizer) ?? 'momentum',
      d.momentum ?? 0.9,
      d.dropout ?? 0,
      typeof d.rngSeed === 'number' ? d.rngSeed : undefined,
      ProNeuralLM.normalizeTokenizerConfig(d.tokenizerConfig)
    );
    ProNeuralLM.restoreSerializedState(m, d);
    return m;
  }

  /**
   * Copy weights, optimizer state, vocabulary maps, history and RNG state from serialized data
   * into a freshly constructed model (shared by the loaders of every architecture).
   */
  protected static restoreSerializedState(m: ProNeuralLM, d: any): void {
    m.embedding = d.embedding;
    m.wHidden = d.wHidden;
    m.wOutput = d.wOutput;
    m.bHidden = d.bHidden;
    m.bOutput = d.bOutput;
    if (typeof d.adamT === 'number') m.adamT = d.adamT;
    if (d.mEmbedding) m.mEmbedding = d.mEmbedding;
    if (d.mWHidden) m.mWHidden = d.mWHidden;
    if (d.mWOutput) m.mWOutput = d.mWOutput;
    if (d.mBHidden) m.mBHidden = d.mBHidden;
    if (d.mBOutput) m.mBOutput = d.mBOutput;
    if (d.aEmbedding) m.aEmbedding = { m: d.aEmbedding.m, v: d.aEmbedding.v };
    if (d.aWHidden) m.aWHidden = { m: d.aWHidden.m, v: d.aWHidden.v };
    if (d.aWOutput) m.aWOutput = { m: d.aWOutput.m, v: d.aWOutput.v };
    if (d.aBHidden) m.aBHidden = { m: d.aBHidden.m, v: d.aBHidden.v };
    if (d.aBOutput) m.aBOutput = { m: d.aBOutput.m, v: d.aBOutput.v };
    if (Array.isArray(d.wordToIdx)) m.wordToIdx = new Map(d.wordToIdx);
    if (Array.isArray(d.idxToWord)) m.idxToWord = new Map(d.idxToWord);
    m.trainingHistory = d.trainingHistory || [];
    if (typeof d.lastUpdatedAt === 'number') {
      m.lastUpdatedAt = d.lastUpdatedAt;
    } else if (m.trainingHistory.length > 0) {
      m.lastUpdatedAt = m.trainingHistory[m.trainingHistory.length - 1]?.timestamp ?? null;
    }
    if (typeof d.rngSeed === 'number') {
      m.rngSeed = d.rngSeed >>> 0;
    }
    const state = typeof d.rngState === 'number' ? (d.rngState as number) >>> 0 : undefined;
    m.rng = makeRng(m.rngSeed, state);
    m.rngState = m.rng.getState();
    m.resetSecondOrderState();
  }

  static loadFromLocalStorage(key: string): ProNeuralLM | null {
    try {
      const data = parseStoredModel(localStorage.getItem(key));
      if (!data) return null;
      return ProNeuralLM.fromJSON(data);
    } catch (e) {
      console.warn('Failed to load model', e);
      return null;
    }
  }
}
