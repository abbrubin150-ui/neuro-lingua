/**
 * AdvancedNeuralLM - Enhanced Neural Language Model
 *
 * Extends ProNeuralLM with advanced mathematical capabilities:
 * - He/Xavier initialization for better convergence
 * - Advanced activation functions (LeakyReLU, ELU, GELU)
 * - Learning rate scheduling (cosine annealing, exponential decay)
 * - L2 regularization (weight decay)
 * - Layer normalization for training stability
 * - Beam search for generation
 * - Improved numerical stability
 *
 * All options act through ProNeuralLM's protected hooks (activate, normalizeHidden,
 * getGradientClipNorm, onEpochStart, ...), so they apply to the standard train() path.
 * Weight tensors owned by the base class are still reached via (this as any).
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import {
  ProNeuralLM,
  type EpochMetrics,
  type Optimizer,
  type TokenizerConfig
} from './ProNeuralLM';
import { ParamOptimizer, type ParamOptimizerKind } from '../training/ParamOptimizer';
import {
  heInit,
  xavierInit,
  leakyRelu,
  leakyReluDerivative,
  elu,
  eluDerivative,
  gelu,
  geluDerivative,
  cosineAnnealingLR,
  exponentialDecayLR,
  warmupCosineAnnealingLR,
  stableSoftmax,
  beamSearch,
  nucleusSampling
} from './MathUtils';
import { greedySample } from '../generation/sampling';
import type { LossMaskConfig } from '../types/project';

export type ActivationFunction = 'relu' | 'leaky_relu' | 'elu' | 'gelu';
export type LRSchedule = 'constant' | 'cosine' | 'exponential' | 'warmup_cosine';
export type InitializationScheme = 'default' | 'xavier' | 'he';

export type AdvancedConfig = {
  // Activation function
  activation?: ActivationFunction;
  leakyReluAlpha?: number;
  eluAlpha?: number;

  // Weight initialization
  initialization?: InitializationScheme;

  // Learning rate scheduling
  lrSchedule?: LRSchedule;
  lrMin?: number;
  lrDecayRate?: number;
  warmupEpochs?: number;

  // Regularization
  weightDecay?: number; // L2 regularization coefficient
  gradientClipNorm?: number;

  // Layer normalization
  useLayerNorm?: boolean;

  // Advanced generation
  beamWidth?: number;
};

const DEFAULT_ADVANCED_CONFIG: Required<AdvancedConfig> = {
  activation: 'relu',
  leakyReluAlpha: 0.01,
  eluAlpha: 1.0,
  initialization: 'he',
  lrSchedule: 'cosine',
  lrMin: 1e-6,
  lrDecayRate: 0.95,
  warmupEpochs: 0,
  weightDecay: 1e-4,
  gradientClipNorm: 5.0,
  useLayerNorm: false,
  beamWidth: 4
};

/** Numerical-stability constant for the hidden-layer LayerNorm. */
const LAYER_NORM_EPSILON = 1e-5;

type EpochCallback = (
  epoch: number,
  metrics: { loss: number; accuracy: number; lr: number }
) => void;

/**
 * AdvancedNeuralLM - Extended neural language model with state-of-the-art techniques.
 *
 * Every option is wired into the regular ProNeuralLM training/inference path through the base
 * class hooks, so `train()` (used by the UI, one epoch per call) and `trainAdvanced()` behave the
 * same way:
 * - activation: φ in the forward pass and φ' in backprop
 * - useLayerNorm: LayerNorm on the hidden pre-activation with exact backward and learned γ/β
 * - lrSchedule: per-epoch learning rate over a horizon declared with setTrainingHorizon()
 * - weightDecay: decoupled L2 shrinkage of wHidden/wOutput after every step
 * - gradientClipNorm: global L2 clipping of each step's gradients
 */
export class AdvancedNeuralLM extends ProNeuralLM {
  private advancedConfig: Required<AdvancedConfig>;
  private baseLearningRate: number;

  // Learning-rate schedule state. The schedule spans `scheduleHorizon` epochs and `scheduleEpoch`
  // counts completed epochs. When a horizon is declared with setTrainingHorizon() the counter
  // persists across train() calls (the UI trains one epoch per call); otherwise every train()
  // call is treated as a complete schedule.
  private scheduleEpoch = 0;
  private scheduleHorizon = 0;
  private horizonExplicit = false;
  private epochCallback: EpochCallback | null = null;

  // Layer normalization parameters (if enabled)
  private layerNormGamma: number[] = [];
  private layerNormBeta: number[] = [];
  private layerNormOptimizer: ParamOptimizer | null = null;

  constructor(
    vocab: string[],
    hiddenSize = 64,
    learningRate = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'momentum',
    momentum = 0.9,
    dropout = 0.0,
    seed = 1337,
    tokenizerConfig?: TokenizerConfig,
    advancedConfig?: AdvancedConfig
  ) {
    super(
      vocab,
      hiddenSize,
      learningRate,
      contextSize,
      optimizer,
      momentum,
      dropout,
      seed,
      tokenizerConfig
    );

    this.advancedConfig = { ...DEFAULT_ADVANCED_CONFIG, ...advancedConfig };
    this.baseLearningRate = learningRate;

    // Reinitialize weights if using advanced initialization
    if (
      this.advancedConfig.initialization === 'he' ||
      this.advancedConfig.initialization === 'xavier'
    ) {
      this.reinitializeWeights();
    }

    // Initialize layer normalization parameters
    if (this.advancedConfig.useLayerNorm) {
      this.initializeLayerNorm();
    }
  }

  /**
   * Reinitialize weights using He or Xavier initialization
   */
  private reinitializeWeights(): void {
    const V = this.getVocabSize();
    const H = (this as any).hiddenSize;
    const rng = (this as any).rng;

    const embedding = (this as any).embedding;
    const wHidden = (this as any).wHidden;
    const wOutput = (this as any).wOutput;

    if (this.advancedConfig.initialization === 'he') {
      // He initialization
      for (let i = 0; i < V; i++) {
        for (let j = 0; j < H; j++) {
          embedding[i][j] = heInit(H, () => rng.next());
        }
      }
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          wHidden[i][j] = heInit(H, () => rng.next());
        }
      }
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < V; j++) {
          wOutput[i][j] = heInit(H, () => rng.next());
        }
      }
    } else {
      // Xavier initialization
      for (let i = 0; i < V; i++) {
        for (let j = 0; j < H; j++) {
          embedding[i][j] = xavierInit(H, H, () => rng.next());
        }
      }
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          wHidden[i][j] = xavierInit(H, H, () => rng.next());
        }
      }
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < V; j++) {
          wOutput[i][j] = xavierInit(H, V, () => rng.next());
        }
      }
    }
  }

  /**
   * Initialize layer normalization parameters (γ = 1, β = 0).
   */
  private initializeLayerNorm(): void {
    const H = this.getHiddenSize();
    this.layerNormGamma = new Array(H).fill(1.0);
    this.layerNormBeta = new Array(H).fill(0.0);
    this.layerNormOptimizer?.reset();
  }

  // ---------------------------------------------------------------------------
  // Hook: activation function
  // ---------------------------------------------------------------------------

  protected activate(x: number): number {
    switch (this.advancedConfig.activation) {
      case 'leaky_relu':
        return leakyRelu(x, this.advancedConfig.leakyReluAlpha);
      case 'elu':
        return elu(x, this.advancedConfig.eluAlpha);
      case 'gelu':
        return gelu(x);
      case 'relu':
      default:
        return x > 0 ? x : 0;
    }
  }

  protected activateDerivative(x: number): number {
    switch (this.advancedConfig.activation) {
      case 'leaky_relu':
        return leakyReluDerivative(x, this.advancedConfig.leakyReluAlpha);
      case 'elu':
        return eluDerivative(x, this.advancedConfig.eluAlpha);
      case 'gelu':
        return geluDerivative(x);
      case 'relu':
      default:
        return x > 0 ? 1 : 0;
    }
  }

  // ---------------------------------------------------------------------------
  // Hook: LayerNorm on the hidden pre-activation  z = γ ⊙ (a − μ)/σ + β
  // ---------------------------------------------------------------------------

  private isLayerNormActive(): boolean {
    return this.advancedConfig.useLayerNorm && this.layerNormGamma.length === this.getHiddenSize();
  }

  private layerNormStats(x: number[]): { xhat: number[]; invStd: number } {
    const n = x.length;
    let mean = 0;
    for (const v of x) mean += v;
    mean /= n;
    let variance = 0;
    for (const v of x) variance += (v - mean) * (v - mean);
    variance /= n;
    const invStd = 1 / Math.sqrt(variance + LAYER_NORM_EPSILON);
    return { xhat: x.map((v) => (v - mean) * invStd), invStd };
  }

  protected normalizeHidden(preAct: number[]): number[] {
    if (!this.isLayerNormActive()) return preAct;
    const { xhat } = this.layerNormStats(preAct);
    return xhat.map((v, i) => v * this.layerNormGamma[i] + this.layerNormBeta[i]);
  }

  /**
   * Exact LayerNorm backward. With x̂ = (a − μ)/σ and g = ∂L/∂x̂ = ∂L/∂z ⊙ γ:
   *   ∂L/∂a = (g − mean(g) − x̂ · mean(g ⊙ x̂)) / σ,   ∂L/∂γ = ∂L/∂z ⊙ x̂,   ∂L/∂β = ∂L/∂z.
   * γ and β are updated here with the same rule as the rest of the model (Adam / Lion /
   * momentum-SGD; second-order optimizers fall back to momentum-SGD for these two vectors).
   */
  protected normalizeHiddenBackward(dZ: number[], preAct: number[], updateParams = true): number[] {
    if (!this.isLayerNormActive()) return dZ;
    const n = preAct.length;
    const { xhat, invStd } = this.layerNormStats(preAct);
    const g = dZ.map((d, i) => d * this.layerNormGamma[i]);
    let meanG = 0;
    let meanGX = 0;
    for (let i = 0; i < n; i++) {
      meanG += g[i];
      meanGX += g[i] * xhat[i];
    }
    meanG /= n;
    meanGX /= n;
    const dPreAct = g.map((gi, i) => invStd * (gi - meanG - xhat[i] * meanGX));
    if (!updateParams) return dPreAct;

    const dGamma = dZ.map((d, i) => d * xhat[i]);
    const dBeta = dZ.slice();
    const opt = this.getLayerNormOptimizer();
    const lr = this.getLearningRate();
    opt.step();
    opt.update('gamma', this.layerNormGamma, dGamma, lr);
    opt.update('beta', this.layerNormBeta, dBeta, lr);
    return dPreAct;
  }

  private getLayerNormOptimizer(): ParamOptimizer {
    const optimizer = (this as any).optimizer as Optimizer;
    const momentum = (this as any).momentum as number;
    const kind: ParamOptimizerKind =
      optimizer === 'adam' ? 'adam' : optimizer === 'lion' ? 'lion' : 'momentum';
    if (!this.layerNormOptimizer) {
      this.layerNormOptimizer = new ParamOptimizer(kind, momentum);
    } else {
      this.layerNormOptimizer.setKind(kind, momentum);
    }
    return this.layerNormOptimizer;
  }

  // ---------------------------------------------------------------------------
  // Hook: global gradient clipping
  // ---------------------------------------------------------------------------

  protected getGradientClipNorm(): number {
    return this.advancedConfig.gradientClipNorm > 0 ? this.advancedConfig.gradientClipNorm : 0;
  }

  // ---------------------------------------------------------------------------
  // Hooks: learning-rate schedule and weight decay
  // ---------------------------------------------------------------------------

  /**
   * Declare the length of the upcoming training run so that the learning-rate schedule spans it
   * even when train() is called one epoch at a time (as the UI does).
   */
  setTrainingHorizon(totalEpochs: number, startEpoch = 0): void {
    this.scheduleHorizon = Math.max(1, Math.floor(totalEpochs));
    this.scheduleEpoch = Math.max(0, Math.floor(startEpoch));
    this.horizonExplicit = true;
  }

  protected onTrainStart(epochs: number): void {
    // Without a declared horizon (or once it is exhausted) each train() call is a full schedule.
    if (!this.horizonExplicit || this.scheduleEpoch >= this.scheduleHorizon) {
      this.horizonExplicit = false;
      this.scheduleEpoch = 0;
      this.scheduleHorizon = Math.max(1, epochs);
    }
  }

  protected onEpochStart(): void {
    this.setLearningRateInternal(this.getCurrentLearningRate());
  }

  protected onTrainingStep(): void {
    if (this.advancedConfig.weightDecay > 0) {
      const lr = this.getLearningRate();
      this.applyL2Regularization((this as any).wHidden, lr);
      this.applyL2Regularization((this as any).wOutput, lr);
    }
  }

  protected onEpochEnd(_epoch: number, _epochs: number, metrics: EpochMetrics): void {
    this.epochCallback?.(this.scheduleEpoch, { ...metrics, lr: this.getLearningRate() });
    this.scheduleEpoch++;
  }

  /**
   * Learning rate for the current schedule position.
   */
  private getCurrentLearningRate(): number {
    const epoch = this.scheduleEpoch;
    const total = this.scheduleHorizon;
    switch (this.advancedConfig.lrSchedule) {
      case 'cosine':
        return cosineAnnealingLR(epoch, total, this.baseLearningRate, this.advancedConfig.lrMin);
      case 'exponential':
        return exponentialDecayLR(epoch, this.baseLearningRate, this.advancedConfig.lrDecayRate);
      case 'warmup_cosine':
        return warmupCosineAnnealingLR(
          epoch,
          total,
          this.baseLearningRate,
          this.advancedConfig.warmupEpochs
        );
      case 'constant':
      default:
        return this.baseLearningRate;
    }
  }

  /**
   * Decoupled weight decay: w ← w − η·λ·w
   */
  private applyL2Regularization(weights: number[][], learningRate: number): void {
    const lambda = this.advancedConfig.weightDecay;
    if (lambda <= 0) return;
    const factor = 1 - learningRate * lambda;
    for (let i = 0; i < weights.length; i++) {
      const row = weights[i];
      for (let j = 0; j < row.length; j++) row[j] *= factor;
    }
  }

  /**
   * Train for `epochs` epochs over one complete learning-rate schedule, reporting per-epoch
   * metrics through `callbacks.onEpochEnd`. Equivalent to setTrainingHorizon(epochs) + train().
   */
  async trainAdvanced(
    text: string,
    epochs = 10,
    callbacks?: {
      onEpochEnd?: EpochCallback;
    },
    lossMaskConfig?: LossMaskConfig
  ): Promise<{ loss: number; accuracy: number; history: any[] }> {
    this.setTrainingHorizon(epochs, 0);
    this.epochCallback = callbacks?.onEpochEnd ?? null;
    try {
      const result = await this.train(text, epochs, lossMaskConfig);
      return { loss: result.loss, accuracy: result.accuracy, history: [...result.history] };
    } finally {
      this.epochCallback = null;
      this.horizonExplicit = false;
    }
  }

  /**
   * Generate text using beam search
   */
  async generateBeamSearch(
    seedText: string,
    maxLen = 25,
    beamWidth?: number,
    temperature = 0.9
  ): Promise<{ text: string; score: number; tokens: number[] }> {
    const width = beamWidth || this.advancedConfig.beamWidth;

    // Tokenize seed text
    const seedToks = (this as any).tokenize(seedText).map((t: string) => (this as any).toIndex(t));

    // Create initial context
    const bosIdx = (this as any).toIndex((this as any).bos);
    const eosIdx = (this as any).toIndex((this as any).eos);
    const contextSize = (this as any).contextSize;

    const initialContext = new Array(contextSize).fill(bosIdx);
    for (const tok of seedToks) {
      initialContext.push(tok);
    }

    // Forward function for beam search
    const forwardFn = async (context: number[]) => {
      const window = context.slice(-contextSize);
      const { logits } = await (this as any).forward(window, false);
      return stableSoftmax(logits, temperature);
    };

    // Perform beam search
    const result = await beamSearch(initialContext, width, maxLen, forwardFn, eosIdx);

    // Convert tokens to text
    const outputTokens = result.tokens.slice(initialContext.length);
    const idxToWord = (this as any).idxToWord;
    const text = this.joinTokens(
      outputTokens
        .filter((idx: number) => idx !== eosIdx)
        .map((idx: number) => idxToWord.get(idx) as string)
    );

    return {
      text,
      score: result.score,
      tokens: outputTokens
    };
  }

  /**
   * Generate text using nucleus sampling (improved version)
   */
  async generateNucleus(
    seedText: string,
    maxLen = 25,
    temperature = 0.9,
    topP = 0.9
  ): Promise<string> {
    const seedToks = (this as any).tokenize(seedText).map((t: string) => (this as any).toIndex(t));
    const bosIdx = (this as any).toIndex((this as any).bos);
    const contextSize = (this as any).contextSize;

    const ctx: number[] = new Array(contextSize).fill(bosIdx);
    for (const t of seedToks) ctx.push(t);

    const out: string[] = [];
    const rng = (this as any).rng;

    while (out.length < maxLen) {
      const window = ctx.slice(-contextSize);
      const { logits } = await (this as any).forward(window, false);
      const probs = stableSoftmax(logits, temperature);

      const idx = nucleusSampling(probs, topP, () => rng.next());

      const idxToWord = (this as any).idxToWord;
      const tok = idxToWord.get(idx)!;
      if (tok === (this as any).eos) break;

      out.push(tok);
      ctx.push(idx);
    }

    return this.joinTokens(out);
  }

  /**
   * Generate text using greedy decoding (always select most likely token)
   */
  async generateGreedy(seedText: string, maxLen = 25): Promise<string> {
    const seedToks = (this as any).tokenize(seedText).map((t: string) => (this as any).toIndex(t));
    const bosIdx = (this as any).toIndex((this as any).bos);
    const contextSize = (this as any).contextSize;

    const ctx: number[] = new Array(contextSize).fill(bosIdx);
    for (const t of seedToks) ctx.push(t);

    const out: string[] = [];

    while (out.length < maxLen) {
      const window = ctx.slice(-contextSize);
      const { logits } = await (this as any).forward(window, false);

      // Use greedy sampling (argmax)
      const idx = greedySample(logits);

      const idxToWord = (this as any).idxToWord;
      const tok = idxToWord.get(idx)!;
      if (tok === (this as any).eos) break;

      out.push(tok);
      ctx.push(idx);
    }

    return this.joinTokens(out);
  }

  /**
   * Generate text using contrastive search/decoding
   * Balances model confidence with diversity to reduce repetition
   *
   * This implementation manually performs contrastive decoding to support async forward passes
   */
  async generateContrastive(
    seedText: string,
    maxLen = 25,
    topK = 4,
    alpha = 0.6
  ): Promise<{ text: string; score: number }> {
    const seedToks = (this as any).tokenize(seedText).map((t: string) => (this as any).toIndex(t));
    const bosIdx = (this as any).toIndex((this as any).bos);
    const contextSize = (this as any).contextSize;

    const ctx: number[] = new Array(contextSize).fill(bosIdx);
    for (const t of seedToks) ctx.push(t);

    const out: number[] = [];
    let totalScore = 0;

    // Cache embeddings for context
    const contextEmbeddings: number[][] = [];
    const embeddingSize = (this as any).embedding[0].length;

    // Helper function for cosine similarity
    const cosineSimilarity = (vec1: number[], vec2: number[]): number => {
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;
      for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
      }
      const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
      return denominator === 0 ? 0 : dotProduct / denominator;
    };

    while (out.length < maxLen) {
      const window = ctx.slice(-contextSize);
      const { logits } = await (this as any).forward(window, false);
      const probs = stableSoftmax(logits, 1.0);

      // Get top-k candidates
      const indexed = probs.map((value, index) => ({ value, index }));
      indexed.sort((a, b) => b.value - a.value);
      const topKCandidates = indexed.slice(0, Math.min(topK, indexed.length));

      let bestToken = topKCandidates[0].index;
      let bestScore = -Infinity;

      // Score each candidate
      for (const candidate of topKCandidates) {
        const tokenId = candidate.index;
        const modelProb = candidate.value;

        // Get embedding for candidate token
        const candidateEmbedding =
          (this as any).embedding[tokenId] || new Array(embeddingSize).fill(0);

        // Compute maximum similarity with context
        let maxSimilarity = 0;
        if (contextEmbeddings.length > 0) {
          for (const contextEmb of contextEmbeddings) {
            const similarity = cosineSimilarity(candidateEmbedding, contextEmb);
            maxSimilarity = Math.max(maxSimilarity, similarity);
          }
        }

        // Contrastive scoring: balance probability and diversity
        const score = (1 - alpha) * modelProb - alpha * maxSimilarity;

        if (score > bestScore) {
          bestScore = score;
          bestToken = tokenId;
        }
      }

      // Check for EOS
      const idxToWord = (this as any).idxToWord;
      const tok = idxToWord.get(bestToken)!;
      if (tok === (this as any).eos) break;

      // Add selected token
      out.push(bestToken);
      ctx.push(bestToken);
      totalScore += bestScore;

      // Cache the embedding
      const selectedEmbedding =
        (this as any).embedding[bestToken] || new Array(embeddingSize).fill(0);
      contextEmbeddings.push([...selectedEmbedding]);
    }

    // Convert tokens to text
    const idxToWord = (this as any).idxToWord;
    const text = this.joinTokens(
      out
        .map((idx: number) => idxToWord.get(idx))
        .filter((word: string | undefined): word is string => word !== undefined)
    );

    return {
      text,
      score: totalScore
    };
  }

  /**
   * Get advanced configuration
   */
  getAdvancedConfig(): Required<AdvancedConfig> {
    return { ...this.advancedConfig };
  }

  /**
   * Update advanced configuration
   */
  setAdvancedConfig(config: Partial<AdvancedConfig>): void {
    this.advancedConfig = { ...this.advancedConfig, ...config };

    // Reinitialize if initialization scheme changed
    if (config.initialization && config.initialization !== 'default') {
      this.reinitializeWeights();
    }

    // Initialize layer norm if enabled
    if (config.useLayerNorm && this.layerNormGamma.length !== this.getHiddenSize()) {
      this.initializeLayerNorm();
    }
  }

  /**
   * Get advanced weights (layer normalization parameters).
   * Used by Cerebro injection system for snapshot/rollback.
   */
  getAdvancedWeights(): { layerNormGamma: number[]; layerNormBeta: number[] } | null {
    if (!this.advancedConfig.useLayerNorm || this.layerNormGamma.length === 0) {
      return null;
    }
    return {
      layerNormGamma: [...this.layerNormGamma],
      layerNormBeta: [...this.layerNormBeta]
    };
  }

  /**
   * Set advanced weights (layer normalization parameters).
   * Used by Cerebro injection system for rollback.
   */
  setAdvancedWeights(weights: { layerNormGamma: number[]; layerNormBeta: number[] }): void {
    this.layerNormGamma = [...weights.layerNormGamma];
    this.layerNormBeta = [...weights.layerNormBeta];
    this.layerNormOptimizer?.reset();
  }

  /**
   * Expand hidden layer by adding k neurons.
   * Overrides base class to also expand layer normalization parameters.
   * Used by Cerebro neuron injection system.
   */
  expandHiddenLayer(k: number, useHeInit = true): void {
    super.expandHiddenLayer(k, useHeInit);

    if (this.advancedConfig.useLayerNorm && this.layerNormGamma.length > 0) {
      for (let i = 0; i < k; i++) {
        this.layerNormGamma.push(1.0);
        this.layerNormBeta.push(0.0);
      }
      this.layerNormOptimizer?.reset();
    }
  }

  getParametersCount(): number {
    const ln = this.isLayerNormActive() ? 2 * this.getHiddenSize() : 0;
    return super.getParametersCount() + ln;
  }

  /**
   * Serialize the model including its advanced configuration, LayerNorm parameters and the
   * learning-rate schedule position. `learningRate` stores the *base* rate (the scheduled value
   * is kept separately) so that reloading a model does not freeze its schedule at the last value.
   */
  toJSON() {
    const base = super.toJSON();
    return {
      ...base,
      architecture: 'advanced' as const,
      learningRate: this.baseLearningRate,
      currentLearningRate: this.getLearningRate(),
      advancedConfig: { ...this.advancedConfig },
      layerNormGamma: [...this.layerNormGamma],
      layerNormBeta: [...this.layerNormBeta],
      schedule: { epoch: this.scheduleEpoch, horizon: this.scheduleHorizon }
    };
  }

  /**
   * Export model with advanced config (alias of toJSON, kept for backwards compatibility).
   */
  toJSONAdvanced() {
    return this.toJSON();
  }

  /**
   * Rebuild an AdvancedNeuralLM from toJSON() output (also accepts exports from older versions,
   * which stored the scheduled rate in `learningRate` and the base rate in `baseLearningRate`).
   */
  static fromJSON(data: any): AdvancedNeuralLM {
    const baseLR =
      typeof data.baseLearningRate === 'number'
        ? data.baseLearningRate
        : (data.learningRate ?? 0.08);
    const model = new AdvancedNeuralLM(
      data.vocab,
      data.hiddenSize,
      baseLR,
      data.contextSize ?? 3,
      (data.optimizer as Optimizer) ?? 'momentum',
      data.momentum ?? 0.9,
      data.dropout ?? 0,
      typeof data.rngSeed === 'number' ? data.rngSeed : undefined,
      ProNeuralLM.normalizeTokenizerConfig(data.tokenizerConfig),
      data.advancedConfig
    );

    AdvancedNeuralLM.restoreSerializedState(model, data);

    if (typeof data.currentLearningRate === 'number') {
      model.setLearningRateInternal(data.currentLearningRate);
    }
    if (Array.isArray(data.layerNormGamma) && Array.isArray(data.layerNormBeta)) {
      if (data.layerNormGamma.length > 0) {
        model.layerNormGamma = [...data.layerNormGamma];
        model.layerNormBeta = [...data.layerNormBeta];
      }
    }
    if (data.schedule && typeof data.schedule.epoch === 'number') {
      model.scheduleEpoch = data.schedule.epoch;
      model.scheduleHorizon = data.schedule.horizon ?? 0;
    } else if (typeof data.currentEpoch === 'number') {
      model.scheduleEpoch = data.currentEpoch;
    }
    return model;
  }

  /**
   * Load model from JSON with advanced config (alias of fromJSON).
   */
  static loadFromJSONAdvanced(data: any): AdvancedNeuralLM {
    return AdvancedNeuralLM.fromJSON(data);
  }

  /**
   * Load an AdvancedNeuralLM saved with saveToLocalStorage(). Returns null when the stored model
   * is missing or is not an advanced model.
   */
  static loadFromLocalStorage(key: string): AdvancedNeuralLM | null {
    try {
      if (typeof localStorage === 'undefined') return null;
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (data?.architecture !== 'advanced') return null;
      return AdvancedNeuralLM.fromJSON(data);
    } catch (error) {
      console.warn('Failed to load advanced model', error);
      return null;
    }
  }

  /**
   * Calculate perplexity on text (exp of the mean next-token cross-entropy).
   */
  async calculatePerplexity(text: string): Promise<number> {
    const result = await this.evaluate(text);
    return result.tokens === 0 ? Infinity : result.perplexity;
  }
}
