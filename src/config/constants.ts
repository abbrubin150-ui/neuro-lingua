import type { Optimizer, TokenizerConfig } from '../lib/ProNeuralLM';
import type { ActivationFunction, LRSchedule, InitializationScheme } from '../lib/AdvancedNeuralLM';
import type { BetaSchedule } from '../losses/information_bottleneck';

/**
 * Storage keys for localStorage
 */
export const STORAGE_KEYS = {
  UI_SETTINGS: 'neuro-lingua-ui-settings-v1',
  MODEL_META: 'neuro-lingua-model-meta-v1',
  TOKENIZER_CONFIG: 'neuro-lingua-tokenizer-config-v1',
  ONBOARDING_DISMISSED: 'neuro-lingua-onboarding-dismissed',
  TRAINING_TEXT: 'neuro-lingua-training-text-v1',
  LEGACY_MODELS: ['neuro-lingua-pro-v32']
} as const;

/**
 * File export filenames
 */
export const EXPORT_FILENAMES = {
  TOKENIZER: 'neuro-lingua-tokenizer.json',
  TRAINING_HISTORY: 'neuro-lingua-training-history.csv'
} as const;

/**
 * Default text shown in training corpus textarea
 */
export const DEFAULT_TRAINING_TEXT =
  'An advanced neural language model trains in the browser. Paste your own English corpus to fine-tune responses.';

/**
 * Default regex pattern for custom tokenizer mode
 */
export const DEFAULT_CUSTOM_TOKENIZER_PATTERN = "[^\\p{L}\\d\\s'-]";

/**
 * Default hyperparameters for model training
 */
export const DEFAULT_HYPERPARAMETERS = {
  hiddenSize: 128,
  epochs: 30,
  learningRate: 0.003,
  optimizer: 'adam' as Optimizer,
  momentum: 0.9,
  dropout: 0.05,
  contextSize: 4,
  seed: 1337,
  resume: true,
  transformer: {
    numHeads: 8,
    numLayers: 4,
    ffHiddenDim: 512,
    attentionDropout: 0.1,
    dropConnectRate: 0.05,
    numKVHeads: 4
  }
} as const;

/**
 * Default generation parameters
 */
export const DEFAULT_GENERATION = {
  temperature: 0.8,
  topK: 20,
  topP: 0.9,
  typicalTau: 0.9,
  samplingMode: 'topp' as 'off' | 'topk' | 'topp' | 'typical' | 'mirostat',
  mirostatTau: 5,
  mirostatEta: 0.1,
  maxTokens: 25,
  useBeamSearch: false,
  beamWidth: 4,
  frequencyPenalty: 0,
  presencePenalty: 0
};

/**
 * Default advanced configuration for AdvancedNeuralLM
 */
export const DEFAULT_ADVANCED_CONFIG = {
  useAdvanced: false,
  activation: 'relu' as ActivationFunction,
  leakyReluAlpha: 0.01,
  eluAlpha: 1.0,
  initialization: 'he' as InitializationScheme,
  lrSchedule: 'cosine' as LRSchedule,
  lrMin: 1e-6,
  lrDecayRate: 0.95,
  warmupEpochs: 0,
  weightDecay: 1e-4,
  gradientClipNorm: 5.0,
  useLayerNorm: false
};

/**
 * Lion optimizer configuration defaults (v4.0)
 *
 * Lion (EvoLved Sign Momentum) is a simple and efficient optimizer that:
 * - Uses 50% less memory than Adam (only one momentum buffer)
 * - Converges 1.5-2× faster
 * - Uses sign of momentum for updates (hence lower learning rate)
 *
 * Reference: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
 */
export const DEFAULT_LION_CONFIG = {
  lionBeta1: 0.9,
  lionBeta2: 0.99,
  lionWeightDecay: 0.01,
  /** Recommended LR for Lion (lower than Adam due to sign() behavior) */
  recommendedLR: 3e-4
};

/**
 * Sophia optimizer configuration defaults (v4.2)
 *
 * Sophia (Second-order Stochastic Optimizer) features:
 * - 2× faster convergence than Adam via curvature-aware updates
 * - Diagonal Hessian estimation with Gauss-Newton approximation
 * - Clipped preconditioned updates for stability
 * - Memory: ~2× parameters (momentum + Hessian diagonal)
 *
 * Reference: Liu et al. (2023) "Sophia: A Scalable Stochastic Second-order Optimizer"
 */
export const DEFAULT_SOPHIA_CONFIG = {
  sophiaBeta1: 0.965,
  sophiaBeta2: 0.99,
  sophiaWeightDecay: 0.01,
  /** Clipping bound for preconditioned updates */
  sophiaRho: 1.0,
  /** Hessian update frequency (every N steps) */
  sophiaHessianFreq: 10,
  /** Recommended LR for Sophia (lower than Adam due to second-order info) */
  recommendedLR: 1e-4
};

/**
 * Sparse Attention configuration defaults (v4.2)
 *
 * Reduces O(n²) attention complexity to O(n log n) or O(n).
 * Enables longer context windows without proportional memory increase.
 */
export const DEFAULT_SPARSE_ATTENTION_CONFIG = {
  /** Default sparse pattern type */
  pattern: 'local' as const,
  /** Local attention window size (half-width) */
  windowSize: 64,
  /** Number of global tokens for BigBird/Longformer */
  numGlobalTokens: 2,
  /** Number of random attention tokens for BigBird */
  numRandomTokens: 3,
  /** Block size for block-sparse attention */
  blockSize: 64,
  /** Enable causal masking by default */
  causal: true
};

/**
 * Mixed Precision configuration defaults (v4.2)
 *
 * FP16/FP32 mixed precision for 2-3× speedup and 50% memory reduction.
 */
export const DEFAULT_MIXED_PRECISION_CONFIG = {
  /** Enable mixed precision when WebGPU available */
  enabled: false,
  /** Initial loss scale (high to catch underflow) */
  initialLossScale: 65536,
  /** Scale growth factor after successful steps */
  scaleGrowthFactor: 2,
  /** Scale backoff factor after overflow */
  scaleBackoffFactor: 0.5,
  /** Steps before scaling up */
  scaleGrowthInterval: 2000
};

/**
 * Default Information Bottleneck configuration
 */
export const DEFAULT_IB_CONFIG = {
  useIB: false,
  betaStart: 0.001,
  betaEnd: 1.0,
  betaSchedule: 'linear' as BetaSchedule,
  ibAlpha: 0.1, // Hybrid loss weight: (1-α)·CE + α·IB
  numBins: 50 // Number of bins for MI estimation
};

/**
 * Default advanced loss function configuration
 *
 * Supports multiple loss types:
 * - cross_entropy: Standard cross-entropy (default)
 * - focal: Focal loss for class imbalance
 * - label_smoothing: Label smoothing CE for regularization
 * - symmetric_ce: Symmetric CE for noise robustness
 */
export const DEFAULT_LOSS_CONFIG = {
  lossFunction: 'cross_entropy' as 'cross_entropy' | 'focal' | 'label_smoothing' | 'symmetric_ce',
  // Focal loss parameters (Lin et al. 2017)
  focalGamma: 2.0, // Focusing parameter - higher = more focus on hard examples
  focalAlpha: 0.25, // Balancing factor
  // Label smoothing parameters
  labelSmoothingEpsilon: 0.1, // Smoothing factor (0 = no smoothing, 1 = uniform)
  // Symmetric CE parameters
  sceBeta: 1.0 // Weight for reverse KL term (α=1 for forward)
};

/**
 * Default tokenizer configuration
 */
export const DEFAULT_TOKENIZER_CONFIG: TokenizerConfig = {
  mode: 'unicode'
} as const;

/**
 * Default BPE Tokenizer configuration (v4.3)
 *
 * Byte Pair Encoding tokenizer for subword tokenization.
 * Provides better coverage than character-level tokenization.
 */
export const DEFAULT_BPE_CONFIG = {
  vocabSize: 1000,
  minFrequency: 2,
  specialTokens: ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] as const,
  padToken: '<PAD>',
  bosToken: '<BOS>',
  eosToken: '<EOS>',
  unknownToken: '<UNK>'
} as const;

/**
 * Default Reproducibility configuration (v4.3)
 *
 * Settings for deterministic training and sampling.
 * Uses xorshift128+ PRNG with splitmix64 initialization.
 */
export const DEFAULT_REPRODUCIBILITY_CONFIG = {
  globalSeed: 1337,
  trainingSeed: 1337,
  samplingSeed: 1338,
  initializationSeed: 1339
} as const;

/**
 * Default Dataset configuration (v4.3)
 *
 * Configuration for dataset splitting and batching.
 */
export const DEFAULT_DATASET_CONFIG = {
  splitConfig: {
    trainRatio: 0.8,
    valRatio: 0.1,
    testRatio: 0.1
  },
  batchConfig: {
    batchSize: 32,
    shuffle: true,
    dropLast: false
  }
} as const;

/**
 * Tolerance levels for reproducibility verification
 * Values represent maximum allowed absolute difference
 */
export const TOLERANCE_LEVELS = {
  strict: 1e-7,
  normal: 1e-5,
  loose: 1e-3
} as const;

/**
 * Constraints for hyperparameter values
 */
export const HYPERPARAMETER_CONSTRAINTS = {
  hiddenSize: { min: 16, max: 256 },
  epochs: { min: 1, max: 200 },
  learningRate: { min: 0.001, max: 1 },
  momentum: { min: 0, max: 0.99 },
  dropout: { min: 0, max: 0.5 },
  contextSize: { min: 2, max: 128 },
  temperature: { min: 0.05, max: 5 },
  topK: { min: 0, max: 1000 },
  topP: { min: 0, max: 0.99 },
  typicalTau: { min: 0.1, max: 1 },
  mirostatTau: { min: 1, max: 10 },
  mirostatEta: { min: 0.01, max: 1 },
  beamWidth: { min: 1, max: 10 },
  frequencyPenalty: { min: 0, max: 2 },
  presencePenalty: { min: 0, max: 2 },
  maxTokens: { min: 1, max: 200 },
  leakyReluAlpha: { min: 0.01, max: 0.3 },
  eluAlpha: { min: 0.1, max: 2.0 },
  lrMin: { min: 1e-8, max: 0.01 },
  lrDecayRate: { min: 0.8, max: 0.99 },
  warmupEpochs: { min: 0, max: 50 },
  weightDecay: { min: 0, max: 0.01 },
  gradientClipNorm: { min: 1, max: 10 },
  transformer: {
    numHeads: { min: 1, max: 16 },
    numLayers: { min: 1, max: 8 },
    ffHiddenDim: { min: 32, max: 2048 },
    attentionDropout: { min: 0, max: 0.5 },
    dropConnectRate: { min: 0, max: 0.5 },
    /**
     * Number of KV heads for Grouped-Query Attention (GQA).
     * Must be a divisor of numHeads. Lower values = more memory savings.
     * - numKVHeads = numHeads: Standard MHA
     * - numKVHeads = 1: Multi-Query Attention (MQA)
     * - numKVHeads = numHeads/4: GQA 4:1 (Llama-3.2 style)
     */
    numKVHeads: { min: 1, max: 16 }
  },
  ib: {
    betaStart: { min: 0.0001, max: 10 },
    betaEnd: { min: 0.0001, max: 10 },
    ibAlpha: { min: 0, max: 1 },
    numBins: { min: 10, max: 200 }
  },
  /**
   * Lion optimizer constraints (v4.0)
   */
  lion: {
    beta1: { min: 0.8, max: 0.99 },
    beta2: { min: 0.9, max: 0.999 },
    weightDecay: { min: 0, max: 0.1 }
  },
  /**
   * Sophia optimizer constraints (v4.2)
   */
  sophia: {
    beta1: { min: 0.9, max: 0.99 },
    beta2: { min: 0.9, max: 0.999 },
    weightDecay: { min: 0, max: 0.1 },
    rho: { min: 0.1, max: 5.0 },
    hessianUpdateFreq: { min: 1, max: 100 }
  },
  /**
   * Sparse Attention constraints (v4.2)
   */
  sparseAttention: {
    windowSize: { min: 1, max: 512 },
    numGlobalTokens: { min: 0, max: 16 },
    numRandomTokens: { min: 0, max: 16 },
    blockSize: { min: 8, max: 256 }
  },
  /**
   * Mixed Precision constraints (v4.2)
   */
  mixedPrecision: {
    initialLossScale: { min: 1, max: 16777216 },
    scaleGrowthFactor: { min: 1.1, max: 4.0 },
    scaleBackoffFactor: { min: 0.1, max: 0.9 },
    scaleGrowthInterval: { min: 100, max: 10000 }
  }
} as const;

/**
 * Minimum vocabulary size required for training
 */
export const MIN_VOCAB_SIZE = 8;

/**
 * Special tokens used by the model
 */
export const SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] as const;

/**
 * UI update delay during training (ms)
 */
export const TRAINING_UI_UPDATE_DELAY = 16;

/**
 * Example corpus text for demonstration
 */
export const EXAMPLE_CORPUS = `Machine learning and artificial intelligence are reshaping the technology landscape. Advanced algorithms learn from data patterns and improve their performance over time.

Artificial neural models emulate the human brain using layers of digital neurons. These technologies enable smart systems that understand language, recognize images, and support decision making.

English-language research communities share open tools, papers, and tutorials so builders everywhere can prototype intelligent assistants.`;
