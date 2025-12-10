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
  hiddenSize: 64,
  epochs: 20,
  learningRate: 0.08,
  optimizer: 'momentum' as Optimizer,
  momentum: 0.9,
  dropout: 0.1,
  contextSize: 3,
  seed: 1337,
  resume: true
};

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
 * Default tokenizer configuration
 */
export const DEFAULT_TOKENIZER_CONFIG: TokenizerConfig = {
  mode: 'unicode'
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
  contextSize: { min: 2, max: 6 },
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
