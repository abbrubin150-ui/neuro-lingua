import type { Optimizer, TokenizerConfig } from '../lib/ProNeuralLM';

/**
 * Storage keys for localStorage
 */
export const STORAGE_KEYS = {
  UI_SETTINGS: 'neuro-lingua-ui-settings-v1',
  MODEL_META: 'neuro-lingua-model-meta-v1',
  TOKENIZER_CONFIG: 'neuro-lingua-tokenizer-config-v1',
  ONBOARDING_DISMISSED: 'neuro-lingua-onboarding-dismissed',
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
  samplingMode: 'topp' as 'off' | 'topk' | 'topp',
  maxTokens: 25
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
  topP: { min: 0, max: 0.99 }
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
