import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ProNeuralLM,
  type Optimizer,
  type TokenizerConfig,
  MODEL_VERSION,
  MODEL_EXPORT_FILENAME
} from '../src/lib/ProNeuralLM';

/**
 * Neuro-Lingua CLI Training Script v2.0
 *
 * Enhanced features:
 * - Tokenizer presets (english, hebrew, multilingual, code)
 * - Σ-SIG governance metadata (project ID, run ID, witness, rationale)
 * - Comprehensive environment variable support
 *
 * Usage:
 *   pnpm train                     # Use defaults
 *   EPOCHS=50 pnpm train           # Custom epochs
 *   TOKENIZER_PRESET=hebrew pnpm train  # Hebrew preset
 *   PROJECT_ID=proj_123 RUN_ID=run_456 WITNESS="John Doe" pnpm train
 *
 * Environment Variables:
 *   EPOCHS          - Training epochs (default: 30)
 *   HIDDEN_SIZE     - Hidden layer size (default: 64)
 *   LEARNING_RATE   - Learning rate (default: 0.08)
 *   CONTEXT_SIZE    - Context window size (default: 3)
 *   OPTIMIZER       - 'momentum' or 'adam' (default: momentum)
 *   MOMENTUM        - Momentum value (default: 0.9)
 *   DROPOUT         - Dropout rate (default: 0.1)
 *   SEED            - Random seed (default: 1337)
 *
 *   TOKENIZER_PRESET - Preset: 'english', 'hebrew', 'multilingual', 'code'
 *   TOKENIZER_MODE   - Mode: 'unicode', 'ascii', 'custom'
 *   TOKENIZER_PATTERN - Custom regex pattern (when mode='custom')
 *
 *   CORPUS_PATH      - Path to training corpus (default: data/corpus.txt)
 *   MODEL_EXPORT_PATH - Path to save model (default: models/neuro-lingua-v324.json)
 *   METRICS_PATH     - Path to save metrics JSON
 *   EXPERIMENT_NAME  - Name for this experiment
 *
 *   PROJECT_ID       - Σ-SIG project identifier
 *   RUN_ID           - Σ-SIG run identifier
 *   WITNESS          - Person/system authorizing training
 *   RATIONALE        - Reason for this training run
 *   EXPIRY           - ISO 8601 expiry date (optional)
 */

const CORPUS_URL = new URL('../data/corpus.txt', import.meta.url);
const OUTPUT_URL = new URL(`../models/${MODEL_EXPORT_FILENAME}`, import.meta.url);
const METRICS_OVERRIDE = process.env.METRICS_PATH ? path.resolve(process.env.METRICS_PATH) : null;
const CORPUS_OVERRIDE = process.env.CORPUS_PATH ? path.resolve(process.env.CORPUS_PATH) : null;
const MODEL_OVERRIDE = process.env.MODEL_EXPORT_PATH
  ? path.resolve(process.env.MODEL_EXPORT_PATH)
  : null;
const EXPERIMENT_NAME = process.env.EXPERIMENT_NAME ?? null;

// Σ-SIG Governance metadata
const PROJECT_ID = process.env.PROJECT_ID ?? null;
const RUN_ID = process.env.RUN_ID ?? null;
const WITNESS = process.env.WITNESS ?? 'cli-training';
const RATIONALE = process.env.RATIONALE ?? 'Automated CLI training';
const EXPIRY = process.env.EXPIRY ?? null;

const DEFAULTS = {
  epochs: 30,
  hiddenSize: 64,
  learningRate: 0.08,
  contextSize: 3,
  optimizer: 'momentum' as Optimizer,
  momentum: 0.9,
  dropout: 0.1,
  seed: 1337
};

/**
 * Tokenizer presets for common use cases
 */
const TOKENIZER_PRESETS: Record<string, TokenizerConfig> = {
  // English: standard Unicode with common English patterns
  english: { mode: 'unicode' },

  // Hebrew: Unicode mode handles Hebrew natively (RTL + nikud)
  hebrew: { mode: 'unicode' },

  // Multilingual: Unicode for mixed-language corpora
  multilingual: { mode: 'unicode' },

  // Code: Preserves programming symbols and whitespace
  code: { mode: 'custom', pattern: '[^\\w\\s{}()\\[\\]<>.,;:!?\'"-]' },

  // ASCII-only: For legacy or restricted environments
  ascii: { mode: 'ascii' }
};

function parseNumber(envValue: string | undefined, fallback: number) {
  if (!envValue) return fallback;
  const parsed = Number(envValue);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOptimizer(envValue: string | undefined): Optimizer {
  return envValue === 'adam' ? 'adam' : 'momentum';
}

function parseTokenizerConfig(): TokenizerConfig {
  // Check for preset first
  const preset = process.env.TOKENIZER_PRESET?.toLowerCase();
  if (preset && TOKENIZER_PRESETS[preset]) {
    return TOKENIZER_PRESETS[preset];
  }

  // Fall back to mode-based configuration
  const fallbackMode = process.env.USE_ASCII_TOKENIZER === 'true' ? 'ascii' : 'unicode';
  const mode = (process.env.TOKENIZER_MODE ?? fallbackMode) as TokenizerConfig['mode'];
  if (mode === 'ascii') return { mode: 'ascii' };
  if (mode === 'custom') {
    const pattern = process.env.TOKENIZER_PATTERN;
    if (pattern && pattern.length > 0) {
      return { mode: 'custom', pattern };
    }
  }
  return { mode: 'unicode' };
}

/**
 * Generate a simple checksum for corpus traceability
 */
function generateChecksum(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

function buildVocabulary(tokens: string[]) {
  const specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
  const uniq = new Set(tokens);
  for (const special of specials) uniq.delete(special.toLowerCase());
  return [...specials, ...Array.from(uniq).sort()];
}

async function main() {
  const tokenizerConfig = parseTokenizerConfig();

  const corpusPath = CORPUS_OVERRIDE ?? fileURLToPath(CORPUS_URL);
  const outputPath = MODEL_OVERRIDE ?? fileURLToPath(OUTPUT_URL);
  const corpus = await fs.readFile(corpusPath, 'utf8');

  const tokens = ProNeuralLM.tokenizeText(corpus, tokenizerConfig);
  if (tokens.length === 0) {
    throw new Error('Corpus is empty after tokenization. Add text to data/corpus.txt.');
  }

  const vocab = buildVocabulary(tokens);

  const epochs = Math.max(1, Math.floor(parseNumber(process.env.EPOCHS, DEFAULTS.epochs)));
  const hiddenSize = Math.max(
    4,
    Math.floor(parseNumber(process.env.HIDDEN_SIZE, DEFAULTS.hiddenSize))
  );
  const contextSize = Math.max(
    1,
    Math.floor(parseNumber(process.env.CONTEXT_SIZE, DEFAULTS.contextSize))
  );
  const learningRate = Math.max(
    1e-4,
    parseNumber(process.env.LEARNING_RATE, DEFAULTS.learningRate)
  );
  const optimizer = parseOptimizer(process.env.OPTIMIZER ?? DEFAULTS.optimizer);
  const momentum = parseNumber(process.env.MOMENTUM, DEFAULTS.momentum);
  const dropout = Math.min(0.5, Math.max(0, parseNumber(process.env.DROPOUT, DEFAULTS.dropout)));
  const seed = Math.floor(parseNumber(process.env.SEED, DEFAULTS.seed));

  console.log(`--- Neuro-Lingua Training v${MODEL_VERSION} ---`);
  console.log(`Corpus path: ${corpusPath}`);
  console.log(`Model artifact: ${outputPath}`);
  if (EXPERIMENT_NAME) {
    console.log(`Experiment: ${EXPERIMENT_NAME}`);
  }
  console.log(`Tokens: ${tokens.length}`);
  console.log(`Vocab size: ${vocab.length}`);

  // Tokenizer info
  const presetUsed = process.env.TOKENIZER_PRESET?.toLowerCase();
  const tokenizerInfo = presetUsed
    ? `${presetUsed} preset (${tokenizerConfig.mode})`
    : `${tokenizerConfig.mode}${tokenizerConfig.mode === 'custom' && tokenizerConfig.pattern ? ` (${tokenizerConfig.pattern})` : ''}`;
  console.log(`Tokenizer: ${tokenizerInfo}`);

  console.log(
    `Hyperparameters → epochs: ${epochs}, hiddenSize: ${hiddenSize}, contextSize: ${contextSize}, learningRate: ${learningRate.toFixed(
      4
    )}, optimizer: ${optimizer}, momentum: ${momentum}, dropout: ${dropout}, seed: ${seed}`
  );

  // Σ-SIG Governance info
  if (PROJECT_ID || RUN_ID) {
    console.log(`\n--- Σ-SIG Governance ---`);
    if (PROJECT_ID) console.log(`Project: ${PROJECT_ID}`);
    if (RUN_ID) console.log(`Run: ${RUN_ID}`);
    console.log(`Witness: ${WITNESS}`);
    console.log(`Rationale: ${RATIONALE}`);
    if (EXPIRY) console.log(`Expiry: ${EXPIRY}`);
  }

  const model = new ProNeuralLM(
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
  const { loss, accuracy, history } = await model.train(corpus, epochs);

  const lastEpoch = history[history.length - 1];
  console.log(
    `Training complete. Avg loss: ${loss.toFixed(4)}, accuracy: ${(accuracy * 100).toFixed(2)}%, last epoch loss: ${
      lastEpoch ? lastEpoch.loss.toFixed(4) : 'n/a'
    }, last epoch accuracy: ${lastEpoch ? (lastEpoch.accuracy * 100).toFixed(2) : 'n/a'}%`
  );

  const corpusChecksum = generateChecksum(corpus);

  const summary = {
    modelVersion: MODEL_VERSION,
    experiment: EXPERIMENT_NAME,
    dataset: corpusPath,
    corpusChecksum,
    modelPath: outputPath,
    hyperparameters: {
      epochs,
      hiddenSize,
      contextSize,
      learningRate,
      optimizer,
      momentum,
      dropout,
      seed
    },
    tokenizer: tokenizerConfig,
    metrics: {
      loss,
      accuracy,
      history
    },
    // Σ-SIG Governance metadata
    governance: {
      projectId: PROJECT_ID,
      runId: RUN_ID,
      witness: WITNESS,
      rationale: RATIONALE,
      expiry: EXPIRY,
      createdAt: new Date().toISOString()
    }
  };

  if (METRICS_OVERRIDE) {
    await fs.mkdir(path.dirname(METRICS_OVERRIDE), { recursive: true });
    await fs.writeFile(METRICS_OVERRIDE, JSON.stringify(summary, null, 2), 'utf8');
    console.log(`Metrics saved to ${METRICS_OVERRIDE}`);
  }

  console.log(JSON.stringify(summary, null, 2));

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(model.toJSON(), null, 2), 'utf8');

  console.log(`Model artifact (${MODEL_EXPORT_FILENAME}) saved to ${outputPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
