import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { ProNeuralLM, type Optimizer } from '../src/lib/ProNeuralLM';

const CORPUS_URL = new URL('../data/corpus.txt', import.meta.url);
const OUTPUT_URL = new URL('../models/neuro-lingua-v322.json', import.meta.url);

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

function parseNumber(envValue: string | undefined, fallback: number) {
  if (!envValue) return fallback;
  const parsed = Number(envValue);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOptimizer(envValue: string | undefined): Optimizer {
  return envValue === 'adam' ? 'adam' : 'momentum';
}

function createTokenizer() {
  const useAscii = process.env.USE_ASCII_TOKENIZER === 'true';
  const pattern = useAscii ? /[^a-z0-9\s'-]/gi : /[^\u0590-\u05FF\w\s'-]/g;
  return (text: string) =>
    text
      .toLowerCase()
      .replace(pattern, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 0);
}

function buildVocabulary(tokens: string[]) {
  const specials = ['<BOS>', '<EOS>', '<UNK>'];
  const uniq = new Set(tokens);
  for (const special of specials) uniq.delete(special.toLowerCase());
  return [...specials, ...Array.from(uniq).sort()];
}

async function main() {
  const tokenizer = createTokenizer();

  const corpusPath = fileURLToPath(CORPUS_URL);
  const outputPath = fileURLToPath(OUTPUT_URL);
  const corpus = await fs.readFile(corpusPath, 'utf8');

  const tokens = tokenizer(corpus);
  if (tokens.length === 0) {
    throw new Error('Corpus is empty after tokenization. Add text to data/corpus.txt.');
  }

  const vocab = buildVocabulary(tokens);

  const epochs = Math.max(1, Math.floor(parseNumber(process.env.EPOCHS, DEFAULTS.epochs)));
  const hiddenSize = Math.max(4, Math.floor(parseNumber(process.env.HIDDEN_SIZE, DEFAULTS.hiddenSize)));
  const contextSize = Math.max(1, Math.floor(parseNumber(process.env.CONTEXT_SIZE, DEFAULTS.contextSize)));
  const learningRate = Math.max(1e-4, parseNumber(process.env.LEARNING_RATE, DEFAULTS.learningRate));
  const optimizer = parseOptimizer(process.env.OPTIMIZER ?? DEFAULTS.optimizer);
  const momentum = parseNumber(process.env.MOMENTUM, DEFAULTS.momentum);
  const dropout = Math.min(0.5, Math.max(0, parseNumber(process.env.DROPOUT, DEFAULTS.dropout)));
  const seed = Math.floor(parseNumber(process.env.SEED, DEFAULTS.seed));

  console.log('--- Neuro-Lingua Training ---');
  console.log(`Corpus path: ${corpusPath}`);
  console.log(`Tokens: ${tokens.length}`);
  console.log(`Vocab size: ${vocab.length}`);
  console.log(
    `Hyperparameters → epochs: ${epochs}, hiddenSize: ${hiddenSize}, contextSize: ${contextSize}, learningRate: ${learningRate.toFixed(
      4
    )}, optimizer: ${optimizer}, momentum: ${momentum}, dropout: ${dropout}, seed: ${seed}`
  );

  const model = new ProNeuralLM(vocab, hiddenSize, learningRate, contextSize, optimizer, momentum, dropout, seed);
  const { loss, accuracy, history } = model.train(corpus, epochs);

  const lastEpoch = history[history.length - 1];
  console.log(
    `Training complete. Avg loss: ${loss.toFixed(4)}, accuracy: ${(accuracy * 100).toFixed(2)}%, last epoch loss: ${
      lastEpoch ? lastEpoch.loss.toFixed(4) : 'n/a'
    }, last epoch accuracy: ${lastEpoch ? (lastEpoch.accuracy * 100).toFixed(2) : 'n/a'}%`
  );

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(model.toJSON(), null, 2), 'utf8');

  console.log(`Model saved to ${outputPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
