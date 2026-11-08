/**
 * Offline model factory: local documents → cleaned corpus → tokenizer → trained model →
 * validated, packaged model with a model card.
 *
 * Everything runs in-process on the CPU. No network request is made and no external AI model or
 * service is used at any step; the only inputs are the documents and the configuration.
 * The pipeline is environment-agnostic (Node CLI or browser); file access lives in the callers.
 */

import { prepareCorpus, type CorpusDocument, type CorpusPrepOptions } from '../data/corpusPrep';
import { trainSubwordModel, subwordVocabulary } from '../tokenizer/subword';
import { ProNeuralLM, type Optimizer, type TokenizerConfig } from '../lib/ProNeuralLM';
import { AdvancedNeuralLM, type AdvancedConfig } from '../lib/AdvancedNeuralLM';
import { TransformerLM, type TransformerConfig } from '../lib/TransformerLM';
import {
  packModel,
  type ModelCard,
  type ModelPackage,
  type TensorPrecision
} from '../lib/modelPackage';

export type FactoryArchitecture = 'transformer' | 'advanced' | 'feedforward';

export interface FactoryConfig {
  name: string;
  description?: string;
  architecture: FactoryArchitecture;
  tokenizer: {
    mode: 'bpe' | 'character' | 'unicode';
    /** Subword vocabulary size for 'bpe' (default 2000). */
    vocabSize?: number;
    lowercase?: boolean;
  };
  model: {
    hiddenSize: number;
    contextSize: number;
    dropout?: number;
    /** Transformer options (layers, heads, KV heads, FFN width, dropouts, stride, decay). */
    transformer?: TransformerConfig;
    /** AdvancedNeuralLM options. */
    advanced?: AdvancedConfig;
  };
  training: {
    epochs: number;
    learningRate: number;
    optimizer?: Optimizer;
    /** Transformer: warm-up steps of the per-step schedule (0 = constant learning rate). */
    warmupSteps?: number;
    /** Stop after this many epochs without a validation improvement (0 = never). */
    patience?: number;
    seed?: number;
  };
  data?: CorpusPrepOptions;
  precision?: TensorPrecision;
}

export interface EpochReport {
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  validationPerplexity: number;
  /** Validation bits per character: comparable across tokenizers. */
  validationBitsPerCharacter: number;
  learningRate: number;
  seconds: number;
  improved: boolean;
}

export interface FactoryCallbacks {
  onLog?: (message: string) => void;
  onProgress?: (done: number, total: number, epoch: number) => void;
  onEpoch?: (report: EpochReport, model: ProNeuralLM) => void | Promise<void>;
}

export interface FactoryResult {
  model: ProNeuralLM;
  package: ModelPackage;
  card: ModelCard;
  history: EpochReport[];
  bestEpoch: number;
}

export const OFFLINE_PROVENANCE: ModelCard['provenance'] = {
  offline: true,
  externalServices: [],
  statement:
    'Trained locally by the Neuro-Lingua model factory from the listed documents only. No external AI model, API or network service was used for data preparation, tokenization, training or evaluation.'
};

export function architectureOf(model: ProNeuralLM): FactoryArchitecture {
  if (model instanceof TransformerLM) return 'transformer';
  if (model instanceof AdvancedNeuralLM) return 'advanced';
  return 'feedforward';
}

/** Held-out metrics, including bits per character (tokenizer-independent). */
export async function evaluateText(
  model: ProNeuralLM,
  text: string
): Promise<{ loss: number; perplexity: number; bitsPerCharacter: number; tokens: number }> {
  const result = await model.evaluate(text);
  const characters = Math.max(1, text.length);
  const bitsPerCharacter =
    result.tokens > 0 ? (result.loss * result.tokens) / characters / Math.LN2 : NaN;
  return {
    loss: result.loss,
    perplexity: result.perplexity,
    bitsPerCharacter,
    tokens: result.tokens
  };
}

function buildTokenizer(
  train: string,
  config: FactoryConfig['tokenizer']
): { tokenizer: TokenizerConfig; summary: Record<string, unknown> } {
  if (config.mode === 'bpe') {
    const bpe = trainSubwordModel(train, {
      vocabSize: config.vocabSize ?? 2000,
      lowercase: config.lowercase ?? false
    });
    return {
      tokenizer: { mode: 'bpe', bpe },
      summary: {
        mode: 'bpe',
        subwordVocabulary: subwordVocabulary(bpe).length,
        merges: bpe.merges.length,
        alphabet: bpe.alphabet.length,
        lowercase: bpe.lowercase
      }
    };
  }
  return { tokenizer: { mode: config.mode }, summary: { mode: config.mode } };
}

function buildModel(
  config: FactoryConfig,
  vocab: string[],
  tokenizer: TokenizerConfig
): ProNeuralLM {
  const seed = config.training.seed ?? 1337;
  const optimizer = config.training.optimizer ?? 'adam';
  const dropout = config.model.dropout ?? 0.05;
  const { hiddenSize, contextSize } = config.model;
  const lr = config.training.learningRate;
  if (config.architecture === 'transformer') {
    return new TransformerLM(
      vocab,
      hiddenSize,
      lr,
      contextSize,
      optimizer,
      0.9,
      dropout,
      seed,
      tokenizer,
      {
        objective: 'causal',
        ...config.model.transformer
      }
    );
  }
  if (config.architecture === 'advanced') {
    return new AdvancedNeuralLM(
      vocab,
      hiddenSize,
      lr,
      contextSize,
      optimizer,
      0.9,
      dropout,
      seed,
      tokenizer,
      config.model.advanced
    );
  }
  return new ProNeuralLM(
    vocab,
    hiddenSize,
    lr,
    contextSize,
    optimizer,
    0.9,
    dropout,
    seed,
    tokenizer
  );
}

/**
 * Run the full pipeline. `initialModel` continues training an existing model (its tokenizer and
 * vocabulary are kept), e.g. to adapt an offline model to new local documents.
 */
export async function runModelFactory(
  documents: CorpusDocument[],
  config: FactoryConfig,
  callbacks: FactoryCallbacks = {},
  initialModel?: ProNeuralLM
): Promise<FactoryResult> {
  const log = callbacks.onLog ?? (() => undefined);
  const started = Date.now();

  const corpus = prepareCorpus(documents, config.data);
  if (corpus.train.length === 0) throw new Error('No training text left after preparation');
  if (corpus.validation.length === 0) throw new Error('No validation text: add more documents');
  log(
    `Data: ${corpus.stats.documents} documents, ${corpus.stats.trainParagraphs} training / ${corpus.stats.validationParagraphs} validation paragraphs, ${corpus.stats.duplicateParagraphs} duplicates removed`
  );

  let model: ProNeuralLM;
  let tokenizerSummary: Record<string, unknown>;
  if (initialModel) {
    model = initialModel;
    const tk = model.getTokenizerConfig();
    tokenizerSummary = { mode: tk.mode, reusedFromInitialModel: true };
    log(`Continuing from an existing ${architectureOf(model)} model (tokenizer kept)`);
  } else {
    const { tokenizer, summary } = buildTokenizer(corpus.train, config.tokenizer);
    tokenizerSummary = summary;
    const vocab = ProNeuralLM.buildVocabulary(corpus.train, tokenizer);
    model = buildModel(config, vocab, tokenizer);
    log(
      `Model: ${config.architecture}, ${model.getParametersCount().toLocaleString('en-US')} parameters, vocabulary ${vocab.length}`
    );
  }
  if (callbacks.onProgress) model.setTrainingProgressCallback(callbacks.onProgress);

  const epochs = Math.max(1, Math.floor(config.training.epochs));
  if (model instanceof TransformerLM) {
    const info = model.getTransformerInfo();
    const stride = Math.max(1, config.model.transformer?.trainingStride ?? 1);
    const windows = Math.ceil(model.countTrainingWindows(corpus.train) / stride);
    const stepsPerEpoch = Math.ceil(windows / Math.max(1, info.batchSize));
    const warmup = config.training.warmupSteps ?? 0;
    if (warmup > 0) {
      model.setLearningRateSchedule({ warmupSteps: warmup, totalSteps: stepsPerEpoch * epochs });
    }
  } else if (model instanceof AdvancedNeuralLM) {
    model.setTrainingHorizon(epochs);
  }

  const patience = Math.max(0, Math.floor(config.training.patience ?? 0));
  const history: EpochReport[] = [];
  let best: { loss: number; epoch: number; json: Record<string, unknown> } | null = null;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const t0 = Date.now();
    const result = await model.train(corpus.train, 1);
    const val = await evaluateText(model, corpus.validation);
    const improved = !best || val.loss < best.loss;
    if (improved) {
      best = {
        loss: val.loss,
        epoch,
        json: JSON.parse(JSON.stringify(model.toJSON())) as Record<string, unknown>
      };
    }
    const report: EpochReport = {
      epoch,
      trainLoss: result.history[result.history.length - 1]?.loss ?? result.loss,
      validationLoss: val.loss,
      validationPerplexity: val.perplexity,
      validationBitsPerCharacter: val.bitsPerCharacter,
      learningRate:
        model instanceof TransformerLM ? model.getScheduledLearningRate() : model.getLearningRate(),
      seconds: (Date.now() - t0) / 1000,
      improved
    };
    history.push(report);
    log(
      `Epoch ${epoch}/${epochs}: train loss ${report.trainLoss.toFixed(3)} · validation loss ${val.loss.toFixed(3)} · ppl ${val.perplexity.toFixed(2)} · ${val.bitsPerCharacter.toFixed(3)} bpc · ${report.seconds.toFixed(1)}s${improved ? ' · best' : ''}`
    );
    await callbacks.onEpoch?.(report, model);
    if (patience > 0 && best && epoch - best.epoch >= patience) {
      log(`Early stopping: no validation improvement for ${patience} epochs`);
      break;
    }
  }
  model.setTrainingProgressCallback(null);

  const bestEntry = history[best!.epoch - 1];
  const card: ModelCard = {
    name: config.name,
    description: config.description,
    createdAt: new Date().toISOString(),
    architecture: architectureOf(model),
    parameters: model.getParametersCount(),
    vocabularySize: model.getVocabSize(),
    contextSize: model.getContextSize(),
    tokenizer: tokenizerSummary,
    data: {
      documents: corpus.stats.documents,
      trainParagraphs: corpus.stats.trainParagraphs,
      validationParagraphs: corpus.stats.validationParagraphs,
      trainCharacters: corpus.stats.trainCharacters,
      validationCharacters: corpus.stats.validationCharacters,
      duplicateParagraphsRemoved: corpus.stats.duplicateParagraphs
    },
    training: {
      epochsRun: history.length,
      epochsPlanned: epochs,
      bestEpoch: best!.epoch,
      learningRate: config.training.learningRate,
      optimizer: config.training.optimizer ?? 'adam',
      warmupSteps: config.training.warmupSteps ?? 0,
      seed: config.training.seed ?? 1337,
      hiddenSize: config.model.hiddenSize,
      contextSize: config.model.contextSize,
      ...(config.model.transformer ? { transformer: config.model.transformer } : {}),
      durationSeconds: Math.round((Date.now() - started) / 1000)
    },
    metrics: {
      validationLoss: bestEntry.validationLoss,
      validationPerplexity: bestEntry.validationPerplexity,
      validationBitsPerCharacter: bestEntry.validationBitsPerCharacter,
      trainLoss: bestEntry.trainLoss,
      history: history.map((h) => ({
        epoch: h.epoch,
        trainLoss: Number(h.trainLoss.toFixed(4)),
        validationLoss: Number(h.validationLoss.toFixed(4)),
        validationBitsPerCharacter: Number(h.validationBitsPerCharacter.toFixed(4))
      }))
    },
    provenance: OFFLINE_PROVENANCE
  };

  // The returned model and package are the best checkpoint (lowest validation loss).
  const pkg = packModel(best!.json, { precision: config.precision ?? 'float32', card });
  const architecture = architectureOf(model);
  const bestModel =
    architecture === 'transformer'
      ? TransformerLM.fromJSON(best!.json)
      : architecture === 'advanced'
        ? AdvancedNeuralLM.fromJSON(best!.json)
        : ProNeuralLM.fromJSON(best!.json);
  return { model: bestModel, package: pkg, card, history, bestEpoch: best!.epoch };
}
