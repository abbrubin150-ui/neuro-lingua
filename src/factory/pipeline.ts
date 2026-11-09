/**
 * Offline model factory: local documents → cleaned corpus → tokenizer → trained model →
 * validated, packaged model with a model card.
 *
 * Everything runs in-process on the CPU. No network request is made and no external AI model or
 * service is used at any step; the only inputs are the documents and the configuration.
 * The pipeline is environment-agnostic (Node CLI or browser); file access lives in the callers.
 */

import {
  fnv1a,
  prepareCorpus,
  type CorpusDocument,
  type CorpusPrepOptions,
  type PreparedCorpus
} from '../data/corpusPrep';
import { trainSubwordModel, subwordVocabulary } from '../tokenizer/subword';
import { answerF1, qaLossMask, qaPairsFromText, sameAnswer, type QaTemplate } from '../data/qa';
import { answerQuestion } from './answer';
import {
  ProNeuralLM,
  type EvaluationResult,
  type Optimizer,
  type RandomState,
  type TokenizerConfig
} from '../lib/ProNeuralLM';
import { AdvancedNeuralLM, type AdvancedConfig } from '../lib/AdvancedNeuralLM';
import {
  TransformerLM,
  type TransformerConfig,
  type TransformerTrainingState
} from '../lib/TransformerLM';
import {
  packModel,
  restoreModel,
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
  /**
   * Question–answer training: the documents hold one pair per paragraph (see src/data/qa.ts)
   * and the loss covers only the answers (and where they stop).
   */
  qa?: QaTemplate;
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

/**
 * Everything besides the model (whose toJSON carries the weights, random stream and history)
 * needed to continue a run exactly after an epoch.
 */
export interface FactoryCheckpoint {
  version: 1;
  epochsDone: number;
  history: EpochReport[];
  best: { epoch: number; loss: number };
  elapsedSeconds: number;
  /** Fingerprint of the prepared training and validation text: a resumed run needs the same. */
  dataFingerprint: string;
  tokenizerSummary: Record<string, unknown>;
  /** Transformer optimizer moments and learning-rate schedule position. */
  modelTraining?: TransformerTrainingState;
  /** Random streams of data-parallel replicas. */
  replicaRandomStates?: RandomState[];
}

export interface FactoryCallbacks {
  onLog?: (message: string) => void;
  onProgress?: (done: number, total: number, epoch: number) => void;
  /** After every epoch, with the state needed to resume the run from this point. */
  onEpoch?: (
    report: EpochReport,
    model: ProNeuralLM,
    checkpoint: FactoryCheckpoint
  ) => void | Promise<void>;
}

/**
 * Where the epochs run. The default trains in-process; the Node CLI can supply a data-parallel
 * backend that spreads every mini-batch over worker processes (src/factory/node/parallel.ts).
 */
export interface TrainingBackend {
  /** Shown in the log, e.g. "in-process" or "4 worker processes". */
  readonly description: string;
  /** Train one epoch on the training text; returns the epoch's training loss. */
  trainEpoch(
    epoch: number,
    onProgress?: (done: number, total: number, epoch: number) => void
  ): Promise<{ loss: number }>;
  /** Evaluate the current model on the validation text. */
  evaluate(): Promise<EvaluationResult>;
  /** Random streams of the backend's own workers, if it has any (for checkpoints). */
  randomStates?(): Promise<RandomState[]>;
  dispose(): Promise<void>;
}

export interface TrainingBackendSetup {
  model: ProNeuralLM;
  trainText: string;
  validationText: string;
  /** Per-token loss masks (Q&A training); absent: every token counts. */
  trainMask?: number[];
  validationMask?: number[];
  seed: number;
  /** Continue the workers' random streams (resuming a run). */
  randomStates?: RandomState[];
}

export type TrainingBackendFactory = (setup: TrainingBackendSetup) => Promise<TrainingBackend>;

/** Train and evaluate on the calling thread. */
export function inProcessBackend({
  model,
  trainText,
  validationText,
  trainMask,
  validationMask
}: TrainingBackendSetup): TrainingBackend {
  return {
    description: 'in-process',
    async trainEpoch(_epoch, onProgress) {
      model.setTrainingProgressCallback(onProgress ?? null);
      const result = await model.train(trainText, 1, trainMask);
      model.setTrainingProgressCallback(null);
      return { loss: result.history[result.history.length - 1]?.loss ?? result.loss };
    },
    evaluate: () => model.evaluate(validationText, validationMask),
    dispose: async () => undefined
  };
}

export interface FactoryOptions {
  /** Training backend (default: in-process). */
  backend?: TrainingBackendFactory;
  /**
   * Continue an interrupted run: the model as it was after `checkpoint.epochsDone` epochs and the
   * best model so far. The documents and configuration must be those of the original run.
   */
  resume?: {
    model: ProNeuralLM;
    checkpoint: FactoryCheckpoint;
    bestModel: Record<string, unknown>;
  };
}

/** Fingerprint of a prepared corpus (both splits). */
export function corpusFingerprint(corpus: PreparedCorpus): string {
  const text = `${corpus.train}\u0000${corpus.validation}`;
  const hex = (n: number) => n.toString(16).padStart(8, '0');
  return hex(fnv1a(text)) + hex(fnv1a(text, 0x9e3779b9));
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

/** Greedy answers to the validation questions compared with the reference answers. */
async function qaExactMatch(model: ProNeuralLM, validation: string, template: QaTemplate) {
  const pairs = qaPairsFromText(validation, template);
  const config = model.getTokenizerConfig();
  let correct = 0;
  let f1 = 0;
  for (const pair of pairs) {
    const answerTokens = ProNeuralLM.tokenizeText(pair.answer, config).length;
    const { text } = await answerQuestion(model, pair.question, template, {
      maxTokens: answerTokens * 2 + 8,
      temperature: 0.1,
      topK: 1
    });
    if (sameAnswer(text, pair.answer)) correct++;
    f1 += answerF1(text, pair.answer);
  }
  const n = pairs.length;
  return { correct, total: n, rate: n ? correct / n : 0, f1: n ? f1 / n : 0 };
}

/** Answer-only loss masks of the training and validation texts (Q&A training). */
function qaMasks(model: ProNeuralLM, train: string, validation: string, template: QaTemplate) {
  const config = model.getTokenizerConfig();
  const t = qaLossMask(train, ProNeuralLM.tokenizeWithOffsets(train, config), template);
  const v = qaLossMask(validation, ProNeuralLM.tokenizeWithOffsets(validation, config), template);
  const answers = (text: string) =>
    text.split('\n\n').filter((p) => p.includes(template.answerLabel)).length;
  return {
    train: t.mask,
    validation: v.mask,
    validationCharacters: v.answerCharacters,
    trainAnswers: answers(train),
    validationAnswers: answers(validation)
  };
}

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
  return withBitsPerCharacter(await model.evaluate(text), text);
}

/**
 * Add bits per character to an evaluation: the total information over the characters of the
 * text (or a given number of characters, e.g. those of the answers in Q&A evaluation).
 */
export function withBitsPerCharacter(
  result: EvaluationResult,
  text: string | number
): { loss: number; perplexity: number; bitsPerCharacter: number; tokens: number } {
  const characters = Math.max(1, typeof text === 'number' ? text : text.length);
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
  initialModel?: ProNeuralLM,
  options: FactoryOptions = {}
): Promise<FactoryResult> {
  const log = callbacks.onLog ?? (() => undefined);
  const resume = options.resume;
  if (resume && initialModel) throw new Error('Resume a run or start from a model, not both');
  const started = Date.now() - (resume?.checkpoint.elapsedSeconds ?? 0) * 1000;

  // Q&A pairs are kept even when short.
  const corpus = prepareCorpus(documents, {
    ...(config.qa ? { minParagraphChars: 1 } : {}),
    ...config.data
  });
  if (corpus.train.length === 0) throw new Error('No training text left after preparation');
  if (corpus.validation.length === 0) throw new Error('No validation text: add more documents');
  log(
    `Data: ${corpus.stats.documents} documents, ${corpus.stats.trainParagraphs} training / ${corpus.stats.validationParagraphs} validation paragraphs, ${corpus.stats.duplicateParagraphs} duplicates removed`
  );
  const dataFingerprint = corpusFingerprint(corpus);

  let model: ProNeuralLM;
  let tokenizerSummary: Record<string, unknown>;
  if (resume) {
    if (resume.checkpoint.dataFingerprint !== dataFingerprint) {
      throw new Error(
        'The prepared training data differs from the interrupted run: resume needs the same documents and data options'
      );
    }
    model = resume.model;
    tokenizerSummary = resume.checkpoint.tokenizerSummary;
    log(`Resuming after epoch ${resume.checkpoint.epochsDone}`);
  } else if (initialModel) {
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
  // After the schedule is configured (which restarts it): continue where the run stopped.
  if (resume?.checkpoint.modelTraining && model instanceof TransformerLM) {
    model.setTrainingState(resume.checkpoint.modelTraining);
  }

  const patience = Math.max(0, Math.floor(config.training.patience ?? 0));
  const history: EpochReport[] = resume ? resume.checkpoint.history.slice() : [];
  let best: { loss: number; epoch: number; json: Record<string, unknown> } | null = resume
    ? { ...resume.checkpoint.best, json: resume.bestModel }
    : null;
  const firstEpoch = (resume?.checkpoint.epochsDone ?? 0) + 1;
  const stoppedEarly = (b: typeof best, epoch: number) =>
    patience > 0 && !!b && epoch - b.epoch >= patience;

  // Q&A: the loss covers the answers only; bits per character are per answer character.
  const masks = config.qa ? qaMasks(model, corpus.train, corpus.validation, config.qa) : null;
  if (masks) {
    log(
      `Q&A training: ${masks.trainAnswers} training / ${masks.validationAnswers} validation answers; loss on answers only`
    );
  }
  const validationCharacters = masks ? masks.validationCharacters : corpus.validation.length;

  const backend = await (options.backend ?? (async (setup) => inProcessBackend(setup)))({
    model,
    trainText: corpus.train,
    validationText: corpus.validation,
    ...(masks ? { trainMask: masks.train, validationMask: masks.validation } : {}),
    seed: config.training.seed ?? 1337,
    ...(resume?.checkpoint.replicaRandomStates
      ? { randomStates: resume.checkpoint.replicaRandomStates }
      : {})
  });
  const backendDescription = backend.description;
  if (backendDescription !== 'in-process') log(`Training on ${backendDescription}`);
  try {
    const alreadyStopped = resume !== undefined && stoppedEarly(best, firstEpoch - 1);
    for (let epoch = firstEpoch; epoch <= epochs && !alreadyStopped; epoch++) {
      const t0 = Date.now();
      const result = await backend.trainEpoch(epoch - 1, callbacks.onProgress);
      const val = withBitsPerCharacter(await backend.evaluate(), validationCharacters);
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
        trainLoss: result.loss,
        validationLoss: val.loss,
        validationPerplexity: val.perplexity,
        validationBitsPerCharacter: val.bitsPerCharacter,
        learningRate:
          model instanceof TransformerLM
            ? model.getScheduledLearningRate()
            : model.getLearningRate(),
        seconds: (Date.now() - t0) / 1000,
        improved
      };
      history.push(report);
      log(
        `Epoch ${epoch}/${epochs}: train loss ${report.trainLoss.toFixed(3)} · validation loss ${val.loss.toFixed(3)} · ppl ${val.perplexity.toFixed(2)} · ${val.bitsPerCharacter.toFixed(3)} bpc · ${report.seconds.toFixed(1)}s${improved ? ' · best' : ''}`
      );
      if (callbacks.onEpoch) {
        const checkpoint: FactoryCheckpoint = {
          version: 1,
          epochsDone: epoch,
          history: history.slice(),
          best: { epoch: best!.epoch, loss: best!.loss },
          elapsedSeconds: (Date.now() - started) / 1000,
          dataFingerprint,
          tokenizerSummary,
          ...(model instanceof TransformerLM ? { modelTraining: model.getTrainingState() } : {}),
          ...(backend.randomStates ? { replicaRandomStates: await backend.randomStates() } : {})
        };
        await callbacks.onEpoch(report, model, checkpoint);
      }
      if (stoppedEarly(best, epoch)) {
        log(`Early stopping: no validation improvement for ${patience} epochs`);
        break;
      }
    }
  } finally {
    await backend.dispose();
  }

  const bestEntry = history[best!.epoch - 1];
  // The returned model and package are the best checkpoint (lowest validation loss).
  const bestModel = restoreModel(best!.json, architectureOf(model));
  // Q&A: answers generated freely (greedy, until the end-of-answer token) for the validation
  // questions, compared with the reference answers. Teacher-forced loss alone cannot show
  // whether the model picks the right answer.
  const exactMatch = config.qa ? await qaExactMatch(bestModel, corpus.validation, config.qa) : null;
  if (exactMatch) {
    log(
      `Validation answers generated exactly right: ${exactMatch.correct}/${exactMatch.total} (${(100 * exactMatch.rate).toFixed(1)}%), word-overlap F1 ${exactMatch.f1.toFixed(2)}`
    );
  }
  const card: ModelCard = {
    name: config.name,
    description: config.description,
    createdAt: new Date().toISOString(),
    architecture: architectureOf(model),
    parameters: model.getParametersCount(),
    vocabularySize: model.getVocabSize(),
    contextSize: model.getContextSize(),
    tokenizer: tokenizerSummary,
    ...(config.qa
      ? {
          chat: {
            format: 'qa' as const,
            questionLabel: config.qa.questionLabel,
            answerLabel: config.qa.answerLabel
          }
        }
      : {}),
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
      execution: backendDescription,
      durationSeconds: Math.round((Date.now() - started) / 1000)
    },
    metrics: {
      validationLoss: bestEntry.validationLoss,
      validationPerplexity: bestEntry.validationPerplexity,
      validationBitsPerCharacter: bestEntry.validationBitsPerCharacter,
      ...(config.qa ? { validationBitsPerCharacterScope: 'answers' } : {}),
      ...(exactMatch
        ? {
            validationExactMatch: Number(exactMatch.rate.toFixed(4)),
            validationAnswerF1: Number(exactMatch.f1.toFixed(4)),
            validationQuestions: exactMatch.total
          }
        : {}),
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

  const pkg = packModel(best!.json, { precision: config.precision ?? 'float32', card });
  return { model: bestModel, package: pkg, card, history, bestEpoch: best!.epoch };
}
