/**
 * Training loop of the Studio workspace, without React: the path of the classic UI's onTrain
 * (held-out validation, early stopping, best-epoch restore, provenance) without the experimental
 * extras (projects, information plane, EMA, WebGPU, loss masks).
 */
import { ProNeuralLM, type TokenizerConfig } from '../lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../lib/AdvancedNeuralLM';
import { TransformerLM } from '../lib/TransformerLM';
import { buildVocab } from '../lib/utils';
import { EarlyStopping, bitsPerCharacter, splitForValidation } from '../lib/validationSplit';
import { evaluateInSlices } from '../lib/evaluateInSlices';
import { recordTraining, textFingerprint, type ModelOrigin } from '../lib/modelExport';
import { restoreModel, modelArchitecture } from '../lib/modelPackage';
import { trainSubwordModel, subwordVocabulary } from '../tokenizer/subword';
import {
  DEFAULT_VALIDATION,
  MIN_VOCAB_SIZE,
  TRAINING_PROGRESS_UPDATE_MS,
  TRAINING_YIELD_INTERVAL_MS,
  defaultTrainingStride
} from '../config/constants';
import type { StudioSettings } from './settings';

export interface EpochReport {
  epoch: number;
  total: number;
  loss: number;
  accuracy: number;
  /** Bits per character on the held-out text (fast chunked score for transformers). */
  validationBpc?: number;
  tokensPerSec: number;
  ms: number;
}

export type TrainEvent =
  | { kind: 'split'; every: number; unit: 'paragraph' | 'sentence'; heldOut: number; total: number }
  | { kind: 'no-split' }
  | { kind: 'tokenizer'; subwords: number }
  | { kind: 'init'; architecture: string; vocab: number; parameters: number }
  | { kind: 'continue'; epochsBefore: number }
  | { kind: 'early-stop'; epoch: number; patience: number }
  | { kind: 'kept'; epoch: number; of: number; bpc: number }
  | { kind: 'best'; epoch: number; of: number; bpc: number };

export interface TrainHooks {
  signal: AbortSignal;
  onEvent?: (event: TrainEvent) => void;
  /** Fraction of the run done, 0–1. */
  onProgress?: (fraction: number) => void;
  onEpoch?: (report: EpochReport, model: ProNeuralLM) => void;
}

export interface TrainRequest {
  text: string;
  settings: StudioSettings;
  /** 'continue' trains `current` further (it must exist); 'new' builds a fresh model. */
  mode: 'new' | 'continue';
  current: ProNeuralLM | null;
  origin: ModelOrigin | undefined;
}

export interface TrainOutcome {
  model: ProNeuralLM;
  origin: ModelOrigin;
  epochsRun: number;
  reason: 'complete' | 'early-stop' | 'stopped';
  /** Epoch whose weights were kept when it was not the last one. */
  keptEpoch: number | null;
  validationBpc: number | null;
  loss: number | null;
  accuracy: number | null;
  durationMs: number;
}

export class TrainError extends Error {
  constructor(
    readonly code: 'empty' | 'vocab' | 'no-model' | 'vocab-mismatch',
    message: string
  ) {
    super(message);
  }
}

function tokenizerFor(settings: StudioSettings): TokenizerConfig {
  if (settings.tokenizer === 'bpe') return { mode: 'bpe', bpeVocabSize: settings.bpeVocabSize };
  return { mode: settings.tokenizer };
}

/** A new, untrained model of the requested architecture over `vocab`. */
export function createModel(
  settings: StudioSettings,
  vocab: string[],
  tokenizer: TokenizerConfig
): ProNeuralLM {
  const s = settings;
  const dropout = Math.min(0.5, Math.max(0, s.dropout));
  if (s.architecture === 'transformer') {
    return new TransformerLM(
      vocab,
      s.hiddenSize,
      s.learningRate,
      s.contextSize,
      s.optimizer,
      s.momentum,
      dropout,
      s.seed,
      tokenizer,
      {
        numLayers: s.transformer.numLayers,
        numHeads: s.transformer.numHeads,
        numKVHeads: s.transformer.numKVHeads,
        ffHiddenDim: s.transformer.ffHiddenDim,
        attentionDropout: s.transformer.attentionDropout,
        dropConnectRate: s.transformer.dropConnectRate,
        trainingStride: defaultTrainingStride(s.contextSize)
      }
    );
  }
  if (s.architecture === 'advanced') {
    return new AdvancedNeuralLM(
      vocab,
      s.hiddenSize,
      s.learningRate,
      s.contextSize,
      s.optimizer,
      s.momentum,
      dropout,
      s.seed,
      tokenizer,
      {
        activation: s.advanced.activation,
        useLayerNorm: s.advanced.useLayerNorm,
        lrSchedule: s.advanced.lrSchedule,
        weightDecay: s.advanced.weightDecay,
        gradientClipNorm: s.advanced.gradientClipNorm,
        initialization: 'he'
      }
    );
  }
  return new ProNeuralLM(
    vocab,
    s.hiddenSize,
    s.learningRate,
    s.contextSize,
    s.optimizer,
    s.momentum,
    dropout,
    s.seed,
    tokenizer
  );
}

function cloneJson(model: ProNeuralLM): Record<string, unknown> {
  return JSON.parse(JSON.stringify(model.toJSON())) as Record<string, unknown>;
}

export async function trainModel(req: TrainRequest, hooks: TrainHooks): Promise<TrainOutcome> {
  const { text, settings, signal } = { ...req, signal: hooks.signal };
  const emit = (e: TrainEvent) => hooks.onEvent?.(e);
  if (!text.trim()) throw new TrainError('empty', 'No training text');
  if (req.mode === 'continue' && !req.current) throw new TrainError('no-model', 'No model');

  const split = settings.validation
    ? splitForValidation(text, { fraction: DEFAULT_VALIDATION.fraction })
    : null;
  const fitText = split?.train ?? text;
  if (split) {
    emit({
      kind: 'split',
      every: split.every,
      unit: split.unit,
      heldOut: split.validation.length,
      total: text.length
    });
  } else if (settings.validation) {
    emit({ kind: 'no-split' });
  }

  let model: ProNeuralLM;
  const continued = req.mode === 'continue';
  if (continued) {
    model = req.current!;
    const tokenizer = model.getTokenizerConfig();
    const vocab = buildVocab(fitText, tokenizer);
    // Word and character vocabularies come from the text: a different text means new tokens the
    // model has no embedding for. Subword vocabularies do not depend on the text.
    if (tokenizer.mode !== 'bpe' && vocab.join('␟') !== model.getVocabSignature()) {
      throw new TrainError('vocab-mismatch', 'The text has tokens that the model does not know');
    }
    emit({ kind: 'continue', epochsBefore: model.getTrainingHistory().length });
  } else {
    let tokenizer = tokenizerFor(settings);
    if (tokenizer.mode === 'bpe') {
      const bpe = trainSubwordModel(fitText, { vocabSize: settings.bpeVocabSize });
      tokenizer = { mode: 'bpe', bpe, bpeVocabSize: settings.bpeVocabSize };
      emit({ kind: 'tokenizer', subwords: subwordVocabulary(bpe).length });
    }
    const vocab = buildVocab(fitText, tokenizer);
    if (vocab.length < MIN_VOCAB_SIZE) {
      throw new TrainError('vocab', `At least ${MIN_VOCAB_SIZE} distinct tokens are needed`);
    }
    model = createModel(settings, vocab, tokenizer);
    emit({
      kind: 'init',
      architecture: modelArchitecture(model),
      vocab: vocab.length,
      parameters: model.getParametersCount()
    });
  }

  const fingerprint = textFingerprint(text);
  let origin = recordTraining(req.origin, { continued, text, epochs: 0, fingerprint });
  const total = Math.max(1, Math.round(settings.epochs));
  if (model instanceof AdvancedNeuralLM) {
    model.setTrainingHorizon(total);
  }
  model.setGPUOps(null);
  model.setInteractiveTraining(TRAINING_YIELD_INTERVAL_MS, signal);
  const totalTokens = ProNeuralLM.tokenizeText(fitText, model.getTokenizerConfig()).length;

  const started = Date.now();
  const earlyStopping = new EarlyStopping(DEFAULT_VALIDATION.patience);
  let best: { json: Record<string, unknown>; loss: number; accuracy: number; bpc: number } | null =
    null;
  let last: { loss: number; accuracy: number; bpc?: number } | null = null;
  let epochsRun = 0;
  let reason: TrainOutcome['reason'] = 'complete';
  let lastProgress = 0;

  try {
    for (let e = 0; e < total; e++) {
      if (signal.aborted) break;
      const epochStart = Date.now();
      model.setTrainingProgressCallback((done, windows) => {
        const now = Date.now();
        if (done < windows && now - lastProgress < TRAINING_PROGRESS_UPDATE_MS) return;
        lastProgress = now;
        hooks.onProgress?.((e + done / windows) / total);
      });
      const res = await model.train(fitText, 1);
      // An epoch interrupted by Stop is not recorded by the model.
      if (signal.aborted) break;
      epochsRun = e + 1;
      const ms = Date.now() - epochStart;

      let bpc: number | undefined;
      let stopEarly = false;
      if (split) {
        const v = await evaluateInSlices(model, split.validation, signal, { fast: true });
        if (v) {
          bpc = bitsPerCharacter(v.loss, v.tokens, split.validation.length);
          if (earlyStopping.record(e + 1, v.loss)) {
            best = { json: cloneJson(model), loss: res.loss, accuracy: res.accuracy, bpc };
          }
          stopEarly = earlyStopping.shouldStop();
        }
      }
      last = { loss: res.loss, accuracy: res.accuracy, bpc };
      origin = recordTraining(req.origin, { continued, text, epochs: epochsRun, fingerprint });
      hooks.onEpoch?.(
        {
          epoch: e + 1,
          total,
          loss: res.loss,
          accuracy: res.accuracy,
          validationBpc: bpc,
          tokensPerSec: ms > 0 ? (totalTokens * 1000) / ms : 0,
          ms
        },
        model
      );
      hooks.onProgress?.((e + 1) / total);
      await new Promise((r) => setTimeout(r, 0));
      if (stopEarly) {
        emit({ kind: 'early-stop', epoch: e + 1, patience: DEFAULT_VALIDATION.patience });
        reason = 'early-stop';
        break;
      }
    }
  } finally {
    model.setInteractiveTraining(0, null);
    model.setTrainingProgressCallback(null);
  }
  if (signal.aborted) reason = 'stopped';

  // Keep the best held-out epoch (also when stopped by hand), reported with its exact score.
  let keptEpoch: number | null = null;
  let validationBpc = last?.bpc ?? null;
  let loss = last?.loss ?? null;
  let accuracy = last?.accuracy ?? null;
  const bestEpoch = earlyStopping.best;
  if (split && best && bestEpoch) {
    if (bestEpoch.epoch < epochsRun) {
      model = restoreModel(best.json);
      keptEpoch = bestEpoch.epoch;
      loss = best.loss;
      accuracy = best.accuracy;
    }
    validationBpc = best.bpc;
    if (!signal.aborted) {
      const exact = await evaluateInSlices(model, split.validation, signal);
      if (exact)
        validationBpc = bitsPerCharacter(exact.loss, exact.tokens, split.validation.length);
    }
    emit({
      kind: keptEpoch !== null ? 'kept' : 'best',
      epoch: bestEpoch.epoch,
      of: epochsRun,
      bpc: validationBpc
    });
  }

  return {
    model,
    origin,
    epochsRun,
    reason,
    keptEpoch,
    validationBpc,
    loss,
    accuracy,
    durationMs: Date.now() - started
  };
}
