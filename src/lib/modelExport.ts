/**
 * Model cards and packages for models trained in the browser app.
 *
 * The card only states what the app knows: the model's own size and tokenizer, the training
 * history recorded by the model, a fingerprint of the text of the last training run, and where the
 * weights came from. The browser app never calls an external AI model or service, so a model it
 * initialised itself is "offline"; a model that started from an imported file inherits that file's
 * provenance, which is unknown when the file carried no model card.
 */

import { fnv1a } from '../data/corpusPrep';
import { subwordVocabulary } from '../tokenizer/subword';
import type { ProNeuralLM, TokenizerConfig } from './ProNeuralLM';
import { modelArchitecture } from './modelPackage';
import {
  packModel,
  type ModelCard,
  type ModelPackage,
  type TensorPrecision
} from './modelPackageCodec';

/** Model the in-app weights started from (absent: initialised by the app itself). */
export interface ModelBase {
  /** File name or description of the starting model. */
  name: string;
  /** Provenance recorded for the starting model; null when unknown. */
  offline: boolean | null;
  externalServices: string[];
  statement?: string;
}

export interface ModelOrigin {
  base?: ModelBase;
  /** Most recent training run in the app. */
  training?: {
    characters: number;
    /** Fingerprint of the training text (see textFingerprint). */
    fingerprint: string;
    epochs: number;
    finishedAt: string;
  };
}

export const BROWSER_PROVENANCE_STATEMENT =
  'Trained in the browser by Neuro-Lingua from the text in its training corpus field. No external AI model, API or network service was used for tokenization, training or evaluation.';

const APP_STATEMENT = 'The browser app itself uses no external AI model, API or network service.';

/** Starting point of a model whose history was not recorded (saved by an older version). */
export const UNRECORDED_BASE: ModelBase = {
  name: 'a model saved without provenance information',
  offline: null,
  externalServices: []
};

/** 64-bit content fingerprint (two FNV-1a passes); identifies a text without storing it. */
export function textFingerprint(text: string): string {
  const hex = (n: number) => n.toString(16).padStart(8, '0');
  return hex(fnv1a(text)) + hex(fnv1a(text, 0x9e3779b9));
}

/** Origin of a model imported from a file (a package's model card supplies its provenance). */
export function originFromImport(fileName: string, card?: ModelCard): ModelOrigin {
  if (!card) return { base: { name: fileName, offline: null, externalServices: [] } };
  return {
    base: {
      name: card.name || fileName,
      offline: typeof card.provenance?.offline === 'boolean' ? card.provenance.offline : null,
      externalServices: Array.isArray(card.provenance?.externalServices)
        ? card.provenance.externalServices.map(String)
        : [],
      ...(typeof card.provenance?.statement === 'string'
        ? { statement: card.provenance.statement }
        : {})
    }
  };
}

/**
 * Origin after a training run. A new model has no base; a continued model keeps the base of the
 * model it continued from (unknown when that model's origin was not recorded).
 */
export function recordTraining(
  previous: ModelOrigin | undefined,
  run: { continued: boolean; text: string; epochs: number; fingerprint?: string; at?: string }
): ModelOrigin {
  const base = run.continued ? (previous ? previous.base : UNRECORDED_BASE) : undefined;
  return {
    ...(base ? { base } : {}),
    training: {
      characters: run.text.length,
      fingerprint: run.fingerprint ?? textFingerprint(run.text),
      epochs: run.epochs,
      finishedAt: run.at ?? new Date().toISOString()
    }
  };
}

export function browserProvenance(origin: ModelOrigin | undefined): ModelCard['provenance'] {
  const base = origin ? origin.base : UNRECORDED_BASE;
  if (!base) {
    return { offline: true, externalServices: [], statement: BROWSER_PROVENANCE_STATEMENT };
  }
  const earlier =
    base.offline === null
      ? 'Its earlier history is unknown.'
      : `Its model card states: "${base.statement ?? ''}"`;
  const action = origin?.training ? 'Training continued' : 'Exported';
  return {
    // Only claim "offline" when the starting model's own card says so.
    offline: base.offline,
    externalServices: base.externalServices,
    statement: `${action} in the browser by Neuro-Lingua, starting from ${
      origin ? `"${base.name}"` : base.name
    }. ${earlier} ${APP_STATEMENT}`
  };
}

/** Short, JSON-friendly description of a tokenizer (subword merges are counted, not listed). */
export function summarizeTokenizer(config: TokenizerConfig): Record<string, unknown> {
  if (config.mode === 'bpe' && config.bpe) {
    return {
      mode: 'bpe',
      subwords: subwordVocabulary(config.bpe).length,
      merges: config.bpe.merges.length,
      lowercase: config.bpe.lowercase
    };
  }
  if (config.mode === 'custom') return { mode: 'custom', pattern: config.pattern ?? '' };
  return { mode: config.mode };
}

export interface BrowserModelDetails {
  name: string;
  description?: string;
  appVersion?: string;
}

/** Model card for the current in-app model. */
export function createBrowserModelCard(
  model: ProNeuralLM,
  origin: ModelOrigin | undefined,
  details: BrowserModelDetails
): ModelCard {
  const history = model.getTrainingHistory();
  const last = history[history.length - 1];
  const run = origin?.training;
  return {
    name: details.name,
    ...(details.description ? { description: details.description } : {}),
    createdAt: new Date().toISOString(),
    architecture: modelArchitecture(model),
    parameters: model.getParametersCount(),
    vocabularySize: model.getVocabSize(),
    contextSize: model.getContextSize(),
    tokenizer: summarizeTokenizer(model.getTokenizerConfig()),
    ...(run
      ? {
          data: {
            lastRunTrainingCharacters: run.characters,
            lastRunTextFingerprint: run.fingerprint
          }
        }
      : {}),
    training: {
      epochsRecorded: history.length,
      ...(run ? { lastRunEpochs: run.epochs, lastRunFinishedAt: run.finishedAt } : {})
    },
    ...(last
      ? {
          metrics: {
            finalEpochTrainingLoss: Number(last.loss.toFixed(4)),
            finalEpochTrainingAccuracy: Number(last.accuracy.toFixed(4)),
            note: 'Measured on the training text during the last epoch; not a held-out evaluation.'
          }
        }
      : {}),
    environment: {
      producer: 'Neuro-Lingua browser app',
      ...(details.appVersion ? { version: details.appVersion } : {})
    },
    provenance: browserProvenance(origin)
  };
}

/** Package the current model (optimizer state stripped) together with its model card. */
export function packageBrowserModel(
  model: ProNeuralLM,
  origin: ModelOrigin | undefined,
  details: BrowserModelDetails,
  precision: TensorPrecision = 'float32'
): ModelPackage {
  const card = createBrowserModelCard(model, origin, details);
  return packModel(model.toJSON() as unknown as Record<string, unknown>, { precision, card });
}

/** File name for an exported package, e.g. "neuro-lingua-transformer-2026-11-08.nlm.json". */
export function packageFileName(name: string): string {
  const slug =
    name
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 80) || 'model';
  return `${slug}.nlm.json`;
}
