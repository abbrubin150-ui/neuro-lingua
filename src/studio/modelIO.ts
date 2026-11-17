/**
 * Model storage, import / export and text generation for the Studio. The Studio uses the same
 * browser storage slots as the classic UI, so a model saved in one opens in the other.
 */
import {
  ProNeuralLM,
  MODEL_STORAGE_KEY,
  MODEL_VERSION,
  TRANSFORMER_MODEL_STORAGE_KEY
} from '../lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../lib/AdvancedNeuralLM';
import { TransformerLM } from '../lib/TransformerLM';
import { StorageManager } from '../lib/storage';
import { STORAGE_KEYS } from '../config/constants';
import { parseModelImport } from '../lib/modelImport';
import { restoreModel, modelArchitecture } from '../lib/modelPackage';
import {
  originFromImport,
  packageBrowserModel,
  packageFileName,
  type ModelOrigin
} from '../lib/modelExport';
import { recommendedNoRepeatNgram } from '../generation/sampling';
import { DEFAULT_GENERATION } from '../config/constants';
import type { ModelMetaStore } from '../types/modelMeta';
import type { Architecture, StudioSettings } from './settings';

export type StorageSlot = 'mlp' | 'transformer';
export type SaveEncoding = 'json' | 'float32' | 'float16' | null;

export function slotOf(architecture: Architecture): StorageSlot {
  return architecture === 'transformer' ? 'transformer' : 'mlp';
}

function keyOf(slot: StorageSlot): string {
  return slot === 'transformer' ? TRANSFORMER_MODEL_STORAGE_KEY : MODEL_STORAGE_KEY;
}

export interface StoredModelInfo {
  slot: StorageSlot;
  present: boolean;
  timestamp?: number;
  vocab?: number;
}

function readMetaStore(): ModelMetaStore {
  const store = StorageManager.get<ModelMetaStore | null>(STORAGE_KEYS.MODEL_META, null);
  return store && typeof store === 'object' ? store : {};
}

/** What is stored in each slot (without restoring the weights). */
export function describeStoredModels(): StoredModelInfo[] {
  const meta = readMetaStore();
  return (['transformer', 'mlp'] as StorageSlot[]).map((slot) => {
    let present = false;
    try {
      present = localStorage.getItem(keyOf(slot)) !== null;
    } catch {
      present = false;
    }
    const m = slot === 'transformer' ? meta.transformer : (meta.advanced ?? meta.feedforward);
    return { slot, present, timestamp: present ? m?.timestamp : undefined, vocab: m?.vocab };
  });
}

export function loadStoredModel(
  slot: StorageSlot
): { model: ProNeuralLM; origin?: ModelOrigin } | null {
  let model: ProNeuralLM | null = null;
  if (slot === 'transformer') {
    model = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
  } else {
    // The serialized `architecture` field decides which class restores an MLP.
    model =
      AdvancedNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY) ??
      ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);
  }
  if (!model) return null;
  const origin = readMetaStore()[modelArchitecture(model)]?.origin;
  return { model, origin };
}

/** Store the model in its slot and record its metadata (shared with the classic UI). */
export function storeModel(
  model: ProNeuralLM,
  origin: ModelOrigin | undefined,
  stats: { loss?: number | null; accuracy?: number | null } = {}
): SaveEncoding {
  const architecture = modelArchitecture(model);
  const encoding = model.saveToLocalStorage(keyOf(slotOf(architecture)));
  if (encoding === null) return null;
  const meta = readMetaStore();
  const previous = meta[architecture];
  meta[architecture] = {
    architecture,
    timestamp: model.getLastUpdatedAt() ?? Date.now(),
    vocab: model.getVocabSize(),
    loss: stats.loss ?? previous?.loss,
    accuracy: stats.accuracy ?? previous?.accuracy,
    perplexity:
      typeof stats.loss === 'number' ? Math.exp(Math.max(1e-8, stats.loss)) : previous?.perplexity,
    origin
  };
  StorageManager.set(STORAGE_KEYS.MODEL_META, meta);
  return encoding;
}

export function deleteStoredModel(slot: StorageSlot): void {
  StorageManager.remove(keyOf(slot));
  const meta = readMetaStore();
  if (slot === 'transformer') delete meta.transformer;
  else {
    delete meta.feedforward;
    delete meta.advanced;
  }
  StorageManager.set(STORAGE_KEYS.MODEL_META, meta);
}

export interface ImportedModel {
  model: ProNeuralLM;
  origin: ModelOrigin;
  format: string;
  name?: string;
}

/** Restore a model from the text of a file (.nlm.json package, plain or trace JSON export). */
export function importModelText(text: string, fileName: string): ImportedModel {
  const parsed = parseModelImport(JSON.parse(text));
  const model = restoreModel(parsed.modelData, parsed.architecture);
  const name = typeof parsed.card?.name === 'string' ? parsed.card.name : undefined;
  return { model, origin: originFromImport(fileName, parsed.card), format: parsed.format, name };
}

/** A compact package with a model card: loads here, in the classic UI and in `pnpm lm`. */
export function exportPackage(
  model: ProNeuralLM,
  origin: ModelOrigin | undefined
): { blob: Blob; text: string; fileName: string; bytes: number } {
  const date = new Date().toISOString().slice(0, 10);
  const name = `neuro-lingua-${modelArchitecture(model)}-${date}`;
  const text = JSON.stringify(
    packageBrowserModel(model, origin, { name, appVersion: MODEL_VERSION })
  );
  return {
    blob: new Blob([text], { type: 'application/json' }),
    text,
    fileName: packageFileName(name),
    bytes: text.length
  };
}

/** The full `toJSON()` (optimizer state included). */
export function exportJson(model: ProNeuralLM): { blob: Blob; fileName: string } {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  return {
    blob: new Blob([JSON.stringify(model.toJSON())], { type: 'application/json' }),
    fileName: `neuro-lingua-${modelArchitecture(model)}-${stamp}.json`
  };
}

export function noRepeatDefault(model: ProNeuralLM): number {
  return recommendedNoRepeatNgram(model.getTokenizerConfig().mode, model.getVocab());
}

export interface GenerationResult {
  text: string;
  tokens: number;
  finishReason: 'eos' | 'length';
  ms: number;
}

export async function generateText(
  model: ProNeuralLM,
  prompt: string,
  g: StudioSettings['generation']
): Promise<GenerationResult> {
  const started = Date.now();
  // Greedy decoding is top-k sampling with k = 1.
  const topK = g.sampling === 'greedy' ? 1 : g.sampling === 'topk' ? g.topK : 0;
  const topP = g.sampling === 'topp' ? g.topP : 0;
  const tau = g.sampling === 'typical' ? g.typicalTau : 0;
  const result = await model.generateDetailed(
    prompt,
    g.maxTokens,
    g.temperature,
    topK,
    topP,
    0,
    0,
    tau,
    0,
    0.1,
    {
      repetitionPenalty: 1,
      repetitionWindow: DEFAULT_GENERATION.repetitionWindow,
      noRepeatNgramSize: g.noRepeatNgram ?? noRepeatDefault(model)
    }
  );
  return { ...result, ms: Date.now() - started };
}

export interface ModelSummary {
  architecture: Architecture;
  vocab: number;
  parameters: number;
  hiddenSize: number;
  contextSize: number;
  tokenizer: string;
  epochsTrained: number;
  layers?: number;
  heads?: number;
  kvHeads?: number;
}

export function summarizeModel(model: ProNeuralLM): ModelSummary {
  const summary: ModelSummary = {
    architecture: modelArchitecture(model),
    vocab: model.getVocabSize(),
    parameters: model.getParametersCount(),
    hiddenSize: model.getHiddenSize(),
    contextSize: model.getContextSize(),
    tokenizer: model.getTokenizerConfig().mode,
    epochsTrained: model.getTrainingHistory().length
  };
  if (model instanceof TransformerLM) {
    const info = model.getTransformerInfo();
    summary.layers = info.numLayers;
    summary.heads = info.numHeads;
    summary.kvHeads = info.numKVHeads;
  }
  return summary;
}
