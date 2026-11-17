/**
 * Settings of the Studio workspace (studio.html): the options that the Studio exposes, their
 * defaults, presets and validation. Pure functions, shared by the UI and the tests.
 *
 * The Studio keeps only the options whose effect has been measured or tested in this repository
 * (architectures, tokenizer, Adam / momentum, held-out validation, sampling with loop control).
 */
import {
  DEFAULT_ADVANCED_CONFIG,
  DEFAULT_GENERATION,
  DEFAULT_HYPERPARAMETERS,
  HYPERPARAMETER_CONSTRAINTS,
  defaultContextSize
} from '../config/constants';
import type { ActivationFunction, LRSchedule } from '../lib/AdvancedNeuralLM';

export type Architecture = 'feedforward' | 'advanced' | 'transformer';
export type StudioTokenizer = 'bpe' | 'character' | 'unicode';
export type StudioOptimizer = 'adam' | 'momentum';
export type SamplingMode = 'greedy' | 'topk' | 'topp' | 'typical';

export interface StudioSettings {
  architecture: Architecture;
  tokenizer: StudioTokenizer;
  bpeVocabSize: number;
  hiddenSize: number;
  contextSize: number;
  epochs: number;
  learningRate: number;
  optimizer: StudioOptimizer;
  momentum: number;
  dropout: number;
  seed: number;
  /** Hold out every 10th sentence / paragraph, stop early and keep the best epoch. */
  validation: boolean;
  transformer: {
    numLayers: number;
    numHeads: number;
    numKVHeads: number;
    ffHiddenDim: number;
    attentionDropout: number;
    dropConnectRate: number;
  };
  advanced: {
    activation: ActivationFunction;
    useLayerNorm: boolean;
    lrSchedule: LRSchedule;
    weightDecay: number;
    gradientClipNorm: number;
  };
  generation: {
    sampling: SamplingMode;
    temperature: number;
    topK: number;
    topP: number;
    typicalTau: number;
    maxTokens: number;
    /** null: the tokenizer-dependent default (`recommendedNoRepeatNgram`). */
    noRepeatNgram: number | null;
  };
}

export const STUDIO_SETTINGS_KEY = 'neuro-lingua-studio-settings-v1';

export const DEFAULT_STUDIO_SETTINGS: StudioSettings = {
  architecture: 'transformer',
  tokenizer: 'bpe',
  bpeVocabSize: 2000,
  hiddenSize: DEFAULT_HYPERPARAMETERS.hiddenSize,
  contextSize: defaultContextSize('transformer'),
  epochs: DEFAULT_HYPERPARAMETERS.epochs,
  learningRate: DEFAULT_HYPERPARAMETERS.learningRate,
  optimizer: 'adam',
  momentum: DEFAULT_HYPERPARAMETERS.momentum,
  dropout: DEFAULT_HYPERPARAMETERS.dropout,
  seed: DEFAULT_HYPERPARAMETERS.seed,
  validation: true,
  transformer: {
    numLayers: DEFAULT_HYPERPARAMETERS.transformer.numLayers,
    numHeads: DEFAULT_HYPERPARAMETERS.transformer.numHeads,
    numKVHeads: DEFAULT_HYPERPARAMETERS.transformer.numKVHeads,
    ffHiddenDim: DEFAULT_HYPERPARAMETERS.transformer.ffHiddenDim,
    attentionDropout: DEFAULT_HYPERPARAMETERS.transformer.attentionDropout,
    dropConnectRate: DEFAULT_HYPERPARAMETERS.transformer.dropConnectRate
  },
  advanced: {
    activation: DEFAULT_ADVANCED_CONFIG.activation,
    useLayerNorm: true,
    lrSchedule: DEFAULT_ADVANCED_CONFIG.lrSchedule,
    weightDecay: DEFAULT_ADVANCED_CONFIG.weightDecay,
    gradientClipNorm: DEFAULT_ADVANCED_CONFIG.gradientClipNorm
  },
  generation: {
    sampling: 'topp',
    temperature: DEFAULT_GENERATION.temperature,
    topK: DEFAULT_GENERATION.topK,
    topP: DEFAULT_GENERATION.topP,
    typicalTau: DEFAULT_GENERATION.typicalTau,
    maxTokens: 60,
    noRepeatNgram: null
  }
};

export type PresetId = 'transformer-tiny' | 'transformer-small' | 'mlp';

/**
 * Starting points. The transformer sizes are the `pnpm lm train` presets measured in
 * docs/benchmarks/MODEL_SIZE.md; the MLP preset is the classic UI's default.
 */
export function applyPreset(settings: StudioSettings, preset: PresetId): StudioSettings {
  if (preset === 'mlp') {
    return {
      ...settings,
      architecture: 'feedforward',
      hiddenSize: 64,
      contextSize: defaultContextSize('feedforward'),
      learningRate: 0.003,
      optimizer: 'adam'
    };
  }
  const tiny = preset === 'transformer-tiny';
  return {
    ...settings,
    architecture: 'transformer',
    hiddenSize: tiny ? 64 : 128,
    contextSize: tiny ? 32 : 64,
    learningRate: 0.003,
    optimizer: 'adam',
    transformer: {
      ...settings.transformer,
      numLayers: tiny ? 2 : 4,
      numHeads: tiny ? 4 : 8,
      numKVHeads: tiny ? 2 : 4,
      ffHiddenDim: tiny ? 128 : 512
    }
  };
}

/** Switch architecture, following its default context unless the user changed it. */
export function withArchitecture(settings: StudioSettings, next: Architecture): StudioSettings {
  if (settings.architecture === next) return settings;
  const followDefault = settings.contextSize === defaultContextSize(settings.architecture);
  return {
    ...settings,
    architecture: next,
    contextSize: followDefault ? defaultContextSize(next) : settings.contextSize
  };
}

const C = HYPERPARAMETER_CONSTRAINTS;

/** Problems that would make training fail or silently change the requested model. */
export function validateSettings(s: StudioSettings): string[] {
  const errors: string[] = [];
  const range = (name: string, v: number, min: number, max: number) => {
    if (!Number.isFinite(v) || v < min || v > max) errors.push(`${name}: ${min}–${max}`);
  };
  range('hiddenSize', s.hiddenSize, C.hiddenSize.min, C.hiddenSize.max);
  range('contextSize', s.contextSize, C.contextSize.min, C.contextSize.max);
  range('epochs', s.epochs, C.epochs.min, C.epochs.max);
  range('learningRate', s.learningRate, 0.0001, 0.1);
  range('dropout', s.dropout, C.dropout.min, C.dropout.max);
  range('momentum', s.momentum, C.momentum.min, C.momentum.max);
  if (!Number.isInteger(s.seed)) errors.push('seed: integer');
  if (s.tokenizer === 'bpe') range('bpeVocabSize', s.bpeVocabSize, 100, 8000);
  if (s.architecture === 'transformer') {
    const t = s.transformer;
    range('numLayers', t.numLayers, C.transformer.numLayers.min, C.transformer.numLayers.max);
    range('numHeads', t.numHeads, C.transformer.numHeads.min, C.transformer.numHeads.max);
    range(
      'ffHiddenDim',
      t.ffHiddenDim,
      C.transformer.ffHiddenDim.min,
      C.transformer.ffHiddenDim.max
    );
    range('attentionDropout', t.attentionDropout, 0, 0.5);
    range('dropConnectRate', t.dropConnectRate, 0, 0.5);
    if (t.numHeads > 0 && s.hiddenSize % t.numHeads !== 0) {
      errors.push('hiddenSize % numHeads ≠ 0');
    }
    if (t.numKVHeads < 1 || t.numKVHeads > t.numHeads || t.numHeads % t.numKVHeads !== 0) {
      errors.push('numHeads % numKVHeads ≠ 0');
    }
  }
  if (s.architecture === 'advanced') {
    range('weightDecay', s.advanced.weightDecay, 0, C.weightDecay.max);
    range(
      'gradientClipNorm',
      s.advanced.gradientClipNorm,
      C.gradientClipNorm.min,
      C.gradientClipNorm.max
    );
  }
  const g = s.generation;
  range('temperature', g.temperature, C.temperature.min, C.temperature.max);
  range('maxTokens', g.maxTokens, 1, 400);
  range('topK', g.topK, 1, C.topK.max);
  range('topP', g.topP, 0.05, C.topP.max);
  range('typicalTau', g.typicalTau, C.typicalTau.min, 0.99);
  if (g.noRepeatNgram !== null) range('noRepeatNgram', g.noRepeatNgram, 0, 32);
  return errors;
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Merge stored settings over the defaults, keeping only fields of the right type (stored data may
 * come from an older version or be edited by hand).
 */
export function mergeSettings(stored: unknown, base: StudioSettings = DEFAULT_STUDIO_SETTINGS) {
  const pick = <T extends Record<string, unknown>>(defaults: T, source: unknown): T => {
    if (!isObject(source)) return { ...defaults };
    const out: Record<string, unknown> = { ...defaults };
    for (const [key, value] of Object.entries(defaults)) {
      const candidate = source[key];
      if (candidate === undefined) continue;
      if (isObject(value)) out[key] = pick(value, candidate);
      else if (value === null) out[key] = typeof candidate === 'number' ? candidate : null;
      else if (typeof candidate === typeof value) out[key] = candidate;
    }
    return out as T;
  };
  const merged = pick(
    base as unknown as Record<string, unknown>,
    stored
  ) as unknown as StudioSettings;
  const oneOf = <T extends string>(v: T, allowed: readonly T[], fallback: T): T =>
    allowed.includes(v) ? v : fallback;
  merged.architecture = oneOf(
    merged.architecture,
    ['feedforward', 'advanced', 'transformer'],
    base.architecture
  );
  merged.tokenizer = oneOf(merged.tokenizer, ['bpe', 'character', 'unicode'], base.tokenizer);
  merged.optimizer = oneOf(merged.optimizer, ['adam', 'momentum'], base.optimizer);
  merged.generation.sampling = oneOf(
    merged.generation.sampling,
    ['greedy', 'topk', 'topp', 'typical'],
    base.generation.sampling
  );
  merged.advanced.activation = oneOf(
    merged.advanced.activation,
    ['relu', 'leaky_relu', 'elu', 'gelu'],
    base.advanced.activation
  );
  merged.advanced.lrSchedule = oneOf(
    merged.advanced.lrSchedule,
    ['constant', 'cosine', 'exponential', 'warmup_cosine'],
    base.advanced.lrSchedule
  );
  return merged;
}

export function loadSettings(): StudioSettings {
  try {
    const raw = localStorage.getItem(STUDIO_SETTINGS_KEY);
    return raw ? mergeSettings(JSON.parse(raw)) : structuredCloneSettings(DEFAULT_STUDIO_SETTINGS);
  } catch {
    return structuredCloneSettings(DEFAULT_STUDIO_SETTINGS);
  }
}

export function saveSettings(settings: StudioSettings): void {
  try {
    localStorage.setItem(STUDIO_SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Storage full or blocked: the settings simply are not remembered.
  }
}

export function structuredCloneSettings(s: StudioSettings): StudioSettings {
  return JSON.parse(JSON.stringify(s)) as StudioSettings;
}
