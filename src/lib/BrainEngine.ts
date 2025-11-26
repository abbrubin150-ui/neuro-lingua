import { StorageManager } from './storage';

export type Mood = 'IDLE' | 'CURIOUS' | 'FOCUSED' | 'OVERLOADED';

export interface BrainStats {
  energy: number;
  clarity: number;
  mood: Mood;
  idleTicks: number;
  updatedAt: number;
}

export type BrainEvent =
  | { type: 'PULSE'; stimulation?: number }
  | { type: 'IDLE' }
  | { type: 'RESET' };

const STORAGE_KEY = 'neuro-lingua-brain-stats-v1';

const DEFAULT_STATS: BrainStats = {
  energy: 0.75,
  clarity: 0.75,
  mood: 'CURIOUS',
  idleTicks: 0,
  updatedAt: Date.now()
};

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const pickMood = (state: Pick<BrainStats, 'energy' | 'clarity' | 'idleTicks'>): Mood => {
  if (state.idleTicks >= 3 || state.energy < 0.25) return 'IDLE';
  if (state.energy > 0.8 && state.clarity > 0.7) return 'FOCUSED';
  if (state.clarity < 0.35) return 'OVERLOADED';
  return 'CURIOUS';
};

export const reduceBrain = (state: BrainStats, event: BrainEvent): BrainStats => {
  const next = { ...state };

  if (event.type === 'RESET') {
    return { ...DEFAULT_STATS, updatedAt: Date.now() };
  }

  if (event.type === 'PULSE') {
    const stimulation = clamp(event.stimulation ?? 0.1, 0, 1);
    next.energy = clamp(state.energy + stimulation * 0.2, 0, 1);
    next.clarity = clamp(state.clarity + stimulation * 0.15, 0, 1);
    next.idleTicks = 0;
  }

  if (event.type === 'IDLE') {
    const decay = Math.max(0.02, Math.min(0.2, state.idleTicks * 0.03));
    next.energy = clamp(state.energy - decay, 0, 1);
    next.clarity = clamp(state.clarity - decay * 1.15, 0, 1);
    next.idleTicks = state.idleTicks + 1;
  }

  next.mood = pickMood({
    energy: next.energy,
    clarity: next.clarity,
    idleTicks: next.idleTicks
  });
  next.updatedAt = Date.now();

  return next;
};

export const loadBrain = (): BrainStats => {
  const stored = StorageManager.get<BrainStats>(STORAGE_KEY, DEFAULT_STATS);
  const normalized: BrainStats = {
    ...DEFAULT_STATS,
    ...stored,
    energy: clamp(stored.energy, 0, 1),
    clarity: clamp(stored.clarity, 0, 1),
    idleTicks: Math.max(0, stored.idleTicks ?? 0),
    updatedAt: stored.updatedAt ?? Date.now()
  };

  return { ...normalized, mood: pickMood(normalized) };
};

export const saveBrain = (stats: BrainStats): boolean => {
  return StorageManager.set(STORAGE_KEY, stats);
};

export const BrainEngine = {
  load: loadBrain,
  save: saveBrain,
  reduce: reduceBrain,
  DEFAULT_STATS
};

export default BrainEngine;
