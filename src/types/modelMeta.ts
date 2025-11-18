import type { Architecture } from '../components/TrainingPanel';

export interface ModelMeta {
  architecture: Architecture;
  timestamp: number;
  vocab: number;
  loss?: number;
  accuracy?: number;
  perplexity?: number;
  tokensPerSec?: number;
  trainingDurationMs?: number;
}

export type ModelMetaStore = Partial<Record<Architecture, ModelMeta>>;
