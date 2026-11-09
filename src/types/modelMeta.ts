import type { Architecture } from '../components/TrainingPanel';
import type { ModelOrigin } from '../lib/modelExport';

export interface ModelMeta {
  architecture: Architecture;
  timestamp: number;
  vocab: number;
  loss?: number;
  accuracy?: number;
  perplexity?: number;
  tokensPerSec?: number;
  trainingDurationMs?: number;
  /** Where the stored model's weights came from (used for exported model cards). */
  origin?: ModelOrigin;
}

export type ModelMetaStore = Partial<Record<Architecture, ModelMeta>>;
