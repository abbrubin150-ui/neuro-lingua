export type InjectionTargetType = 'ffn' | 'mlp' | 'adapter' | 'transformer';

export interface InjectionTarget {
  modelId: string;
  layerId: string;
  type: InjectionTargetType;
  dModel: number;
  hiddenSize: number; // before injection
  /** Optional transformer-specific metadata */
  metadata?: {
    numLayers?: number;
    numHeads?: number;
    ffHiddenDim?: number;
    vocabSize?: number;
    contextSize?: number;
  };
}

export interface InjectionProposal {
  target: InjectionTarget;
  k: number; // neurons to add
  method: 'residual_eig' | 'random_he' | 'svd_local';
  epsilon: number; // residual threshold
  minGain: number; // delta energy threshold
  orthPenalty: number; // soft orth regularizer
  createdAt: string;
}

export interface InjectionEvent {
  proposal: InjectionProposal;
  accepted: boolean;
  metricsPre: Record<string, number>;
  metricsPost: Record<string, number>;
  delta: Record<string, number>;
  seed: number;
  runId: string;
}

export type CerebroBubbleTag = 'body' | 'desire' | 'risk' | 'value' | 'action' | 'other';

export interface CerebroBubble {
  id: string;
  label: string;
  embedding: number[];
  activation: number; // 0..1
  tag?: CerebroBubbleTag;
  ts: number;
  members?: string[]; // token ids or sample ids
}
