/**
 * Dataset Abstraction Types
 */

export interface SplitConfig {
  trainRatio: number;
  valRatio: number;
  testRatio: number;
  seed: number;
  stratify?: boolean;
}

export const DEFAULT_SPLIT_CONFIG: SplitConfig = {
  trainRatio: 0.8,
  valRatio: 0.1,
  testRatio: 0.1,
  seed: 42,
  stratify: false
};

export interface DatasetSchema {
  name: string;
  version: string;
  description: string;
  source?: string;
  license?: string;
  language: string;
  createdAt: string;
  modifiedAt: string;
  stats: DatasetStats;
  format: 'text' | 'jsonl' | 'csv';
  encoding: 'utf-8' | 'ascii';
  hasLabels: boolean;
  labelSchema?: LabelSchema;
}

export interface DatasetStats {
  totalSamples: number;
  totalTokens: number;
  totalCharacters: number;
  uniqueTokens: number;
  averageTokensPerSample: number;
  medianTokensPerSample: number;
  maxTokensPerSample: number;
  minTokensPerSample: number;
}

export interface LabelSchema {
  type: 'classification' | 'regression' | 'sequence';
  classes?: string[];
  numClasses?: number;
}

export interface DataSample {
  id: string;
  text: string;
  tokens?: string[];
  label?: string | number;
  metadata?: Record<string, unknown>;
}

export interface DatasetSplit {
  name: 'train' | 'val' | 'test';
  samples: DataSample[];
  stats: DatasetStats;
  hash: string;
}

export interface Dataset {
  schema: DatasetSchema;
  splits: {
    train: DatasetSplit;
    val: DatasetSplit;
    test: DatasetSplit;
  };
  hash: string;
  splitConfig: SplitConfig;
  shuffleSeed: number;
  originalOrder: string[];
}

export interface DatasetArtifact {
  version: string;
  hash: string;
  createdAt: string;
  schema: DatasetSchema;
  splitConfig: SplitConfig;
  splits: {
    train: SerializedSplit;
    val: SerializedSplit;
    test: SerializedSplit;
  };
}

export interface SerializedSplit {
  name: 'train' | 'val' | 'test';
  sampleCount: number;
  stats: DatasetStats;
  hash: string;
  sampleIds: string[];
}

export interface DatasetLoadOptions {
  maxSamples?: number;
  shuffle?: boolean;
  seed?: number;
  tokenizer?: 'unicode' | 'ascii' | 'bpe';
}

export interface BatchConfig {
  batchSize: number;
  shuffle: boolean;
  seed: number;
  dropLast: boolean;
}

export interface Batch {
  samples: DataSample[];
  indices: number[];
  batchIndex: number;
  isLast: boolean;
}

export interface BatchIteratorState {
  currentBatch: number;
  totalBatches: number;
  epoch: number;
  rngState: [number, number];
  shuffledIndices: number[];
}

export interface DatasetBuilderConfig {
  name: string;
  description?: string;
  language?: string;
  splitConfig?: SplitConfig;
  seed?: number;
}

export interface DatasetProgress {
  operation: 'loading' | 'tokenizing' | 'splitting' | 'hashing';
  current: number;
  total: number;
  message?: string;
}

export type DatasetProgressCallback = (progress: DatasetProgress) => void;
