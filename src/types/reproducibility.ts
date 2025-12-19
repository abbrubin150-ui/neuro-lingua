/**
 * Determinism & Reproducibility Types
 */

export interface PRNGState {
  seed: number;
  state0: number;
  state1: number;
  callCount: number;
}

export interface ConfigHash {
  hash: string;
  algorithm: 'sha256';
  timestamp: string;
  components: {
    hyperparameters: string;
    architecture: string;
    tokenizer: string;
    seed: string;
  };
}

export interface WeightHash {
  hash: string;
  algorithm: 'sha256';
  timestamp: string;
  parameterCount: number;
  layerHashes: Record<string, string>;
}

export interface ReproducibilityManifest {
  version: string;
  createdAt: string;
  globalSeed: number;
  trainingSeed: number;
  samplingSeed: number;
  configHash: ConfigHash;
  initialWeightsHash: WeightHash;
  finalWeightsHash?: WeightHash;
  corpusHash: string;
  tokenizerHash: string;
  epochChecksums: string[];
  environment: {
    platform: string;
    userAgent: string;
    timestamp: string;
    numericPrecision: 'float32' | 'float64';
  };
}

export interface ReplayVerification {
  success: boolean;
  configMatch: boolean;
  seedMatch: boolean;
  corpusMatch: boolean;
  tokenizerMatch: boolean;
  epochMatchCount: number;
  totalEpochs: number;
  maxLossDelta: number;
  tolerance: number;
  errors: string[];
}

export interface SamplingConfig {
  seed: number;
  temperature: number;
  topK?: number;
  topP?: number;
  typicalP?: number;
  mirostatTau?: number;
}

export interface SamplingState {
  rngState: PRNGState;
  tokensGenerated: number;
  lastTokenId: number;
  mirostatMu?: number;
}

export interface ReproducibilitySnapshot {
  timestamp: string;
  rngState: PRNGState;
  epoch: number;
  step: number;
  loss: number;
  weightsHash: string;
}

export const TOLERANCE_LEVELS = {
  strict: 1e-10,
  normal: 1e-6,
  relaxed: 1e-4,
  loose: 1e-2
} as const;

export type ToleranceLevel = keyof typeof TOLERANCE_LEVELS;
