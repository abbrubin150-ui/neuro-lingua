/**
 * BPE Tokenizer Types
 * Browser-native Byte Pair Encoding tokenizer with versioning and metrics
 */

export interface BPEMergeRule {
  left: string;
  right: string;
  merged: string;
  priority: number;
}

export interface VocabEntry {
  token: string;
  frequency: number;
  isMerge: boolean;
}

export interface BPEConfig {
  vocabSize: number;
  minFrequency: number;
  specialTokens: string[];
  unknownToken: string;
  padToken: string;
  bosToken: string;
  eosToken: string;
}

export interface BPEArtifact {
  version: string;
  hash: string;
  createdAt: string;
  config: BPEConfig;
  merges: BPEMergeRule[];
  vocab: VocabEntry[];
  vocabToId: Record<string, number>;
  idToVocab: Record<number, string>;
  trainingCorpusStats: {
    totalTokens: number;
    uniqueWords: number;
    corpusHash: string;
  };
}

export interface TokenizerMetrics {
  coverage: number;
  entropy: number;
  fertility: number;
  compressionRatio: number;
  vocabUtilization: number;
  unknownRate: number;
}

export interface BPETrainingProgress {
  currentMerges: number;
  targetMerges: number;
  currentVocabSize: number;
  targetVocabSize: number;
  lastMerge?: BPEMergeRule;
  elapsedMs: number;
}

export type BPEProgressCallback = (progress: BPETrainingProgress) => void;

export interface EncodingResult {
  ids: number[];
  tokens: string[];
  offsets: Array<[number, number]>;
}

export interface DecodingOptions {
  skipSpecialTokens: boolean;
  cleanupSpaces: boolean;
}

export const DEFAULT_BPE_CONFIG: BPEConfig = {
  vocabSize: 1000,
  minFrequency: 2,
  specialTokens: ['<PAD>', '<BOS>', '<EOS>', '<UNK>'],
  unknownToken: '<UNK>',
  padToken: '<PAD>',
  bosToken: '<BOS>',
  eosToken: '<EOS>'
};
