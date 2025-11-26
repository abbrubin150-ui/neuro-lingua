/**
 * Model Compression Module
 *
 * Provides three compression techniques for neural language models:
 * 1. Int8 Quantization - 4x size reduction
 * 2. Knowledge Distillation - Train smaller student from larger teacher
 * 3. Low-Rank Approximation - SVD-based matrix factorization
 *
 * Usage:
 *   import { quantizeModel, distillKnowledge, compressModelLowRank } from './compression';
 */

// Quantization exports
export {
  quantizeArray,
  quantizeMatrix,
  dequantizeArray,
  dequantizeMatrix,
  calculateQuantizationError,
  calculateCompressionRatio,
  estimateQuantizedSize,
  serializeQuantizedWeights,
  deserializeQuantizedWeights
} from './quantization';

export type {
  QuantizationParams,
  QuantizedWeights,
  QuantizedModel
} from './quantization';

// Distillation exports
export {
  softmaxWithTemperature,
  klDivergence,
  crossEntropyLoss,
  distillationLoss,
  getTeacherSoftTargets,
  calculateModelSize,
  estimateCompressionRatio,
  distillKnowledge,
  compareModels,
  DEFAULT_DISTILLATION_CONFIG
} from './distillation';

export type {
  TeacherModel,
  DistillationConfig,
  DistillationResult
} from './distillation';

// Low-rank approximation exports
export {
  transpose,
  matmul,
  frobeniusNorm,
  simplifiedSVD,
  lowRankApproximation,
  reconstructFromLowRank,
  approximationError,
  findOptimalRank,
  compressModelLowRank,
  serializeLowRankWeights,
  deserializeLowRankWeights
} from './lowrank';

export type {
  SVDResult,
  LowRankWeights,
  LowRankModel
} from './lowrank';

/**
 * Unified compression interface
 */
export interface CompressionOptions {
  method: 'quantization' | 'distillation' | 'lowrank' | 'hybrid';

  // Quantization options
  quantizationMethod?: 'symmetric' | 'asymmetric';

  // Distillation options
  distillationConfig?: any; // DistillationConfig

  // Low-rank options
  rank?: number;
  targetCompressionRatio?: number;
}

export interface CompressionResult {
  method: string;
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  approximationError?: number;
  metadata: Record<string, any>;
}
