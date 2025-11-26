/**
 * Unified Model Compression Utility
 *
 * Provides high-level interface for compressing ProNeuralLM models
 * using quantization, distillation, or low-rank approximation.
 */

import type { ProNeuralLM } from '../lib/ProNeuralLM';
import { quantizeMatrix, dequantizeMatrix, type QuantizedModel } from './quantization';
import {
  distillKnowledge,
  type DistillationConfig,
  DEFAULT_DISTILLATION_CONFIG
} from './distillation';
import { compressModelLowRank, findOptimalRank } from './lowrank';

export type CompressionMethod = 'quantization' | 'distillation' | 'lowrank';

export interface CompressionConfig {
  method: CompressionMethod;

  // Quantization settings
  quantizationBits?: 8 | 16; // Currently only 8-bit supported

  // Distillation settings
  distillation?: Partial<DistillationConfig>;

  // Low-rank settings
  rank?: number;
  targetCompressionRatio?: number; // e.g., 2.0 for 2x compression
}

export interface CompressionResult {
  method: CompressionMethod;
  originalSize: number; // in bytes
  compressedSize: number; // in bytes
  compressionRatio: number;
  approximationError?: number;
  compressedModel?: any;
  metadata: {
    timestamp: number;
    [key: string]: any;
  };
}

/**
 * Estimate model size in bytes (for comparison)
 */
function estimateModelSize(model: ProNeuralLM): number {
  const json = model.toJSON();

  // Estimate based on parameter count (float32 = 4 bytes each)
  let paramCount = 0;

  // Embedding
  paramCount += json.embedding.length * json.embedding[0].length;

  // Hidden weights
  paramCount += json.wHidden.length * json.wHidden[0].length;

  // Output weights
  paramCount += json.wOutput.length * json.wOutput[0].length;

  // Biases
  paramCount += json.bHidden.length + json.bOutput.length;

  return paramCount * 4; // 4 bytes per float32
}

/**
 * Compress model using int8 quantization
 */
export function compressWithQuantization(model: ProNeuralLM): CompressionResult {
  const startTime = performance.now();
  const json = model.toJSON();

  // Quantize all weight matrices
  const quantizedEmbedding = quantizeMatrix(json.embedding);
  const quantizedWHidden = quantizeMatrix(json.wHidden);
  const quantizedWOutput = quantizeMatrix(json.wOutput);

  // For biases, quantize as 1D arrays (treated as Nx1 matrix)
  const quantizedBHidden = quantizeMatrix([json.bHidden]);
  const quantizedBOutput = quantizeMatrix([json.bOutput]);

  const compressedModel: QuantizedModel = {
    embedding: quantizedEmbedding,
    wHidden: quantizedWHidden,
    wOutput: quantizedWOutput,
    bHidden: quantizedBHidden,
    bOutput: quantizedBOutput,
    quantizationMethod: 'symmetric',
    compressionRatio: 0 // Will calculate below
  };

  // Calculate sizes
  const originalSize = estimateModelSize(model);

  // Compressed size: int8 arrays + metadata
  const compressedSize =
    quantizedEmbedding.values.length +
    quantizedWHidden.values.length +
    quantizedWOutput.values.length +
    quantizedBHidden.values.length +
    quantizedBOutput.values.length +
    1000; // Overhead for metadata

  const compressionRatio = originalSize / compressedSize;
  compressedModel.compressionRatio = compressionRatio;

  // Test reconstruction to measure error
  const reconstructedEmbedding = dequantizeMatrix(quantizedEmbedding);
  let totalError = 0;
  let count = 0;

  for (let i = 0; i < json.embedding.length; i++) {
    for (let j = 0; j < json.embedding[i].length; j++) {
      const diff = json.embedding[i][j] - reconstructedEmbedding[i][j];
      totalError += diff * diff;
      count++;
    }
  }

  const approximationError = Math.sqrt(totalError / count);

  return {
    method: 'quantization',
    originalSize,
    compressedSize,
    compressionRatio,
    approximationError,
    compressedModel,
    metadata: {
      timestamp: Date.now(),
      processingTime: performance.now() - startTime,
      quantizationBits: 8,
      method: 'symmetric'
    }
  };
}

/**
 * Compress model using knowledge distillation
 */
export async function compressWithDistillation(
  model: ProNeuralLM,
  corpus: string,
  config: Partial<DistillationConfig> = {}
): Promise<CompressionResult> {
  const startTime = performance.now();

  const distillConfig: DistillationConfig = {
    ...DEFAULT_DISTILLATION_CONFIG,
    ...config
  };

  const result = await distillKnowledge(model, corpus, distillConfig);

  const originalSize = estimateModelSize(model);
  const compressedSize = estimateModelSize(result.studentModel);

  return {
    method: 'distillation',
    originalSize,
    compressedSize,
    compressionRatio: result.compressionRatio,
    compressedModel: result.studentModel,
    metadata: {
      timestamp: Date.now(),
      processingTime: performance.now() - startTime,
      studentHiddenSize: distillConfig.studentHiddenSize,
      temperature: distillConfig.temperature,
      alpha: distillConfig.alpha,
      accuracyRetention: result.accuracyRetention,
      finalLoss: result.finalLoss
    }
  };
}

/**
 * Compress model using low-rank approximation
 */
export function compressWithLowRank(
  model: ProNeuralLM,
  config: { rank?: number; targetCompressionRatio?: number }
): CompressionResult {
  const startTime = performance.now();
  const json = model.toJSON();

  // Determine rank
  let rank = config.rank;

  if (!rank && config.targetCompressionRatio) {
    // Calculate rank needed for target ratio
    const m = json.wHidden.length;
    const n = json.wHidden[0].length;
    rank = findOptimalRank(m, n, config.targetCompressionRatio);
  }

  if (!rank) {
    // Default: use rank that gives ~2x compression
    const m = json.wHidden.length;
    const n = json.wHidden[0].length;
    rank = Math.floor(Math.min(m, n) / 2);
  }

  // Compress weight matrices
  const { compressed, errors } = compressModelLowRank(
    json.embedding,
    json.wHidden,
    json.wOutput,
    rank
  );

  const originalSize = estimateModelSize(model);

  // Estimate compressed size
  const compressedSize =
    (compressed.embedding.U.length * rank +
      rank * compressed.embedding.originalShape[1] +
      compressed.wHidden.U.length * rank +
      rank * compressed.wHidden.originalShape[1] +
      compressed.wOutput.U.length * rank +
      rank * compressed.wOutput.originalShape[1]) *
    4; // float32

  const compressionRatio = compressed.totalCompressionRatio;

  return {
    method: 'lowrank',
    originalSize,
    compressedSize,
    compressionRatio,
    approximationError: compressed.approximationError,
    compressedModel: compressed,
    metadata: {
      timestamp: Date.now(),
      processingTime: performance.now() - startTime,
      rank,
      errors: errors,
      targetCompressionRatio: config.targetCompressionRatio
    }
  };
}

/**
 * Main compression function - automatically selects best method
 */
export async function compressModel(
  model: ProNeuralLM,
  config: CompressionConfig,
  corpus?: string
): Promise<CompressionResult> {
  switch (config.method) {
    case 'quantization':
      return compressWithQuantization(model);

    case 'distillation':
      if (!corpus) {
        throw new Error('Corpus required for knowledge distillation');
      }
      return await compressWithDistillation(model, corpus, config.distillation);

    case 'lowrank':
      return compressWithLowRank(model, {
        rank: config.rank,
        targetCompressionRatio: config.targetCompressionRatio
      });

    default:
      throw new Error(`Unknown compression method: ${config.method}`);
  }
}

/**
 * Export compressed model to JSON
 */
export function exportCompressedModel(result: CompressionResult): string {
  const exportData = {
    compressionInfo: {
      method: result.method,
      originalSize: result.originalSize,
      compressedSize: result.compressedSize,
      compressionRatio: result.compressionRatio,
      approximationError: result.approximationError,
      metadata: result.metadata
    },
    model: result.compressedModel
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Compare compression methods side-by-side
 */
export async function compareCompressionMethods(
  model: ProNeuralLM,
  corpus: string
): Promise<{
  quantization: CompressionResult;
  lowrank: CompressionResult;
  distillation?: CompressionResult;
}> {
  const quantization = compressWithQuantization(model);
  const lowrank = compressWithLowRank(model, { targetCompressionRatio: 2.0 });

  // Distillation is optional (takes longer)
  let distillation: CompressionResult | undefined;
  try {
    distillation = await compressWithDistillation(model, corpus, {
      studentHiddenSize: Math.floor(model.getHiddenSize() / 2),
      epochs: 20
    });
  } catch (e) {
    console.warn('Distillation failed:', e);
  }

  return {
    quantization,
    lowrank,
    distillation
  };
}
