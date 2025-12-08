import type { InjectableLayer } from './InjectableLayer';
import type { InjectionTarget } from '../../types/injection';
import type { TransformerLM } from '../TransformerLM';
import type { RMSNormState } from '../RMSNorm';

/**
 * Adapter that wraps TransformerLM to implement InjectableLayer interface.
 * Handles transformer-specific state including attention weights, feed-forward
 * layers, position embeddings, and batch renormalization states.
 *
 * Key differences from feedforward adapters:
 * - Manages multiple transformer layers (attention + FF)
 * - Handles position embeddings expansion
 * - Preserves batch renorm state during rollback
 */
export class TransformerLMAdapter implements InjectableLayer {
  private model: TransformerLM;
  private modelId: string;
  private maxHiddenSize: number;

  constructor(model: TransformerLM, modelId: string, maxHiddenSize = 512) {
    this.model = model;
    this.modelId = modelId;
    this.maxHiddenSize = maxHiddenSize;
  }

  getTarget(): InjectionTarget {
    const hiddenSize = this.model.getHiddenSize();
    const vocabSize = this.model.getVocabSize();
    const contextSize = this.model.getContextSize();
    const transformerInfo = this.model.getTransformerInfo();

    return {
      modelId: this.modelId,
      layerId: 'transformer-0',
      type: 'transformer',
      dModel: hiddenSize, // In transformers, dModel is the hidden/embedding dimension
      hiddenSize,
      // Additional transformer-specific info
      metadata: {
        numLayers: transformerInfo.numLayers,
        numHeads: transformerInfo.numHeads,
        ffHiddenDim: transformerInfo.ffHiddenDim,
        vocabSize,
        contextSize
      }
    };
  }

  canInject(k: number): boolean {
    const currentSize = this.model.getHiddenSize();
    const newSize = currentSize + k;
    const transformerInfo = this.model.getTransformerInfo();

    if (k <= 0) return false;
    if (newSize > this.maxHiddenSize) return false;

    // Check if new size is compatible with attention heads
    // The model dimension must be divisible by number of heads
    // The expandHiddenLayer method will normalize heads if needed
    // But we check that it won't reduce to less than 1 head
    if (newSize < transformerInfo.numHeads) return false;

    return true;
  }

  inject(k: number, init: 'random_he' | 'residual_eig'): void {
    if (!this.canInject(k)) {
      throw new Error(
        `Cannot inject ${k} neurons: would exceed max hidden size ${this.maxHiddenSize}`
      );
    }

    const useHe = init === 'random_he' || init === 'residual_eig';
    this.model.expandHiddenLayer(k, useHe);
  }

  exportWeights(): Float32Array[] {
    const weights = this.model.getWeights();
    const transformerWeights = this.model.getTransformerWeights();

    // Base weights (indices 0-4)
    const baseArrays: Float32Array[] = [
      new Float32Array(weights.wHidden.flat()),
      new Float32Array(weights.bHidden),
      new Float32Array(weights.wOutput.flat()),
      new Float32Array(weights.bOutput),
      new Float32Array(weights.embedding.flat())
    ];

    // Position embeddings (index 5)
    baseArrays.push(new Float32Array(transformerWeights.positionEmbeddings.flat()));

    // For each layer, add attention weights and FF weights
    for (let i = 0; i < transformerWeights.attentionWeights.length; i++) {
      const attn = transformerWeights.attentionWeights[i];
      const ff1 = transformerWeights.ffWeights1[i];
      const ff2 = transformerWeights.ffWeights2[i];
      const renorm = transformerWeights.renormStates[i];

      // Attention: query, key, value
      baseArrays.push(new Float32Array(attn.query.flat()));
      baseArrays.push(new Float32Array(attn.key.flat()));
      baseArrays.push(new Float32Array(attn.value.flat()));

      // Feed-forward
      baseArrays.push(new Float32Array(ff1.flat()));
      baseArrays.push(new Float32Array(ff2.flat()));

      // Renorm state
      baseArrays.push(new Float32Array(renorm.gamma));
      baseArrays.push(new Float32Array([renorm.epsilon]));
    }

    return baseArrays;
  }

  importWeights(weights: Float32Array[]): void {
    if (weights.length < 6) {
      throw new Error('Invalid weights array: expected at least 6 arrays');
    }

    const vocabSize = this.model.getVocabSize();
    const contextSize = this.model.getContextSize();
    const inputDim = vocabSize * contextSize;
    const transformerInfo = this.model.getTransformerInfo();
    const numLayers = transformerInfo.numLayers;
    const ffHiddenDim = transformerInfo.ffHiddenDim;

    // Infer hidden size from bHidden
    const bHidden = Array.from(weights[1]);
    const hiddenSize = bHidden.length;

    // Reshape wHidden
    const wHidden: number[][] = [];
    const wHiddenFlat = Array.from(weights[0]);
    for (let i = 0; i < hiddenSize; i++) {
      wHidden.push(wHiddenFlat.slice(i * inputDim, (i + 1) * inputDim));
    }

    // Reshape wOutput
    const wOutput: number[][] = [];
    const wOutputFlat = Array.from(weights[2]);
    for (let i = 0; i < vocabSize; i++) {
      wOutput.push(wOutputFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
    }

    const bOutput = Array.from(weights[3]);

    // Reshape embedding
    const embedding: number[][] = [];
    const embeddingFlat = Array.from(weights[4]);
    const embeddingDim = Math.floor(embeddingFlat.length / vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      embedding.push(embeddingFlat.slice(i * embeddingDim, (i + 1) * embeddingDim));
    }

    // Set base weights
    this.model.setWeights({
      wHidden,
      bHidden,
      wOutput,
      bOutput,
      embedding
    });

    // Reshape position embeddings (index 5)
    const posEmbFlat = Array.from(weights[5]);
    const maxSeqLength = Math.floor(posEmbFlat.length / hiddenSize);
    const positionEmbeddings: number[][] = [];
    for (let i = 0; i < maxSeqLength; i++) {
      positionEmbeddings.push(posEmbFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
    }

    // Rebuild transformer weights
    const attentionWeights: Array<{ query: number[][]; key: number[][]; value: number[][] }> = [];
    const ffWeights1: number[][][] = [];
    const ffWeights2: number[][][] = [];
    const renormStates: RMSNormState[] = [];

    // 7 arrays per layer: query, key, value, ff1, ff2, gamma, epsilon
    const arraysPerLayer = 7;
    const baseIndex = 6;

    for (let layer = 0; layer < numLayers; layer++) {
      const offset = baseIndex + layer * arraysPerLayer;

      // Reshape attention weights [hiddenSize x hiddenSize]
      const queryFlat = Array.from(weights[offset]);
      const keyFlat = Array.from(weights[offset + 1]);
      const valueFlat = Array.from(weights[offset + 2]);

      const query: number[][] = [];
      const key: number[][] = [];
      const value: number[][] = [];
      for (let i = 0; i < hiddenSize; i++) {
        query.push(queryFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
        key.push(keyFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
        value.push(valueFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
      }
      attentionWeights.push({ query, key, value });

      // Reshape FF weights
      // ff1: [hiddenSize x ffHiddenDim]
      const ff1Flat = Array.from(weights[offset + 3]);
      const ff1Layer: number[][] = [];
      for (let i = 0; i < hiddenSize; i++) {
        ff1Layer.push(ff1Flat.slice(i * ffHiddenDim, (i + 1) * ffHiddenDim));
      }
      ffWeights1.push(ff1Layer);

      // ff2: [ffHiddenDim x hiddenSize]
      const ff2Flat = Array.from(weights[offset + 4]);
      const ff2Layer: number[][] = [];
      for (let i = 0; i < ffHiddenDim; i++) {
        ff2Layer.push(ff2Flat.slice(i * hiddenSize, (i + 1) * hiddenSize));
      }
      ffWeights2.push(ff2Layer);

      // Renorm state
      const gamma = Array.from(weights[offset + 5]);
      const epsilonArr = Array.from(weights[offset + 6]);
      renormStates.push({
        gamma,
        epsilon: epsilonArr[0] ?? 1e-6
      });
    }

    // Set transformer-specific weights
    this.model.setTransformerWeights({
      attentionWeights,
      ffWeights1,
      ffWeights2,
      positionEmbeddings,
      renormStates
    });
  }
}

/**
 * Factory function to create a TransformerLMAdapter
 */
export function createTransformerLMAdapter(
  model: TransformerLM,
  modelId?: string,
  maxHiddenSize?: number
): TransformerLMAdapter {
  const id = modelId ?? `transformer-model-${Date.now()}`;
  return new TransformerLMAdapter(model, id, maxHiddenSize);
}
