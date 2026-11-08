import type { InjectableLayer } from './InjectableLayer';
import type { InjectionTarget } from '../../types/injection';
import { decodeBaseWeights, unflatten } from './weightCodec';
import type { TransformerLM, TransformerRMSStates } from '../TransformerLM';

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

    // Every head must grow by the same number of dimensions so that the learned head structure
    // is kept (and a rollback restores the same layout): k must be a multiple of numHeads.
    if (k % transformerInfo.numHeads !== 0) return false;

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

      // Renorm state (split attention/FFN)
      baseArrays.push(new Float32Array(renorm.attention.gamma));
      baseArrays.push(new Float32Array([renorm.attention.epsilon]));
      baseArrays.push(new Float32Array(renorm.ffn.gamma));
      baseArrays.push(new Float32Array([renorm.ffn.epsilon]));
    }

    // Final RMSNorm of the causal objective (one trailing array)
    if (transformerWeights.finalNorm) {
      baseArrays.push(new Float32Array(transformerWeights.finalNorm.gamma));
    }

    return baseArrays;
  }

  importWeights(weights: Float32Array[]): void {
    if (weights.length < 6) {
      throw new Error('Invalid weights array: expected at least 6 arrays');
    }

    // Shapes are inferred from the snapshot itself: the live model may have been widened since.
    const vocabSize = this.model.getVocabSize();
    const numLayers = this.model.getTransformerInfo().numLayers;
    const base = decodeBaseWeights(weights, vocabSize);
    const D = base.bHidden.length;
    this.model.setWeights(base);

    const positionEmbeddings = unflatten(weights[5], weights[5].length / D);

    // Per layer: 9 arrays (query, key, value, ff1, ff2, attn γ/ε, ffn γ/ε) or 7 (legacy shared
    // γ/ε), optionally followed by one final-norm array.
    const baseIndex = 6;
    const remaining = weights.length - baseIndex;
    let arraysPerLayer = 0;
    let hasFinalNorm = false;
    for (const per of [9, 7]) {
      const extra = remaining - per * numLayers;
      if (extra === 0 || extra === 1) {
        arraysPerLayer = per;
        hasFinalNorm = extra === 1;
        break;
      }
    }
    if (arraysPerLayer === 0) {
      throw new Error('Invalid weights array: unexpected transformer payload');
    }

    const attentionWeights: Array<{ query: number[][]; key: number[][]; value: number[][] }> = [];
    const ffWeights1: number[][][] = [];
    const ffWeights2: number[][][] = [];
    const renormStates: TransformerRMSStates[] = [];

    for (let layer = 0; layer < numLayers; layer++) {
      const offset = baseIndex + layer * arraysPerLayer;
      // Q: D×D, K/V: D×kvDim (GQA), ff1: D×2F (SwiGLU), ff2: F×D
      attentionWeights.push({
        query: unflatten(weights[offset], D),
        key: unflatten(weights[offset + 1], D),
        value: unflatten(weights[offset + 2], D)
      });
      ffWeights1.push(unflatten(weights[offset + 3], D));
      ffWeights2.push(unflatten(weights[offset + 4], weights[offset + 4].length / D));

      if (arraysPerLayer === 7) {
        const gamma = Array.from(weights[offset + 5]);
        const epsilon = Array.from(weights[offset + 6])[0] ?? 1e-6;
        renormStates.push({
          attention: { gamma: [...gamma], epsilon },
          ffn: { gamma: [...gamma], epsilon }
        });
      } else {
        renormStates.push({
          attention: {
            gamma: Array.from(weights[offset + 5]),
            epsilon: Array.from(weights[offset + 6])[0] ?? 1e-6
          },
          ffn: {
            gamma: Array.from(weights[offset + 7]),
            epsilon: Array.from(weights[offset + 8])[0] ?? 1e-6
          }
        });
      }
    }

    this.model.setTransformerWeights({
      attentionWeights,
      ffWeights1,
      ffWeights2,
      positionEmbeddings,
      renormStates,
      finalNorm: hasFinalNorm
        ? { gamma: Array.from(weights[weights.length - 1]), epsilon: 1e-6 }
        : null
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
