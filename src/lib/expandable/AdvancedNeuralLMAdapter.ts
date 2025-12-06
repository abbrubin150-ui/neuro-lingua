import type { InjectableLayer } from './InjectableLayer';
import type { InjectionTarget } from '../../types/injection';
import type { AdvancedNeuralLM } from '../AdvancedNeuralLM';

/**
 * Adapter that wraps AdvancedNeuralLM to implement InjectableLayer interface.
 * Handles additional state like layer normalization parameters.
 *
 * Extends the base adapter pattern to support:
 * - Layer normalization gamma/beta parameters
 * - Advanced config preservation during rollback
 */
export class AdvancedNeuralLMAdapter implements InjectableLayer {
  private model: AdvancedNeuralLM;
  private modelId: string;
  private maxHiddenSize: number;

  constructor(model: AdvancedNeuralLM, modelId: string, maxHiddenSize = 512) {
    this.model = model;
    this.modelId = modelId;
    this.maxHiddenSize = maxHiddenSize;
  }

  getTarget(): InjectionTarget {
    const hiddenSize = this.model.getHiddenSize();
    const vocabSize = this.model.getVocabSize();
    const contextSize = this.model.getContextSize();

    return {
      modelId: this.modelId,
      layerId: 'hidden-0',
      type: 'ffn',
      dModel: vocabSize * contextSize,
      hiddenSize
    };
  }

  canInject(k: number): boolean {
    const currentSize = this.model.getHiddenSize();
    const newSize = currentSize + k;

    if (k <= 0) return false;
    if (newSize > this.maxHiddenSize) return false;

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
    const advancedWeights = this.model.getAdvancedWeights?.();

    // Base weights (same as ProNeuralLMAdapter)
    const baseArrays = [
      new Float32Array(weights.wHidden.flat()),
      new Float32Array(weights.bHidden),
      new Float32Array(weights.wOutput.flat()),
      new Float32Array(weights.bOutput),
      new Float32Array(weights.embedding.flat())
    ];

    // Add layer norm parameters if they exist
    if (advancedWeights?.layerNormGamma && advancedWeights?.layerNormBeta) {
      baseArrays.push(new Float32Array(advancedWeights.layerNormGamma));
      baseArrays.push(new Float32Array(advancedWeights.layerNormBeta));
    }

    return baseArrays;
  }

  importWeights(weights: Float32Array[]): void {
    if (weights.length < 5) {
      throw new Error('Invalid weights array: expected at least 5 arrays');
    }

    const vocabSize = this.model.getVocabSize();
    const contextSize = this.model.getContextSize();
    const inputDim = vocabSize * contextSize;

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

    // Restore layer norm parameters if present
    if (weights.length >= 7 && this.model.setAdvancedWeights) {
      const layerNormGamma = Array.from(weights[5]);
      const layerNormBeta = Array.from(weights[6]);
      this.model.setAdvancedWeights({ layerNormGamma, layerNormBeta });
    }
  }
}

/**
 * Factory function to create an AdvancedNeuralLMAdapter
 */
export function createAdvancedNeuralLMAdapter(
  model: AdvancedNeuralLM,
  modelId?: string,
  maxHiddenSize?: number
): AdvancedNeuralLMAdapter {
  const id = modelId ?? `advanced-model-${Date.now()}`;
  return new AdvancedNeuralLMAdapter(model, id, maxHiddenSize);
}
