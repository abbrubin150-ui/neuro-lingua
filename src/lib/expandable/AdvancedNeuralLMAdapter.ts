import type { InjectableLayer } from './InjectableLayer';
import type { InjectionTarget } from '../../types/injection';
import type { AdvancedNeuralLM } from '../AdvancedNeuralLM';
import { decodeBaseWeights } from './weightCodec';

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

    return {
      modelId: this.modelId,
      layerId: 'hidden-0',
      type: 'ffn',
      // Hidden units read the mean context embedding (dimension = hiddenSize in ProNeuralLM),
      // so their weight rows live in that space.
      dModel: hiddenSize,
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
    // Shapes are inferred from the snapshot (the live model may have been widened since).
    this.model.setWeights(decodeBaseWeights(weights, this.model.getVocabSize()));

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
