import type { InjectableLayer } from './InjectableLayer';
import type { InjectionTarget } from '../../types/injection';
import type { ProNeuralLM } from '../ProNeuralLM';

/**
 * Adapter that wraps ProNeuralLM to implement InjectableLayer interface.
 * Enables Cerebro neuron injection system to work with feedforward models.
 */
export class ProNeuralLMAdapter implements InjectableLayer {
  private model: ProNeuralLM;
  private modelId: string;
  private maxHiddenSize: number;

  constructor(model: ProNeuralLM, modelId: string, maxHiddenSize = 512) {
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
      dModel: vocabSize * contextSize, // Input dimension (context embeddings)
      hiddenSize
    };
  }

  canInject(k: number): boolean {
    const currentSize = this.model.getHiddenSize();
    const newSize = currentSize + k;

    // Check bounds
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

    // Use He initialization by default for ReLU networks
    const useHe = init === 'random_he' || init === 'residual_eig';
    this.model.expandHiddenLayer(k, useHe);
  }

  exportWeights(): Float32Array[] {
    const weights = this.model.getWeights();

    // Convert to Float32Arrays for the injection system
    return [
      new Float32Array(weights.wHidden.flat()),
      new Float32Array(weights.bHidden),
      new Float32Array(weights.wOutput.flat()),
      new Float32Array(weights.bOutput),
      new Float32Array(weights.embedding.flat())
    ];
  }

  importWeights(weights: Float32Array[]): void {
    if (weights.length < 5) {
      throw new Error('Invalid weights array: expected 5 arrays');
    }

    const vocabSize = this.model.getVocabSize();
    const contextSize = this.model.getContextSize();
    const inputDim = vocabSize * contextSize;

    // Infer hidden size from bHidden (1D array where length = hiddenSize)
    const bHidden = Array.from(weights[1]);
    const hiddenSize = bHidden.length;

    // Reshape flat arrays back to matrices
    const wHidden: number[][] = [];
    const wHiddenFlat = Array.from(weights[0]);
    for (let i = 0; i < hiddenSize; i++) {
      wHidden.push(wHiddenFlat.slice(i * inputDim, (i + 1) * inputDim));
    }

    const wOutput: number[][] = [];
    const wOutputFlat = Array.from(weights[2]);
    for (let i = 0; i < vocabSize; i++) {
      wOutput.push(wOutputFlat.slice(i * hiddenSize, (i + 1) * hiddenSize));
    }

    const bOutput = Array.from(weights[3]);

    const embedding: number[][] = [];
    const embeddingFlat = Array.from(weights[4]);
    const embeddingDim = Math.floor(embeddingFlat.length / vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      embedding.push(embeddingFlat.slice(i * embeddingDim, (i + 1) * embeddingDim));
    }

    this.model.setWeights({
      wHidden,
      bHidden,
      wOutput,
      bOutput,
      embedding
    });
  }
}

/**
 * Factory function to create a ProNeuralLMAdapter
 */
export function createProNeuralLMAdapter(
  model: ProNeuralLM,
  modelId?: string,
  maxHiddenSize?: number
): ProNeuralLMAdapter {
  const id = modelId ?? `model-${Date.now()}`;
  return new ProNeuralLMAdapter(model, id, maxHiddenSize);
}
