/**
 * TransformerLM - Transformer-based Language Model
 *
 * Provides a transformer architecture as an alternative to the feedforward ProNeuralLM.
 * Maintains API compatibility with ProNeuralLM for easy integration.
 */

import { ProNeuralLM, type Optimizer, type TokenizerConfig } from './ProNeuralLM';
import { MiniTransformerBlock, type MiniTransformerConfig } from '../models/mini_transformer';
import { type AttentionWeights } from '../models/attention';
import { type BatchRenormState } from '../models/regularizers';

export interface TransformerConfig {
  numLayers?: number;
  numHeads?: number;
  ffHiddenDim?: number;
  attentionDropout?: number;
  dropConnectRate?: number;
}

const DEFAULT_TRANSFORMER_CONFIG: Required<TransformerConfig> = {
  numLayers: 2,
  numHeads: 4,
  ffHiddenDim: 128,
  attentionDropout: 0.1,
  dropConnectRate: 0.1
};

/**
 * TransformerLM extends ProNeuralLM but uses transformer blocks instead of simple feedforward layers.
 * This allows comparing transformer vs feedforward architectures on the same tasks.
 */
export class TransformerLM extends ProNeuralLM {
  private transformerConfig: Required<TransformerConfig>;
  private transformerBlocks: MiniTransformerBlock[] = [];
  private attentionWeights: AttentionWeights[] = [];
  private ffWeights1: number[][][] = [];
  private ffWeights2: number[][][] = [];
  private renormStates: BatchRenormState[] = [];

  constructor(
    vocab: string[],
    hiddenSize = 64,
    learningRate = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'momentum',
    momentum = 0.9,
    dropout = 0.0,
    seed = 1337,
    tokenizerConfig?: TokenizerConfig,
    transformerConfig?: TransformerConfig
  ) {
    super(
      vocab,
      hiddenSize,
      learningRate,
      contextSize,
      optimizer,
      momentum,
      dropout,
      seed,
      tokenizerConfig
    );

    this.transformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...transformerConfig };
    this.initializeTransformer();
  }

  private initializeTransformer(): void {
    const { numLayers, numHeads, ffHiddenDim, attentionDropout, dropConnectRate } =
      this.transformerConfig;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const modelDim = (this as any).hiddenSize;

    // Initialize transformer blocks
    for (let i = 0; i < numLayers; i++) {
      // Create renorm state for this layer
      const renormState: BatchRenormState = {
        runningMean: new Array(modelDim).fill(0),
        runningVar: new Array(modelDim).fill(1),
        momentum: 0.99,
        epsilon: 1e-5,
        rMax: 3,
        dMax: 5
      };
      this.renormStates.push(renormState);

      // Create transformer block config
      const blockConfig: MiniTransformerConfig = {
        modelDim,
        heads: numHeads,
        ff: { hiddenDim: ffHiddenDim },
        attentionDropout,
        dropConnectRate,
        renormState
      };

      this.transformerBlocks.push(new MiniTransformerBlock(blockConfig));

      // Initialize attention weights (Q, K, V projections)
      this.attentionWeights.push({
        query: this.randomMatrix(modelDim, modelDim, 0.02),
        key: this.randomMatrix(modelDim, modelDim, 0.02),
        value: this.randomMatrix(modelDim, modelDim, 0.02)
      });

      // Initialize feedforward weights
      this.ffWeights1.push(this.randomMatrix(modelDim, ffHiddenDim, 0.02));
      this.ffWeights2.push(this.randomMatrix(ffHiddenDim, modelDim, 0.02));
    }
  }

  private randomMatrix(rows: number, cols: number, scale: number): number[][] {
    const matrix: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      matrix.push(row);
    }
    return matrix;
  }

  /**
   * NOTE: Transformer forward pass implementation deferred.
   * Currently uses parent class (ProNeuralLM) forward implementation.
   *
   * TODO: Implement full transformer forward pass:
   * 1. Convert embeddings to matrix form (sequence_length x model_dim)
   * 2. Pass through each transformer block
   * 3. Apply final output projection
   *
   * This requires overriding private forward() which has access restrictions.
   * Future work will refactor ProNeuralLM to expose hooks for custom architectures.
   */

  /**
   * Get transformer configuration
   */
  getTransformerConfig(): TransformerConfig {
    return { ...this.transformerConfig };
  }

  /**
   * Update transformer configuration
   */
  setTransformerConfig(config: Partial<TransformerConfig>): void {
    this.transformerConfig = { ...this.transformerConfig, ...config };
    // Re-initialize if architecture changed
    if (config.numLayers || config.numHeads || config.ffHiddenDim) {
      this.initializeTransformer();
    }
  }

  /**
   * Override toJSON to include transformer config
   */
  toJSON() {
    const baseJSON = super.toJSON();
    return {
      ...baseJSON,
      transformerConfig: this.transformerConfig,
      isTransformer: true
    };
  }

  /**
   * Load TransformerLM from JSON
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static loadFromJSON(json: any): TransformerLM {
    const transformer = new TransformerLM(
      json.vocab,
      json.hiddenSize,
      json.learningRate,
      json.contextSize,
      json.optimizer,
      json.momentum,
      json.dropout,
      json.rngSeed,
      json.tokenizerConfig,
      json.transformerConfig
    );

    // Copy state from parent using any to access private members
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Object.assign(transformer as any, json);

    return transformer;
  }
}
