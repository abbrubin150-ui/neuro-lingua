/**
 * TransformerLM - Transformer-based Neural Language Model
 *
 * A character-level language model using transformer architecture:
 * - Multi-head self-attention mechanism
 * - Position-aware embeddings
 * - Feed-forward layers with residual connections
 * - Layer normalization
 * - Compatible with ProNeuralLM interface for seamless UI integration
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { ProNeuralLM, type Optimizer, type TokenizerConfig } from './ProNeuralLM';
import { MiniTransformerBlock, type MiniTransformerConfig } from '../models/mini_transformer';
import type { BatchRenormState } from '../models/regularizers';

export type TransformerConfig = {
  numLayers?: number;
  numHeads?: number;
  ffHiddenDim?: number;
  attentionDropout?: number;
  dropConnectRate?: number;
};

const DEFAULT_TRANSFORMER_CONFIG: Required<TransformerConfig> = {
  numLayers: 2,
  numHeads: 4,
  ffHiddenDim: 128,
  attentionDropout: 0.1,
  dropConnectRate: 0.1
};

/**
 * TransformerLM - Full transformer language model
 */
export class TransformerLM extends ProNeuralLM {
  private transformerConfig: Required<TransformerConfig>;
  private transformerLayers: MiniTransformerBlock[] = [];
  private positionEmbeddings: number[][] = [];
  private maxSeqLength = 128;

  // Transformer-specific weights (stored in addition to base embeddings)
  private attentionWeights: {
    query: number[][][];
    key: number[][][];
    value: number[][][];
  }[] = [];

  private ffWeights1: number[][][] = [];
  private ffWeights2: number[][][] = [];

  // Batch renorm state for each layer
  private renormStates: BatchRenormState[] = [];

  constructor(
    vocab: string[],
    hiddenSize = 64,
    lr = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'adam',
    momentum = 0.9,
    dropout = 0.1,
    seed = 1337,
    tokenizerConfig: TokenizerConfig = { mode: 'unicode' },
    transformerConfig: TransformerConfig = {}
  ) {
    // Initialize base ProNeuralLM (provides embedding layer and basic infrastructure)
    super(vocab, hiddenSize, lr, contextSize, optimizer, momentum, dropout, seed, tokenizerConfig);

    this.transformerConfig = { ...DEFAULT_TRANSFORMER_CONFIG, ...transformerConfig };
    this.initializeTransformerLayers();
    this.initializePositionEmbeddings();
  }

  /**
   * Initialize transformer layers
   */
  private initializeTransformerLayers(): void {
    const { numLayers, numHeads, ffHiddenDim, attentionDropout, dropConnectRate } = this.transformerConfig;
    const modelDim = this.getHiddenSize();

    this.transformerLayers = [];
    this.attentionWeights = [];
    this.ffWeights1 = [];
    this.ffWeights2 = [];
    this.renormStates = [];

    for (let i = 0; i < numLayers; i++) {
      const renormState: BatchRenormState = {
        runningMean: new Array(modelDim).fill(0),
        runningVar: new Array(modelDim).fill(1),
        momentum: 0.99,
        epsilon: 1e-5,
        rMax: 3.0,
        dMax: 5.0
      };

      const config: MiniTransformerConfig = {
        modelDim,
        heads: numHeads,
        ff: {
          hiddenDim: ffHiddenDim,
          activation: (x: number) => Math.max(0, x) // ReLU
        },
        attentionDropout,
        dropConnectRate,
        renormState
      };

      this.transformerLayers.push(new MiniTransformerBlock(config));
      this.renormStates.push(renormState);

      // Initialize attention weights (Q, K, V)
      const headDim = modelDim / numHeads;
      this.attentionWeights.push({
        query: this.initWeightMatrix([numHeads, modelDim, headDim]),
        key: this.initWeightMatrix([numHeads, modelDim, headDim]),
        value: this.initWeightMatrix([numHeads, modelDim, headDim])
      });

      // Initialize feedforward weights
      this.ffWeights1.push(this.initWeightMatrix([modelDim, ffHiddenDim]));
      this.ffWeights2.push(this.initWeightMatrix([ffHiddenDim, modelDim]));
    }
  }

  /**
   * Initialize position embeddings
   */
  private initializePositionEmbeddings(): void {
    const modelDim = this.getHiddenSize();
    this.positionEmbeddings = [];

    for (let pos = 0; pos < this.maxSeqLength; pos++) {
      const embedding = new Array(modelDim);
      for (let i = 0; i < modelDim; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / modelDim);
        embedding[i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
      this.positionEmbeddings.push(embedding);
    }
  }

  /**
   * Initialize weight matrix with He initialization
   */
  private initWeightMatrix(shape: number[]): any {
    if (shape.length === 2) {
      const [rows, cols] = shape;
      const scale = Math.sqrt(2.0 / rows);
      return Array(rows)
        .fill(0)
        .map(() =>
          Array(cols)
            .fill(0)
            .map(() => (Math.random() - 0.5) * 2 * scale)
        );
    } else if (shape.length === 3) {
      const [depth, rows, cols] = shape;
      return Array(depth)
        .fill(0)
        .map(() => this.initWeightMatrix([rows, cols]));
    }
    return [];
  }

  /**
   * Add position embeddings to input embeddings
   */
  private addPositionEmbeddings(embeddings: number[][]): number[][] {
    return embeddings.map((emb, pos) => {
      if (pos >= this.maxSeqLength) return emb;
      const posEmb = this.positionEmbeddings[pos];
      return emb.map((val, i) => val + posEmb[i]);
    });
  }

  /**
   * Transformer forward pass (placeholder for Phase 2 implementation)
   * @private
   */
  private async transformerForward(inputEmbeddings: number[][]): Promise<number[][]> {
    // Add position embeddings
    let hidden = this.addPositionEmbeddings(inputEmbeddings);

    // TODO Phase 2: Implement full transformer forward pass with proper attention mechanism
    // For now, the transformer layers are initialized but not used in training
    // The base ProNeuralLM feedforward architecture is used instead

    return hidden;
  }

  /**
   * Get architecture type for display
   */
  getArchitectureType(): string {
    return 'Transformer';
  }

  /**
   * Get transformer-specific info
   */
  getTransformerInfo(): {
    numLayers: number;
    numHeads: number;
    ffHiddenDim: number;
  } {
    return {
      numLayers: this.transformerConfig.numLayers,
      numHeads: this.transformerConfig.numHeads,
      ffHiddenDim: this.transformerConfig.ffHiddenDim
    };
  }

  /**
   * Train method - uses base class training for Phase 1
   * (Full transformer training with self-attention backpropagation will be added in Phase 2)
   */
  async train(text: string, epochs = 1): Promise<{
    readonly loss: number;
    readonly accuracy: number;
    readonly history: { loss: number; accuracy: number; timestamp: number }[];
  }> {
    // TODO: Implement full transformer backpropagation in Phase 2
    // For Phase 1, we use the base class training as a proof of concept
    // The transformer layers are initialized but the actual training still uses
    // the feedforward architecture from ProNeuralLM
    return super.train(text, epochs);
  }

  /**
   * Get hidden size (model dimension)
   */
  private getHiddenSize(): number {
    return (this as any).hiddenSize;
  }
}
