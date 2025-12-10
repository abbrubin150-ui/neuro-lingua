/**
 * Lion Optimizer (EvoLved Sign Momentum)
 *
 * Lion is a simple and effective optimizer discovered through program search.
 * It uses the sign of the momentum for updates, leading to:
 * - 50% less memory than Adam (only one momentum buffer)
 * - 1.5-2× faster convergence
 * - More stable training with lower learning rates
 *
 * Algorithm:
 *   update = sign(β₁ × m + (1 - β₁) × g)
 *   θ ← θ - η × update - η × λ × θ  (with weight decay)
 *   m ← β₂ × m + (1 - β₂) × g
 *
 * Reference: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
 *
 * @module training/LionOptimizer
 */

export interface LionConfig {
  /** Learning rate (default: 3e-4, lower than Adam due to sign() behavior) */
  lr?: number;
  /** Momentum decay for update computation (default: 0.9) */
  beta1?: number;
  /** Momentum decay for state update (default: 0.99) */
  beta2?: number;
  /** Weight decay / L2 regularization (default: 0.01) */
  weightDecay?: number;
}

export interface LionState {
  /** Momentum buffer for matrices (keyed by parameter name) */
  momentumMatrices: Map<string, number[][]>;
  /** Momentum buffer for vectors (keyed by parameter name) */
  momentumVectors: Map<string, number[]>;
}

const DEFAULT_LION_CONFIG: Required<LionConfig> = {
  lr: 3e-4,
  beta1: 0.9,
  beta2: 0.99,
  weightDecay: 0.01
};

/**
 * Lion optimizer class for standalone usage
 */
export class LionOptimizer {
  private config: Required<LionConfig>;
  private state: LionState;

  constructor(config: LionConfig = {}) {
    this.config = { ...DEFAULT_LION_CONFIG, ...config };
    this.state = {
      momentumMatrices: new Map(),
      momentumVectors: new Map()
    };
  }

  /**
   * Get or initialize momentum buffer for a matrix parameter
   */
  private getMatrixMomentum(key: string, rows: number, cols: number): number[][] {
    if (!this.state.momentumMatrices.has(key)) {
      const m: number[][] = [];
      for (let i = 0; i < rows; i++) {
        m.push(new Array(cols).fill(0));
      }
      this.state.momentumMatrices.set(key, m);
    }
    return this.state.momentumMatrices.get(key)!;
  }

  /**
   * Get or initialize momentum buffer for a vector parameter
   */
  private getVectorMomentum(key: string, length: number): number[] {
    if (!this.state.momentumVectors.has(key)) {
      this.state.momentumVectors.set(key, new Array(length).fill(0));
    }
    return this.state.momentumVectors.get(key)!;
  }

  /**
   * Apply Lion update to a matrix parameter
   *
   * @param W - Weight matrix to update (modified in place)
   * @param G - Gradient matrix
   * @param key - Unique identifier for this parameter
   */
  updateMatrix(W: number[][], G: number[][], key: string): void {
    const { lr, beta1, beta2, weightDecay } = this.config;
    const M = this.getMatrixMomentum(key, W.length, W[0]?.length ?? 0);

    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      const gRow = G[i];
      const mRow = M[i];

      for (let j = 0; j < row.length; j++) {
        const g = gRow[j];

        // Compute update direction: sign(β₁m + (1-β₁)g)
        const interpolated = beta1 * mRow[j] + (1 - beta1) * g;
        const update = Math.sign(interpolated);

        // Apply update with weight decay: θ -= η × sign + η × λ × θ
        row[j] -= lr * update + lr * weightDecay * row[j];

        // Update momentum: m = β₂m + (1-β₂)g
        mRow[j] = beta2 * mRow[j] + (1 - beta2) * g;
      }
    }
  }

  /**
   * Apply Lion update to a vector parameter
   *
   * @param b - Bias/parameter vector to update (modified in place)
   * @param g - Gradient vector
   * @param key - Unique identifier for this parameter
   */
  updateVector(b: number[], g: number[], key: string): void {
    const { lr, beta1, beta2, weightDecay } = this.config;
    const m = this.getVectorMomentum(key, b.length);

    for (let i = 0; i < b.length; i++) {
      const gi = g[i];

      // Compute update direction: sign(β₁m + (1-β₁)g)
      const interpolated = beta1 * m[i] + (1 - beta1) * gi;
      const update = Math.sign(interpolated);

      // Apply update with weight decay: θ -= η × sign + η × λ × θ
      b[i] -= lr * update + lr * weightDecay * b[i];

      // Update momentum: m = β₂m + (1-β₂)g
      m[i] = beta2 * m[i] + (1 - beta2) * gi;
    }
  }

  /**
   * Apply Lion update to a single row of a matrix (for sparse updates)
   *
   * @param W - Weight matrix
   * @param rowIdx - Row index to update
   * @param gRow - Gradient for the row
   * @param key - Unique identifier for this parameter
   */
  updateRow(W: number[][], rowIdx: number, gRow: number[], key: string): void {
    const { lr, beta1, beta2, weightDecay } = this.config;
    const M = this.getMatrixMomentum(key, W.length, W[0]?.length ?? 0);

    const row = W[rowIdx];
    const mRow = M[rowIdx];

    for (let j = 0; j < row.length; j++) {
      const g = gRow[j];

      // Compute update direction: sign(β₁m + (1-β₁)g)
      const interpolated = beta1 * mRow[j] + (1 - beta1) * g;
      const update = Math.sign(interpolated);

      // Apply update with weight decay
      row[j] -= lr * update + lr * weightDecay * row[j];

      // Update momentum
      mRow[j] = beta2 * mRow[j] + (1 - beta2) * g;
    }
  }

  /**
   * Reset optimizer state (e.g., when starting new training)
   */
  reset(): void {
    this.state.momentumMatrices.clear();
    this.state.momentumVectors.clear();
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<LionConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration (e.g., for learning rate scheduling)
   */
  setConfig(config: Partial<LionConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Export optimizer state for serialization
   */
  exportState(): {
    config: Required<LionConfig>;
    momentumMatrices: Record<string, number[][]>;
    momentumVectors: Record<string, number[]>;
  } {
    return {
      config: this.config,
      momentumMatrices: Object.fromEntries(this.state.momentumMatrices),
      momentumVectors: Object.fromEntries(this.state.momentumVectors)
    };
  }

  /**
   * Import optimizer state from serialization
   */
  importState(state: {
    config?: LionConfig;
    momentumMatrices?: Record<string, number[][]>;
    momentumVectors?: Record<string, number[]>;
  }): void {
    if (state.config) {
      this.config = { ...DEFAULT_LION_CONFIG, ...state.config };
    }
    if (state.momentumMatrices) {
      this.state.momentumMatrices = new Map(Object.entries(state.momentumMatrices));
    }
    if (state.momentumVectors) {
      this.state.momentumVectors = new Map(Object.entries(state.momentumVectors));
    }
  }
}

/**
 * Default Lion configuration optimized for language models
 */
export const LION_DEFAULTS = {
  lr: 3e-4,
  beta1: 0.9,
  beta2: 0.99,
  weightDecay: 0.01
} as const;

/**
 * Lion configuration recommended for fine-tuning
 */
export const LION_FINETUNE_DEFAULTS = {
  lr: 1e-4,
  beta1: 0.9,
  beta2: 0.99,
  weightDecay: 0.01
} as const;
