/**
 * Sophia Optimizer (Second-order Stochastic Optimizer)
 *
 * Sophia is a lightweight second-order optimizer that uses diagonal Hessian
 * estimates for adaptive per-parameter learning rate scaling.
 *
 * Key advantages over Adam/Lion:
 * - 2× faster convergence (fewer epochs to reach target loss)
 * - Better generalization via curvature-aware updates
 * - Automatic learning rate adaptation per dimension
 * - Memory: ~2× parameters (momentum + Hessian diagonal)
 *
 * Algorithm (Sophia-G variant with Gauss-Newton Hessian):
 *   m_t = β₁ × m_{t-1} + (1 - β₁) × g_t                  (momentum)
 *   h_t = β₂ × h_{t-1} + (1 - β₂) × (g_t)²              (Gauss-Newton Hessian diagonal)
 *   θ_{t+1} = θ_t - η × clip(m_t / max(h_t, ε), -ρ, ρ)  (clipped update)
 *   θ_{t+1} -= η × λ × θ_t                               (weight decay)
 *
 * The clipping bound ρ prevents extreme updates when h_t is small.
 *
 * Reference: Liu et al. (2023) "Sophia: A Scalable Stochastic Second-order Optimizer
 *            for Language Model Pre-training" https://arxiv.org/abs/2305.14342
 *
 * @module training/SophiaOptimizer
 */

export interface SophiaConfig {
  /** Learning rate (default: 1e-4, lower than Adam due to second-order info) */
  lr?: number;
  /** Momentum decay (default: 0.965) */
  beta1?: number;
  /** Hessian diagonal EMA decay (default: 0.99) */
  beta2?: number;
  /** Weight decay / L2 regularization (default: 0.01) */
  weightDecay?: number;
  /** Numerical stability epsilon (default: 1e-12) */
  epsilon?: number;
  /** Update clipping bound ρ (default: 1.0) */
  rho?: number;
  /** Hessian update frequency (update every k steps, default: 10) */
  hessianUpdateFreq?: number;
  /** Use Hutchinson estimator for Hessian (stochastic, default: false) */
  useHutchinson?: boolean;
}

export interface SophiaState {
  /** First moment (momentum) for matrices */
  momentumMatrices: Map<string, number[][]>;
  /** First moment (momentum) for vectors */
  momentumVectors: Map<string, number[]>;
  /** Hessian diagonal estimates for matrices */
  hessianMatrices: Map<string, number[][]>;
  /** Hessian diagonal estimates for vectors */
  hessianVectors: Map<string, number[]>;
  /** Current step count */
  step: number;
}

const DEFAULT_SOPHIA_CONFIG: Required<SophiaConfig> = {
  lr: 1e-4,
  beta1: 0.965,
  beta2: 0.99,
  weightDecay: 0.01,
  epsilon: 1e-12,
  rho: 1.0,
  hessianUpdateFreq: 10,
  useHutchinson: false
};

/**
 * Sophia optimizer class implementing second-order stochastic optimization
 */
export class SophiaOptimizer {
  private config: Required<SophiaConfig>;
  private state: SophiaState;

  constructor(config: SophiaConfig = {}) {
    this.config = { ...DEFAULT_SOPHIA_CONFIG, ...config };
    this.state = {
      momentumMatrices: new Map(),
      momentumVectors: new Map(),
      hessianMatrices: new Map(),
      hessianVectors: new Map(),
      step: 0
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
   * Get or initialize Hessian diagonal estimate for a matrix parameter
   */
  private getMatrixHessian(key: string, rows: number, cols: number): number[][] {
    if (!this.state.hessianMatrices.has(key)) {
      const h: number[][] = [];
      for (let i = 0; i < rows; i++) {
        // Initialize with small positive value for stability
        h.push(new Array(cols).fill(this.config.epsilon));
      }
      this.state.hessianMatrices.set(key, h);
    }
    return this.state.hessianMatrices.get(key)!;
  }

  /**
   * Get or initialize Hessian diagonal estimate for a vector parameter
   */
  private getVectorHessian(key: string, length: number): number[] {
    if (!this.state.hessianVectors.has(key)) {
      this.state.hessianVectors.set(key, new Array(length).fill(this.config.epsilon));
    }
    return this.state.hessianVectors.get(key)!;
  }

  /**
   * Clip value to [-rho, rho] range
   */
  private clip(value: number): number {
    const { rho } = this.config;
    return Math.max(-rho, Math.min(rho, value));
  }

  /**
   * Check if Hessian should be updated this step
   */
  private shouldUpdateHessian(): boolean {
    return this.state.step % this.config.hessianUpdateFreq === 0;
  }

  /**
   * Apply Sophia update to a matrix parameter
   *
   * @param W - Weight matrix to update (modified in place)
   * @param G - Gradient matrix
   * @param key - Unique identifier for this parameter
   * @param loss - Optional current loss value for Hutchinson estimator
   */
  updateMatrix(W: number[][], G: number[][], key: string, loss?: number): void {
    const { lr, beta1, beta2, weightDecay, epsilon } = this.config;
    const rows = W.length;
    const cols = W[0]?.length ?? 0;

    const M = this.getMatrixMomentum(key, rows, cols);
    const H = this.getMatrixHessian(key, rows, cols);

    const updateHessian = this.shouldUpdateHessian();

    for (let i = 0; i < rows; i++) {
      const row = W[i];
      const gRow = G[i];
      const mRow = M[i];
      const hRow = H[i];

      for (let j = 0; j < cols; j++) {
        const g = gRow[j];

        // Update momentum: m = β₁m + (1-β₁)g
        mRow[j] = beta1 * mRow[j] + (1 - beta1) * g;

        // Update Hessian diagonal estimate (Gauss-Newton approximation): h = β₂h + (1-β₂)g²
        if (updateHessian) {
          hRow[j] = beta2 * hRow[j] + (1 - beta2) * g * g;
        }

        // Compute preconditioned update with clipping: clip(m / max(h, ε), -ρ, ρ)
        const hessianScale = Math.max(hRow[j], epsilon);
        const preconUpdate = this.clip(mRow[j] / hessianScale);

        // Apply update: θ -= η × preconUpdate
        row[j] -= lr * preconUpdate;

        // Apply weight decay: θ -= η × λ × θ
        row[j] -= lr * weightDecay * row[j];
      }
    }
  }

  /**
   * Apply Sophia update to a vector parameter
   *
   * @param b - Bias/parameter vector to update (modified in place)
   * @param g - Gradient vector
   * @param key - Unique identifier for this parameter
   */
  updateVector(b: number[], g: number[], key: string): void {
    const { lr, beta1, beta2, weightDecay, epsilon } = this.config;
    const length = b.length;

    const m = this.getVectorMomentum(key, length);
    const h = this.getVectorHessian(key, length);

    const updateHessian = this.shouldUpdateHessian();

    for (let i = 0; i < length; i++) {
      const gi = g[i];

      // Update momentum: m = β₁m + (1-β₁)g
      m[i] = beta1 * m[i] + (1 - beta1) * gi;

      // Update Hessian diagonal estimate (Gauss-Newton): h = β₂h + (1-β₂)g²
      if (updateHessian) {
        h[i] = beta2 * h[i] + (1 - beta2) * gi * gi;
      }

      // Compute preconditioned update with clipping
      const hessianScale = Math.max(h[i], epsilon);
      const preconUpdate = this.clip(m[i] / hessianScale);

      // Apply update: θ -= η × preconUpdate
      b[i] -= lr * preconUpdate;

      // Apply weight decay: θ -= η × λ × θ
      b[i] -= lr * weightDecay * b[i];
    }
  }

  /**
   * Apply Sophia update to a single row of a matrix (for sparse/embedding updates)
   *
   * @param W - Weight matrix
   * @param rowIdx - Row index to update
   * @param gRow - Gradient for the row
   * @param key - Unique identifier for this parameter
   */
  updateRow(W: number[][], rowIdx: number, gRow: number[], key: string): void {
    const { lr, beta1, beta2, weightDecay, epsilon } = this.config;
    const rows = W.length;
    const cols = W[0]?.length ?? 0;

    const M = this.getMatrixMomentum(key, rows, cols);
    const H = this.getMatrixHessian(key, rows, cols);

    const row = W[rowIdx];
    const mRow = M[rowIdx];
    const hRow = H[rowIdx];

    const updateHessian = this.shouldUpdateHessian();

    for (let j = 0; j < cols; j++) {
      const g = gRow[j];

      // Update momentum
      mRow[j] = beta1 * mRow[j] + (1 - beta1) * g;

      // Update Hessian diagonal
      if (updateHessian) {
        hRow[j] = beta2 * hRow[j] + (1 - beta2) * g * g;
      }

      // Compute preconditioned update with clipping
      const hessianScale = Math.max(hRow[j], epsilon);
      const preconUpdate = this.clip(mRow[j] / hessianScale);

      // Apply update
      row[j] -= lr * preconUpdate;
      row[j] -= lr * weightDecay * row[j];
    }
  }

  /**
   * Increment step counter (call after each mini-batch)
   */
  step(): void {
    this.state.step++;
  }

  /**
   * Get current step count
   */
  getStep(): number {
    return this.state.step;
  }

  /**
   * Reset optimizer state (e.g., when starting new training)
   */
  reset(): void {
    this.state.momentumMatrices.clear();
    this.state.momentumVectors.clear();
    this.state.hessianMatrices.clear();
    this.state.hessianVectors.clear();
    this.state.step = 0;
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<SophiaConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration (e.g., for learning rate scheduling)
   */
  setConfig(config: Partial<SophiaConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current Hessian diagonal statistics for monitoring
   */
  getHessianStats(): {
    matrices: { key: string; mean: number; max: number; min: number }[];
    vectors: { key: string; mean: number; max: number; min: number }[];
  } {
    const matrixStats: { key: string; mean: number; max: number; min: number }[] = [];
    const vectorStats: { key: string; mean: number; max: number; min: number }[] = [];

    for (const [key, matrix] of this.state.hessianMatrices) {
      const flat = matrix.flat();
      const sum = flat.reduce((a, b) => a + b, 0);
      matrixStats.push({
        key,
        mean: sum / flat.length,
        max: Math.max(...flat),
        min: Math.min(...flat)
      });
    }

    for (const [key, vector] of this.state.hessianVectors) {
      const sum = vector.reduce((a, b) => a + b, 0);
      vectorStats.push({
        key,
        mean: sum / vector.length,
        max: Math.max(...vector),
        min: Math.min(...vector)
      });
    }

    return { matrices: matrixStats, vectors: vectorStats };
  }

  /**
   * Export optimizer state for serialization
   */
  exportState(): {
    config: Required<SophiaConfig>;
    momentumMatrices: Record<string, number[][]>;
    momentumVectors: Record<string, number[]>;
    hessianMatrices: Record<string, number[][]>;
    hessianVectors: Record<string, number[]>;
    step: number;
  } {
    return {
      config: this.config,
      momentumMatrices: Object.fromEntries(this.state.momentumMatrices),
      momentumVectors: Object.fromEntries(this.state.momentumVectors),
      hessianMatrices: Object.fromEntries(this.state.hessianMatrices),
      hessianVectors: Object.fromEntries(this.state.hessianVectors),
      step: this.state.step
    };
  }

  /**
   * Import optimizer state from serialization
   */
  importState(state: {
    config?: SophiaConfig;
    momentumMatrices?: Record<string, number[][]>;
    momentumVectors?: Record<string, number[]>;
    hessianMatrices?: Record<string, number[][]>;
    hessianVectors?: Record<string, number[]>;
    step?: number;
  }): void {
    if (state.config) {
      this.config = { ...DEFAULT_SOPHIA_CONFIG, ...state.config };
    }
    if (state.momentumMatrices) {
      this.state.momentumMatrices = new Map(Object.entries(state.momentumMatrices));
    }
    if (state.momentumVectors) {
      this.state.momentumVectors = new Map(Object.entries(state.momentumVectors));
    }
    if (state.hessianMatrices) {
      this.state.hessianMatrices = new Map(Object.entries(state.hessianMatrices));
    }
    if (state.hessianVectors) {
      this.state.hessianVectors = new Map(Object.entries(state.hessianVectors));
    }
    if (state.step !== undefined) {
      this.state.step = state.step;
    }
  }
}

/**
 * Default Sophia configuration optimized for language models
 */
export const SOPHIA_DEFAULTS = {
  lr: 1e-4,
  beta1: 0.965,
  beta2: 0.99,
  weightDecay: 0.01,
  epsilon: 1e-12,
  rho: 1.0,
  hessianUpdateFreq: 10
} as const;

/**
 * Sophia configuration recommended for fine-tuning
 * (lower learning rate, less aggressive updates)
 */
export const SOPHIA_FINETUNE_DEFAULTS = {
  lr: 5e-5,
  beta1: 0.965,
  beta2: 0.99,
  weightDecay: 0.01,
  epsilon: 1e-12,
  rho: 0.5,
  hessianUpdateFreq: 20
} as const;

/**
 * Sophia configuration for aggressive training
 * (higher learning rate, more frequent Hessian updates)
 */
export const SOPHIA_AGGRESSIVE_DEFAULTS = {
  lr: 2e-4,
  beta1: 0.9,
  beta2: 0.95,
  weightDecay: 0.01,
  epsilon: 1e-12,
  rho: 2.0,
  hessianUpdateFreq: 5
} as const;
