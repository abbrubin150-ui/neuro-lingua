/**
 * K-FAC (Kronecker-Factored Approximate Curvature) Optimizer
 *
 * K-FAC approximates the Fisher Information Matrix (natural gradient) using
 * Kronecker factorization: F ≈ A ⊗ G where:
 * - A is the covariance of activations (input statistics)
 * - G is the covariance of gradients (output statistics)
 *
 * This allows efficient approximate natural gradient descent:
 * θ ← θ - η × F^{-1} ∇L ≈ θ - η × (A^{-1} ⊗ G^{-1}) vec(∇L)
 *
 * Benefits:
 * - Automatic per-layer learning rate scaling
 * - Better convergence than first-order methods
 * - Invariant to reparameterization
 *
 * References:
 * - Martens & Grosse (2015) "Optimizing Neural Networks with Kronecker-factored
 *   Approximate Curvature"
 * - Grosse & Martens (2016) "A Kronecker-factored approximate Fisher matrix for
 *   convolution layers"
 *
 * @module training/KFACOptimizer
 */

export interface KFACConfig {
  /** Base learning rate (default: 0.01) */
  lr?: number;
  /** Exponential moving average decay for statistics (default: 0.95) */
  emaDecay?: number;
  /** Damping factor for numerical stability (default: 0.001) */
  damping?: number;
  /** Weight decay / L2 regularization (default: 0) */
  weightDecay?: number;
  /** Update statistics every N steps (default: 1) */
  statisticsUpdateInterval?: number;
  /** Invert curvature matrices every N steps (default: 10) */
  inversionInterval?: number;
  /** Clip curvature eigenvalues to this minimum (default: 1e-7) */
  minEigenvalue?: number;
  /** Use momentum on preconditioned gradients (default: 0.9) */
  momentum?: number;
}

interface LayerStatistics {
  /** Input activation covariance (A matrix) */
  A: number[][];
  /** Output gradient covariance (G matrix) */
  G: number[][];
  /** Inverse of A (cached) */
  AInv: number[][] | null;
  /** Inverse of G (cached) */
  GInv: number[][] | null;
  /** Momentum buffer */
  momentum: number[][] | null;
  /** Number of samples accumulated */
  sampleCount: number;
}

export interface KFACState {
  /** Layer statistics keyed by parameter name */
  layerStats: Map<string, LayerStatistics>;
  /** Current step count */
  step: number;
}

const DEFAULT_KFAC_CONFIG: Required<KFACConfig> = {
  lr: 0.01,
  emaDecay: 0.95,
  damping: 0.001,
  weightDecay: 0,
  statisticsUpdateInterval: 1,
  inversionInterval: 10,
  minEigenvalue: 1e-7,
  momentum: 0.9
};

// ============================================================================
// Matrix Utilities
// ============================================================================

/**
 * Create identity matrix
 */
function eye(n: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 1;
    result.push(row);
  }
  return result;
}

/**
 * Create zero matrix
 */
function zeros(rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result.push(new Array(cols).fill(0));
  }
  return result;
}

/**
 * Matrix addition: C = A + B
 */
function matAdd(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((val, j) => val + B[i][j]));
}

/**
 * Scalar multiplication: C = α × A
 */
function matScale(A: number[][], alpha: number): number[][] {
  return A.map((row) => row.map((val) => val * alpha));
}

/**
 * Matrix multiplication: C = A × B
 */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B[0].length;
  const k = B.length;

  const C: number[][] = zeros(m, n);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) {
        sum += A[i][l] * B[l][j];
      }
      C[i][j] = sum;
    }
  }

  return C;
}

/**
 * Matrix transpose
 */
function transpose(A: number[][]): number[][] {
  if (A.length === 0) return [];
  return A[0].map((_, i) => A.map((row) => row[i]));
}

/**
 * Outer product: A = x × y^T
 */
function outerProduct(x: number[], y: number[]): number[][] {
  const m = x.length;
  const n = y.length;
  const result: number[][] = [];

  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(x[i] * y[j]);
    }
    result.push(row);
  }

  return result;
}

/**
 * Add damping to diagonal: A + λI
 */
function addDamping(A: number[][], damping: number): number[][] {
  return A.map((row, i) => row.map((val, j) => (i === j ? val + damping : val)));
}

/**
 * Cholesky decomposition: A = L × L^T
 * Returns lower triangular matrix L
 */
function cholesky(A: number[][]): number[][] | null {
  const n = A.length;
  const L = zeros(n, n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];

      for (let k = 0; k < j; k++) {
        sum -= L[i][k] * L[j][k];
      }

      if (i === j) {
        if (sum <= 0) return null; // Not positive definite
        L[i][j] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }

  return L;
}

/**
 * Solve L × x = b for lower triangular L (forward substitution)
 */
function forwardSubstitution(L: number[][], b: number[]): number[] {
  const n = L.length;
  const x = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) {
      sum -= L[i][j] * x[j];
    }
    x[i] = sum / L[i][i];
  }

  return x;
}

/**
 * Solve L^T × x = b for lower triangular L (backward substitution)
 */
function backwardSubstitution(L: number[][], b: number[]): number[] {
  const n = L.length;
  const x = new Array(n).fill(0);

  for (let i = n - 1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i + 1; j < n; j++) {
      sum -= L[j][i] * x[j];
    }
    x[i] = sum / L[i][i];
  }

  return x;
}

/**
 * Compute matrix inverse using Cholesky decomposition
 * For symmetric positive definite matrices
 */
function invertSPD(A: number[][], damping: number, minEigenvalue: number): number[][] {
  const dampedA = addDamping(A, damping);
  const n = dampedA.length;

  // Try Cholesky decomposition first
  const L = cholesky(dampedA);

  if (L) {
    // A^{-1} = (L L^T)^{-1}
    // Solve for each column of identity matrix
    const AInv = zeros(n, n);
    for (let col = 0; col < n; col++) {
      const e = new Array(n).fill(0);
      e[col] = 1;
      const y = forwardSubstitution(L, e);
      const x = backwardSubstitution(L, y);
      for (let row = 0; row < n; row++) {
        AInv[row][col] = x[row];
      }
    }
    return AInv;
  }

  // Fallback: diagonal approximation with eigenvalue clipping
  const AInv = zeros(n, n);
  for (let i = 0; i < n; i++) {
    const eigenEst = Math.max(dampedA[i][i], minEigenvalue);
    AInv[i][i] = 1 / eigenEst;
  }
  return AInv;
}

// ============================================================================
// K-FAC Optimizer Class
// ============================================================================

export class KFACOptimizer {
  private config: Required<KFACConfig>;
  private state: KFACState;

  constructor(config: KFACConfig = {}) {
    this.config = { ...DEFAULT_KFAC_CONFIG, ...config };
    this.state = {
      layerStats: new Map(),
      step: 0
    };
  }

  /**
   * Initialize or get layer statistics
   */
  private getLayerStats(key: string, inputDim: number, outputDim: number): LayerStatistics {
    if (!this.state.layerStats.has(key)) {
      this.state.layerStats.set(key, {
        A: zeros(inputDim, inputDim),
        G: zeros(outputDim, outputDim),
        AInv: null,
        GInv: null,
        momentum: null,
        sampleCount: 0
      });
    }
    return this.state.layerStats.get(key)!;
  }

  /**
   * Update activation statistics (A matrix)
   *
   * A ← ρA + (1-ρ) × (1/n) Σ a_i a_i^T
   *
   * @param key Layer identifier
   * @param activations Input activations [batch, inputDim]
   */
  updateActivationStatistics(key: string, activations: number[][]): void {
    const inputDim = activations[0]?.length ?? 0;
    if (inputDim === 0) return;

    const stats = this.getLayerStats(key, inputDim, 1);
    const { emaDecay } = this.config;
    const batchSize = activations.length;

    // Compute batch covariance
    const batchCov = zeros(inputDim, inputDim);
    for (const activation of activations) {
      const outer = outerProduct(activation, activation);
      for (let i = 0; i < inputDim; i++) {
        for (let j = 0; j < inputDim; j++) {
          batchCov[i][j] += outer[i][j];
        }
      }
    }

    // Scale by batch size
    for (let i = 0; i < inputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        batchCov[i][j] /= batchSize;
      }
    }

    // EMA update
    if (stats.sampleCount === 0) {
      stats.A = batchCov;
    } else {
      stats.A = matAdd(matScale(stats.A, emaDecay), matScale(batchCov, 1 - emaDecay));
    }

    stats.sampleCount++;
    stats.AInv = null; // Invalidate cached inverse
  }

  /**
   * Update gradient statistics (G matrix)
   *
   * G ← ρG + (1-ρ) × (1/n) Σ g_i g_i^T
   *
   * @param key Layer identifier
   * @param outputGradients Output gradients [batch, outputDim]
   */
  updateGradientStatistics(key: string, outputGradients: number[][]): void {
    const outputDim = outputGradients[0]?.length ?? 0;
    if (outputDim === 0) return;

    const stats = this.getLayerStats(key, 1, outputDim);
    const { emaDecay } = this.config;
    const batchSize = outputGradients.length;

    // Compute batch covariance
    const batchCov = zeros(outputDim, outputDim);
    for (const gradient of outputGradients) {
      const outer = outerProduct(gradient, gradient);
      for (let i = 0; i < outputDim; i++) {
        for (let j = 0; j < outputDim; j++) {
          batchCov[i][j] += outer[i][j];
        }
      }
    }

    // Scale by batch size
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < outputDim; j++) {
        batchCov[i][j] /= batchSize;
      }
    }

    // EMA update
    if (stats.sampleCount === 0) {
      stats.G = batchCov;
    } else {
      stats.G = matAdd(matScale(stats.G, emaDecay), matScale(batchCov, 1 - emaDecay));
    }

    stats.GInv = null; // Invalidate cached inverse
  }

  /**
   * Update both activation and gradient statistics
   */
  updateStatistics(
    key: string,
    activations: number[][],
    outputGradients: number[][],
    inputDim: number,
    outputDim: number
  ): void {
    if (this.state.step % this.config.statisticsUpdateInterval !== 0) {
      return;
    }

    const stats = this.getLayerStats(key, inputDim, outputDim);
    const { emaDecay } = this.config;
    const batchSize = activations.length;

    // Compute activation covariance
    const batchA = zeros(inputDim, inputDim);
    for (const activation of activations) {
      const outer = outerProduct(activation, activation);
      for (let i = 0; i < inputDim; i++) {
        for (let j = 0; j < inputDim; j++) {
          batchA[i][j] += outer[i][j];
        }
      }
    }

    // Compute gradient covariance
    const batchG = zeros(outputDim, outputDim);
    for (const gradient of outputGradients) {
      const outer = outerProduct(gradient, gradient);
      for (let i = 0; i < outputDim; i++) {
        for (let j = 0; j < outputDim; j++) {
          batchG[i][j] += outer[i][j];
        }
      }
    }

    // Scale by batch size
    for (let i = 0; i < inputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        batchA[i][j] /= batchSize;
      }
    }
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < outputDim; j++) {
        batchG[i][j] /= batchSize;
      }
    }

    // EMA update
    if (stats.sampleCount === 0) {
      stats.A = batchA;
      stats.G = batchG;
    } else {
      stats.A = matAdd(matScale(stats.A, emaDecay), matScale(batchA, 1 - emaDecay));
      stats.G = matAdd(matScale(stats.G, emaDecay), matScale(batchG, 1 - emaDecay));
    }

    stats.sampleCount++;
    stats.AInv = null;
    stats.GInv = null;
  }

  /**
   * Update curvature matrix inverses
   */
  private updateInverses(stats: LayerStatistics): void {
    const { damping, minEigenvalue } = this.config;

    if (stats.A.length > 0) {
      stats.AInv = invertSPD(stats.A, damping, minEigenvalue);
    }
    if (stats.G.length > 0) {
      stats.GInv = invertSPD(stats.G, damping, minEigenvalue);
    }
  }

  /**
   * Apply K-FAC update to a matrix parameter
   *
   * Preconditioned gradient: G' = A^{-1} × G × G^{-1}
   * Update: W ← W - η × G' - η × λ × W
   *
   * @param W Weight matrix to update [outputDim, inputDim]
   * @param G Gradient matrix [outputDim, inputDim]
   * @param key Layer identifier
   */
  updateMatrix(W: number[][], G: number[][], key: string): void {
    const outputDim = W.length;
    const inputDim = W[0]?.length ?? 0;
    if (outputDim === 0 || inputDim === 0) return;

    const stats = this.getLayerStats(key, inputDim, outputDim);
    const { lr, weightDecay, momentum, inversionInterval } = this.config;

    // Update inverses periodically
    if (this.state.step % inversionInterval === 0 || !stats.AInv || !stats.GInv) {
      this.updateInverses(stats);
    }

    // If we don't have statistics yet, fall back to standard gradient descent
    if (!stats.AInv || !stats.GInv) {
      for (let i = 0; i < outputDim; i++) {
        for (let j = 0; j < inputDim; j++) {
          W[i][j] -= lr * G[i][j] + lr * weightDecay * W[i][j];
        }
      }
      return;
    }

    // Compute preconditioned gradient: G' = G^{-1} × G × A^{-1}
    // Using the Kronecker product property:
    // vec(A^{-1} G G^{-1}) = (G^{-1} ⊗ A^{-1}) vec(G)
    const temp = matMul(stats.GInv, G);
    const preconditioned = matMul(temp, stats.AInv);

    // Initialize momentum buffer if needed
    if (!stats.momentum) {
      stats.momentum = zeros(outputDim, inputDim);
    }

    // Apply update with momentum
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        // Momentum update
        stats.momentum[i][j] = momentum * stats.momentum[i][j] + (1 - momentum) * preconditioned[i][j];

        // Weight update with weight decay
        W[i][j] -= lr * stats.momentum[i][j] + lr * weightDecay * W[i][j];
      }
    }
  }

  /**
   * Apply K-FAC update to a vector parameter (bias)
   * Uses diagonal approximation for simplicity
   *
   * @param b Bias vector to update
   * @param g Gradient vector
   * @param key Parameter identifier
   */
  updateVector(b: number[], g: number[], key: string): void {
    const { lr, weightDecay, momentum, damping } = this.config;

    const stats = this.getLayerStats(key + '_bias', b.length, 1);

    // Initialize momentum if needed
    if (!stats.momentum) {
      stats.momentum = [new Array(b.length).fill(0)];
    }

    const momentumBuf = stats.momentum[0];

    for (let i = 0; i < b.length; i++) {
      // Simple diagonal preconditioning
      const preconditioned = g[i] / (damping + Math.abs(g[i]));

      // Momentum update
      momentumBuf[i] = momentum * momentumBuf[i] + (1 - momentum) * preconditioned;

      // Weight update
      b[i] -= lr * momentumBuf[i] + lr * weightDecay * b[i];
    }
  }

  /**
   * Increment step counter (call after each batch)
   */
  step(): void {
    this.state.step++;
  }

  /**
   * Reset optimizer state
   */
  reset(): void {
    this.state.layerStats.clear();
    this.state.step = 0;
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<KFACConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<KFACConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current step count
   */
  getStep(): number {
    return this.state.step;
  }

  /**
   * Export optimizer state for serialization
   */
  exportState(): {
    config: Required<KFACConfig>;
    step: number;
    layerStats: Record<
      string,
      {
        A: number[][];
        G: number[][];
        sampleCount: number;
      }
    >;
  } {
    const layerStats: Record<
      string,
      {
        A: number[][];
        G: number[][];
        sampleCount: number;
      }
    > = {};

    for (const [key, stats] of this.state.layerStats) {
      layerStats[key] = {
        A: stats.A,
        G: stats.G,
        sampleCount: stats.sampleCount
      };
    }

    return {
      config: this.config,
      step: this.state.step,
      layerStats
    };
  }

  /**
   * Import optimizer state from serialization
   */
  importState(state: {
    config?: KFACConfig;
    step?: number;
    layerStats?: Record<
      string,
      {
        A: number[][];
        G: number[][];
        sampleCount: number;
      }
    >;
  }): void {
    if (state.config) {
      this.config = { ...DEFAULT_KFAC_CONFIG, ...state.config };
    }
    if (state.step !== undefined) {
      this.state.step = state.step;
    }
    if (state.layerStats) {
      this.state.layerStats.clear();
      for (const [key, stats] of Object.entries(state.layerStats)) {
        this.state.layerStats.set(key, {
          A: stats.A,
          G: stats.G,
          AInv: null,
          GInv: null,
          momentum: null,
          sampleCount: stats.sampleCount
        });
      }
    }
  }

  /**
   * Get diagnostic information about curvature estimates
   */
  getDiagnostics(key: string): {
    activationCondition: number;
    gradientCondition: number;
    estimatedLR: number;
  } | null {
    const stats = this.state.layerStats.get(key);
    if (!stats || stats.sampleCount === 0) return null;

    // Estimate condition numbers from diagonal
    const diagA = stats.A.map((row, i) => row[i]);
    const diagG = stats.G.map((row, i) => row[i]);

    const maxA = Math.max(...diagA.map(Math.abs));
    const minA = Math.min(...diagA.map((v) => Math.max(Math.abs(v), 1e-10)));
    const maxG = Math.max(...diagG.map(Math.abs));
    const minG = Math.min(...diagG.map((v) => Math.max(Math.abs(v), 1e-10)));

    const activationCondition = maxA / minA;
    const gradientCondition = maxG / minG;

    // Estimate optimal learning rate based on curvature
    const estimatedLR = 1 / Math.sqrt(maxA * maxG + this.config.damping);

    return {
      activationCondition,
      gradientCondition,
      estimatedLR
    };
  }
}

/**
 * Default K-FAC configuration optimized for neural language models
 */
export const KFAC_DEFAULTS = {
  lr: 0.01,
  emaDecay: 0.95,
  damping: 0.001,
  weightDecay: 0,
  statisticsUpdateInterval: 1,
  inversionInterval: 10,
  minEigenvalue: 1e-7,
  momentum: 0.9
} as const;

/**
 * K-FAC configuration for fine-tuning (more conservative)
 */
export const KFAC_FINETUNE_DEFAULTS = {
  lr: 0.003,
  emaDecay: 0.99,
  damping: 0.01,
  weightDecay: 1e-4,
  statisticsUpdateInterval: 1,
  inversionInterval: 20,
  minEigenvalue: 1e-6,
  momentum: 0.95
} as const;
