/**
 * RMSNorm (Root Mean Square Normalization)
 * Simpler alternative to LayerNorm used in T5, LLaMA, PaLM
 *
 * Key benefits:
 * - 20% less memory (only γ, no β)
 * - 2x faster (no mean calculation)
 * - Equivalent performance to LayerNorm in practice
 *
 * Formula: RMSNorm(x) = (x / RMS(x)) ⊙ γ
 * where RMS(x) = sqrt(mean(x²) + ε)
 *
 * Reference: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
 */

export interface RMSNormState {
  gamma: number[]; // Scale parameters (learnable)
  epsilon: number; // Numerical stability constant
}

export class RMSNorm {
  private gamma: number[];
  private epsilon: number;
  private dimension: number;

  // Gradients for backprop
  private gammaGrad: number[] | null = null;

  // Cache for backward pass
  private lastInput: number[] | null = null;
  private lastRMS: number | null = null;
  private lastNormalized: number[] | null = null;

  /**
   * Create RMSNorm layer
   * @param dimension - Input dimension
   * @param epsilon - Small constant for numerical stability (default: 1e-6)
   */
  constructor(dimension: number, epsilon: number = 1e-6) {
    this.dimension = dimension;
    this.epsilon = epsilon;

    // Initialize gamma to ones (no normalization initially)
    this.gamma = new Array(dimension).fill(1.0);
    this.gammaGrad = new Array(dimension).fill(0.0);
  }

  /**
   * Forward pass: normalize input
   * @param x - Input vector (length = dimension)
   * @returns Normalized output
   */
  forward(x: number[]): number[] {
    if (x.length !== this.dimension) {
      throw new Error(
        `Input dimension ${x.length} doesn't match layer dimension ${this.dimension}`
      );
    }

    // Compute RMS: sqrt(mean(x²) + ε)
    let sumSquares = 0;
    for (let i = 0; i < this.dimension; i++) {
      sumSquares += x[i] * x[i];
    }
    const rms = Math.sqrt(sumSquares / this.dimension + this.epsilon);

    // Normalize and scale: (x / RMS) ⊙ γ
    const normalized = new Array(this.dimension);
    const output = new Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      normalized[i] = x[i] / rms;
      output[i] = normalized[i] * this.gamma[i];
    }

    // Cache for backward pass
    this.lastInput = [...x];
    this.lastRMS = rms;
    this.lastNormalized = normalized;

    return output;
  }

  /**
   * Backward pass: compute gradients
   * @param gradOutput - Gradient from next layer
   * @returns Gradient with respect to input
   */
  backward(gradOutput: number[]): number[] {
    if (!this.lastInput || !this.lastRMS || !this.lastNormalized) {
      throw new Error('Must call forward() before backward()');
    }

    const x = this.lastInput;
    const rms = this.lastRMS;
    const normalized = this.lastNormalized;

    // Gradient w.r.t. gamma: ∂L/∂γ = ∂L/∂y ⊙ normalized
    if (!this.gammaGrad) {
      this.gammaGrad = new Array(this.dimension).fill(0);
    }
    for (let i = 0; i < this.dimension; i++) {
      this.gammaGrad[i] += gradOutput[i] * normalized[i];
    }

    // Gradient w.r.t. normalized: ∂L/∂normalized = ∂L/∂y ⊙ γ
    const gradNormalized = new Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      gradNormalized[i] = gradOutput[i] * this.gamma[i];
    }

    // Gradient w.r.t. input: chain rule through normalization
    // d(x/RMS)/dx = 1/RMS - (x/RMS) * (x/RMS³) / dimension
    const gradInput = new Array(this.dimension);
    const rms3 = rms * rms * rms;

    let dotProduct = 0;
    for (let i = 0; i < this.dimension; i++) {
      dotProduct += gradNormalized[i] * x[i];
    }

    for (let i = 0; i < this.dimension; i++) {
      // Derivative of normalization
      const dNorm = 1.0 / rms - (x[i] * x[i]) / (this.dimension * rms3);
      gradInput[i] = gradNormalized[i] * dNorm;

      // Account for contribution through RMS
      const dRMS = -(normalized[i] / rms) * (x[i] / this.dimension);
      gradInput[i] += dotProduct * dRMS;
    }

    return gradInput;
  }

  /**
   * Update parameters with gradients
   * @param learningRate - Learning rate for gradient descent
   */
  updateParameters(learningRate: number): void {
    if (!this.gammaGrad) return;

    for (let i = 0; i < this.dimension; i++) {
      this.gamma[i] -= learningRate * this.gammaGrad[i];
    }
  }

  /**
   * Clear accumulated gradients
   */
  zeroGradients(): void {
    if (this.gammaGrad) {
      this.gammaGrad.fill(0);
    }
  }

  /**
   * Get number of trainable parameters
   */
  getParameterCount(): number {
    return this.dimension; // Only gamma (no beta like LayerNorm)
  }

  /**
   * Export layer state for serialization
   */
  exportState(): RMSNormState {
    return {
      gamma: [...this.gamma],
      epsilon: this.epsilon
    };
  }

  /**
   * Import layer state from serialization
   */
  static loadState(state: RMSNormState, dimension: number): RMSNorm {
    const layer = new RMSNorm(dimension, state.epsilon);
    layer.gamma = [...state.gamma];
    return layer;
  }

  /**
   * Reset layer to initial state
   */
  reset(): void {
    this.gamma.fill(1.0);
    this.gammaGrad?.fill(0.0);
    this.lastInput = null;
    this.lastRMS = null;
    this.lastNormalized = null;
  }
}

/**
 * Standalone RMSNorm function (no gradients)
 * Useful for inference or when you don't need backprop
 */
export function rmsNorm(x: number[], gamma: number[], epsilon: number = 1e-6): number[] {
  const dimension = x.length;

  // Compute RMS
  let sumSquares = 0;
  for (let i = 0; i < dimension; i++) {
    sumSquares += x[i] * x[i];
  }
  const rms = Math.sqrt(sumSquares / dimension + epsilon);

  // Normalize and scale
  const output = new Array(dimension);
  for (let i = 0; i < dimension; i++) {
    output[i] = (x[i] / rms) * gamma[i];
  }

  return output;
}

/**
 * Batch RMSNorm - process multiple vectors at once
 * More efficient for batch training
 */
export function batchRMSNorm(
  batch: number[][],
  gamma: number[],
  epsilon: number = 1e-6
): number[][] {
  return batch.map((x) => rmsNorm(x, gamma, epsilon));
}
