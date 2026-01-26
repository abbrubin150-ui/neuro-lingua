/**
 * EMAWeights - Exponential Moving Average (Polyak Averaging) of Model Weights
 *
 * Maintains a shadow copy of model parameters that tracks an exponential
 * moving average of the training weights:
 *
 *   shadow_t = decay * shadow_{t-1} + (1 - decay) * theta_t
 *
 * Benefits:
 * - Smoother convergence and improved generalization
 * - Better validation/test metrics (use EMA weights for evaluation)
 * - Reduces sensitivity to learning rate and optimizer noise
 *
 * Typical decay values: 0.999 (fast updates), 0.9999 (slow, stable)
 *
 * Usage:
 *   const ema = new EMATracker(0.999);
 *   // After each training step:
 *   ema.update(model.getWeights());
 *   // For evaluation:
 *   const originalWeights = model.getWeights();
 *   model.setWeights(ema.getShadowWeights());
 *   // ... run evaluation ...
 *   model.setWeights(originalWeights);
 */

/**
 * Configuration for EMA weight averaging
 */
export interface EMAConfig {
  /** EMA decay factor (0.99-0.9999). Higher = slower updates, smoother average. */
  decay: number;
  /** Whether EMA is enabled */
  enabled: boolean;
}

/**
 * Weight snapshot structure matching ProNeuralLM.getWeights()
 */
export interface WeightSnapshot {
  embedding: number[][];
  wHidden: number[][];
  wOutput: number[][];
  bHidden: number[];
  bOutput: number[];
}

/**
 * Deep-clone a weight snapshot
 */
function cloneWeights(w: WeightSnapshot): WeightSnapshot {
  return {
    embedding: w.embedding.map((row) => [...row]),
    wHidden: w.wHidden.map((row) => [...row]),
    wOutput: w.wOutput.map((row) => [...row]),
    bHidden: [...w.bHidden],
    bOutput: [...w.bOutput]
  };
}

/**
 * Apply EMA update to a matrix: shadow = decay * shadow + (1-decay) * current
 */
function emaMatrix(shadow: number[][], current: number[][], decay: number): void {
  const oneMinusDecay = 1 - decay;
  for (let i = 0; i < shadow.length; i++) {
    const sRow = shadow[i];
    const cRow = current[i];
    for (let j = 0; j < sRow.length; j++) {
      sRow[j] = decay * sRow[j] + oneMinusDecay * cRow[j];
    }
  }
}

/**
 * Apply EMA update to a vector: shadow = decay * shadow + (1-decay) * current
 */
function emaVector(shadow: number[], current: number[], decay: number): void {
  const oneMinusDecay = 1 - decay;
  for (let i = 0; i < shadow.length; i++) {
    shadow[i] = decay * shadow[i] + oneMinusDecay * current[i];
  }
}

/**
 * EMA Weight Tracker
 *
 * Maintains shadow weights as an exponential moving average of model parameters.
 */
export class EMATracker {
  private decay: number;
  private shadowWeights: WeightSnapshot | null = null;
  private numUpdates = 0;

  constructor(decay: number) {
    this.decay = Math.max(0, Math.min(1, decay));
  }

  /**
   * Update shadow weights with the current model weights.
   * On the first call, initializes the shadow to the current weights.
   */
  update(currentWeights: WeightSnapshot): void {
    if (this.shadowWeights === null) {
      // First update: initialize shadow to current weights
      this.shadowWeights = cloneWeights(currentWeights);
      this.numUpdates = 1;
      return;
    }

    this.numUpdates++;

    // Apply EMA update: shadow = decay * shadow + (1-decay) * current
    emaMatrix(this.shadowWeights.embedding, currentWeights.embedding, this.decay);
    emaMatrix(this.shadowWeights.wHidden, currentWeights.wHidden, this.decay);
    emaMatrix(this.shadowWeights.wOutput, currentWeights.wOutput, this.decay);
    emaVector(this.shadowWeights.bHidden, currentWeights.bHidden, this.decay);
    emaVector(this.shadowWeights.bOutput, currentWeights.bOutput, this.decay);
  }

  /**
   * Get a deep copy of the current shadow (EMA) weights.
   * Returns null if no updates have been applied yet.
   */
  getShadowWeights(): WeightSnapshot | null {
    if (this.shadowWeights === null) return null;
    return cloneWeights(this.shadowWeights);
  }

  /**
   * Get the current decay value.
   */
  getDecay(): number {
    return this.decay;
  }

  /**
   * Update the decay value.
   */
  setDecay(decay: number): void {
    this.decay = Math.max(0, Math.min(1, decay));
  }

  /**
   * Get the number of EMA updates performed.
   */
  getNumUpdates(): number {
    return this.numUpdates;
  }

  /**
   * Reset the tracker (clears shadow weights).
   */
  reset(): void {
    this.shadowWeights = null;
    this.numUpdates = 0;
  }

  /**
   * Check if the tracker has been initialized with weights.
   */
  isInitialized(): boolean {
    return this.shadowWeights !== null;
  }

  /**
   * Export the EMA state for serialization.
   */
  exportState(): {
    decay: number;
    shadowWeights: WeightSnapshot | null;
    numUpdates: number;
  } {
    return {
      decay: this.decay,
      shadowWeights: this.shadowWeights ? cloneWeights(this.shadowWeights) : null,
      numUpdates: this.numUpdates
    };
  }

  /**
   * Import a previously exported EMA state.
   */
  importState(state: {
    decay: number;
    shadowWeights: WeightSnapshot | null;
    numUpdates: number;
  }): void {
    this.decay = state.decay;
    this.shadowWeights = state.shadowWeights ? cloneWeights(state.shadowWeights) : null;
    this.numUpdates = state.numUpdates;
  }
}
