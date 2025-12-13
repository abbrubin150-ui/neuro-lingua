/**
 * Mixed Precision Training Support
 *
 * Implements FP16/FP32 mixed precision training for 2-3× speedup and 50% memory reduction.
 *
 * Key concepts:
 * - FP16 (half precision): Used for forward/backward pass, 2× faster on GPU
 * - FP32 (single precision): Used for master weights and critical accumulators
 * - Loss scaling: Prevents gradient underflow in FP16 by scaling loss before backward
 * - Dynamic loss scaling: Automatically adjusts scale to prevent overflow/underflow
 *
 * Memory layout:
 * - Master weights (FP32): ~100% of original
 * - FP16 working copies: ~50% of original
 * - Total: ~150% vs ~200% for pure FP32 (saves memory when gradient checkpointing)
 *
 * Numerical stability:
 * - Accumulations in FP32 (running sums, softmax, layer norm)
 * - Gradient accumulation in FP32
 * - Loss computation in FP32
 *
 * Reference: Micikevicius et al. (2018) "Mixed Precision Training"
 *            https://arxiv.org/abs/1710.03740
 *
 * @module backend/mixed_precision
 */

/**
 * Configuration for mixed precision training
 */
export interface MixedPrecisionConfig {
  /** Enable mixed precision (default: true when WebGPU available) */
  enabled: boolean;
  /** Initial loss scale (default: 65536, high to catch underflow) */
  initialLossScale: number;
  /** Factor to increase scale by after successful steps (default: 2) */
  scaleGrowthFactor: number;
  /** Factor to decrease scale by after overflow (default: 0.5) */
  scaleBackoffFactor: number;
  /** Number of successful steps before increasing scale (default: 2000) */
  scaleGrowthInterval: number;
  /** Minimum loss scale before disabling (default: 1) */
  minLossScale: number;
  /** Maximum loss scale (default: 2^24 = 16777216) */
  maxLossScale: number;
  /** Skip batch on overflow instead of scaling down (default: false) */
  skipOnOverflow: boolean;
}

/**
 * State for dynamic loss scaling
 */
export interface LossScalingState {
  /** Current loss scale */
  scale: number;
  /** Number of consecutive successful steps */
  successfulSteps: number;
  /** Number of overflows encountered */
  overflowCount: number;
  /** Whether last step had overflow */
  lastOverflow: boolean;
}

/**
 * Default mixed precision configuration
 */
export const DEFAULT_MIXED_PRECISION_CONFIG: MixedPrecisionConfig = {
  enabled: true,
  initialLossScale: 65536, // 2^16
  scaleGrowthFactor: 2,
  scaleBackoffFactor: 0.5,
  scaleGrowthInterval: 2000,
  minLossScale: 1,
  maxLossScale: 16777216, // 2^24
  skipOnOverflow: false
};

// FP16 constants
const FP16_MAX = 65504; // Maximum representable value in FP16
const FP16_MIN_NORMAL = 6.103515625e-5; // Minimum normal positive FP16
const _FP16_EPSILON = 0.0009765625; // FP16 machine epsilon (reserved for future use)

/**
 * Convert FP32 value to FP16 representation
 * Uses rounding to nearest even (IEEE 754 default)
 *
 * @param value - Float32 value
 * @returns Float16 representation as Uint16
 */
export function float32ToFloat16Bits(value: number): number {
  // Handle special cases
  if (Number.isNaN(value)) {
    return 0x7e00; // FP16 NaN
  }
  if (!Number.isFinite(value)) {
    return value > 0 ? 0x7c00 : 0xfc00; // +/- Infinity
  }
  if (value === 0) {
    return Object.is(value, -0) ? 0x8000 : 0x0000; // Preserve sign of zero
  }

  // Get sign bit
  const sign = value < 0 ? 0x8000 : 0;
  const absValue = Math.abs(value);

  // Handle overflow to infinity
  if (absValue > FP16_MAX) {
    return sign | 0x7c00;
  }

  // Handle underflow to zero (subnormal)
  if (absValue < FP16_MIN_NORMAL) {
    // Convert to subnormal representation
    const subnormal = Math.round(absValue / 2 ** -24);
    return sign | subnormal;
  }

  // Normal number conversion
  const log2 = Math.floor(Math.log2(absValue));
  const exponent = log2 + 15; // FP16 bias is 15

  // Handle exponent overflow
  if (exponent >= 31) {
    return sign | 0x7c00; // Infinity
  }

  // Calculate mantissa (10 bits)
  const mantissa = absValue / 2 ** log2 - 1;
  const mantissaBits = Math.round(mantissa * 1024); // 2^10 = 1024

  return sign | (exponent << 10) | mantissaBits;
}

/**
 * Convert FP16 bits to FP32 value
 *
 * @param bits - Float16 representation as Uint16
 * @returns Float32 value
 */
export function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;

  if (exponent === 0) {
    // Subnormal or zero
    if (mantissa === 0) {
      return sign * 0;
    }
    // Subnormal: value = sign * 2^-14 * (mantissa/1024)
    return sign * 2 ** -14 * (mantissa / 1024);
  }

  if (exponent === 31) {
    // Infinity or NaN
    if (mantissa === 0) {
      return sign * Infinity;
    }
    return NaN;
  }

  // Normal number: value = sign * 2^(exp-15) * (1 + mantissa/1024)
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

/**
 * Convert Float32Array to Float16 Uint16Array
 *
 * @param fp32 - Input Float32Array
 * @returns Uint16Array with FP16 values
 */
export function float32ArrayToFloat16(fp32: Float32Array): Uint16Array {
  const fp16 = new Uint16Array(fp32.length);
  for (let i = 0; i < fp32.length; i++) {
    fp16[i] = float32ToFloat16Bits(fp32[i]);
  }
  return fp16;
}

/**
 * Convert Float16 Uint16Array to Float32Array
 *
 * @param fp16 - Input Uint16Array with FP16 values
 * @returns Float32Array
 */
export function float16ArrayToFloat32(fp16: Uint16Array): Float32Array {
  const fp32 = new Float32Array(fp16.length);
  for (let i = 0; i < fp16.length; i++) {
    fp32[i] = float16BitsToFloat32(fp16[i]);
  }
  return fp32;
}

/**
 * Check if value can be safely represented in FP16
 *
 * @param value - Value to check
 * @returns True if value is within FP16 range
 */
export function isSafeForFP16(value: number): boolean {
  const absValue = Math.abs(value);
  return absValue <= FP16_MAX && (absValue === 0 || absValue >= FP16_MIN_NORMAL * 0.5);
}

/**
 * Check array for FP16 overflow/underflow
 *
 * @param values - Array of values to check
 * @returns Object with overflow and underflow info
 */
export function checkFP16Safety(values: number[] | Float32Array): {
  hasOverflow: boolean;
  hasUnderflow: boolean;
  overflowCount: number;
  underflowCount: number;
  maxAbs: number;
  minAbs: number;
} {
  let overflowCount = 0;
  let underflowCount = 0;
  let maxAbs = 0;
  let minAbs = Infinity;

  for (const value of values) {
    const absValue = Math.abs(value);

    if (absValue > maxAbs) maxAbs = absValue;
    if (absValue > 0 && absValue < minAbs) minAbs = absValue;

    if (absValue > FP16_MAX) {
      overflowCount++;
    } else if (absValue > 0 && absValue < FP16_MIN_NORMAL * 0.5) {
      underflowCount++;
    }
  }

  return {
    hasOverflow: overflowCount > 0,
    hasUnderflow: underflowCount > 0,
    overflowCount,
    underflowCount,
    maxAbs,
    minAbs: minAbs === Infinity ? 0 : minAbs
  };
}

/**
 * Dynamic Loss Scaler for mixed precision training
 *
 * Automatically adjusts loss scaling to prevent gradient underflow in FP16
 * while avoiding overflow.
 */
export class DynamicLossScaler {
  private config: MixedPrecisionConfig;
  private state: LossScalingState;

  constructor(config: Partial<MixedPrecisionConfig> = {}) {
    this.config = { ...DEFAULT_MIXED_PRECISION_CONFIG, ...config };
    this.state = {
      scale: this.config.initialLossScale,
      successfulSteps: 0,
      overflowCount: 0,
      lastOverflow: false
    };
  }

  /**
   * Get current loss scale
   */
  getScale(): number {
    return this.state.scale;
  }

  /**
   * Get current state for monitoring
   */
  getState(): LossScalingState {
    return { ...this.state };
  }

  /**
   * Scale loss value for backward pass
   *
   * @param loss - Original loss value
   * @returns Scaled loss value
   */
  scaleLoss(loss: number): number {
    return loss * this.state.scale;
  }

  /**
   * Unscale gradients after backward pass
   *
   * @param gradients - Array of gradient values
   * @returns Unscaled gradients (modifies in place for efficiency)
   */
  unscaleGradients(gradients: number[]): number[] {
    const invScale = 1.0 / this.state.scale;
    for (let i = 0; i < gradients.length; i++) {
      gradients[i] *= invScale;
    }
    return gradients;
  }

  /**
   * Unscale gradient matrix
   *
   * @param gradients - 2D gradient matrix
   * @returns Unscaled gradients (modifies in place)
   */
  unscaleGradientMatrix(gradients: number[][]): number[][] {
    const invScale = 1.0 / this.state.scale;
    for (let i = 0; i < gradients.length; i++) {
      for (let j = 0; j < gradients[i].length; j++) {
        gradients[i][j] *= invScale;
      }
    }
    return gradients;
  }

  /**
   * Check if gradients have overflow and update scaling
   *
   * @param gradients - Gradient values to check
   * @returns True if gradients are valid (no overflow), false if overflow detected
   */
  checkAndUpdateScale(gradients: number[] | Float32Array): boolean {
    // Check for overflow (NaN or Inf)
    let hasOverflow = false;
    for (const g of gradients) {
      if (!Number.isFinite(g)) {
        hasOverflow = true;
        break;
      }
    }

    this.state.lastOverflow = hasOverflow;

    if (hasOverflow) {
      // Overflow detected: scale down
      this.state.overflowCount++;
      this.state.successfulSteps = 0;

      const newScale = this.state.scale * this.config.scaleBackoffFactor;
      this.state.scale = Math.max(newScale, this.config.minLossScale);

      return false;
    } else {
      // No overflow: increment successful steps
      this.state.successfulSteps++;

      // Scale up if enough successful steps
      if (this.state.successfulSteps >= this.config.scaleGrowthInterval) {
        this.state.successfulSteps = 0;

        const newScale = this.state.scale * this.config.scaleGrowthFactor;
        this.state.scale = Math.min(newScale, this.config.maxLossScale);
      }

      return true;
    }
  }

  /**
   * Check gradient matrix for overflow
   *
   * @param gradients - 2D gradient matrix
   * @returns True if valid, false if overflow
   */
  checkMatrixAndUpdateScale(gradients: number[][]): boolean {
    // Flatten and check
    const flat: number[] = [];
    for (const row of gradients) {
      flat.push(...row);
    }
    return this.checkAndUpdateScale(flat);
  }

  /**
   * Reset scaler state
   */
  reset(): void {
    this.state = {
      scale: this.config.initialLossScale,
      successfulSteps: 0,
      overflowCount: 0,
      lastOverflow: false
    };
  }

  /**
   * Export state for serialization
   */
  exportState(): { config: MixedPrecisionConfig; state: LossScalingState } {
    return {
      config: { ...this.config },
      state: { ...this.state }
    };
  }

  /**
   * Import state from serialization
   */
  importState(data: {
    config?: Partial<MixedPrecisionConfig>;
    state?: Partial<LossScalingState>;
  }): void {
    if (data.config) {
      this.config = { ...this.config, ...data.config };
    }
    if (data.state) {
      this.state = { ...this.state, ...data.state };
    }
  }
}

/**
 * Mixed Precision Tensor wrapper for managing FP16/FP32 conversions
 */
export class MixedPrecisionTensor {
  /** FP32 master weights (used for updates) */
  private masterData: Float32Array;
  /** FP16 working copy (used for forward/backward) */
  private workingData: Uint16Array | null = null;
  /** Shape of the tensor */
  public readonly shape: number[];
  /** Whether FP16 copy is dirty (needs sync from master) */
  private dirty: boolean = true;

  constructor(data: Float32Array, shape: number[]) {
    this.masterData = data;
    this.shape = shape;
  }

  /**
   * Get FP32 master data (for optimizer updates)
   */
  getMasterData(): Float32Array {
    return this.masterData;
  }

  /**
   * Get FP16 working data (for forward/backward)
   * Syncs from master if dirty
   */
  getWorkingData(): Uint16Array {
    if (this.dirty || this.workingData === null) {
      this.workingData = float32ArrayToFloat16(this.masterData);
      this.dirty = false;
    }
    return this.workingData;
  }

  /**
   * Update master data and mark working copy as dirty
   */
  updateMaster(data: Float32Array): void {
    this.masterData = data;
    this.dirty = true;
  }

  /**
   * Mark working copy as dirty (call after optimizer update)
   */
  markDirty(): void {
    this.dirty = true;
  }

  /**
   * Get total element count
   */
  get size(): number {
    return this.masterData.length;
  }

  /**
   * Get memory usage in bytes
   */
  getMemoryUsage(): { master: number; working: number; total: number } {
    const masterBytes = this.masterData.byteLength;
    const workingBytes = this.workingData ? this.workingData.byteLength : 0;
    return {
      master: masterBytes,
      working: workingBytes,
      total: masterBytes + workingBytes
    };
  }
}

/**
 * Utility functions for mixed precision arithmetic
 */
export const MixedPrecisionOps = {
  /**
   * Perform FP16-safe matrix-vector multiplication
   * Uses FP32 accumulation for numerical stability
   */
  matvec(matrix: number[][], vector: number[]): number[] {
    const result: number[] = [];
    for (let i = 0; i < matrix.length; i++) {
      let sum = 0; // FP32 accumulator
      for (let j = 0; j < vector.length; j++) {
        sum += matrix[i][j] * vector[j];
      }
      result.push(sum);
    }
    return result;
  },

  /**
   * Perform stable softmax in mixed precision
   * Uses FP32 for the log-sum-exp calculation
   */
  softmax(logits: number[]): number[] {
    // Find max for numerical stability (in FP32)
    let max = -Infinity;
    for (const x of logits) {
      if (x > max) max = x;
    }

    // Compute exp(x - max) and sum (in FP32)
    const exps: number[] = [];
    let sum = 0;
    for (const x of logits) {
      const e = Math.exp(x - max);
      exps.push(e);
      sum += e;
    }

    // Normalize (in FP32, result can be FP16)
    const result: number[] = [];
    for (const e of exps) {
      result.push(e / sum);
    }

    return result;
  },

  /**
   * Compute layer normalization in mixed precision
   * Uses FP32 for mean/variance computation
   */
  layerNorm(input: number[], gamma: number[], beta: number[], epsilon: number = 1e-5): number[] {
    // Compute mean (FP32)
    let mean = 0;
    for (const x of input) {
      mean += x;
    }
    mean /= input.length;

    // Compute variance (FP32)
    let variance = 0;
    for (const x of input) {
      const diff = x - mean;
      variance += diff * diff;
    }
    variance /= input.length;

    // Normalize and scale (result can be FP16)
    const invStd = 1.0 / Math.sqrt(variance + epsilon);
    const result: number[] = [];
    for (let i = 0; i < input.length; i++) {
      const normalized = (input[i] - mean) * invStd;
      result.push(normalized * gamma[i] + beta[i]);
    }

    return result;
  },

  /**
   * Clip gradients to FP16-safe range
   */
  clipGradients(gradients: number[], maxNorm: number = FP16_MAX * 0.9): number[] {
    let norm = 0;
    for (const g of gradients) {
      norm += g * g;
    }
    norm = Math.sqrt(norm);

    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (let i = 0; i < gradients.length; i++) {
        gradients[i] *= scale;
      }
    }

    return gradients;
  }
};

/**
 * WebGPU shader code for FP16 operations (WGSL)
 * Can be used when WebGPU is available for true hardware FP16
 */
export const FP16_SHADERS = {
  /**
   * Matrix multiplication shader using f16 (requires WebGPU shader-f16 extension)
   */
  matmul: `
    enable f16;

    @group(0) @binding(0) var<storage, read> a: array<f16>;
    @group(0) @binding(1) var<storage, read> b: array<f16>;
    @group(0) @binding(2) var<storage, read_write> result: array<f16>;
    @group(0) @binding(3) var<uniform> dims: vec3<u32>; // M, N, K

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let M = dims.x;
      let N = dims.y;
      let K = dims.z;

      let row = global_id.x;
      let col = global_id.y;

      if (row >= M || col >= N) { return; }

      // Use f32 accumulator for stability
      var sum: f32 = 0.0;
      for (var k: u32 = 0; k < K; k++) {
        let a_val = f32(a[row * K + k]);
        let b_val = f32(b[k * N + col]);
        sum += a_val * b_val;
      }

      result[row * N + col] = f16(sum);
    }
  `,

  /**
   * Softmax shader with FP32 accumulation
   */
  softmax: `
    enable f16;

    @group(0) @binding(0) var<storage, read> input: array<f16>;
    @group(0) @binding(1) var<storage, read_write> output: array<f16>;
    @group(0) @binding(2) var<uniform> length: u32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      if (global_id.x != 0) { return; }

      // Find max (FP32)
      var max_val: f32 = f32(input[0]);
      for (var i: u32 = 1; i < length; i++) {
        let val = f32(input[i]);
        if (val > max_val) { max_val = val; }
      }

      // Compute exp and sum (FP32)
      var sum: f32 = 0.0;
      for (var i: u32 = 0; i < length; i++) {
        let exp_val = exp(f32(input[i]) - max_val);
        output[i] = f16(exp_val);
        sum += exp_val;
      }

      // Normalize
      let inv_sum = 1.0 / sum;
      for (var i: u32 = 0; i < length; i++) {
        output[i] = f16(f32(output[i]) * inv_sum);
      }
    }
  `,

  /**
   * ReLU activation in FP16
   */
  relu: `
    enable f16;

    @group(0) @binding(0) var<storage, read> input: array<f16>;
    @group(0) @binding(1) var<storage, read_write> output: array<f16>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let i = global_id.x;
      let val = input[i];
      output[i] = select(f16(0.0), val, val > f16(0.0));
    }
  `,

  /**
   * Gradient scaling for loss scaler
   */
  scaleGradients: `
    enable f16;

    @group(0) @binding(0) var<storage, read_write> gradients: array<f16>;
    @group(0) @binding(1) var<uniform> inv_scale: f32;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let i = global_id.x;
      gradients[i] = f16(f32(gradients[i]) * inv_scale);
    }
  `
};

/**
 * Check if WebGPU supports FP16 operations
 */
export async function checkFP16Support(): Promise<{
  supported: boolean;
  shaderF16: boolean;
  storageF16: boolean;
}> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return { supported: false, shaderF16: false, storageF16: false };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      return { supported: false, shaderF16: false, storageF16: false };
    }

    const features = adapter.features;
    const shaderF16 = features.has('shader-f16');
    const storageF16 = features.has('float16' as GPUFeatureName) || shaderF16;

    return {
      supported: shaderF16 || storageF16,
      shaderF16,
      storageF16
    };
  } catch {
    return { supported: false, shaderF16: false, storageF16: false };
  }
}
