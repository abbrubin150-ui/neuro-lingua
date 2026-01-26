/**
 * GradientClipping - Unified gradient clipping module
 *
 * Provides two clipping strategies:
 * - Global norm: Scales all gradients by a single factor so that their combined
 *   L2 norm does not exceed `maxNorm`. This preserves gradient direction.
 * - Per-parameter: Clips each parameter's gradient independently to `maxNorm`.
 *   Useful when individual layers have very different gradient magnitudes.
 *
 * Both strategies can be combined (global first, then per-parameter).
 */

/**
 * Clipping mode for gradient clipping.
 * - 'global_norm': Clip by the global L2 norm of all gradients combined
 * - 'per_parameter': Clip each parameter gradient independently
 * - 'both': Apply global norm clipping first, then per-parameter clipping
 */
export type GradientClipMode = 'global_norm' | 'per_parameter' | 'both';

/**
 * Configuration for gradient clipping
 */
export interface GradientClipConfig {
  /** Clipping mode */
  mode: GradientClipMode;
  /** Maximum allowed norm (applies to both global and per-parameter) */
  maxNorm: number;
  /** Optional separate max norm for per-parameter clipping when mode='both' */
  perParamMaxNorm?: number;
}

/**
 * Statistics returned after clipping
 */
export interface GradientClipStats {
  /** L2 norm of all gradients before clipping */
  globalNormBefore: number;
  /** L2 norm of all gradients after clipping */
  globalNormAfter: number;
  /** Whether clipping was actually applied */
  wasClipped: boolean;
  /** Scale factor applied (1.0 = no clipping) */
  scaleFactor: number;
  /** Number of parameters whose gradients were clipped (per-parameter mode) */
  numParamsClipped: number;
}

/**
 * Compute the global L2 norm across all gradient matrices and vectors.
 */
function computeGlobalNorm(matGrads: number[][][], vecGrads: number[][]): number {
  let sumSq = 0;
  for (const mat of matGrads) {
    for (const row of mat) {
      for (const g of row) {
        sumSq += g * g;
      }
    }
  }
  for (const vec of vecGrads) {
    for (const g of vec) {
      sumSq += g * g;
    }
  }
  return Math.sqrt(sumSq);
}

/**
 * Compute the L2 norm of a single matrix gradient.
 */
function matrixNorm(mat: number[][]): number {
  let sumSq = 0;
  for (const row of mat) {
    for (const g of row) {
      sumSq += g * g;
    }
  }
  return Math.sqrt(sumSq);
}

/**
 * Compute the L2 norm of a single vector gradient.
 */
function vectorNorm(vec: number[]): number {
  let sumSq = 0;
  for (const g of vec) {
    sumSq += g * g;
  }
  return Math.sqrt(sumSq);
}

/**
 * Clip gradients by global norm.
 * All gradient tensors are scaled by the same factor so that their combined
 * L2 norm does not exceed `maxNorm`.
 *
 * Modifies gradients in place.
 */
export function clipByGlobalNorm(
  matGrads: number[][][],
  vecGrads: number[][],
  maxNorm: number
): { globalNorm: number; scaleFactor: number; wasClipped: boolean } {
  const globalNorm = computeGlobalNorm(matGrads, vecGrads);

  if (globalNorm <= maxNorm || globalNorm === 0) {
    return { globalNorm, scaleFactor: 1, wasClipped: false };
  }

  const scaleFactor = maxNorm / globalNorm;

  for (const mat of matGrads) {
    for (const row of mat) {
      for (let i = 0; i < row.length; i++) {
        row[i] *= scaleFactor;
      }
    }
  }
  for (const vec of vecGrads) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] *= scaleFactor;
    }
  }

  return { globalNorm, scaleFactor, wasClipped: true };
}

/**
 * Clip each parameter's gradient independently by its L2 norm.
 *
 * Modifies gradients in place.
 */
export function clipByPerParameter(
  matGrads: number[][][],
  vecGrads: number[][],
  maxNorm: number
): { numClipped: number } {
  let numClipped = 0;

  for (const mat of matGrads) {
    const norm = matrixNorm(mat);
    if (norm > maxNorm && norm > 0) {
      const scale = maxNorm / norm;
      for (const row of mat) {
        for (let i = 0; i < row.length; i++) {
          row[i] *= scale;
        }
      }
      numClipped++;
    }
  }

  for (const vec of vecGrads) {
    const norm = vectorNorm(vec);
    if (norm > maxNorm && norm > 0) {
      const scale = maxNorm / norm;
      for (let i = 0; i < vec.length; i++) {
        vec[i] *= scale;
      }
      numClipped++;
    }
  }

  return { numClipped };
}

/**
 * Unified gradient clipping entry point.
 *
 * @param matGrads - Array of matrix gradients (2D arrays), modified in place
 * @param vecGrads - Array of vector gradients (1D arrays), modified in place
 * @param config - Clipping configuration
 * @returns Clipping statistics
 */
export function clipGradients(
  matGrads: number[][][],
  vecGrads: number[][],
  config: GradientClipConfig
): GradientClipStats {
  const globalNormBefore = computeGlobalNorm(matGrads, vecGrads);
  let scaleFactor = 1;
  let wasClipped = false;
  let numParamsClipped = 0;

  if (config.mode === 'global_norm' || config.mode === 'both') {
    const result = clipByGlobalNorm(matGrads, vecGrads, config.maxNorm);
    scaleFactor = result.scaleFactor;
    wasClipped = result.wasClipped;
  }

  if (config.mode === 'per_parameter' || config.mode === 'both') {
    const perParamNorm = config.perParamMaxNorm ?? config.maxNorm;
    const result = clipByPerParameter(matGrads, vecGrads, perParamNorm);
    numParamsClipped = result.numClipped;
    if (result.numClipped > 0) {
      wasClipped = true;
    }
  }

  const globalNormAfter = computeGlobalNorm(matGrads, vecGrads);

  return {
    globalNormBefore,
    globalNormAfter,
    wasClipped,
    scaleFactor,
    numParamsClipped
  };
}
