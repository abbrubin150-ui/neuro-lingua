/**
 * Spectral and Lyapunov analysis utilities.
 *
 * This module implements spectral radius computation via power iteration and
 * Lyapunov stability diagnostics for discrete-time linearized systems. The
 * algorithms are numerically robust and document the assumptions under which
 * the returned indicators are valid.
 */

export type Matrix = number[][];

export interface SpectralRadiusOptions {
  /** Maximum power-iteration steps. */
  maxIterations?: number;
  /** Convergence tolerance on consecutive eigenvalue estimates. */
  tolerance?: number;
  /** Optional initial vector for the power iteration. */
  initialVector?: number[];
}

export interface SpectralRadiusResult {
  spectralRadius: number;
  iterations: number;
  converged: boolean;
  tolerance: number;
}

export interface LyapunovAnalysisOptions extends SpectralRadiusOptions {
  /**
   * Number of trajectory steps for Lyapunov exponent estimation. Defaults to
   * `10 * matrix.length` to give the average direction time to converge.
   */
  steps?: number;
  /**
   * Small perturbation added to the initialization vector when estimating the
   * dominant Lyapunov exponent.
   */
  perturbation?: number;
  /**
   * Indicates whether the dynamics are discrete (default) or continuous. The
   * stability criterion differs slightly (|λ| < 1 vs Re(λ) < 0).
   */
  discreteTime?: boolean;
}

export interface LyapunovAnalysisResult extends SpectralRadiusResult {
  /** Estimated largest Lyapunov exponent (base-e). */
  lyapunovExponent: number;
  /** Whether the linearized system satisfies the Lyapunov stability criterion. */
  stable: boolean;
  /** Margin to the stability boundary (|λ|max for discrete, Re(λ) for continuous). */
  stabilityMargin: number;
  /** Explicit list of modelling assumptions. */
  assumptions: string[];
}

const DEFAULT_SPECTRAL_RADIUS_OPTIONS: Required<
  Pick<SpectralRadiusOptions, 'maxIterations' | 'tolerance'>
> = {
  maxIterations: 1024,
  tolerance: 1e-9
};

const DEFAULT_LYAPUNOV_OPTIONS: Required<
  Pick<LyapunovAnalysisOptions, 'steps' | 'perturbation' | 'discreteTime'>
> = {
  steps: 0,
  perturbation: 1e-8,
  discreteTime: true
};

function isSquareMatrix(matrix: Matrix): boolean {
  if (matrix.length === 0) return false;
  const cols = matrix[0].length;
  if (cols === 0) return false;
  return matrix.every((row) => row.length === cols);
}

function multiplyMatrixVector(matrix: Matrix, vector: number[]): number[] {
  const result = new Array(matrix.length).fill(0);
  for (let i = 0; i < matrix.length; i++) {
    const row = matrix[i];
    let sum = 0;
    for (let j = 0; j < row.length; j++) sum += row[j] * vector[j];
    result[i] = sum;
  }
  return result;
}

function vectorNorm(vector: number[]): number {
  let sum = 0;
  for (const v of vector) sum += v * v;
  return Math.sqrt(sum);
}

function normalizeVector(vector: number[]): number[] {
  const norm = vectorNorm(vector);
  if (norm === 0) return vector.map(() => 0);
  return vector.map((v) => v / norm);
}

function rayleighQuotient(matrix: Matrix, vector: number[]): number {
  const Av = multiplyMatrixVector(matrix, vector);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < vector.length; i++) {
    numerator += vector[i] * Av[i];
    denominator += vector[i] * vector[i];
  }
  if (denominator === 0) return 0;
  return numerator / denominator;
}

function defaultInitialVector(size: number): number[] {
  return new Array(size).fill(1 / Math.sqrt(size));
}

/**
 * Compute the spectral radius of a square matrix via the power method.
 *
 * The returned value is an approximation of the dominant eigenvalue magnitude.
 * The implementation guards against zero matrices, non-square inputs, and
 * non-convergence by returning the best estimate so far together with metadata.
 */
export function spectralRadius(
  matrix: Matrix,
  options: SpectralRadiusOptions = {}
): SpectralRadiusResult {
  if (!isSquareMatrix(matrix)) {
    throw new Error('spectralRadius: expected a non-empty square matrix.');
  }

  const size = matrix.length;
  const { maxIterations, tolerance } = { ...DEFAULT_SPECTRAL_RADIUS_OPTIONS, ...options };
  const initial = options.initialVector ?? defaultInitialVector(size);
  let v = normalizeVector(initial.slice(0, size));
  let eigenvalue = 0;
  let converged = false;

  for (let iter = 1; iter <= maxIterations; iter++) {
    const Av = multiplyMatrixVector(matrix, v);
    const norm = vectorNorm(Av);
    if (!Number.isFinite(norm) || norm === 0) {
      return { spectralRadius: 0, iterations: iter, converged: false, tolerance };
    }
    const next = Av.map((x) => x / norm);
    const lambda = Math.abs(rayleighQuotient(matrix, next));
    if (Math.abs(lambda - eigenvalue) < tolerance) {
      eigenvalue = lambda;
      converged = true;
      return { spectralRadius: eigenvalue, iterations: iter, converged, tolerance };
    }
    eigenvalue = lambda;
    v = next;
  }

  return { spectralRadius: eigenvalue, iterations: maxIterations, converged, tolerance };
}

/**
 * Analyse Lyapunov stability for a linearized discrete or continuous system.
 *
 * The matrix represents the Jacobian of the system around an equilibrium. We
 * approximate the dominant Lyapunov exponent by repeatedly applying the
 * Jacobian to a random perturbation, measuring the exponential growth rate.
 */
export function analyzeLyapunov(
  matrix: Matrix,
  options: LyapunovAnalysisOptions = {}
): LyapunovAnalysisResult {
  const { maxIterations, tolerance } = { ...DEFAULT_SPECTRAL_RADIUS_OPTIONS, ...options };
  const merged = { ...DEFAULT_LYAPUNOV_OPTIONS, ...options };
  const steps = merged.steps > 0 ? merged.steps : maxIterations * matrix.length;
  const perturbation = merged.perturbation;

  const spectral = spectralRadius(matrix, {
    maxIterations,
    tolerance,
    initialVector: options.initialVector
  });
  const assumptions = [
    'Linearization is performed around a fixed point of the dynamics.',
    merged.discreteTime
      ? 'Stability is assessed under discrete-time evolution (|λ|max < 1).'
      : 'Stability is assessed under continuous-time evolution (Re(λ) < 0).',
    'The power method approximates the dominant eigenvalue; sub-dominant modes are neglected.',
    'Noise and external forcing are not modelled explicitly.'
  ];

  const size = matrix.length;
  let direction = defaultInitialVector(size).map((v) => v + perturbation);
  direction = normalizeVector(direction);
  let logNormSum = 0;

  for (let step = 0; step < steps; step++) {
    const next = multiplyMatrixVector(matrix, direction);
    const norm = vectorNorm(next) || Number.EPSILON;
    logNormSum += Math.log(norm);
    direction = next.map((v) => v / norm);
  }

  const exponent = steps > 0 ? logNormSum / steps : 0;
  let stabilityMargin: number;
  let stable: boolean;
  if (merged.discreteTime) {
    stabilityMargin = 1 - spectral.spectralRadius;
    stable = spectral.spectralRadius < 1 - tolerance;
  } else {
    // For continuous-time systems we approximate Re(λ) ≈ log(ρ(A))
    const realPartEstimate = Math.log(Math.max(spectral.spectralRadius, Number.EPSILON));
    stabilityMargin = -realPartEstimate;
    stable = realPartEstimate < -tolerance;
  }

  return {
    ...spectral,
    lyapunovExponent: exponent,
    stable,
    stabilityMargin,
    assumptions
  };
}

/**
 * Enumerates the modelling assumptions used in this module so that downstream
 * documentation can cite them without executing code.
 */
export const ANALYSIS_ASSUMPTIONS = Object.freeze([
  'Linearization around a stationary point ensures the Jacobian captures first-order dynamics.',
  'Power iteration assumes a dominant eigenvalue with magnitude strictly larger than others.',
  'Spectral radius < 1 implies asymptotic stability for discrete-time linear systems.',
  'Fisher information style approximations treat gradient outer products as Hessian surrogates.'
]);
