/**
 * Information-geometric statistics utilities.
 *
 * The functions in this module extend the curvature diagnostics in
 * `src/math/analysis.ts` by making explicit the link between empirical Fisher
 * information and Hessian norms. They operate on lightweight vector/matrix
 * structures to avoid introducing heavy linear algebra dependencies.
 */

export type Vector = number[];
export type Matrix = number[][];

export interface FisherStatistics {
  /** Empirical Fisher information matrix averaged across the provided samples. */
  fisher: Matrix;
  /** Frobenius norm of the empirical Fisher (a proxy for Hessian magnitude). */
  frobeniusNorm: number;
  /** Trace of the empirical Fisher, corresponding to the squared gradient norm. */
  trace: number;
  /** Upper bound on the spectral norm obtained via Gershgorin discs. */
  spectralNormBound: number;
  /** Diagnostic notes clarifying the geometric interpretation. */
  notes: string[];
}

export interface FisherFromGradientsOptions {
  /** Numerical stability floor applied before inverting curvature quantities. */
  epsilon?: number;
}

function zeros(rows: number, cols: number): Matrix {
  const matrix: Matrix = [];
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols).fill(0);
    matrix.push(row);
  }
  return matrix;
}

function addOuterProduct(target: Matrix, vector: Vector): void {
  for (let i = 0; i < target.length; i++) {
    const row = target[i];
    for (let j = 0; j < row.length; j++) {
      row[j] += vector[i] * vector[j];
    }
  }
}

function scaleMatrix(matrix: Matrix, scalar: number): void {
  for (const row of matrix) {
    for (let j = 0; j < row.length; j++) {
      row[j] *= scalar;
    }
  }
}

function ensureFiniteGradients(gradients: Vector[]): void {
  for (const grad of gradients) {
    for (const value of grad) {
      if (!Number.isFinite(value)) {
        throw new Error('Fisher statistics require finite gradient entries.');
      }
    }
  }
}

function matrixTrace(matrix: Matrix): number {
  let trace = 0;
  for (let i = 0; i < matrix.length; i++) trace += matrix[i][i];
  return trace;
}

function frobeniusNorm(matrix: Matrix): number {
  let sum = 0;
  for (const row of matrix) {
    for (const value of row) sum += value * value;
  }
  return Math.sqrt(sum);
}

function gershgorinSpectralBound(matrix: Matrix): number {
  let bound = 0;
  for (let i = 0; i < matrix.length; i++) {
    let radius = 0;
    const row = matrix[i];
    for (let j = 0; j < row.length; j++) {
      if (i === j) continue;
      radius += Math.abs(row[j]);
    }
    const disc = Math.abs(row[i]) + radius;
    if (disc > bound) bound = disc;
  }
  return bound;
}

function averageGradients(gradients: Vector[]): Vector {
  const dim = gradients[0].length;
  const avg = new Array(dim).fill(0);
  for (const grad of gradients) {
    for (let j = 0; j < dim; j++) avg[j] += grad[j];
  }
  const inv = 1 / gradients.length;
  for (let j = 0; j < dim; j++) avg[j] *= inv;
  return avg;
}

/**
 * Compute the empirical Fisher information matrix from per-sample gradients.
 *
 * The implementation mirrors the diagonal computation in `computeDiagonalHessian`
 * but keeps the full matrix so that we can reason about global curvature norms.
 */
export function empiricalFisherFromGradients(
  gradients: Vector[],
  options: FisherFromGradientsOptions = {}
): Matrix {
  if (gradients.length === 0) throw new Error('No gradients provided.');
  ensureFiniteGradients(gradients);
  const dim = gradients[0].length;
  const fisher = zeros(dim, dim);
  for (const grad of gradients) {
    if (grad.length !== dim) {
      throw new Error('All gradients must share the same dimensionality.');
    }
    addOuterProduct(fisher, grad);
  }
  const inv = 1 / gradients.length;
  scaleMatrix(fisher, inv);
  const epsilon = options.epsilon ?? 0;
  if (epsilon > 0) {
    for (let i = 0; i < dim; i++) fisher[i][i] += epsilon;
  }
  return fisher;
}

/**
 * Link the empirical Fisher matrix to Hessian norms for curvature diagnostics.
 *
 * Given per-sample gradients, this function returns the empirical Fisher matrix
 * together with norm-based summaries that upper-bound the true Hessian norm
 * under the common Gauss–Newton approximation $H \approx F$.
 */
export function fisherHessianStatistics(
  gradients: Vector[],
  options: FisherFromGradientsOptions = {}
): FisherStatistics {
  const fisher = empiricalFisherFromGradients(gradients, options);
  const trace = matrixTrace(fisher);
  const frob = frobeniusNorm(fisher);
  const spectralBound = gershgorinSpectralBound(fisher);
  const meanGrad = averageGradients(gradients);
  const meanGradNormSq = meanGrad.reduce((acc, value) => acc + value * value, 0);
  const notes = [
    'Trace equals the average squared gradient norm (empirical Fisher diagonal).',
    'Frobenius norm upper-bounds the Gauss–Newton Hessian under well-specified models.',
    'Gershgorin discs provide a cheap spectral-norm bound, highlighting sharp directions.',
    `Mean gradient norm squared: ${meanGradNormSq.toFixed(6)}`
  ];
  return {
    fisher,
    trace,
    frobeniusNorm: frob,
    spectralNormBound: spectralBound,
    notes
  };
}

/**
 * Convert a gradient vector into a curvature-aware scaling using the empirical
 * Fisher diagonal. This mirrors `computeDiagonalHessian` but keeps the conceptual
 * connection to the Fisher information explicit.
 */
export function fisherDiagonalScaling(gradient: Vector, epsilon = 1e-9): Vector {
  ensureFiniteGradients([gradient]);
  return gradient.map((value) => Math.max(value * value, epsilon));
}

/**
 * Evaluate the quadratic form induced by the empirical Fisher to compare with
 * Hessian-based trust-region criteria.
 */
export function fisherQuadraticForm(fisher: Matrix, vector: Vector): number {
  if (fisher.length !== vector.length) {
    throw new Error('Vector dimension must match Fisher matrix size.');
  }
  let value = 0;
  for (let i = 0; i < fisher.length; i++) {
    const row = fisher[i];
    let acc = 0;
    for (let j = 0; j < row.length; j++) {
      acc += row[j] * vector[j];
    }
    value += vector[i] * acc;
  }
  return value;
}
