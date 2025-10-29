/**
 * Second-order optimization utilities integrated with the training loop.
 *
 * The module provides:
 * - Hessian (diagonal) approximations from gradient outer products.
 * - Damped Newton steps using the diagonal Hessian.
 * - Limited-memory BFGS (quasi-Newton) directions with curvature safeguards.
 * - Helpers to flatten and reconstruct parameter structures used by the models.
 */

export type ParameterStructure = Record<string, number[] | number[][]>;

export interface SegmentMeta {
  key: string;
  type: 'vector' | 'matrix';
  length: number;
  rows?: number;
  cols?: number;
}

export interface FlattenedStructure {
  vector: Float64Array;
  meta: SegmentMeta[];
}

export interface SecondOrderConfig {
  damping?: number;
  epsilon?: number;
  maxHistory?: number;
  learningRate?: number;
}

export interface LBFGSPair {
  s: Float64Array;
  y: Float64Array;
  rho: number;
}

export interface SecondOrderState {
  lastParams?: Float64Array;
  lastGrad?: Float64Array;
  history: LBFGSPair[];
  maxHistory: number;
  dimension?: number;
  iteration: number;
}

const DEFAULT_CONFIG: Required<Pick<SecondOrderConfig, 'damping' | 'epsilon' | 'maxHistory'>> = {
  damping: 1e-4,
  epsilon: 1e-8,
  maxHistory: 7
};

function assertFiniteVector(vector: Float64Array): void {
  for (let i = 0; i < vector.length; i++) {
    if (!Number.isFinite(vector[i])) {
      throw new Error(`Non-finite value encountered in optimization vector at index ${i}.`);
    }
  }
}

export function flattenStructure(structure: ParameterStructure): FlattenedStructure {
  const meta: SegmentMeta[] = [];
  const values: number[] = [];

  for (const [key, value] of Object.entries(structure)) {
    if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0])) {
      const matrix = value as number[][];
      const rows = matrix.length;
      const cols = matrix[0]?.length ?? 0;
      meta.push({ key, type: 'matrix', length: rows * cols, rows, cols });
      for (const row of matrix) {
        if (row.length !== cols) {
          throw new Error(`flattenStructure: matrix ${key} has inconsistent row lengths.`);
        }
        for (const cell of row) values.push(cell);
      }
    } else {
      const vector = value as number[];
      meta.push({ key, type: 'vector', length: vector.length });
      for (const cell of vector) values.push(cell);
    }
  }

  return { vector: Float64Array.from(values), meta };
}

export function flattenGradients(structure: ParameterStructure, meta: SegmentMeta[]): Float64Array {
  const values: number[] = [];
  for (const segment of meta) {
    const value = structure[segment.key];
    if (!value) {
      values.push(...new Array(segment.length).fill(0));
      continue;
    }
    if (segment.type === 'matrix') {
      const matrix = value as number[][];
      if (matrix.length !== segment.rows) {
        throw new Error(`flattenGradients: gradient for ${segment.key} has unexpected row count.`);
      }
      for (let i = 0; i < segment.rows!; i++) {
        const row = matrix[i];
        if (row.length !== segment.cols) {
          throw new Error(`flattenGradients: gradient for ${segment.key} has inconsistent columns.`);
        }
        for (const cell of row) values.push(cell);
      }
    } else {
      const vector = value as number[];
      if (vector.length !== segment.length) {
        throw new Error(`flattenGradients: gradient for ${segment.key} has mismatched length.`);
      }
      for (const cell of vector) values.push(cell);
    }
  }
  return Float64Array.from(values);
}

export function applyUpdateVector(
  structure: ParameterStructure,
  update: Float64Array,
  meta: SegmentMeta[]
): void {
  assertFiniteVector(update);
  let offset = 0;
  for (const segment of meta) {
    const target = structure[segment.key];
    if (!target) {
      offset += segment.length;
      continue;
    }
    if (segment.type === 'matrix') {
      const matrix = target as number[][];
      for (let i = 0; i < segment.rows!; i++) {
        const row = matrix[i];
        for (let j = 0; j < segment.cols!; j++) {
          row[j] += update[offset++];
        }
      }
    } else {
      const vector = target as number[];
      for (let j = 0; j < segment.length; j++) {
        vector[j] += update[offset++];
      }
    }
  }
}

export function computeDiagonalHessian(grad: Float64Array, epsilon = 1e-9): Float64Array {
  const diag = new Float64Array(grad.length);
  for (let i = 0; i < grad.length; i++) {
    const g = grad[i];
    diag[i] = Math.max(g * g, epsilon);
  }
  return diag;
}

export function dampedNewtonStep(
  grad: Float64Array,
  diagHessian: Float64Array,
  config: SecondOrderConfig
): Float64Array {
  const { damping, learningRate } = { ...DEFAULT_CONFIG, ...config };
  const step = new Float64Array(grad.length);
  for (let i = 0; i < grad.length; i++) {
    const denom = diagHessian[i] + damping;
    step[i] = -((learningRate ?? 1) * grad[i]) / denom;
  }
  return step;
}

export function createSecondOrderState(maxHistory = DEFAULT_CONFIG.maxHistory): SecondOrderState {
  return {
    history: [],
    maxHistory,
    iteration: 0
  };
}

function ensureHistoryCapacity(state: SecondOrderState): void {
  while (state.history.length > state.maxHistory) {
    state.history.shift();
  }
}

function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function subtract(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function scale(vector: Float64Array, scalar: number): Float64Array {
  const out = new Float64Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] * scalar;
  return out;
}

function addInPlace(target: Float64Array, source: Float64Array, scaleFactor: number): void {
  for (let i = 0; i < target.length; i++) target[i] += scaleFactor * source[i];
}

function twoLoopRecursion(state: SecondOrderState, grad: Float64Array): Float64Array {
  const q = grad.slice() as Float64Array;
  const alphas: number[] = [];

  for (let i = state.history.length - 1; i >= 0; i--) {
    const { s, y, rho } = state.history[i];
    const alpha = rho * dot(s, q);
    alphas.push(alpha);
    addInPlace(q, y, -alpha);
  }

  let scaleFactor = 1;
  if (state.history.length > 0) {
    const { s, y } = state.history[state.history.length - 1];
    const yy = dot(y, y);
    if (yy > state.history.length * Number.EPSILON) {
      scaleFactor = dot(y, s) / yy;
    }
  }

  let z = scale(q, scaleFactor);
  for (let i = 0; i < state.history.length; i++) {
    const { s, y, rho } = state.history[i];
    const alpha = alphas[alphas.length - 1 - i];
    const beta = rho * dot(y, z);
    addInPlace(z, s, alpha - beta);
  }

  for (let i = 0; i < z.length; i++) z[i] = -z[i];
  return z;
}

export function quasiNewtonStep(
  params: Float64Array,
  grad: Float64Array,
  state: SecondOrderState,
  config: SecondOrderConfig
): Float64Array {
  const merged = { ...DEFAULT_CONFIG, ...config };
  assertFiniteVector(grad);
  state.maxHistory = merged.maxHistory;
  if (state.dimension !== undefined && state.dimension !== grad.length) {
    state.history = [];
    state.lastParams = undefined;
    state.lastGrad = undefined;
  }
  state.dimension = grad.length;

  let direction: Float64Array;
  if (!state.lastParams || !state.lastGrad) {
    direction = scale(grad, -1);
  } else {
    const s = subtract(params, state.lastParams);
    const y = subtract(grad, state.lastGrad);
    const sy = dot(s, y);
    if (Math.abs(sy) > merged.epsilon) {
      const pair: LBFGSPair = { s, y, rho: 1 / sy };
      state.history.push(pair);
      ensureHistoryCapacity(state);
    }
    direction = twoLoopRecursion(state, grad);
  }

  state.lastParams = params.slice() as Float64Array;
  state.lastGrad = grad.slice() as Float64Array;
  state.iteration += 1;

  const lr = config.learningRate ?? 1;
  const step = scale(direction, lr);
  assertFiniteVector(step);
  return step;
}
