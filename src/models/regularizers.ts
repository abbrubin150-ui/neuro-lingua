/**
 * Regularisation utilities used by the miniature recurrent/transformer stack.
 * DropConnect randomly masks weight elements whereas Batch Renormalisation
 * rescales per-batch statistics to stay close to long-term moving averages.
 */

export type Matrix = number[][];

export interface DropConnectConfig {
  rate: number; // probability of dropping a connection
  seed?: number;
}

export function applyDropConnect(matrix: Matrix, config: DropConnectConfig): Matrix {
  const { rate, seed } = config;
  if (rate <= 0) return matrix.map((row) => row.slice());
  const rng = mulberry32(seed ?? Date.now());
  const keepProb = 1 - rate;
  return matrix.map((row) => row.map((value) => (rng() < keepProb ? value / keepProb : 0)));
}

function mulberry32(seed: number): () => number {
  let t = seed + 0x6d2b79f5;
  return () => {
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface BatchRenormState {
  runningMean: number[];
  runningVar: number[];
  momentum: number;
  epsilon: number;
  rMax: number;
  dMax: number;
}

export interface BatchRenormResult {
  normalized: Matrix;
  r: number[];
  d: number[];
}

export function batchRenormalize(inputs: Matrix, state: BatchRenormState): BatchRenormResult {
  if (inputs.length === 0 || inputs[0].length === 0) {
    return { normalized: inputs, r: [], d: [] };
  }
  const batchSize = inputs.length;
  const features = inputs[0].length;
  const batchMean = new Array(features).fill(0);
  const batchVar = new Array(features).fill(0);

  for (const row of inputs) {
    row.forEach((value, idx) => {
      batchMean[idx] += value;
      batchVar[idx] += value * value;
    });
  }

  for (let i = 0; i < features; i++) {
    batchMean[i] /= batchSize;
    batchVar[i] = batchVar[i] / batchSize - batchMean[i] ** 2;
  }

  state.runningMean = state.runningMean.map((mean, idx) =>
    state.momentum * mean + (1 - state.momentum) * batchMean[idx]
  );
  state.runningVar = state.runningVar.map((variance, idx) =>
    state.momentum * variance + (1 - state.momentum) * batchVar[idx]
  );

  const r: number[] = [];
  const d: number[] = [];
  for (let i = 0; i < features; i++) {
    const std = Math.sqrt(batchVar[i] + state.epsilon);
    const runningStd = Math.sqrt(state.runningVar[i] + state.epsilon);
    const rVal = Math.min(state.rMax, Math.max(1 / state.rMax, std / runningStd));
    const dVal = Math.max(-state.dMax, Math.min(state.dMax, (batchMean[i] - state.runningMean[i]) / runningStd));
    r.push(rVal);
    d.push(dVal);
  }

  const normalized = inputs.map((row) =>
    row.map((value, idx) => ((value - batchMean[idx]) / Math.sqrt(batchVar[idx] + state.epsilon)) * r[idx] + d[idx])
  );

  return { normalized, r, d };
}
