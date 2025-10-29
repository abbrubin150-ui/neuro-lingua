import { logSumExp } from '../lib/MathUtils';
import { normalizeLogWeights } from '../generation/sampling';

export type SamplingRng = () => number;

export type WeightArray = number[] | number[][];

export type WeightSnapshot = Record<string, WeightArray>;

export interface WeightSample {
  weights: WeightSnapshot;
  logWeight: number;
  source: 'bootstrap' | 'bayesian';
}

export interface BootstrapConfig {
  numSamples: number;
  rng?: SamplingRng;
}

export interface BayesianSamplingConfig {
  numSamples: number;
  priorMean?: Float32Array;
  priorStd?: number;
  likelihoodStd?: number;
  rng?: SamplingRng;
}

export interface PosteriorMonteCarloConfig extends BayesianSamplingConfig {
  bootstrapSamples?: number;
}

export interface MonteCarloResult {
  mean: number;
  variance: number;
  effectiveSampleSize: number;
  values: number[];
  weights: number[];
}

function defaultRng(): number {
  return Math.random();
}

interface WeightStructureEntry {
  key: string;
  shape: number[];
}

function describeSnapshot(snapshot: WeightSnapshot): WeightStructureEntry[] {
  return Object.entries(snapshot).map(([key, value]) => {
    if (Array.isArray(value) && value.length > 0 && Array.isArray(value[0])) {
      const matrix = value as number[][];
      return { key, shape: [matrix.length, matrix[0]?.length ?? 0] };
    }
    return { key, shape: [(value as number[]).length] };
  });
}

function flattenSnapshot(
  snapshot: WeightSnapshot,
  structure: WeightStructureEntry[]
): Float32Array {
  const values: number[] = [];
  for (const entry of structure) {
    const data = snapshot[entry.key];
    if (entry.shape.length === 1) {
      values.push(...((data as number[]) ?? []));
    } else if (entry.shape.length === 2) {
      const matrix = data as number[][];
      for (const row of matrix) {
        values.push(...row);
      }
    }
  }
  return new Float32Array(values);
}

function unflattenSnapshot(flat: Float32Array, structure: WeightStructureEntry[]): WeightSnapshot {
  const snapshot: WeightSnapshot = {};
  let offset = 0;
  for (const entry of structure) {
    if (entry.shape.length === 1) {
      const length = entry.shape[0];
      const slice = flat.slice(offset, offset + length);
      snapshot[entry.key] = Array.from(slice);
      offset += length;
    } else if (entry.shape.length === 2) {
      const [rows, cols] = entry.shape;
      const matrix: number[][] = [];
      for (let r = 0; r < rows; r++) {
        const slice = flat.slice(offset, offset + cols);
        matrix.push(Array.from(slice));
        offset += cols;
      }
      snapshot[entry.key] = matrix;
    } else {
      throw new Error(`Unsupported tensor rank ${entry.shape.length}`);
    }
  }
  return snapshot;
}

function resampleVector(vector: number[], rng: SamplingRng): number[] {
  if (vector.length === 0) return [];
  const result = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    const index = Math.floor(rng() * vector.length);
    result[i] = vector[index];
  }
  return result;
}

function resampleMatrix(matrix: number[][], rng: SamplingRng): number[][] {
  if (matrix.length === 0) return [];
  return matrix.map(() => {
    const index = Math.floor(rng() * matrix.length);
    return [...matrix[index]];
  });
}

function gaussianRandom(rng: SamplingRng): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gaussianLogPdf(x: number, mean: number, std: number): number {
  const diff = x - mean;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(std) - (diff * diff) / (2 * std * std);
}

export function bootstrapWeightSamples(
  snapshot: WeightSnapshot,
  config: BootstrapConfig
): WeightSample[] {
  const { numSamples, rng = defaultRng } = config;
  if (numSamples <= 0) {
    throw new Error('numSamples must be positive for bootstrap sampling.');
  }
  const structure = describeSnapshot(snapshot);
  const samples: WeightSample[] = [];
  for (let i = 0; i < numSamples; i++) {
    const sample: WeightSnapshot = {};
    for (const entry of structure) {
      const data = snapshot[entry.key];
      if (entry.shape.length === 1) {
        sample[entry.key] = resampleVector(data as number[], rng);
      } else if (entry.shape.length === 2) {
        sample[entry.key] = resampleMatrix(data as number[][], rng);
      }
    }
    samples.push({ weights: sample, logWeight: 0, source: 'bootstrap' });
  }
  return samples;
}

export function bayesianWeightSamples(
  snapshot: WeightSnapshot,
  config: BayesianSamplingConfig
): WeightSample[] {
  const { numSamples, priorMean, priorStd = 1, likelihoodStd = 1, rng = defaultRng } = config;
  if (numSamples <= 0) {
    throw new Error('numSamples must be positive for Bayesian sampling.');
  }
  const structure = describeSnapshot(snapshot);
  const flat = flattenSnapshot(snapshot, structure);
  const prior = priorMean ?? new Float32Array(flat.length).fill(0);
  if (prior.length !== flat.length) {
    throw new Error('priorMean must match flattened weight size.');
  }

  const priorVar = priorStd * priorStd;
  const likelihoodVar = likelihoodStd * likelihoodStd;
  const posteriorVar = 1 / (1 / Math.max(priorVar, 1e-12) + 1 / Math.max(likelihoodVar, 1e-12));
  const posteriorStd = Math.sqrt(posteriorVar);

  const samples: WeightSample[] = [];
  for (let i = 0; i < numSamples; i++) {
    const flatSample = new Float32Array(flat.length);
    let logProb = 0;
    for (let j = 0; j < flat.length; j++) {
      const weight = flat[j];
      const postMean =
        posteriorVar *
        (weight / Math.max(likelihoodVar, 1e-12) + prior[j] / Math.max(priorVar, 1e-12));
      const draw = postMean + posteriorStd * gaussianRandom(rng);
      flatSample[j] = draw;
      logProb += gaussianLogPdf(draw, postMean, posteriorStd);
    }
    samples.push({
      weights: unflattenSnapshot(flatSample, structure),
      logWeight: logProb,
      source: 'bayesian'
    });
  }
  return samples;
}

export function monteCarloExperiment(
  samples: WeightSample[],
  evaluate: (weights: WeightSnapshot) => number
): MonteCarloResult {
  if (samples.length === 0) {
    throw new Error('Monte Carlo experiment requires at least one sample.');
  }
  const logWeights = samples.map((sample) => sample.logWeight);
  const weights = normalizeLogWeights(logWeights);
  const values = samples.map((sample) => evaluate(sample.weights));
  const mean = values.reduce((acc, value, index) => acc + value * weights[index], 0);
  const variance = values.reduce((acc, value, index) => {
    const diff = value - mean;
    return acc + weights[index] * diff * diff;
  }, 0);
  const weightSquares = weights.reduce((acc, value) => acc + value * value, 0);
  const effectiveSampleSize = weightSquares > 0 ? 1 / weightSquares : 0;
  return { mean, variance, effectiveSampleSize, values, weights };
}

export function posteriorMonteCarlo(
  snapshot: WeightSnapshot,
  evaluate: (weights: WeightSnapshot) => number,
  config: PosteriorMonteCarloConfig
): MonteCarloResult {
  const { bootstrapSamples = 0, ...bayesianConfig } = config;
  const allSamples: WeightSample[] = [];
  if (bootstrapSamples > 0) {
    allSamples.push(
      ...bootstrapWeightSamples(snapshot, { numSamples: bootstrapSamples, rng: bayesianConfig.rng })
    );
  }
  if (bayesianConfig.numSamples > 0) {
    allSamples.push(...bayesianWeightSamples(snapshot, bayesianConfig));
  }
  return monteCarloExperiment(allSamples, evaluate);
}

export function aggregateLogEvidence(logWeights: number[]): number {
  if (logWeights.length === 0) return Number.NEGATIVE_INFINITY;
  return logSumExp(logWeights) - Math.log(logWeights.length);
}
