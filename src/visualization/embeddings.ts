import { UMAP } from 'umap-js';
import TSNE from 'tsne-js';

export type NormalisationMode = 'none' | 'l2' | 'zscore';

export interface ProjectionSummary {
  min: [number, number];
  max: [number, number];
  mean: [number, number];
}

export interface ProjectionResult {
  coordinates: number[][];
  summary: ProjectionSummary;
  metadata: Record<string, unknown>;
}

export interface TSNEProjectionOptions {
  perplexity?: number;
  epsilon?: number;
  iterations?: number;
  normalise?: NormalisationMode;
}

export interface UMAPProjectionOptions {
  nNeighbors?: number;
  minDist?: number;
  spread?: number;
  learningRate?: number;
  normalise?: NormalisationMode;
}

function clone2d(matrix: ReadonlyArray<ReadonlyArray<number>>): number[][] {
  return matrix.map((row) => row.slice());
}

function normaliseRow(row: number[], mode: NormalisationMode): number[] {
  if (mode === 'none') {
    return row.slice();
  }
  if (mode === 'l2') {
    const norm = Math.sqrt(row.reduce((acc, value) => acc + value * value, 0));
    if (!Number.isFinite(norm) || norm === 0) {
      return row.slice();
    }
    return row.map((value) => value / norm);
  }
  const mean = row.reduce((acc, value) => acc + value, 0) / row.length;
  const variance = row.reduce((acc, value) => acc + (value - mean) ** 2, 0) / row.length;
  const std = Math.sqrt(variance) || 1;
  return row.map((value) => (value - mean) / std);
}

export function normaliseEmbeddings(
  embeddings: ReadonlyArray<ReadonlyArray<number>>,
  mode: NormalisationMode = 'zscore'
): number[][] {
  return embeddings.map((row) => normaliseRow(row.slice(), mode));
}

function summariseCoordinates(points: number[][]): ProjectionSummary {
  if (points.length === 0) {
    return {
      min: [0, 0],
      max: [0, 0],
      mean: [0, 0]
    };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let sumX = 0;
  let sumY = 0;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    sumX += x;
    sumY += y;
  }
  const n = points.length || 1;
  return {
    min: [minX, minY],
    max: [maxX, maxY],
    mean: [sumX / n, sumY / n]
  };
}

export function projectWithTSNE(
  embeddings: ReadonlyArray<ReadonlyArray<number>>,
  options: TSNEProjectionOptions = {}
): ProjectionResult {
  const normalised = options.normalise ? normaliseEmbeddings(embeddings, options.normalise) : clone2d(embeddings);
  const perplexity = options.perplexity ?? Math.min(30, Math.max(5, normalised.length - 1));
  const iterations = options.iterations ?? 750;
  const learningRate = options.epsilon ?? 100;
  const tsne = new TSNE({
    dim: 2,
    perplexity,
    earlyExaggeration: 4.0,
    learningRate,
    nIter: iterations,
    metric: 'euclidean'
  });
  tsne.init({ data: normalised, type: 'dense' });
  tsne.run();
  const coordinates = tsne.getOutputScaled();
  return {
    coordinates,
    summary: summariseCoordinates(coordinates),
    metadata: {
      method: 'tsne',
      perplexity,
      epsilon: learningRate,
      iterations,
      normalise: options.normalise ?? 'none'
    }
  };
}

export function projectWithUMAP(
  embeddings: ReadonlyArray<ReadonlyArray<number>>,
  options: UMAPProjectionOptions = {}
): ProjectionResult {
  const normalised = options.normalise ? normaliseEmbeddings(embeddings, options.normalise) : clone2d(embeddings);
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: options.nNeighbors ?? Math.min(15, Math.max(2, normalised.length - 1)),
    minDist: options.minDist ?? 0.1,
    spread: options.spread ?? 1.0,
    learningRate: options.learningRate ?? 1.0
  });
  const coordinates = umap.fit(normalised);
  return {
    coordinates,
    summary: summariseCoordinates(coordinates),
    metadata: {
      method: 'umap',
      nNeighbors: options.nNeighbors ?? 'auto',
      minDist: options.minDist ?? 0.1,
      spread: options.spread ?? 1.0,
      learningRate: options.learningRate ?? 1.0,
      normalise: options.normalise ?? 'none'
    }
  };
}

export function compareProjections(
  embeddings: ReadonlyArray<ReadonlyArray<number>>,
  tsneOptions: TSNEProjectionOptions = {},
  umapOptions: UMAPProjectionOptions = {}
): { tsne: ProjectionResult; umap: ProjectionResult } {
  const tsneResult = projectWithTSNE(embeddings, tsneOptions);
  const umapResult = projectWithUMAP(embeddings, umapOptions);
  return { tsne: tsneResult, umap: umapResult };
}
