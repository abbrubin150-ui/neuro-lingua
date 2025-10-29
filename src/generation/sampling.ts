import { clip, logSumExp, stableSoftmax } from '../lib/MathUtils';

export type SamplingRng = () => number;

export interface SamplingOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
  minProbability?: number;
  rng?: SamplingRng;
}

export interface BeamSearchOptions {
  beamWidth: number;
  maxLength: number;
  eosToken?: number;
  temperature?: number;
  lengthPenalty?: number;
  rng?: SamplingRng;
}

export interface BeamSearchState {
  tokens: number[];
  logProbability: number;
  completed: boolean;
}

export interface BeamSearchResult {
  tokens: number[];
  score: number;
  probability: number;
}

export type LogitsFn = (prefix: number[]) => number[];

function defaultRng(): number {
  return Math.random();
}

function argMax(values: number[]): number {
  if (values.length === 0) return -1;
  let maxIndex = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[maxIndex]) {
      maxIndex = i;
    }
  }
  return maxIndex;
}

function renormalize(probs: number[]): number[] {
  const sum = probs.reduce((acc, value) => acc + value, 0);
  if (!isFinite(sum) || sum <= 0) {
    const fallback = new Array(probs.length).fill(0);
    const idx = argMax(probs);
    if (idx >= 0) fallback[idx] = 1;
    return fallback;
  }
  return probs.map((value) => value / sum);
}

function applyTopK(probs: number[], k: number): number[] {
  if (k <= 0 || k >= probs.length) {
    return [...probs];
  }
  const indices = probs.map((_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  const threshold = probs[indices[Math.max(0, k - 1)]];
  const filtered = probs.map((value) => (value >= threshold ? value : 0));
  return renormalize(filtered);
}

function applyTopP(probs: number[], topP: number): number[] {
  if (topP <= 0 || topP >= 1) {
    return [...probs];
  }
  const indexed = probs.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => b.value - a.value);
  let cumulative = 0;
  const keep = new Set<number>();
  for (const item of indexed) {
    keep.add(item.index);
    cumulative += item.value;
    if (cumulative >= topP) break;
  }
  const filtered = probs.map((value, index) => (keep.has(index) ? value : 0));
  return renormalize(filtered);
}

function applyMinProbability(probs: number[], minProbability = 0): number[] {
  if (minProbability <= 0) return [...probs];
  const filtered = probs.map((value) => (value >= minProbability ? value : 0));
  return renormalize(filtered);
}

export function normalizeLogWeights(logWeights: number[]): number[] {
  if (logWeights.length === 0) return [];
  const norm = logSumExp(logWeights);
  return logWeights.map((w) => Math.exp(w - norm));
}

export function sampleCategorical(probs: number[], rng: SamplingRng = defaultRng): number {
  let r = clip(rng(), 0, 0.999999);
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

export function temperatureSample(
  logits: number[],
  temperature = 1,
  rng: SamplingRng = defaultRng
): number {
  return sampleFromLogits(logits, { temperature, rng });
}

export function topKSample(logits: number[], topK: number, options: SamplingOptions = {}): number {
  return sampleFromLogits(logits, { ...options, topK });
}

export function nucleusSample(
  logits: number[],
  topP: number,
  options: SamplingOptions = {}
): number {
  return sampleFromLogits(logits, { ...options, topP });
}

export function sampleFromLogits(logits: number[], options: SamplingOptions = {}): number {
  if (logits.length === 0) {
    throw new Error('Cannot sample from empty logits.');
  }

  const { temperature = 1, topK = 0, topP = 0, minProbability = 0, rng = defaultRng } = options;

  const scaled = logits.map((value) => value / clip(temperature, 0.05, 5));
  let probs = stableSoftmax(scaled);
  probs = applyMinProbability(probs, minProbability);
  probs = applyTopK(probs, topK);
  probs = applyTopP(probs, topP);

  if (probs.every((value) => value === 0)) {
    const index = argMax(logits);
    return index >= 0 ? index : 0;
  }

  return sampleCategorical(probs, rng);
}

export function beamSearch(
  step: LogitsFn,
  startTokens: number[],
  options: BeamSearchOptions
): BeamSearchResult[] {
  const {
    beamWidth,
    maxLength,
    eosToken,
    temperature = 1,
    lengthPenalty = 1,
    rng = defaultRng
  } = options;

  if (beamWidth <= 0) {
    throw new Error('beamWidth must be positive.');
  }
  if (maxLength <= 0) {
    throw new Error('maxLength must be positive.');
  }

  const active: BeamSearchState[] = [
    { tokens: [...startTokens], logProbability: 0, completed: false }
  ];
  const finished: BeamSearchState[] = [];

  while (active.length > 0 && finished.length < beamWidth) {
    const newBeams: BeamSearchState[] = [];
    for (const beam of active) {
      if (beam.completed || beam.tokens.length >= startTokens.length + maxLength) {
        finished.push(beam);
        continue;
      }

      const logits = step(beam.tokens);
      if (logits.length === 0) {
        finished.push({ ...beam, completed: true });
        continue;
      }

      const scaled = logits.map((value) => value / clip(temperature, 0.05, 5));
      const norm = logSumExp(scaled);
      const logProbs = scaled.map((value) => value - norm);

      const ranking = logProbs.map((value, index) => ({
        index,
        score: value + 1e-8 * (rng() - 0.5)
      }));
      ranking.sort((a, b) => b.score - a.score);
      const candidates = ranking.slice(0, beamWidth * 2);

      for (const candidate of candidates) {
        const index = candidate.index;
        const logProb = logProbs[index];
        const tokens = [...beam.tokens, index];
        const completed = eosToken !== undefined && index === eosToken;
        newBeams.push({
          tokens,
          completed,
          logProbability: beam.logProbability + logProb
        });
      }
    }

    if (newBeams.length === 0) break;

    newBeams.sort((a, b) => b.logProbability - a.logProbability);
    active.length = 0;
    for (const beam of newBeams.slice(0, beamWidth)) {
      if (beam.completed) {
        finished.push(beam);
      } else {
        active.push(beam);
      }
    }

    if (active.length === 0) {
      while (finished.length < beamWidth && newBeams.length > finished.length) {
        const beam = newBeams[finished.length];
        finished.push({ ...beam, completed: true });
      }
    }
  }

  const scored = finished.map((beam) => {
    const length = Math.max(1, beam.tokens.length - startTokens.length);
    const score = beam.logProbability / Math.pow(length, lengthPenalty);
    return { ...beam, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const logWeights = scored.map((beam) => beam.logProbability);
  const weights = normalizeLogWeights(logWeights);

  return scored.map((beam, index) => ({
    tokens: beam.tokens.slice(startTokens.length),
    score: beam.score,
    probability: weights[index]
  }));
}

export function monteCarloSample<T>(
  logitsFn: () => number[],
  evaluator: (index: number) => T,
  options: SamplingOptions & { trials: number }
): T[] {
  const { trials, ...sampling } = options;
  if (trials <= 0) {
    throw new Error('Monte Carlo sampling requires at least one trial.');
  }
  const results: T[] = [];
  for (let i = 0; i < trials; i++) {
    const logits = logitsFn();
    const index = sampleFromLogits(logits, sampling);
    results.push(evaluator(index));
  }
  return results;
}
