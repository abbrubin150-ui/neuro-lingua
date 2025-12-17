/**
 * Determinism & Reproducibility Layer
 */

import {
  PRNGState,
  ConfigHash,
  WeightHash,
  ReproducibilityManifest,
  ReplayVerification,
  SamplingConfig,
  SamplingState,
  ReproducibilitySnapshot,
  TOLERANCE_LEVELS,
  ToleranceLevel
} from '../types/reproducibility';

async function sha256(data: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

function quickHash(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

export class SeededRNG {
  private state: [number, number];
  private initialSeed: number;
  private callCount: number = 0;

  constructor(seed: number = 1337) {
    this.initialSeed = seed >>> 0;
    this.state = this.initState(this.initialSeed);
  }

  private initState(seed: number): [number, number] {
    let s = seed >>> 0;
    const mix = (): number => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    };
    return [mix(), mix()];
  }

  next(): number {
    this.callCount++;
    let s0 = this.state[0];
    let s1 = this.state[1];
    const result = (s0 + s1) >>> 0;
    s1 ^= s0;
    this.state[0] = ((s0 << 23) | (s0 >>> 9)) ^ s1 ^ (s1 << 17);
    this.state[1] = (s1 << 26) | (s1 >>> 6);
    return result / 4294967296;
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  nextGaussian(mean: number = 0, stddev: number = 1): number {
    const u1 = this.next();
    const u2 = this.next();
    const z0 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    return z0 * stddev + mean;
  }

  nextArray(length: number): number[] {
    const arr: number[] = new Array(length);
    for (let i = 0; i < length; i++) arr[i] = this.next();
    return arr;
  }

  nextGaussianArray(length: number, mean: number = 0, stddev: number = 1): number[] {
    const arr: number[] = new Array(length);
    for (let i = 0; i < length; i++) arr[i] = this.nextGaussian(mean, stddev);
    return arr;
  }

  shuffle<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i);
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  getState(): PRNGState {
    return { seed: this.initialSeed, state0: this.state[0], state1: this.state[1], callCount: this.callCount };
  }

  setState(savedState: PRNGState): void {
    this.initialSeed = savedState.seed;
    this.state = [savedState.state0 >>> 0, savedState.state1 >>> 0];
    this.callCount = savedState.callCount;
  }

  reset(): void {
    this.state = this.initState(this.initialSeed);
    this.callCount = 0;
  }

  fork(seedOffset: number = 1): SeededRNG {
    return new SeededRNG(this.initialSeed + seedOffset);
  }

  getCallCount(): number { return this.callCount; }
}

export class SeedManager {
  private globalSeed: number;
  private trainRng: SeededRNG;
  private samplingRng: SeededRNG;
  private initRng: SeededRNG;

  constructor(seed: number = 1337) {
    this.globalSeed = seed >>> 0;
    this.trainRng = new SeededRNG(this.globalSeed);
    this.samplingRng = new SeededRNG(this.globalSeed + 1);
    this.initRng = new SeededRNG(this.globalSeed + 2);
  }

  getTrainRng(): SeededRNG { return this.trainRng; }
  getSamplingRng(): SeededRNG { return this.samplingRng; }
  getInitRng(): SeededRNG { return this.initRng; }

  reset(): void {
    this.trainRng.reset();
    this.samplingRng.reset();
    this.initRng.reset();
  }

  getSeed(): number { return this.globalSeed; }

  getState(): { globalSeed: number; train: PRNGState; sampling: PRNGState; init: PRNGState } {
    return {
      globalSeed: this.globalSeed,
      train: this.trainRng.getState(),
      sampling: this.samplingRng.getState(),
      init: this.initRng.getState()
    };
  }

  restoreState(state: { globalSeed: number; train: PRNGState; sampling: PRNGState; init: PRNGState }): void {
    this.globalSeed = state.globalSeed;
    this.trainRng.setState(state.train);
    this.samplingRng.setState(state.sampling);
    this.initRng.setState(state.init);
  }
}

export class SeededSampler {
  private rng: SeededRNG;
  private config: SamplingConfig;
  private state: SamplingState;

  constructor(config: SamplingConfig) {
    this.config = config;
    this.rng = new SeededRNG(config.seed);
    this.state = {
      rngState: this.rng.getState(),
      tokensGenerated: 0,
      lastTokenId: -1,
      mirostatMu: config.mirostatTau ?? 5.0
    };
  }

  sample(logits: number[]): number {
    const scaledLogits = logits.map((l) => l / this.config.temperature);
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    let probs = expLogits.map((e) => e / sumExp);

    if (this.config.topK !== undefined && this.config.topK > 0) {
      probs = this.applyTopK(probs, this.config.topK);
    }
    if (this.config.topP !== undefined && this.config.topP < 1.0) {
      probs = this.applyTopP(probs, this.config.topP);
    }

    const tokenId = this.sampleFromProbs(probs);
    this.state.tokensGenerated++;
    this.state.lastTokenId = tokenId;
    this.state.rngState = this.rng.getState();
    return tokenId;
  }

  private applyTopK(probs: number[], k: number): number[] {
    const indexed = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    const filtered = new Array(probs.length).fill(0);
    let sum = 0;
    for (let i = 0; i < Math.min(k, indexed.length); i++) {
      filtered[indexed[i].i] = indexed[i].p;
      sum += indexed[i].p;
    }
    return filtered.map((p) => p / sum);
  }

  private applyTopP(probs: number[], p: number): number[] {
    const indexed = probs.map((prob, i) => ({ prob, i })).sort((a, b) => b.prob - a.prob);
    const filtered = new Array(probs.length).fill(0);
    let cumSum = 0, sum = 0;
    for (const { prob, i } of indexed) {
      if (cumSum >= p && sum > 0) break;
      filtered[i] = prob;
      cumSum += prob;
      sum += prob;
    }
    return filtered.map((prob) => prob / sum);
  }

  private sampleFromProbs(probs: number[]): number {
    const r = this.rng.next();
    let cumSum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i];
      if (r < cumSum) return i;
    }
    return probs.length - 1;
  }

  getState(): SamplingState { return { ...this.state }; }
  setState(state: SamplingState): void {
    this.state = { ...state };
    this.rng.setState(state.rngState);
  }

  reset(): void {
    this.rng = new SeededRNG(this.config.seed);
    this.state = {
      rngState: this.rng.getState(),
      tokensGenerated: 0,
      lastTokenId: -1,
      mirostatMu: this.config.mirostatTau ?? 5.0
    };
  }
}

export async function hashWeights(weights: { [layer: string]: Float32Array | number[][] }): Promise<WeightHash> {
  const layerHashes: Record<string, string> = {};
  const allData: string[] = [];
  for (const [name, data] of Object.entries(weights)) {
    const flatData = Array.isArray(data[0]) ? (data as number[][]).flat() : Array.from(data as Float32Array);
    const layerStr = flatData.map((v) => v.toFixed(8)).join(',');
    layerHashes[name] = quickHash(layerStr);
    allData.push(layerStr);
  }
  const combinedHash = await sha256(allData.join('|'));
  return {
    hash: combinedHash,
    algorithm: 'sha256',
    timestamp: new Date().toISOString(),
    parameterCount: Object.values(weights).reduce(
      (sum, w) => sum + (Array.isArray(w[0]) ? (w as number[][]).flat().length : (w as Float32Array).length), 0
    ),
    layerHashes
  };
}

export async function hashConfig(config: {
  hyperparameters: Record<string, unknown>;
  architecture: string;
  tokenizer: Record<string, unknown>;
  seed: number;
}): Promise<ConfigHash> {
  const components = {
    hyperparameters: quickHash(JSON.stringify(config.hyperparameters)),
    architecture: quickHash(config.architecture),
    tokenizer: quickHash(JSON.stringify(config.tokenizer)),
    seed: quickHash(String(config.seed))
  };
  const combinedHash = await sha256(Object.values(components).join('|'));
  return { hash: combinedHash, algorithm: 'sha256', timestamp: new Date().toISOString(), components };
}

export async function createManifest(params: {
  seed: number;
  config: Record<string, unknown>;
  tokenizer: Record<string, unknown>;
  corpus: string;
  initialWeights: { [layer: string]: Float32Array | number[][] };
  finalWeights?: { [layer: string]: Float32Array | number[][] };
  epochLosses: number[];
}): Promise<ReproducibilityManifest> {
  const configHash = await hashConfig({
    hyperparameters: params.config,
    architecture: String(params.config.architecture || 'unknown'),
    tokenizer: params.tokenizer,
    seed: params.seed
  });
  const initialWeightsHash = await hashWeights(params.initialWeights);
  const finalWeightsHash = params.finalWeights ? await hashWeights(params.finalWeights) : undefined;
  const corpusHash = await sha256(params.corpus);
  const tokenizerHash = await sha256(JSON.stringify(params.tokenizer));
  const epochChecksums = await Promise.all(
    params.epochLosses.map((loss, i) => sha256(`epoch:${i}:loss:${loss.toFixed(10)}`))
  );
  return {
    version: '1.0.0',
    createdAt: new Date().toISOString(),
    globalSeed: params.seed,
    trainingSeed: params.seed,
    samplingSeed: params.seed + 1,
    configHash,
    initialWeightsHash,
    finalWeightsHash,
    corpusHash,
    tokenizerHash,
    epochChecksums,
    environment: {
      platform: typeof navigator !== 'undefined' ? navigator.platform : 'node',
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
      timestamp: new Date().toISOString(),
      numericPrecision: 'float32'
    }
  };
}

export async function verifyReplay(
  manifest: ReproducibilityManifest,
  replay: { seed: number; config: Record<string, unknown>; tokenizer: Record<string, unknown>; corpus: string; epochLosses: number[] },
  tolerance: ToleranceLevel = 'normal'
): Promise<ReplayVerification> {
  const errors: string[] = [];
  const tol = TOLERANCE_LEVELS[tolerance];

  const seedMatch = manifest.globalSeed === replay.seed;
  if (!seedMatch) errors.push(`Seed mismatch: expected ${manifest.globalSeed}, got ${replay.seed}`);

  const corpusHash = await sha256(replay.corpus);
  const corpusMatch = corpusHash === manifest.corpusHash;
  if (!corpusMatch) errors.push('Corpus hash mismatch');

  const tokenizerHash = await sha256(JSON.stringify(replay.tokenizer));
  const tokenizerMatch = tokenizerHash === manifest.tokenizerHash;
  if (!tokenizerMatch) errors.push('Tokenizer hash mismatch');

  const replayConfigHash = await hashConfig({
    hyperparameters: replay.config,
    architecture: String(replay.config.architecture || 'unknown'),
    tokenizer: replay.tokenizer,
    seed: replay.seed
  });
  const configMatch = replayConfigHash.hash === manifest.configHash.hash;
  if (!configMatch) errors.push('Config hash mismatch');

  let epochMatchCount = 0;
  let maxLossDelta = 0;
  const totalEpochs = Math.min(manifest.epochChecksums.length, replay.epochLosses.length);

  for (let i = 0; i < totalEpochs; i++) {
    const replayChecksum = await sha256(`epoch:${i}:loss:${replay.epochLosses[i].toFixed(10)}`);
    if (replayChecksum === manifest.epochChecksums[i]) epochMatchCount++;
  }

  const success = seedMatch && corpusMatch && tokenizerMatch && configMatch && epochMatchCount >= totalEpochs * 0.9;
  return { success, configMatch, seedMatch, corpusMatch, tokenizerMatch, epochMatchCount, totalEpochs, maxLossDelta, tolerance: tol, errors };
}

export async function takeSnapshot(
  rng: SeededRNG,
  epoch: number,
  step: number,
  loss: number,
  weights: { [layer: string]: Float32Array | number[][] }
): Promise<ReproducibilitySnapshot> {
  const weightsHash = await hashWeights(weights);
  return { timestamp: new Date().toISOString(), rngState: rng.getState(), epoch, step, loss, weightsHash: weightsHash.hash };
}

export function withinTolerance(a: number, b: number, tolerance: ToleranceLevel = 'normal'): boolean {
  return Math.abs(a - b) <= TOLERANCE_LEVELS[tolerance];
}

export function arraysWithinTolerance(a: number[], b: number[], tolerance: ToleranceLevel = 'normal'): boolean {
  if (a.length !== b.length) return false;
  const tol = TOLERANCE_LEVELS[tolerance];
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

export { sha256, quickHash };
