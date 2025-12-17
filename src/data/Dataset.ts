/**
 * Dataset Abstraction
 */

import {
  Dataset,
  DatasetSchema,
  DatasetSplit,
  DatasetStats,
  DataSample,
  SplitConfig,
  DatasetArtifact,
  SerializedSplit,
  DatasetBuilderConfig,
  BatchConfig,
  Batch,
  BatchIteratorState,
  DatasetProgressCallback,
  DEFAULT_SPLIT_CONFIG
} from '../types/dataset';
import { SeededRNG } from '../lib/Reproducibility';

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

function generateId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
}

function tokenize(text: string): string[] {
  return text.toLowerCase().replace(/[^\p{L}\d\s'-]/gu, ' ').split(/\s+/).filter((t) => t.length > 0);
}

function calculateStats(samples: DataSample[]): DatasetStats {
  if (samples.length === 0) {
    return { totalSamples: 0, totalTokens: 0, totalCharacters: 0, uniqueTokens: 0, averageTokensPerSample: 0, medianTokensPerSample: 0, maxTokensPerSample: 0, minTokensPerSample: 0 };
  }
  const allTokens: string[] = [];
  const tokenCounts: number[] = [];
  let totalChars = 0;

  for (const sample of samples) {
    const tokens = sample.tokens || tokenize(sample.text);
    allTokens.push(...tokens);
    tokenCounts.push(tokens.length);
    totalChars += sample.text.length;
  }

  const sortedCounts = [...tokenCounts].sort((a, b) => a - b);
  const mid = Math.floor(sortedCounts.length / 2);
  const median = sortedCounts.length % 2 === 0 ? (sortedCounts[mid - 1] + sortedCounts[mid]) / 2 : sortedCounts[mid];

  return {
    totalSamples: samples.length,
    totalTokens: allTokens.length,
    totalCharacters: totalChars,
    uniqueTokens: new Set(allTokens).size,
    averageTokensPerSample: allTokens.length / samples.length,
    medianTokensPerSample: median,
    maxTokensPerSample: Math.max(...tokenCounts),
    minTokensPerSample: Math.min(...tokenCounts)
  };
}

export class DatasetBuilder {
  private config: DatasetBuilderConfig;
  private samples: DataSample[] = [];
  private rng: SeededRNG;

  constructor(config: DatasetBuilderConfig) {
    this.config = {
      name: config.name,
      description: config.description ?? '',
      language: config.language ?? 'en',
      splitConfig: config.splitConfig ?? DEFAULT_SPLIT_CONFIG,
      seed: config.seed ?? 42
    };
    this.rng = new SeededRNG(this.config.seed!);
  }

  addText(text: string, label?: string | number, metadata?: Record<string, unknown>): this {
    const tokens = tokenize(text);
    this.samples.push({ id: generateId(), text, tokens, label, metadata });
    return this;
  }

  addTexts(texts: string[], labels?: (string | number)[]): this {
    for (let i = 0; i < texts.length; i++) {
      this.addText(texts[i], labels?.[i]);
    }
    return this;
  }

  addCorpus(corpus: string, delimiter: string | RegExp = '\n'): this {
    const texts = corpus.split(delimiter).filter((t) => t.trim().length > 0);
    return this.addTexts(texts);
  }

  addJsonl(jsonl: string, textField: string = 'text', labelField?: string): this {
    const lines = jsonl.split('\n').filter((l) => l.trim().length > 0);
    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        const text = obj[textField];
        const label = labelField ? obj[labelField] : undefined;
        if (text) this.addText(text, label, obj);
      } catch { /* skip invalid JSON */ }
    }
    return this;
  }

  async build(onProgress?: DatasetProgressCallback): Promise<Dataset> {
    if (this.samples.length === 0) throw new Error('No samples added to dataset');

    onProgress?.({ operation: 'tokenizing', current: 0, total: this.samples.length, message: 'Tokenizing samples...' });

    for (let i = 0; i < this.samples.length; i++) {
      if (!this.samples[i].tokens) this.samples[i].tokens = tokenize(this.samples[i].text);
      if (onProgress && i % 100 === 0) onProgress({ operation: 'tokenizing', current: i, total: this.samples.length });
    }

    onProgress?.({ operation: 'splitting', current: 0, total: 3, message: 'Creating splits...' });

    const shuffled = [...this.samples];
    this.rng.shuffle(shuffled);

    const splitConfig = this.config.splitConfig!;
    const total = shuffled.length;
    const trainEnd = Math.floor(total * splitConfig.trainRatio);
    const valEnd = trainEnd + Math.floor(total * splitConfig.valRatio);

    const trainSamples = shuffled.slice(0, trainEnd);
    const valSamples = shuffled.slice(trainEnd, valEnd);
    const testSamples = shuffled.slice(valEnd);

    onProgress?.({ operation: 'hashing', current: 0, total: 4, message: 'Computing hashes...' });

    const trainSplit = await this.createSplit('train', trainSamples);
    onProgress?.({ operation: 'hashing', current: 1, total: 4 });

    const valSplit = await this.createSplit('val', valSamples);
    onProgress?.({ operation: 'hashing', current: 2, total: 4 });

    const testSplit = await this.createSplit('test', testSamples);
    onProgress?.({ operation: 'hashing', current: 3, total: 4 });

    const combinedHash = await sha256(`${trainSplit.hash}|${valSplit.hash}|${testSplit.hash}`);
    onProgress?.({ operation: 'hashing', current: 4, total: 4 });

    const schema: DatasetSchema = {
      name: this.config.name,
      version: '1.0.0',
      description: this.config.description!,
      language: this.config.language!,
      createdAt: new Date().toISOString(),
      modifiedAt: new Date().toISOString(),
      stats: calculateStats(this.samples),
      format: 'text',
      encoding: 'utf-8',
      hasLabels: this.samples.some((s) => s.label !== undefined),
      labelSchema: this.samples.some((s) => s.label !== undefined)
        ? { type: 'classification', classes: [...new Set(this.samples.map((s) => String(s.label)).filter(Boolean))] }
        : undefined
    };

    return {
      schema,
      splits: { train: trainSplit, val: valSplit, test: testSplit },
      hash: combinedHash,
      splitConfig,
      shuffleSeed: this.config.seed!,
      originalOrder: this.samples.map((s) => s.id)
    };
  }

  private async createSplit(name: 'train' | 'val' | 'test', samples: DataSample[]): Promise<DatasetSplit> {
    const samplesStr = JSON.stringify(samples.map((s) => s.id + s.text));
    const hash = await sha256(samplesStr);
    return { name, samples, stats: calculateStats(samples), hash };
  }
}

export class BatchIterator {
  private split: DatasetSplit;
  private config: BatchConfig;
  private rng: SeededRNG;
  private state: BatchIteratorState;

  constructor(split: DatasetSplit, config: Partial<BatchConfig> = {}) {
    this.split = split;
    this.config = { batchSize: 32, shuffle: true, seed: 42, dropLast: false, ...config };
    this.rng = new SeededRNG(this.config.seed);
    this.state = this.initState();
  }

  private initState(): BatchIteratorState {
    const indices = Array.from({ length: this.split.samples.length }, (_, i) => i);
    if (this.config.shuffle) this.rng.shuffle(indices);
    const totalBatches = this.config.dropLast ? Math.floor(indices.length / this.config.batchSize) : Math.ceil(indices.length / this.config.batchSize);
    const rngState = this.rng.getState();
    return { currentBatch: 0, totalBatches, epoch: 0, rngState: [rngState.state0, rngState.state1], shuffledIndices: indices };
  }

  next(): Batch | null {
    if (this.state.currentBatch >= this.state.totalBatches) return null;
    const startIdx = this.state.currentBatch * this.config.batchSize;
    const endIdx = Math.min(startIdx + this.config.batchSize, this.split.samples.length);
    const batchIndices = this.state.shuffledIndices.slice(startIdx, endIdx);
    const samples = batchIndices.map((i) => this.split.samples[i]);
    const batch: Batch = { samples, indices: batchIndices, batchIndex: this.state.currentBatch, isLast: this.state.currentBatch === this.state.totalBatches - 1 };
    this.state.currentBatch++;
    return batch;
  }

  hasNext(): boolean { return this.state.currentBatch < this.state.totalBatches; }

  nextEpoch(): void {
    this.state.epoch++;
    this.state.currentBatch = 0;
    if (this.config.shuffle) {
      const indices = Array.from({ length: this.split.samples.length }, (_, i) => i);
      this.rng.shuffle(indices);
      this.state.shuffledIndices = indices;
      const rngState = this.rng.getState();
      this.state.rngState = [rngState.state0, rngState.state1];
    }
  }

  reset(): void {
    this.rng = new SeededRNG(this.config.seed);
    this.state = this.initState();
  }

  getState(): BatchIteratorState { return { ...this.state }; }
  setState(state: BatchIteratorState): void { this.state = { ...state }; }

  *[Symbol.iterator](): Generator<Batch, void, unknown> {
    while (this.hasNext()) {
      const batch = this.next();
      if (batch) yield batch;
    }
  }
}

export async function exportDatasetArtifact(dataset: Dataset): Promise<DatasetArtifact> {
  const serializeSplit = (split: DatasetSplit): SerializedSplit => ({
    name: split.name,
    sampleCount: split.samples.length,
    stats: split.stats,
    hash: split.hash,
    sampleIds: split.samples.map((s) => s.id)
  });
  return {
    version: '1.0.0',
    hash: dataset.hash,
    createdAt: dataset.schema.createdAt,
    schema: dataset.schema,
    splitConfig: dataset.splitConfig,
    splits: { train: serializeSplit(dataset.splits.train), val: serializeSplit(dataset.splits.val), test: serializeSplit(dataset.splits.test) }
  };
}

export async function verifyDataset(dataset: Dataset, artifact: DatasetArtifact): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];
  if (dataset.hash !== artifact.hash) errors.push('Dataset hash mismatch');
  for (const splitName of ['train', 'val', 'test'] as const) {
    if (dataset.splits[splitName].hash !== artifact.splits[splitName].hash) errors.push(`${splitName} split hash mismatch`);
    if (dataset.splits[splitName].samples.length !== artifact.splits[splitName].sampleCount) errors.push(`${splitName} split sample count mismatch`);
  }
  return { valid: errors.length === 0, errors };
}

export async function mergeDatasets(datasets: Dataset[], name: string, seed: number = 42): Promise<Dataset> {
  const builder = new DatasetBuilder({ name, seed });
  for (const dataset of datasets) {
    for (const sample of dataset.splits.train.samples) builder.addText(sample.text, sample.label, sample.metadata);
    for (const sample of dataset.splits.val.samples) builder.addText(sample.text, sample.label, sample.metadata);
    for (const sample of dataset.splits.test.samples) builder.addText(sample.text, sample.label, sample.metadata);
  }
  return builder.build();
}

export async function fromCorpus(corpus: string, name: string, options: { delimiter?: string | RegExp; splitConfig?: SplitConfig; seed?: number } = {}): Promise<Dataset> {
  const builder = new DatasetBuilder({ name, splitConfig: options.splitConfig, seed: options.seed });
  builder.addCorpus(corpus, options.delimiter || '\n');
  return builder.build();
}

export function sampleDataset(dataset: Dataset, fraction: number, seed: number = 42): Dataset {
  const rng = new SeededRNG(seed);
  const sampleSplit = (split: DatasetSplit, frac: number): DatasetSplit => {
    const n = Math.floor(split.samples.length * frac);
    const shuffled = [...split.samples];
    rng.shuffle(shuffled);
    const sampled = shuffled.slice(0, n);
    return { ...split, samples: sampled, stats: calculateStats(sampled), hash: quickHash(JSON.stringify(sampled.map((s) => s.id))) };
  };
  return { ...dataset, splits: { train: sampleSplit(dataset.splits.train, fraction), val: sampleSplit(dataset.splits.val, fraction), test: sampleSplit(dataset.splits.test, fraction) }, hash: quickHash(`sampled-${fraction}-${seed}`) };
}

export { calculateStats, tokenize, sha256, quickHash, generateId, DEFAULT_SPLIT_CONFIG };
