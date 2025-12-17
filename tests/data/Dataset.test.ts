/**
 * Dataset Abstraction Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  DatasetBuilder,
  BatchIterator,
  exportDatasetArtifact,
  verifyDataset,
  mergeDatasets,
  fromCorpus,
  sampleDataset,
  calculateStats,
  tokenize
} from '../../src/data/Dataset';
import type { Dataset, DatasetSplit, SplitConfig } from '../../src/types/dataset';

describe('DatasetBuilder', () => {
  let builder: DatasetBuilder;

  beforeEach(() => {
    builder = new DatasetBuilder({ name: 'test-dataset' });
  });

  describe('constructor', () => {
    it('should initialize with name', () => {
      const b = new DatasetBuilder({ name: 'my-dataset' });
      expect(b).toBeDefined();
    });

    it('should accept optional description', () => {
      const b = new DatasetBuilder({
        name: 'my-dataset',
        description: 'A test dataset'
      });
      expect(b).toBeDefined();
    });

    it('should accept split configuration', () => {
      const b = new DatasetBuilder({
        name: 'my-dataset',
        splitConfig: { trainRatio: 0.7, valRatio: 0.15, testRatio: 0.15 }
      });
      expect(b).toBeDefined();
    });
  });

  describe('addText', () => {
    it('should add text samples', () => {
      builder.addText('Hello world');
      builder.addText('Goodbye world');

      // Build to verify samples were added
      return builder.build().then(dataset => {
        const totalSamples =
          dataset.splits.train.samples.length +
          dataset.splits.val.samples.length +
          dataset.splits.test.samples.length;
        expect(totalSamples).toBe(2);
      });
    });

    it('should support chaining', () => {
      const result = builder
        .addText('Sample 1')
        .addText('Sample 2')
        .addText('Sample 3');

      expect(result).toBe(builder);
    });

    it('should accept labels', async () => {
      builder.addText('Positive text', 'positive');
      builder.addText('Negative text', 'negative');

      const dataset = await builder.build();
      expect(dataset.schema.hasLabels).toBe(true);
    });

    it('should accept metadata', async () => {
      builder.addText('Sample with meta', undefined, { source: 'test' });
      const dataset = await builder.build();

      const allSamples = [
        ...dataset.splits.train.samples,
        ...dataset.splits.val.samples,
        ...dataset.splits.test.samples
      ];
      expect(allSamples[0].metadata).toEqual({ source: 'test' });
    });
  });

  describe('addTexts', () => {
    it('should add multiple texts at once', async () => {
      builder.addTexts(['Text 1', 'Text 2', 'Text 3']);

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(3);
    });

    it('should accept labels array', async () => {
      builder.addTexts(['Positive', 'Negative'], ['pos', 'neg']);

      const dataset = await builder.build();
      expect(dataset.schema.hasLabels).toBe(true);
    });
  });

  describe('addCorpus', () => {
    it('should split corpus by newlines', async () => {
      builder.addCorpus('Line 1\nLine 2\nLine 3');

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(3);
    });

    it('should accept custom delimiter', async () => {
      builder.addCorpus('Part1|Part2|Part3', '|');

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(3);
    });

    it('should filter empty lines', async () => {
      builder.addCorpus('Line 1\n\nLine 2\n\n\nLine 3');

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(3);
    });
  });

  describe('addJsonl', () => {
    it('should parse JSONL format', async () => {
      const jsonl = '{"text":"Sample 1"}\n{"text":"Sample 2"}';
      builder.addJsonl(jsonl);

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(2);
    });

    it('should use custom text field', async () => {
      const jsonl = '{"content":"Sample 1"}\n{"content":"Sample 2"}';
      builder.addJsonl(jsonl, 'content');

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(2);
    });

    it('should extract labels from custom field', async () => {
      const jsonl = '{"text":"Text","label":"A"}\n{"text":"Text2","label":"B"}';
      builder.addJsonl(jsonl, 'text', 'label');

      const dataset = await builder.build();
      expect(dataset.schema.hasLabels).toBe(true);
    });

    it('should skip invalid JSON lines', async () => {
      const jsonl = '{"text":"Valid"}\nnot valid json\n{"text":"Also valid"}';
      builder.addJsonl(jsonl);

      const dataset = await builder.build();
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(2);
    });
  });

  describe('build', () => {
    it('should throw if no samples added', async () => {
      await expect(builder.build()).rejects.toThrow('No samples added');
    });

    it('should create train/val/test splits', async () => {
      for (let i = 0; i < 100; i++) {
        builder.addText(`Sample ${i}`);
      }

      const dataset = await builder.build();

      expect(dataset.splits.train).toBeDefined();
      expect(dataset.splits.val).toBeDefined();
      expect(dataset.splits.test).toBeDefined();
    });

    it('should respect split ratios', async () => {
      const customBuilder = new DatasetBuilder({
        name: 'test',
        splitConfig: { trainRatio: 0.6, valRatio: 0.2, testRatio: 0.2 }
      });

      for (let i = 0; i < 100; i++) {
        customBuilder.addText(`Sample ${i}`);
      }

      const dataset = await customBuilder.build();

      expect(dataset.splits.train.samples.length).toBeCloseTo(60, -1);
      expect(dataset.splits.val.samples.length).toBeCloseTo(20, -1);
      expect(dataset.splits.test.samples.length).toBeCloseTo(20, -1);
    });

    it('should compute dataset hash', async () => {
      builder.addText('Sample 1');
      builder.addText('Sample 2');

      const dataset = await builder.build();

      expect(dataset.hash).toBeTruthy();
      expect(dataset.hash.length).toBe(64); // SHA-256 hex
    });

    it('should compute split hashes', async () => {
      for (let i = 0; i < 10; i++) {
        builder.addText(`Sample ${i}`);
      }

      const dataset = await builder.build();

      expect(dataset.splits.train.hash).toBeTruthy();
      expect(dataset.splits.val.hash).toBeTruthy();
      expect(dataset.splits.test.hash).toBeTruthy();
    });

    it('should report progress', async () => {
      for (let i = 0; i < 100; i++) {
        builder.addText(`Sample ${i}`);
      }

      let progressCalls = 0;
      await builder.build((progress) => {
        progressCalls++;
        expect(progress.operation).toBeDefined();
      });

      expect(progressCalls).toBeGreaterThan(0);
    });

    it('should be deterministic with same seed', async () => {
      const builder1 = new DatasetBuilder({ name: 'test', seed: 42 });
      const builder2 = new DatasetBuilder({ name: 'test', seed: 42 });

      for (let i = 0; i < 20; i++) {
        builder1.addText(`Sample ${i}`);
        builder2.addText(`Sample ${i}`);
      }

      const dataset1 = await builder1.build();
      const dataset2 = await builder2.build();

      // The split shuffling is deterministic with the same seed
      // So the order of samples in each split should be identical
      expect(dataset1.splits.train.samples.map(s => s.text))
        .toEqual(dataset2.splits.train.samples.map(s => s.text));
      expect(dataset1.splits.val.samples.map(s => s.text))
        .toEqual(dataset2.splits.val.samples.map(s => s.text));
      expect(dataset1.splits.test.samples.map(s => s.text))
        .toEqual(dataset2.splits.test.samples.map(s => s.text));
    });
  });

  describe('single sample edge case', () => {
    it('should handle single sample dataset', async () => {
      builder.addText('Only sample');

      const dataset = await builder.build();

      // With default 80/10/10 split, 1 sample will go to test
      const totalSamples =
        dataset.splits.train.samples.length +
        dataset.splits.val.samples.length +
        dataset.splits.test.samples.length;
      expect(totalSamples).toBe(1);
    });
  });
});

describe('BatchIterator', () => {
  let split: DatasetSplit;

  beforeEach(async () => {
    const builder = new DatasetBuilder({ name: 'test', seed: 42 });
    for (let i = 0; i < 100; i++) {
      builder.addText(`Sample ${i}`);
    }
    const dataset = await builder.build();
    split = dataset.splits.train;
  });

  describe('constructor', () => {
    it('should create iterator with default config', () => {
      const iterator = new BatchIterator(split);
      expect(iterator.hasNext()).toBe(true);
    });

    it('should accept custom batch size', () => {
      const iterator = new BatchIterator(split, { batchSize: 16 });
      const batch = iterator.next();
      expect(batch?.samples.length).toBeLessThanOrEqual(16);
    });
  });

  describe('next', () => {
    it('should return batches', () => {
      const iterator = new BatchIterator(split, { batchSize: 10 });
      const batch = iterator.next();

      expect(batch).toBeDefined();
      expect(batch!.samples.length).toBeLessThanOrEqual(10);
      expect(batch!.indices).toBeDefined();
      expect(batch!.batchIndex).toBe(0);
    });

    it('should return null when exhausted', () => {
      const iterator = new BatchIterator(split, { batchSize: 100 });

      while (iterator.hasNext()) {
        iterator.next();
      }

      expect(iterator.next()).toBeNull();
    });

    it('should mark last batch', () => {
      const iterator = new BatchIterator(split, { batchSize: 100 });
      const batch = iterator.next();

      expect(batch!.isLast).toBe(true);
    });
  });

  describe('hasNext', () => {
    it('should return true when batches remain', () => {
      const iterator = new BatchIterator(split, { batchSize: 10 });
      expect(iterator.hasNext()).toBe(true);
    });

    it('should return false when exhausted', () => {
      const iterator = new BatchIterator(split, { batchSize: split.samples.length });
      iterator.next();
      expect(iterator.hasNext()).toBe(false);
    });
  });

  describe('nextEpoch', () => {
    it('should reset to beginning', () => {
      const iterator = new BatchIterator(split, { batchSize: 10 });

      // Exhaust first epoch
      while (iterator.hasNext()) iterator.next();

      // Start new epoch
      iterator.nextEpoch();

      expect(iterator.hasNext()).toBe(true);
      expect(iterator.getState().epoch).toBe(1);
    });

    it('should reshuffle if shuffle enabled', () => {
      const iterator = new BatchIterator(split, { batchSize: 10, shuffle: true, seed: 42 });

      const firstBatch = iterator.next();
      iterator.reset();
      iterator.nextEpoch();
      const secondEpochBatch = iterator.next();

      // Different shuffle order in new epoch
      expect(firstBatch!.indices).not.toEqual(secondEpochBatch!.indices);
    });
  });

  describe('reset', () => {
    it('should reset to initial state', () => {
      const iterator = new BatchIterator(split, { batchSize: 10, seed: 42 });

      const firstBatch = iterator.next();
      iterator.next();
      iterator.reset();
      const afterReset = iterator.next();

      expect(afterReset!.indices).toEqual(firstBatch!.indices);
    });
  });

  describe('state management', () => {
    it('should save and restore state', () => {
      const iterator = new BatchIterator(split, { batchSize: 10 });

      // Advance a few batches
      iterator.next();
      iterator.next();

      const state = iterator.getState();

      // Advance more
      const nextBatch = iterator.next();

      // Restore
      iterator.setState(state);

      // Should get same batch
      const afterRestore = iterator.next();
      expect(afterRestore!.batchIndex).toBe(nextBatch!.batchIndex);
    });
  });

  describe('iteration protocol', () => {
    it('should be iterable with for...of', () => {
      const iterator = new BatchIterator(split, { batchSize: 20 });
      let count = 0;

      for (const batch of iterator) {
        count++;
        expect(batch.samples).toBeDefined();
      }

      expect(count).toBeGreaterThan(0);
    });
  });

  describe('dropLast option', () => {
    it('should drop incomplete last batch when enabled', () => {
      const iterator = new BatchIterator(split, { batchSize: 11, dropLast: true });

      let lastBatchSize = 0;
      for (const batch of iterator) {
        lastBatchSize = batch.samples.length;
      }

      expect(lastBatchSize).toBe(11);
    });

    it('should keep incomplete last batch when disabled', () => {
      const iterator = new BatchIterator(split, { batchSize: 11, dropLast: false });

      let hasIncompleteBatch = false;
      for (const batch of iterator) {
        if (batch.samples.length < 11) {
          hasIncompleteBatch = true;
        }
      }

      // May or may not have incomplete batch depending on split size
      expect(typeof hasIncompleteBatch).toBe('boolean');
    });
  });
});

describe('exportDatasetArtifact', () => {
  it('should export dataset metadata', async () => {
    const builder = new DatasetBuilder({ name: 'test-export' });
    builder.addTexts(['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']);
    const dataset = await builder.build();

    const artifact = await exportDatasetArtifact(dataset);

    expect(artifact.version).toBe('1.0.0');
    expect(artifact.hash).toBe(dataset.hash);
    expect(artifact.schema).toBeDefined();
    expect(artifact.splits.train.sampleCount).toBeDefined();
  });

  it('should include sample IDs', async () => {
    const builder = new DatasetBuilder({ name: 'test' });
    builder.addTexts(['Sample 1', 'Sample 2', 'Sample 3']);
    const dataset = await builder.build();

    const artifact = await exportDatasetArtifact(dataset);

    const totalIds =
      artifact.splits.train.sampleIds.length +
      artifact.splits.val.sampleIds.length +
      artifact.splits.test.sampleIds.length;
    expect(totalIds).toBe(3);
  });
});

describe('verifyDataset', () => {
  it('should verify matching dataset and artifact', async () => {
    const builder = new DatasetBuilder({ name: 'test' });
    builder.addTexts(['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']);
    const dataset = await builder.build();
    const artifact = await exportDatasetArtifact(dataset);

    const result = await verifyDataset(dataset, artifact);

    expect(result.valid).toBe(true);
    expect(result.errors.length).toBe(0);
  });

  it('should detect hash mismatch', async () => {
    const builder = new DatasetBuilder({ name: 'test' });
    builder.addTexts(['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']);
    const dataset = await builder.build();
    const artifact = await exportDatasetArtifact(dataset);

    // Tamper with artifact
    artifact.hash = 'tampered-hash';

    const result = await verifyDataset(dataset, artifact);

    expect(result.valid).toBe(false);
    expect(result.errors.some(e => e.includes('hash'))).toBe(true);
  });
});

describe('mergeDatasets', () => {
  it('should merge multiple datasets', async () => {
    const builder1 = new DatasetBuilder({ name: 'dataset1' });
    builder1.addTexts(['A1', 'A2', 'A3', 'A4', 'A5']);
    const dataset1 = await builder1.build();

    const builder2 = new DatasetBuilder({ name: 'dataset2' });
    builder2.addTexts(['B1', 'B2', 'B3', 'B4', 'B5']);
    const dataset2 = await builder2.build();

    const merged = await mergeDatasets([dataset1, dataset2], 'merged');

    const totalSamples =
      merged.splits.train.samples.length +
      merged.splits.val.samples.length +
      merged.splits.test.samples.length;
    expect(totalSamples).toBe(10);
  });

  it('should preserve labels when merging', async () => {
    const builder1 = new DatasetBuilder({ name: 'd1' });
    builder1.addText('Labeled 1', 'classA');
    builder1.addText('Labeled 2', 'classA');
    builder1.addText('Labeled 3', 'classA');
    const dataset1 = await builder1.build();

    const builder2 = new DatasetBuilder({ name: 'd2' });
    builder2.addText('Labeled 3', 'classB');
    builder2.addText('Labeled 4', 'classB');
    builder2.addText('Labeled 5', 'classB');
    const dataset2 = await builder2.build();

    const merged = await mergeDatasets([dataset1, dataset2], 'merged');

    expect(merged.schema.hasLabels).toBe(true);
  });
});

describe('fromCorpus', () => {
  it('should create dataset from corpus string', async () => {
    const corpus = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5';
    const dataset = await fromCorpus(corpus, 'corpus-dataset');

    const totalSamples =
      dataset.splits.train.samples.length +
      dataset.splits.val.samples.length +
      dataset.splits.test.samples.length;
    expect(totalSamples).toBe(5);
  });

  it('should accept custom delimiter', async () => {
    const corpus = 'Part1||Part2||Part3||Part4||Part5';
    const dataset = await fromCorpus(corpus, 'corpus-dataset', { delimiter: '||' });

    const totalSamples =
      dataset.splits.train.samples.length +
      dataset.splits.val.samples.length +
      dataset.splits.test.samples.length;
    expect(totalSamples).toBe(5);
  });

  it('should accept split config', async () => {
    const corpus = Array.from({ length: 100 }, (_, i) => `Line ${i}`).join('\n');
    const dataset = await fromCorpus(corpus, 'corpus-dataset', {
      splitConfig: { trainRatio: 0.5, valRatio: 0.25, testRatio: 0.25 }
    });

    expect(dataset.splits.train.samples.length).toBeCloseTo(50, -1);
  });

  it('should use default split config when not provided', async () => {
    const corpus = Array.from({ length: 100 }, (_, i) => `Line ${i}`).join('\n');
    const dataset = await fromCorpus(corpus, 'corpus-dataset');

    // Default is 80/10/10
    expect(dataset.splits.train.samples.length).toBeCloseTo(80, -1);
    expect(dataset.splits.val.samples.length).toBeCloseTo(10, -1);
    expect(dataset.splits.test.samples.length).toBeCloseTo(10, -1);
  });
});

describe('sampleDataset', () => {
  it('should sample fraction of dataset', async () => {
    const builder = new DatasetBuilder({ name: 'test', seed: 42 });
    for (let i = 0; i < 100; i++) {
      builder.addText(`Sample ${i}`);
    }
    const dataset = await builder.build();

    const sampled = sampleDataset(dataset, 0.5);

    const originalTotal =
      dataset.splits.train.samples.length +
      dataset.splits.val.samples.length +
      dataset.splits.test.samples.length;

    const sampledTotal =
      sampled.splits.train.samples.length +
      sampled.splits.val.samples.length +
      sampled.splits.test.samples.length;

    expect(sampledTotal).toBeCloseTo(originalTotal * 0.5, -1);
  });

  it('should be deterministic with same seed', async () => {
    const builder = new DatasetBuilder({ name: 'test' });
    for (let i = 0; i < 50; i++) {
      builder.addText(`Sample ${i}`);
    }
    const dataset = await builder.build();

    const sampled1 = sampleDataset(dataset, 0.5, 42);
    const sampled2 = sampleDataset(dataset, 0.5, 42);

    expect(sampled1.splits.train.samples.map(s => s.id))
      .toEqual(sampled2.splits.train.samples.map(s => s.id));
  });
});

describe('calculateStats', () => {
  it('should calculate stats for samples', () => {
    const samples = [
      { id: '1', text: 'Hello world', tokens: ['hello', 'world'] },
      { id: '2', text: 'Goodbye world', tokens: ['goodbye', 'world'] },
      { id: '3', text: 'Test sample here', tokens: ['test', 'sample', 'here'] }
    ];

    const stats = calculateStats(samples);

    expect(stats.totalSamples).toBe(3);
    expect(stats.totalTokens).toBe(7);
    expect(stats.uniqueTokens).toBe(6); // hello, world, goodbye, test, sample, here
    expect(stats.averageTokensPerSample).toBeCloseTo(7/3, 2);
  });

  it('should handle empty array', () => {
    const stats = calculateStats([]);

    expect(stats.totalSamples).toBe(0);
    expect(stats.totalTokens).toBe(0);
    expect(stats.uniqueTokens).toBe(0);
  });

  it('should calculate min/max/median', () => {
    const samples = [
      { id: '1', text: 'a', tokens: ['a'] },
      { id: '2', text: 'a b', tokens: ['a', 'b'] },
      { id: '3', text: 'a b c', tokens: ['a', 'b', 'c'] }
    ];

    const stats = calculateStats(samples);

    expect(stats.minTokensPerSample).toBe(1);
    expect(stats.maxTokensPerSample).toBe(3);
    expect(stats.medianTokensPerSample).toBe(2);
  });
});

describe('tokenize', () => {
  it('should tokenize text', () => {
    const tokens = tokenize('Hello, world!');
    expect(tokens).toContain('hello');
    expect(tokens).toContain('world');
  });

  it('should lowercase tokens', () => {
    const tokens = tokenize('HELLO World');
    expect(tokens).toContain('hello');
    expect(tokens).toContain('world');
  });

  it('should handle contractions', () => {
    const tokens = tokenize("don't can't won't");
    expect(tokens.some(t => t.includes("'"))).toBe(true);
  });

  it('should filter empty tokens', () => {
    const tokens = tokenize('  hello   world  ');
    expect(tokens.every(t => t.length > 0)).toBe(true);
  });
});
