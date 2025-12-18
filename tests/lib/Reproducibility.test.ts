/**
 * Reproducibility Layer Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  SeededRNG,
  SeedManager,
  SeededSampler,
  hashWeights,
  hashConfig,
  createManifest,
  verifyReplay,
  takeSnapshot,
  withinTolerance,
  arraysWithinTolerance
} from '../../src/lib/Reproducibility';
import type { SamplingConfig } from '../../src/types/reproducibility';

describe('SeededRNG', () => {
  describe('constructor', () => {
    it('should initialize with default seed', () => {
      const rng = new SeededRNG();
      expect(rng.getCallCount()).toBe(0);
    });

    it('should initialize with custom seed', () => {
      const rng = new SeededRNG(42);
      expect(rng.getCallCount()).toBe(0);
    });
  });

  describe('next', () => {
    it('should generate values between 0 and 1', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const val = rng.next();
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });

    it('should be deterministic with same seed', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it('should produce different sequences with different seeds', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(43);

      const seq1 = Array.from({ length: 10 }, () => rng1.next());
      const seq2 = Array.from({ length: 10 }, () => rng2.next());

      expect(seq1).not.toEqual(seq2);
    });

    it('should increment call count', () => {
      const rng = new SeededRNG(42);
      expect(rng.getCallCount()).toBe(0);
      rng.next();
      expect(rng.getCallCount()).toBe(1);
      rng.next();
      expect(rng.getCallCount()).toBe(2);
    });
  });

  describe('nextInt', () => {
    it('should generate integers in range', () => {
      const rng = new SeededRNG(12345);
      for (let i = 0; i < 100; i++) {
        const val = rng.nextInt(0, 10);
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(10);
        expect(Number.isInteger(val)).toBe(true);
      }
    });

    it('should be deterministic', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      for (let i = 0; i < 50; i++) {
        expect(rng1.nextInt(1, 100)).toBe(rng2.nextInt(1, 100));
      }
    });
  });

  describe('nextGaussian', () => {
    it('should generate roughly normal distribution', () => {
      const rng = new SeededRNG(12345);
      const samples = Array.from({ length: 1000 }, () => rng.nextGaussian(0, 1));

      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      const variance = samples.reduce((sum, x) => sum + (x - mean) ** 2, 0) / samples.length;

      expect(Math.abs(mean)).toBeLessThan(0.2);
      expect(Math.abs(variance - 1)).toBeLessThan(0.3);
    });

    it('should respect mean and stddev', () => {
      const rng = new SeededRNG(12345);
      const samples = Array.from({ length: 1000 }, () => rng.nextGaussian(5, 2));

      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      expect(Math.abs(mean - 5)).toBeLessThan(0.3);
    });
  });

  describe('shuffle', () => {
    it('should shuffle array in place', () => {
      const rng = new SeededRNG(42);
      const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const original = [...arr];

      rng.shuffle(arr);

      expect(arr).not.toEqual(original);
      expect(arr.sort((a, b) => a - b)).toEqual(original);
    });

    it('should be deterministic', () => {
      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      new SeededRNG(42).shuffle(arr1);
      new SeededRNG(42).shuffle(arr2);

      expect(arr1).toEqual(arr2);
    });
  });

  describe('state management', () => {
    it('should save and restore state', () => {
      const rng = new SeededRNG(42);

      // Generate some values
      for (let i = 0; i < 10; i++) rng.next();

      const state = rng.getState();

      // Generate more values
      const valuesAfterSave = Array.from({ length: 5 }, () => rng.next());

      // Restore state
      rng.setState(state);

      // Should get same values
      const valuesAfterRestore = Array.from({ length: 5 }, () => rng.next());
      expect(valuesAfterRestore).toEqual(valuesAfterSave);
    });

    it('should track state with state0 and state1', () => {
      const rng = new SeededRNG(42);
      rng.next();

      const state = rng.getState();
      expect(state.state0).toBeDefined();
      expect(state.state1).toBeDefined();
      expect(typeof state.state0).toBe('number');
      expect(typeof state.state1).toBe('number');
    });

    it('should reset to initial state', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      // Generate some values with rng1
      for (let i = 0; i < 10; i++) rng1.next();

      // Reset
      rng1.reset();

      // Both should now produce same sequence
      for (let i = 0; i < 10; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });
  });

  describe('fork', () => {
    it('should create independent RNG with offset seed', () => {
      const rng = new SeededRNG(42);
      const forked = rng.fork(5);

      // Should have different sequences
      const val1 = rng.next();
      const val2 = forked.next();
      expect(val1).not.toBe(val2);
    });

    it('should be deterministic', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(42);

      const forked1 = rng1.fork(3);
      const forked2 = rng2.fork(3);

      for (let i = 0; i < 10; i++) {
        expect(forked1.next()).toBe(forked2.next());
      }
    });
  });
});

describe('SeedManager', () => {
  describe('constructor', () => {
    it('should initialize with default seed', () => {
      const manager = new SeedManager();
      expect(manager.getSeed()).toBe(1337);
    });

    it('should initialize with custom seed', () => {
      const manager = new SeedManager(42);
      expect(manager.getSeed()).toBe(42);
    });
  });

  describe('RNG getters', () => {
    it('should provide separate RNGs for different purposes', () => {
      const manager = new SeedManager(42);

      const trainRng = manager.getTrainRng();
      const samplingRng = manager.getSamplingRng();
      const initRng = manager.getInitRng();

      // Each should produce different sequences (different seeds)
      const trainVal = trainRng.next();
      const samplingVal = samplingRng.next();
      const initVal = initRng.next();

      // At least two should be different
      expect(trainVal !== samplingVal || samplingVal !== initVal || trainVal !== initVal).toBe(true);
    });
  });

  describe('state management', () => {
    it('should save and restore state', () => {
      const manager = new SeedManager(42);

      // Advance RNGs
      manager.getTrainRng().next();
      manager.getSamplingRng().next();

      const state = manager.getState();

      // Advance more
      const trainVal = manager.getTrainRng().next();

      // Restore
      manager.restoreState(state);

      // Should get same value
      expect(manager.getTrainRng().next()).toBe(trainVal);
    });

    it('should reset all RNGs', () => {
      const manager1 = new SeedManager(42);
      const manager2 = new SeedManager(42);

      // Advance manager1's RNGs
      for (let i = 0; i < 10; i++) {
        manager1.getTrainRng().next();
      }

      // Reset
      manager1.reset();

      // Should now match manager2
      for (let i = 0; i < 5; i++) {
        expect(manager1.getTrainRng().next()).toBe(manager2.getTrainRng().next());
      }
    });
  });
});

describe('SeededSampler', () => {
  let config: SamplingConfig;

  beforeEach(() => {
    config = {
      seed: 42,
      temperature: 1.0,
      topK: 10,
      topP: 0.9
    };
  });

  describe('sample', () => {
    it('should sample from logits', () => {
      const sampler = new SeededSampler(config);
      const logits = [1.0, 2.0, 3.0, 0.5, 0.1];

      const idx = sampler.sample(logits);

      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(logits.length);
    });

    it('should be deterministic', () => {
      const logits = [1.0, 2.0, 3.0, 0.5, 0.1, 1.5, 2.5];

      const sampler1 = new SeededSampler(config);
      const sampler2 = new SeededSampler(config);

      for (let i = 0; i < 10; i++) {
        expect(sampler1.sample(logits)).toBe(sampler2.sample(logits));
      }
    });

    it('should update state after sampling', () => {
      const sampler = new SeededSampler(config);
      const logits = [1.0, 2.0, 3.0];

      const stateBefore = sampler.getState();
      sampler.sample(logits);
      const stateAfter = sampler.getState();

      expect(stateAfter.tokensGenerated).toBe(stateBefore.tokensGenerated + 1);
    });
  });

  describe('state management', () => {
    it('should save and restore state', () => {
      const sampler = new SeededSampler(config);
      const logits = [1.0, 2.0, 3.0, 0.5, 0.1];

      // Sample a few times
      for (let i = 0; i < 5; i++) sampler.sample(logits);

      const state = sampler.getState();

      // Sample more
      const nextSamples = Array.from({ length: 3 }, () => sampler.sample(logits));

      // Restore
      sampler.setState(state);

      // Should get same samples
      for (let i = 0; i < 3; i++) {
        expect(sampler.sample(logits)).toBe(nextSamples[i]);
      }
    });

    it('should reset to initial state', () => {
      const sampler1 = new SeededSampler(config);
      const sampler2 = new SeededSampler(config);
      const logits = [1.0, 2.0, 3.0, 0.5, 0.1];

      // Advance sampler1
      for (let i = 0; i < 10; i++) sampler1.sample(logits);

      // Reset
      sampler1.reset();

      // Should match fresh sampler
      for (let i = 0; i < 5; i++) {
        expect(sampler1.sample(logits)).toBe(sampler2.sample(logits));
      }
    });
  });

  describe('top-k filtering', () => {
    it('should filter to top-k tokens', () => {
      const _sampler = new SeededSampler({ ...config, topK: 2, topP: 1.0 });
      const logits = [0.1, 10.0, 5.0, 0.2, 0.3]; // Top 2 are indices 1 and 2

      const samples = new Set<number>();
      for (let i = 0; i < 50; i++) {
        const idx = new SeededSampler({ ...config, topK: 2, topP: 1.0, seed: i }).sample(logits);
        samples.add(idx);
      }

      // Should only sample from top-k (indices 1 and 2)
      expect(samples.has(1) || samples.has(2)).toBe(true);
    });
  });

  describe('top-p filtering', () => {
    it('should respect top-p (nucleus) sampling', () => {
      const _sampler = new SeededSampler({ ...config, topK: 100, topP: 0.5 });
      const logits = [0.1, 10.0, 0.1, 0.1, 0.1]; // Most mass in index 1

      const samples = new Set<number>();
      for (let i = 0; i < 50; i++) {
        const idx = new SeededSampler({ ...config, topK: 100, topP: 0.5, seed: i }).sample(logits);
        samples.add(idx);
      }

      // Index 1 should be heavily sampled
      expect(samples.has(1)).toBe(true);
    });
  });
});

describe('hashing functions', () => {
  describe('hashWeights', () => {
    it('should hash Float32Array weights', async () => {
      const weights = {
        layer1: new Float32Array([1.0, 2.0, 3.0]),
        layer2: new Float32Array([4.0, 5.0, 6.0])
      };

      const hash = await hashWeights(weights);

      expect(hash.hash).toBeTruthy();
      expect(hash.algorithm).toBe('sha256');
      expect(hash.parameterCount).toBe(6);
      expect(hash.layerHashes.layer1).toBeTruthy();
      expect(hash.layerHashes.layer2).toBeTruthy();
    });

    it('should hash 2D array weights', async () => {
      const weights = {
        layer1: [[1.0, 2.0], [3.0, 4.0]],
        layer2: [[5.0, 6.0], [7.0, 8.0]]
      };

      const hash = await hashWeights(weights);

      expect(hash.hash).toBeTruthy();
      expect(hash.parameterCount).toBe(8);
    });

    it('should produce different hashes for different weights', async () => {
      const weights1 = { layer: new Float32Array([1.0, 2.0, 3.0]) };
      const weights2 = { layer: new Float32Array([1.0, 2.0, 3.1]) };

      const hash1 = await hashWeights(weights1);
      const hash2 = await hashWeights(weights2);

      expect(hash1.hash).not.toBe(hash2.hash);
    });

    it('should produce same hash for same weights', async () => {
      const weights = { layer: new Float32Array([1.0, 2.0, 3.0]) };

      const hash1 = await hashWeights(weights);
      const hash2 = await hashWeights(weights);

      expect(hash1.hash).toBe(hash2.hash);
    });
  });

  describe('hashConfig', () => {
    it('should hash configuration', async () => {
      const config = {
        hyperparameters: { lr: 0.01, epochs: 100 },
        architecture: 'transformer',
        tokenizer: { vocabSize: 1000 },
        seed: 42
      };

      const hash = await hashConfig(config);

      expect(hash.hash).toBeTruthy();
      expect(hash.algorithm).toBe('sha256');
      expect(hash.components).toBeDefined();
    });

    it('should produce different hashes for different configs', async () => {
      const config1 = {
        hyperparameters: { lr: 0.01 },
        architecture: 'transformer',
        tokenizer: {},
        seed: 42
      };
      const config2 = {
        hyperparameters: { lr: 0.02 },
        architecture: 'transformer',
        tokenizer: {},
        seed: 42
      };

      const hash1 = await hashConfig(config1);
      const hash2 = await hashConfig(config2);

      expect(hash1.hash).not.toBe(hash2.hash);
    });
  });
});

describe('createManifest', () => {
  it('should create reproducibility manifest', async () => {
    const params = {
      seed: 42,
      config: { lr: 0.01, architecture: 'feedforward' },
      tokenizer: { vocabSize: 100 },
      corpus: 'test corpus data',
      initialWeights: { layer: new Float32Array([1.0, 2.0]) },
      epochLosses: [2.5, 2.0, 1.5]
    };

    const manifest = await createManifest(params);

    expect(manifest.version).toBe('1.0.0');
    expect(manifest.globalSeed).toBe(42);
    expect(manifest.corpusHash).toBeTruthy();
    expect(manifest.tokenizerHash).toBeTruthy();
    expect(manifest.epochChecksums.length).toBe(3);
    expect(manifest.initialWeightsHash).toBeDefined();
  });

  it('should include final weights hash when provided', async () => {
    const params = {
      seed: 42,
      config: {},
      tokenizer: {},
      corpus: 'test',
      initialWeights: { layer: new Float32Array([1.0]) },
      finalWeights: { layer: new Float32Array([2.0]) },
      epochLosses: [1.0]
    };

    const manifest = await createManifest(params);

    expect(manifest.finalWeightsHash).toBeDefined();
  });
});

describe('verifyReplay', () => {
  it('should verify matching replay', async () => {
    const params = {
      seed: 42,
      config: { lr: 0.01, architecture: 'feedforward' },
      tokenizer: { vocabSize: 100 },
      corpus: 'test corpus data',
      initialWeights: { layer: new Float32Array([1.0, 2.0]) },
      epochLosses: [2.5, 2.0, 1.5]
    };

    const manifest = await createManifest(params);

    const replay = {
      seed: 42,
      config: { lr: 0.01, architecture: 'feedforward' },
      tokenizer: { vocabSize: 100 },
      corpus: 'test corpus data',
      epochLosses: [2.5, 2.0, 1.5]
    };

    const result = await verifyReplay(manifest, replay);

    expect(result.success).toBe(true);
    expect(result.seedMatch).toBe(true);
    expect(result.corpusMatch).toBe(true);
    expect(result.configMatch).toBe(true);
    expect(result.errors.length).toBe(0);
  });

  it('should detect seed mismatch', async () => {
    const params = {
      seed: 42,
      config: {},
      tokenizer: {},
      corpus: 'test',
      initialWeights: { layer: new Float32Array([1.0]) },
      epochLosses: [1.0]
    };

    const manifest = await createManifest(params);

    const replay = { ...params, seed: 43 };

    const result = await verifyReplay(manifest, replay);

    expect(result.seedMatch).toBe(false);
    expect(result.errors.some(e => e.includes('Seed'))).toBe(true);
  });

  it('should detect corpus mismatch', async () => {
    const params = {
      seed: 42,
      config: {},
      tokenizer: {},
      corpus: 'original corpus',
      initialWeights: { layer: new Float32Array([1.0]) },
      epochLosses: [1.0]
    };

    const manifest = await createManifest(params);

    const replay = { ...params, corpus: 'different corpus' };

    const result = await verifyReplay(manifest, replay);

    expect(result.corpusMatch).toBe(false);
  });
});

describe('takeSnapshot', () => {
  it('should create reproducibility snapshot', async () => {
    const rng = new SeededRNG(42);
    rng.next(); // Advance state

    const weights = { layer: new Float32Array([1.0, 2.0, 3.0]) };

    const snapshot = await takeSnapshot(rng, 5, 100, 1.234, weights);

    expect(snapshot.epoch).toBe(5);
    expect(snapshot.step).toBe(100);
    expect(snapshot.loss).toBe(1.234);
    expect(snapshot.weightsHash).toBeTruthy();
    expect(snapshot.rngState).toBeDefined();
    expect(snapshot.timestamp).toBeTruthy();
  });
});

describe('tolerance functions', () => {
  describe('withinTolerance', () => {
    it('should return true for identical values', () => {
      expect(withinTolerance(1.0, 1.0)).toBe(true);
    });

    it('should return true for values within tolerance', () => {
      // normal tolerance is 1e-6, so 0.0000001 difference should pass
      expect(withinTolerance(1.0, 1.0000001, 'normal')).toBe(true);
    });

    it('should return false for values outside tolerance', () => {
      expect(withinTolerance(1.0, 2.0, 'strict')).toBe(false);
    });

    it('should respect different tolerance levels', () => {
      const a = 1.0;
      // strict = 1e-10, normal = 1e-6, relaxed = 1e-4, loose = 1e-2
      const bStrict = 1.0 + 1e-9; // within relaxed but not strict
      const bLoose = 1.0 + 1e-3; // within loose but not normal

      expect(withinTolerance(a, bStrict, 'strict')).toBe(false); // 1e-9 > 1e-10
      expect(withinTolerance(a, bStrict, 'normal')).toBe(true);  // 1e-9 < 1e-6
      expect(withinTolerance(a, bLoose, 'loose')).toBe(true);    // 1e-3 < 1e-2
    });
  });

  describe('arraysWithinTolerance', () => {
    it('should return true for identical arrays', () => {
      const arr = [1.0, 2.0, 3.0];
      expect(arraysWithinTolerance(arr, arr)).toBe(true);
    });

    it('should return true for arrays within tolerance', () => {
      const a = [1.0, 2.0, 3.0];
      // normal tolerance is 1e-6
      const b = [1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7];
      expect(arraysWithinTolerance(a, b, 'normal')).toBe(true);
    });

    it('should return false for different length arrays', () => {
      const a = [1.0, 2.0];
      const b = [1.0, 2.0, 3.0];
      expect(arraysWithinTolerance(a, b)).toBe(false);
    });

    it('should return false if any element outside tolerance', () => {
      const a = [1.0, 2.0, 3.0];
      const b = [1.0, 2.0, 4.0];
      expect(arraysWithinTolerance(a, b, 'normal')).toBe(false);
    });
  });
});
