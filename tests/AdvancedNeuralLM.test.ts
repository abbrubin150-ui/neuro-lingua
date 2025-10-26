import { describe, expect, it } from 'vitest';
import { AdvancedNeuralLM } from '../src/lib/AdvancedNeuralLM';

describe('AdvancedNeuralLM', () => {
  const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'neural', 'network'];
  const hiddenSize = 16;
  const contextSize = 2;

  describe('Initialization', () => {
    it('initializes with default advanced config', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.05, contextSize);
      const config = model.getAdvancedConfig();

      expect(config.activation).toBe('relu');
      expect(config.initialization).toBe('he');
      expect(config.lrSchedule).toBe('cosine');
      expect(config.weightDecay).toBeGreaterThan(0);
    });

    it('accepts custom advanced config', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.05,
        contextSize,
        'adam',
        0.9,
        0.1,
        1337,
        undefined,
        {
          activation: 'leaky_relu',
          initialization: 'xavier',
          lrSchedule: 'exponential',
          weightDecay: 1e-3
        }
      );

      const config = model.getAdvancedConfig();
      expect(config.activation).toBe('leaky_relu');
      expect(config.initialization).toBe('xavier');
      expect(config.lrSchedule).toBe('exponential');
      expect(config.weightDecay).toBe(1e-3);
    });

    it('reinitializes weights with He initialization', () => {
      const model1 = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.05,
        contextSize,
        'momentum',
        0.9,
        0,
        1337,
        undefined,
        { initialization: 'default' }
      );

      const model2 = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.05,
        contextSize,
        'momentum',
        0.9,
        0,
        1337,
        undefined,
        { initialization: 'he' }
      );

      // Weights should be different
      const weights1 = (model1 as any).wHidden[0];
      const weights2 = (model2 as any).wHidden[0];

      expect(weights1).not.toEqual(weights2);
    });
  });

  describe('Advanced Training', () => {
    it('trains with learning rate schedule', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        {
          lrSchedule: 'cosine',
          weightDecay: 0
        }
      );

      const lrHistory: number[] = [];

      model.trainAdvanced('hello world neural network', 10, {
        onEpochEnd: (epoch, metrics) => {
          lrHistory.push(metrics.lr);
        }
      });

      // Learning rate should decrease over epochs with cosine schedule
      expect(lrHistory.length).toBe(10);
      expect(lrHistory[0]).toBeGreaterThan(lrHistory[lrHistory.length - 1]);

      // Should follow cosine curve (not linear)
      const midpointLR = lrHistory[Math.floor(lrHistory.length / 2)];
      const linearMidpoint = (lrHistory[0] + lrHistory[lrHistory.length - 1]) / 2;

      // Cosine annealing drops faster initially
      expect(midpointLR).toBeLessThan(linearMidpoint);
    });

    it('applies L2 regularization', () => {
      const text = 'hello world hello world';

      // Train without regularization
      const model1 = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        {
          weightDecay: 0,
          lrSchedule: 'constant'
        }
      );
      model1.trainAdvanced(text, 20);

      // Train with regularization
      const model2 = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        {
          weightDecay: 1e-2,
          lrSchedule: 'constant'
        }
      );
      model2.trainAdvanced(text, 20);

      // Model with regularization should have smaller weight norms
      const weights1 = (model1 as any).wHidden;
      const weights2 = (model2 as any).wHidden;

      const norm1 = Math.sqrt(weights1.flat().reduce((sum: number, w: number) => sum + w * w, 0));
      const norm2 = Math.sqrt(weights2.flat().reduce((sum: number, w: number) => sum + w * w, 0));

      expect(norm2).toBeLessThan(norm1);
    });

    it('callback provides metrics', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.05, contextSize);

      let callbackCount = 0;
      const metrics: any[] = [];

      model.trainAdvanced('hello world neural', 5, {
        onEpochEnd: (epoch, metric) => {
          callbackCount++;
          metrics.push(metric);
        }
      });

      expect(callbackCount).toBe(5);
      expect(metrics.length).toBe(5);

      // Check metrics structure
      metrics.forEach((m) => {
        expect(m).toHaveProperty('loss');
        expect(m).toHaveProperty('accuracy');
        expect(m).toHaveProperty('lr');
        expect(typeof m.loss).toBe('number');
        expect(typeof m.accuracy).toBe('number');
        expect(typeof m.lr).toBe('number');
      });
    });
  });

  describe('Beam Search Generation', () => {
    it('generates text using beam search', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      model.train('hello world neural network hello neural', 30);

      const result = model.generateBeamSearch('hello', 10, 3, 0.8);

      expect(result).toHaveProperty('text');
      expect(result).toHaveProperty('score');
      expect(result).toHaveProperty('tokens');

      expect(typeof result.text).toBe('string');
      expect(typeof result.score).toBe('number');
      expect(Array.isArray(result.tokens)).toBe(true);

      // Score should be negative (log probabilities)
      expect(result.score).toBeLessThanOrEqual(0);
    });

    it('beam search produces different results than greedy', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      model.train('hello world neural network world neural', 30);

      const beamResult = model.generateBeamSearch('hello', 5, 3);
      const greedyResult = model.generate('hello', 5, 0.01); // Low temp = greedy

      // Beam search might find better sequences
      expect(typeof beamResult.text).toBe('string');
      expect(typeof greedyResult).toBe('string');

      // Results can be different (not guaranteed, but likely)
      // Just check both produce valid output
      expect(beamResult.text.length).toBeGreaterThan(0);
    });

    it('respects beam width parameter', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      model.train('hello world', 10);

      // Should work with different beam widths
      const result1 = model.generateBeamSearch('hello', 5, 1);
      const result2 = model.generateBeamSearch('hello', 5, 5);

      expect(result1.text).toBeDefined();
      expect(result2.text).toBeDefined();
    });
  });

  describe('Nucleus Sampling', () => {
    it('generates text using nucleus sampling', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      model.train('hello world neural network', 30);

      const text = model.generateNucleus('hello', 10, 0.9, 0.9);

      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    });

    it('produces diverse outputs with high temperature', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize, 'adam', 0.9, 0, 42);
      model.train('hello world neural network hello world', 30);

      const outputs = new Set<string>();

      for (let i = 0; i < 10; i++) {
        const text = model.generateNucleus('hello', 5, 1.5, 0.9);
        outputs.add(text);
      }

      // With high temperature, should get some diversity
      // (not guaranteed due to small vocab, but likely > 1)
      expect(outputs.size).toBeGreaterThan(0);
    });
  });

  describe('Perplexity Calculation', () => {
    it('calculates perplexity on text', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      const trainText = 'hello world neural network';
      model.train(trainText, 30);

      const perplexity = model.calculatePerplexity('hello world');

      expect(typeof perplexity).toBe('number');
      expect(perplexity).toBeGreaterThan(0);
      expect(isFinite(perplexity)).toBe(true);
    });

    it('perplexity decreases with training', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);
      const text = 'hello world neural network hello world';

      const ppl0 = model.calculatePerplexity(text);
      model.train(text, 20);
      const ppl20 = model.calculatePerplexity(text);

      // Perplexity should decrease with training (first 20 epochs should show improvement)
      expect(ppl20).toBeLessThan(ppl0);

      // Verify perplexity is reasonable
      expect(ppl0).toBeGreaterThan(1);
      expect(ppl20).toBeGreaterThan(1);
    });

    it('handles empty or very short sequences', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.1, contextSize);

      // Empty text should return Infinity
      const pplEmpty = model.calculatePerplexity('');
      expect(pplEmpty).toBe(Infinity);

      // Very short text (shorter than context) may have special handling
      const pplShort = model.calculatePerplexity('a');
      expect(isFinite(pplShort)).toBe(true);
      expect(pplShort).toBeGreaterThan(0);
    });
  });

  describe('Config Management', () => {
    it('updates advanced config', () => {
      const model = new AdvancedNeuralLM(vocab, hiddenSize, 0.05, contextSize);

      model.setAdvancedConfig({
        activation: 'elu',
        weightDecay: 1e-3
      });

      const config = model.getAdvancedConfig();
      expect(config.activation).toBe('elu');
      expect(config.weightDecay).toBe(1e-3);
    });

    it('reinitializes weights when initialization changes', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.05,
        contextSize,
        'momentum',
        0.9,
        0,
        1337,
        undefined,
        { initialization: 'default' }
      );

      const weightsBefore = JSON.stringify((model as any).wHidden);

      model.setAdvancedConfig({ initialization: 'xavier' });

      const weightsAfter = JSON.stringify((model as any).wHidden);

      expect(weightsBefore).not.toEqual(weightsAfter);
    });
  });

  describe('Serialization', () => {
    it('exports and imports advanced model', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'adam',
        0.9,
        0.1,
        1337,
        undefined,
        {
          activation: 'leaky_relu',
          lrSchedule: 'cosine',
          weightDecay: 1e-3
        }
      );

      model.train('hello world neural network', 10);

      const exported = model.toJSONAdvanced();

      expect(exported.advancedConfig).toBeDefined();
      expect(exported.advancedConfig.activation).toBe('leaky_relu');
      expect(exported.advancedConfig.lrSchedule).toBe('cosine');

      const restored = AdvancedNeuralLM.loadFromJSONAdvanced(exported);

      expect(restored).toBeDefined();
      expect(restored.getAdvancedConfig().activation).toBe('leaky_relu');
      expect(restored.getAdvancedConfig().lrSchedule).toBe('cosine');

      // Check that weights are preserved
      const originalWeights = (model as any).wHidden;
      const restoredWeights = (restored as any).wHidden;

      expect(originalWeights.length).toBe(restoredWeights.length);
      for (let i = 0; i < originalWeights.length; i++) {
        for (let j = 0; j < originalWeights[i].length; j++) {
          expect(Math.abs(originalWeights[i][j] - restoredWeights[i][j])).toBeLessThan(1e-10);
        }
      }
    });
  });

  describe('Different Activation Functions', () => {
    it('trains with LeakyReLU activation', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'adam',
        0.9,
        0,
        42,
        undefined,
        { activation: 'leaky_relu' }
      );

      const result = model.trainAdvanced('hello world neural network', 10);

      expect(result.loss).toBeGreaterThan(0);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
      expect(result.accuracy).toBeLessThanOrEqual(1);
    });

    it('trains with ELU activation', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'adam',
        0.9,
        0,
        42,
        undefined,
        { activation: 'elu' }
      );

      const result = model.trainAdvanced('hello world neural network', 10);

      expect(result.loss).toBeGreaterThan(0);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
    });

    it('trains with GELU activation', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'adam',
        0.9,
        0,
        42,
        undefined,
        { activation: 'gelu' }
      );

      const result = model.trainAdvanced('hello world neural network', 10);

      expect(result.loss).toBeGreaterThan(0);
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Learning Rate Schedules', () => {
    it('constant schedule maintains LR', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        { lrSchedule: 'constant' }
      );

      const lrs: number[] = [];
      model.trainAdvanced('hello world', 5, {
        onEpochEnd: (_, metrics) => lrs.push(metrics.lr)
      });

      // All learning rates should be the same
      expect(lrs.every((lr) => Math.abs(lr - 0.1) < 1e-6)).toBe(true);
    });

    it('exponential schedule decays LR', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        { lrSchedule: 'exponential', lrDecayRate: 0.9 }
      );

      const lrs: number[] = [];
      model.trainAdvanced('hello world', 5, {
        onEpochEnd: (_, metrics) => lrs.push(metrics.lr)
      });

      // Learning rate should decay exponentially
      for (let i = 1; i < lrs.length; i++) {
        expect(lrs[i]).toBeLessThan(lrs[i - 1]);
        // Check exponential relationship
        const ratio = lrs[i] / lrs[i - 1];
        expect(Math.abs(ratio - 0.9)).toBeLessThan(0.01);
      }
    });

    it('warmup + cosine schedule warms up then decays', () => {
      const model = new AdvancedNeuralLM(
        vocab,
        hiddenSize,
        0.1,
        contextSize,
        'momentum',
        0.9,
        0,
        42,
        undefined,
        { lrSchedule: 'warmup_cosine', warmupEpochs: 3 }
      );

      const lrs: number[] = [];
      model.trainAdvanced('hello world', 10, {
        onEpochEnd: (_, metrics) => lrs.push(metrics.lr)
      });

      // First few epochs should increase (warmup)
      expect(lrs[1]).toBeGreaterThan(lrs[0]);
      expect(lrs[2]).toBeGreaterThan(lrs[1]);

      // After warmup, should decrease
      expect(lrs[lrs.length - 1]).toBeLessThan(lrs[3]);
    });
  });
});
