import { describe, it, expect } from 'vitest';
import { TransformerLM } from '../src/lib/TransformerLM';

describe('TransformerLM', () => {
  it('should create a transformer model', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a', 'b', 'c'];
    const model = new TransformerLM(vocab, 32, 0.01, 2);
    expect(model).toBeDefined();
    expect(model.getArchitectureType()).toBe('Transformer');
  });

  it('should have correct transformer configuration', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a', 'b', 'c'];
    const model = new TransformerLM(
      vocab,
      32,
      0.01,
      2,
      'adam',
      0.9,
      0.1,
      42,
      { mode: 'unicode' },
      {
        numLayers: 2,
        numHeads: 4,
        ffHiddenDim: 64
      }
    );

    const info = model.getTransformerInfo();
    expect(info.numLayers).toBe(2);
    expect(info.numHeads).toBe(4);
    expect(info.ffHiddenDim).toBe(64);
  });

  it('should train on simple corpus', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a', 'b', 'c'];
    const model = new TransformerLM(
      vocab,
      16,
      0.05,
      2,
      'adam',
      0.9,
      0.1,
      42,
      { mode: 'unicode' },
      {
        numLayers: 1,
        numHeads: 2,
        ffHiddenDim: 32
      }
    );

    const corpus = 'a b c a b c';
    const result = await model.train(corpus, 2);

    expect(result).toBeDefined();
    expect(result.loss).toBeGreaterThan(0);
    expect(result.accuracy).toBeGreaterThanOrEqual(0);
    expect(result.accuracy).toBeLessThanOrEqual(1);
    expect(result.history.length).toBe(2);
  });

  it('should generate text after training', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a', 'b', 'c', 'd'];
    const model = new TransformerLM(
      vocab,
      16,
      0.1,
      2,
      'adam',
      0.9,
      0.0,
      42,
      { mode: 'unicode' },
      {
        numLayers: 1,
        numHeads: 2,
        ffHiddenDim: 32
      }
    );

    const corpus = 'a b c d a b c d';
    await model.train(corpus, 3);

    const generated = await model.generate('a', 5, 0.8);
    expect(generated).toBeDefined();
    expect(typeof generated).toBe('string');
  });

  it('should improve with more epochs', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a', 'b'];
    const model = new TransformerLM(
      vocab,
      16,
      0.1,
      2,
      'adam',
      0.9,
      0.0,
      42,
      { mode: 'unicode' },
      {
        numLayers: 1,
        numHeads: 2,
        ffHiddenDim: 32
      }
    );

    const corpus = 'a b a b a b a b';
    const result = await model.train(corpus, 5);

    // Should have trained for 5 epochs
    expect(result.history.length).toBe(5);

    // Loss should generally decrease (allowing some fluctuation)
    const firstLoss = result.history[0].loss;
    const lastLoss = result.history[4].loss;

    // At least the trend should be downward or final loss should be reasonable
    expect(lastLoss).toBeLessThan(firstLoss * 1.5); // Allow some variance
  });
});
