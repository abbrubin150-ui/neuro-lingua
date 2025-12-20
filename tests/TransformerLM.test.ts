import { describe, it, expect } from 'vitest';
import { TransformerLM } from '../src/lib/TransformerLM';
import { MiniTransformerBlock } from '../src/models/mini_transformer';
import type { AttentionWeights } from '../src/models/attention';

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
    expect(generated.length).toBeGreaterThan(0);
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

  it('applies RMSNorm pre-norm residuals without distorting identity when weights are zero', async () => {
    const block = new MiniTransformerBlock({
      modelDim: 4,
      heads: 2,
      ff: { hiddenDim: 4 },
      attentionDropout: 0,
      dropConnectRate: 0,
      attentionRms: { gamma: new Array(4).fill(1), epsilon: 1e-6 },
      ffnRms: { gamma: new Array(4).fill(1), epsilon: 1e-6 },
      numKVHeads: 2
    });

    const inputs = [
      [1, 1, 1, 1],
      [2, 2, 2, 2]
    ];
    const zeroAttention: AttentionWeights = {
      query: Array.from({ length: 4 }, () => new Array(4).fill(0)),
      key: Array.from({ length: 4 }, () => new Array(4).fill(0)),
      value: Array.from({ length: 4 }, () => new Array(4).fill(0))
    };
    const zeroFf1 = Array.from({ length: 4 }, () => new Array(8).fill(0));
    const zeroFf2 = Array.from({ length: 4 }, () => new Array(4).fill(0));

    const output = await block.forward(inputs, zeroAttention, zeroFf1, zeroFf2, { positions: [0, 1] });

    expect(output).toHaveLength(inputs.length);
    output.forEach((row, rowIdx) => {
      row.forEach((value, colIdx) => {
        expect(value).toBeCloseTo(inputs[rowIdx][colIdx]);
      });
    });
  });

  it('keeps RMS-normalized forward passes numerically stable across iterations', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'x', 'y'];
    const model = new TransformerLM(vocab, 16, 0.05, 2, 'adam', 0.9, 0.05, 13, { mode: 'unicode' }, {
      numLayers: 1,
      numHeads: 2,
      ffHiddenDim: 32
    });

    const samples = [
      [0, 4],
      [0, 5]
    ];

    for (const sample of samples) {
      const cache = await (model as any).transformerForwardPass(sample, false);
      expect(cache.h.every(Number.isFinite)).toBe(true);
      expect(cache.transformerHidden.flat().every(Number.isFinite)).toBe(true);
    }

    const corpus = 'x y x y x y x y';
    const result = await model.train(corpus, 3);
    expect(result.history[2].loss).toBeLessThan(result.history[0].loss * 1.1);
  });

  it('serializes transformer-specific state', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', '<PAD>', 'a'];
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

    const json = model.toJSON() as unknown as {
      architecture?: string;
      transformer?: { config: { numHeads: number } };
    };

    expect(json.architecture).toBe('transformer');
    expect(json.transformer).toBeDefined();
    expect(json.transformer?.config.numHeads).toBe(2);
  });
});
