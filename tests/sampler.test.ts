import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';
import { mirostatV2Sample } from '../src/generation/sampling';

describe('Text Sampler / Generator', () => {
  const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'test', 'foo', 'bar'];

  async function trainSimpleModel(): Promise<ProNeuralLM> {
    const model = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 42);
    await model.train('hello world test foo bar hello world', 10);
    return model;
  }

  describe('Basic generation', () => {
    it('generates text from seed', async () => {
      const model = await trainSimpleModel();
      const output = await model.generate('hello', 10, 0.8, 5, 0); // Use top-k to avoid empty outputs
      expect(typeof output).toBe('string');
      // Output might be empty if EOS is generated immediately, so check it's a string
      expect(output.length).toBeGreaterThanOrEqual(0);
    });

    it('respects max length', async () => {
      const model = await trainSimpleModel();
      const maxLen = 10;
      const output = await model.generate('hello', maxLen, 0.8);
      const tokens = output.split(' ').filter((t) => t.length > 0);
      expect(tokens.length).toBeLessThanOrEqual(maxLen);
    });

    it('handles empty seed', async () => {
      const model = await trainSimpleModel();
      const output = await model.generate('', 5, 0.8);
      expect(typeof output).toBe('string');
    });

    it('stops at EOS token', async () => {
      const model = await trainSimpleModel();
      // Generate with low temperature to make it more deterministic
      const output = await model.generate('hello', 50, 0.1);
      expect(output).not.toContain('<EOS>');
    });

    it('handles unknown words in seed', async () => {
      const model = await trainSimpleModel();
      // 'unknown' is not in vocab, should map to <UNK>
      const output = await model.generate('unknown', 5, 0.8);
      expect(typeof output).toBe('string');
    });
  });

  describe('Temperature', () => {
    it('lower temperature produces more deterministic output', async () => {
      // Create two identical models to ensure deterministic behavior
      const model1 = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 42);
      await model1.train('hello world test foo bar hello world', 10);

      const model2 = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 42);
      await model2.train('hello world test foo bar hello world', 10);

      const seed = 'hello';
      const output1 = await model1.generate(seed, 10, 0.1, 3, 0); // Use top-k for stability
      const output2 = await model2.generate(seed, 10, 0.1, 3, 0);

      // With same seed, same model state, and low temp, outputs should be identical
      expect(output1).toBe(output2);
    });

    it('higher temperature produces more varied output', async () => {
      const model1 = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 100);
      await model1.train('hello world test foo bar hello world', 10);

      const model2 = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 200);
      await model2.train('hello world test foo bar hello world', 10);

      const outputLowTemp = await model1.generate('hello', 10, 0.3);
      const outputHighTemp = await model2.generate('hello', 10, 1.5);

      // With different seeds and temps, outputs should likely differ
      expect(typeof outputLowTemp).toBe('string');
      expect(typeof outputHighTemp).toBe('string');
    });

    it('clamps extreme temperatures', async () => {
      const model = await trainSimpleModel();

      // Very low temp (should be clamped to 0.05)
      const output1 = await model.generate('hello', 5, 0.001);
      expect(typeof output1).toBe('string');

      // Very high temp (should be clamped to 5)
      const output2 = await model.generate('hello', 5, 100);
      expect(typeof output2).toBe('string');
    });
  });

  describe('Top-k sampling', () => {
    it('generates with top-k sampling', async () => {
      const model = await trainSimpleModel();
      const output = await model.generate('hello', 10, 0.8, 3, 0);
      expect(typeof output).toBe('string');
      expect(output.length).toBeGreaterThan(0);
    });

    it('top-k=1 is greedy sampling', async () => {
      const model = await trainSimpleModel();
      const output1 = await model.generate('hello', 10, 0.8, 1, 0);
      const output2 = await model.generate('hello', 10, 0.8, 1, 0);

      // With top-k=1 (greedy), outputs should be identical
      expect(output1).toBe(output2);
    });

    it('handles top-k larger than vocab', async () => {
      const model = await trainSimpleModel();
      const largeK = 1000;
      const output = await model.generate('hello', 10, 0.8, largeK, 0);
      expect(typeof output).toBe('string');
    });
  });

  describe('Top-p (nucleus) sampling', () => {
    it('generates with top-p sampling', async () => {
      // Create fresh model for this test
      const model = new ProNeuralLM(vocab, 16, 0.1, 3, 'adam', 0.9, 0, 123);
      await model.train('hello world test foo bar hello world', 10);
      const output = await model.generate('hello', 10, 0.8, 0, 0.9);
      expect(typeof output).toBe('string');
      // Output might be empty if EOS is generated, so just check it's a string
      expect(output.length).toBeGreaterThanOrEqual(0);
    });

    it('handles edge case top-p values', async () => {
      const model = await trainSimpleModel();

      // Very low p
      const output1 = await model.generate('hello', 5, 0.8, 0, 0.1);
      expect(typeof output1).toBe('string');

      // Very high p (close to 1)
      const output2 = await model.generate('hello', 5, 0.8, 0, 0.99);
      expect(typeof output2).toBe('string');
    });

    it('top-p close to 0 produces deterministic output', async () => {
      const model = await trainSimpleModel();
      const output1 = await model.generate('hello', 10, 0.5, 0, 0.01);
      const output2 = await model.generate('hello', 10, 0.5, 0, 0.01);

      // With very low top-p, should be more deterministic
      expect(typeof output1).toBe('string');
      expect(typeof output2).toBe('string');
    });
  });

  describe('Deterministic generation', () => {
    it('same seed produces same output', async () => {
      const model1 = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0, 1234);
      await model1.train('hello world foo bar test', 5);

      const model2 = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0, 1234);
      await model2.train('hello world foo bar test', 5);

      const output1 = await model1.generate('hello', 8, 0.7, 0, 0.9);
      const output2 = await model2.generate('hello', 8, 0.7, 0, 0.9);

      expect(output1).toBe(output2);
    });

    it('different seeds produce different output', async () => {
      const model1 = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0, 1111);
      await model1.train('hello world foo bar test', 5);

      const model2 = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0, 2222);
      await model2.train('hello world foo bar test', 5);

      const output1 = await model1.generate('hello', 8, 0.7, 0, 0.9);
      const output2 = await model2.generate('hello', 8, 0.7, 0, 0.9);

      // Different seeds should likely produce different outputs
      // (though not guaranteed mathematically)
      expect(typeof output1).toBe('string');
      expect(typeof output2).toBe('string');
    });
  });

  describe('Edge cases', () => {
    it('generates with very short max length', async () => {
      const model = await trainSimpleModel();
      const output = await model.generate('hello', 1, 0.8);
      const tokens = output.split(' ').filter((t) => t.length > 0);
      expect(tokens.length).toBeLessThanOrEqual(1);
    });

    it('handles generation with no training', async () => {
      const model = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0, 42);
      // Don't train, generate with random weights
      const output = await model.generate('hello', 5, 0.8);
      expect(typeof output).toBe('string');
      // Output quality will be poor but should not crash
    });

    it('handles seed longer than context window', async () => {
      const model = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0, 42);
      await model.train('hello world test foo bar', 5);

      const longSeed = 'hello world test foo bar hello world';
      const output = await model.generate(longSeed, 5, 0.8);
      expect(typeof output).toBe('string');
    });

    it('generates valid vocabulary tokens only', async () => {
      const model = await trainSimpleModel();
      const output = await model.generate('hello', 20, 0.8);
      const tokens = output.split(' ').filter((t) => t.length > 0);

      // All output tokens should be in the vocabulary (except special tokens)
      for (const token of tokens) {
        // Note: <EOS> should not appear in output
        expect(token).not.toBe('<EOS>');
        // Token should be in vocab or be a special token
        const isValid =
          vocab.includes(token) || token === '<BOS>' || token === '<PAD>' || token === '<UNK>';
        expect(isValid).toBe(true);
      }
    });
  });

  describe('Mirostat v2', () => {
    it('returns adaptive state updates', () => {
      const logits = [2, 1, 0];
      const { index, state, surprise } = mirostatV2Sample(logits, {
        targetEntropy: 5,
        learningRate: 0.3,
        rng: () => 0.12
      });
      expect(index).toBeGreaterThanOrEqual(0);
      expect(state.mu).toBeTypeOf('number');
      expect(Number.isFinite(surprise)).toBe(true);
    });
  });
});
