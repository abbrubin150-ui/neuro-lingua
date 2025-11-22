import { describe, it, expect, beforeEach } from 'vitest';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../src/lib/TransformerLM';

describe('Embedding Extraction', () => {
  describe('ProNeuralLM', () => {
    let model: ProNeuralLM;
    const vocab = ['hello', 'world', 'test', 'neural', 'network'];
    const hiddenSize = 16;

    beforeEach(() => {
      model = new ProNeuralLM(vocab, hiddenSize);
    });

    it('should return embeddings with correct shape', () => {
      const embeddings = model.getEmbeddings();
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(vocab.length);
      expect(embeddings[0].length).toBe(hiddenSize);
    });

    it('should return a copy of embeddings (not reference)', () => {
      const embeddings1 = model.getEmbeddings();
      const embeddings2 = model.getEmbeddings();

      expect(embeddings1).not.toBe(embeddings2);
      expect(embeddings1[0]).not.toBe(embeddings2[0]);
    });

    it('should return vocabulary array', () => {
      const vocabCopy = model.getVocab();
      expect(vocabCopy).toBeDefined();
      expect(vocabCopy.length).toBe(vocab.length);
      expect(vocabCopy).toEqual(vocab);
    });

    it('should return a copy of vocabulary (not reference)', () => {
      const vocab1 = model.getVocab();
      const vocab2 = model.getVocab();

      expect(vocab1).not.toBe(vocab2);
    });

    it('should return initial embeddings from untrained model', () => {
      const embeddings = model.getEmbeddings();
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(vocab.length);
      expect(embeddings[0].length).toBe(hiddenSize);

      // Check that embeddings are initialized (not all zeros)
      const hasNonZero = embeddings.some((row) => row.some((val) => Math.abs(val) > 1e-6));
      expect(hasNonZero).toBe(true);
    });
  });

  describe('AdvancedNeuralLM', () => {
    let model: AdvancedNeuralLM;
    const vocab = ['hello', 'world', 'test', 'neural', 'network'];
    const hiddenSize = 16;

    beforeEach(() => {
      model = new AdvancedNeuralLM(vocab, hiddenSize);
    });

    it('should inherit getEmbeddings from ProNeuralLM', () => {
      const embeddings = model.getEmbeddings();
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(vocab.length);
      expect(embeddings[0].length).toBe(hiddenSize);
    });

    it('should inherit getVocab from ProNeuralLM', () => {
      const vocabCopy = model.getVocab();
      expect(vocabCopy).toBeDefined();
      expect(vocabCopy.length).toBe(vocab.length);
      expect(vocabCopy).toEqual(vocab);
    });
  });

  describe('TransformerLM', () => {
    let model: TransformerLM;
    const vocab = ['hello', 'world', 'test', 'neural', 'network'];
    const hiddenSize = 16;

    beforeEach(() => {
      // TransformerLM constructor: (vocab, hiddenSize, lr, contextSize, optimizer, momentum, dropout, seed, tokenizerConfig, transformerConfig)
      model = new TransformerLM(
        vocab,
        hiddenSize, // embedding dimension
        0.01, // learning rate
        3, // context size
        'adam', // optimizer
        0.9, // momentum
        0.0, // dropout
        1337, // seed
        { mode: 'unicode' }, // tokenizer config
        {
          // transformer config
          numHeads: 2,
          numLayers: 1,
          ffHiddenDim: 32,
          attentionDropout: 0.0,
          dropConnectRate: 0.0
        }
      );
    });

    it('should inherit getEmbeddings from ProNeuralLM', () => {
      const embeddings = model.getEmbeddings();
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(vocab.length);
      expect(embeddings[0].length).toBe(hiddenSize);
    });

    it('should inherit getVocab from ProNeuralLM', () => {
      const vocabCopy = model.getVocab();
      expect(vocabCopy).toBeDefined();
      expect(vocabCopy.length).toBe(vocab.length);
      expect(vocabCopy).toEqual(vocab);
    });
  });

  describe('Edge Cases', () => {
    it('should handle minimal vocabulary', () => {
      const vocab = ['a', 'b', 'c'];
      const model = new ProNeuralLM(vocab, 8);

      const embeddings = model.getEmbeddings();
      expect(embeddings.length).toBe(3);
      expect(embeddings[0].length).toBe(8);

      const vocabCopy = model.getVocab();
      expect(vocabCopy).toEqual(vocab);
    });

    it('should handle large hidden size', () => {
      const vocab = ['hello', 'world'];
      const hiddenSize = 128;
      const model = new ProNeuralLM(vocab, hiddenSize);

      const embeddings = model.getEmbeddings();
      expect(embeddings.length).toBe(2);
      expect(embeddings[0].length).toBe(128);
    });

    it('should preserve special tokens in vocabulary', () => {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
      const model = new ProNeuralLM(vocab, 16);

      const vocabCopy = model.getVocab();
      expect(vocabCopy).toEqual(vocab);
      expect(vocabCopy[0]).toBe('<PAD>');
      expect(vocabCopy[1]).toBe('<BOS>');
      expect(vocabCopy[2]).toBe('<EOS>');
      expect(vocabCopy[3]).toBe('<UNK>');
    });
  });
});
