/**
 * Comprehensive Model Export/Import Tests
 *
 * These tests verify that models can be exported to JSON and re-imported
 * with full integrity of:
 * - Network weights (embedding, hidden, output)
 * - Optimizer states (momentum, adam)
 * - RNG state (reproducibility)
 * - Training history
 * - Prediction consistency
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ProNeuralLM, MODEL_STORAGE_KEY, TRANSFORMER_MODEL_STORAGE_KEY } from '../../src/lib/ProNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';

// Mock localStorage
const storage = new Map<string, string>();
const localStorageMock = {
  getItem(key: string) {
    return storage.has(key) ? storage.get(key)! : null;
  },
  setItem(key: string, value: string) {
    storage.set(key, value);
  },
  removeItem(key: string) {
    storage.delete(key);
  },
  clear() {
    storage.clear();
  }
};

Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Test corpus for training
const testCorpus = 'hello world hello neural network hello world neural models';

// Create vocabulary from corpus
function createVocab(corpus: string): string[] {
  const tokens = ProNeuralLM.tokenizeText(corpus);
  return ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ...Array.from(new Set(tokens))];
}

describe('ProNeuralLM Export/Import', () => {
  let vocab: string[];

  beforeEach(() => {
    storage.clear();
    vocab = createVocab(testCorpus);
  });

  describe('toJSON serialization', () => {
    it('should serialize all essential model fields', () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0.1, 42);
      const json = model.toJSON();

      // Verify metadata
      expect(json.version).toBeDefined();
      expect(json.vocab).toEqual(vocab);
      expect(json.hiddenSize).toBe(16);
      expect(json.learningRate).toBe(0.05);
      expect(json.contextSize).toBe(2);
      expect(json.optimizer).toBe('momentum');
      expect(json.momentum).toBe(0.9);
      expect(json.dropout).toBeCloseTo(0.1);
      expect(json.rngSeed).toBe(42);

      // Verify weights exist and have correct dimensions
      // embedding: V x H (vocab_length x hidden_size)
      expect(json.embedding).toHaveLength(vocab.length);
      expect(json.embedding[0]).toHaveLength(16); // embeddingDim = hiddenSize
      // wHidden: H x H (hidden_size x hidden_size)
      expect(json.wHidden).toHaveLength(16);
      expect(json.wHidden[0]).toHaveLength(16);
      // wOutput: H x V (hidden_size x vocab_length)
      expect(json.wOutput).toHaveLength(16);
      expect(json.wOutput[0]).toHaveLength(vocab.length);
      // biases
      expect(json.bHidden).toHaveLength(16);
      expect(json.bOutput).toHaveLength(vocab.length);

      // Verify vocabulary maps
      expect(json.wordToIdx).toBeInstanceOf(Array);
      expect(json.idxToWord).toBeInstanceOf(Array);
    });

    it('should preserve RNG state for reproducibility', async () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      // Train a bit to advance RNG state
      await model.train(testCorpus, 2);

      const json1 = model.toJSON();
      const json2 = model.toJSON();

      expect(json1.rngSeed).toBe(json2.rngSeed);
      expect(json1.rngState).toBe(json2.rngState);
    });

    it('should preserve training history', async () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await model.train(testCorpus, 3);

      const json = model.toJSON();

      expect(json.trainingHistory).toHaveLength(3);
      expect(json.trainingHistory[0]).toHaveProperty('loss');
      expect(json.trainingHistory[0]).toHaveProperty('accuracy');
      expect(json.trainingHistory[0]).toHaveProperty('timestamp');

      // Verify loss decreases across epochs
      expect(json.trainingHistory[2].loss).toBeLessThanOrEqual(json.trainingHistory[0].loss);
    });

    it('should preserve optimizer state for adam', async () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'adam', 0.9, 0, 42);

      await model.train(testCorpus, 3);

      const json = model.toJSON();

      expect(json.adamT).toBeGreaterThan(0);
      expect(json.aEmbedding.m).toBeDefined();
      expect(json.aEmbedding.v).toBeDefined();
      expect(json.aWHidden.m).toBeDefined();
      expect(json.aWHidden.v).toBeDefined();
    });

    it('should preserve momentum state', async () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await model.train(testCorpus, 3);

      const json = model.toJSON();

      expect(json.mEmbedding).toBeDefined();
      expect(json.mWHidden).toBeDefined();
      expect(json.mWOutput).toBeDefined();
    });
  });

  describe('saveToLocalStorage / loadFromLocalStorage round-trip', () => {
    it('should save and load model with identical weights', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 3);

      original.saveToLocalStorage(MODEL_STORAGE_KEY);
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();

      const origJson = original.toJSON();
      const loadedJson = loaded!.toJSON();

      // Verify all weights are identical
      expect(loadedJson.embedding).toEqual(origJson.embedding);
      expect(loadedJson.wHidden).toEqual(origJson.wHidden);
      expect(loadedJson.wOutput).toEqual(origJson.wOutput);
      expect(loadedJson.bHidden).toEqual(origJson.bHidden);
      expect(loadedJson.bOutput).toEqual(origJson.bOutput);
    });

    it('should produce identical forward pass after load', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 10);

      original.saveToLocalStorage(MODEL_STORAGE_KEY);
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();

      // Test that logits are identical after load
      const testContext = [1, 2, 3]; // BOS, word1, word2
      const originalLogits = await original.getLogitsForContext(testContext);
      const loadedLogits = await loaded!.getLogitsForContext(testContext);

      // Verify logits are identical
      expect(loadedLogits.length).toBe(originalLogits.length);
      for (let i = 0; i < originalLogits.length; i++) {
        expect(loadedLogits[i]).toBeCloseTo(originalLogits[i], 10);
      }
    });

    it('should preserve training history after load', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 4);

      const originalHistory = original.toJSON().trainingHistory;

      original.saveToLocalStorage(MODEL_STORAGE_KEY);
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);

      expect(loaded!.toJSON().trainingHistory).toEqual(originalHistory);
    });

    it('should preserve optimizer state through round-trip (adam)', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'adam', 0.9, 0, 42);

      await original.train(testCorpus, 3);

      const originalJson = original.toJSON();

      original.saveToLocalStorage(MODEL_STORAGE_KEY);
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);

      const loadedJson = loaded!.toJSON();

      expect(loadedJson.adamT).toBe(originalJson.adamT);
      expect(loadedJson.aEmbedding).toEqual(originalJson.aEmbedding);
      expect(loadedJson.aWHidden).toEqual(originalJson.aWHidden);
    });

    it('should allow continued training after load', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 2);

      original.saveToLocalStorage(MODEL_STORAGE_KEY);
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();

      const lossBefore = loaded!.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

      // Continue training
      await loaded!.train(testCorpus, 3);

      const lossAfter = loaded!.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

      // Loss should continue to decrease
      expect(lossAfter).toBeLessThanOrEqual(lossBefore);
      expect(loaded!.toJSON().trainingHistory).toHaveLength(5);
    });

    it('should handle missing localStorage key', () => {
      const loaded = ProNeuralLM.loadFromLocalStorage('non-existent-key');
      expect(loaded).toBeNull();
    });

    it('should handle corrupted JSON gracefully', () => {
      storage.set(MODEL_STORAGE_KEY, 'not valid json {{{');
      const loaded = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);
      expect(loaded).toBeNull();
    });
  });

  describe('JSON round-trip (without localStorage)', () => {
    it('should preserve model through JSON.stringify/parse cycle', async () => {
      const original = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 3);

      const json = original.toJSON();
      const serialized = JSON.stringify(json);
      const parsed = JSON.parse(serialized);

      // Verify all fields survive JSON round-trip
      expect(parsed.version).toBe(json.version);
      expect(parsed.vocab).toEqual(json.vocab);
      expect(parsed.hiddenSize).toBe(json.hiddenSize);
      expect(parsed.embedding).toEqual(json.embedding);
      expect(parsed.wHidden).toEqual(json.wHidden);
      expect(parsed.trainingHistory).toEqual(json.trainingHistory);
    });

    it('should handle special numeric values', () => {
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);
      const json = model.toJSON();

      // Verify no NaN or Infinity in weights
      for (const row of json.embedding) {
        for (const val of row) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }

      for (const row of json.wHidden) {
        for (const val of row) {
          expect(Number.isFinite(val)).toBe(true);
        }
      }
    });
  });
});

describe('TransformerLM Export/Import', () => {
  let vocab: string[];

  beforeEach(() => {
    storage.clear();
    vocab = createVocab(testCorpus);
  });

  describe('toJSON serialization', () => {
    it('should include transformer-specific fields', () => {
      const model = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);
      const json = model.toJSON();

      expect(json.architecture).toBe('transformer');
      expect(json.transformer).toBeDefined();
      expect(json.transformer.config).toBeDefined();
      expect(json.transformer.attentionWeights).toBeDefined();
      expect(json.transformer.ffWeights1).toBeDefined();
      expect(json.transformer.ffWeights2).toBeDefined();
    });

    it('should serialize attention weights correctly', () => {
      const model = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, {
        numLayers: 2,
        numHeads: 2
      });

      const json = model.toJSON();

      // Verify attention weights for each layer
      expect(json.transformer.attentionWeights).toHaveLength(2);

      for (const layer of json.transformer.attentionWeights) {
        // Transformer uses query/key/value naming
        expect(layer.query).toBeDefined();
        expect(layer.key).toBeDefined();
        expect(layer.value).toBeDefined();
      }
    });

    it('should preserve transformer config', () => {
      const config = {
        numLayers: 2,
        numHeads: 4,
        ffHiddenDim: 32,
        attentionDropout: 0.1
      };

      const model = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, config);
      const json = model.toJSON();

      expect(json.transformer.config.numLayers).toBe(2);
      expect(json.transformer.config.numHeads).toBe(4);
      expect(json.transformer.config.ffHiddenDim).toBe(32);
    });
  });

  describe('saveToLocalStorage / loadFromLocalStorage round-trip', () => {
    it('should save and load transformer model correctly', async () => {
      const original = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, {
        numLayers: 1,
        numHeads: 2
      });

      await original.train(testCorpus, 2);

      original.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();
      expect(loaded!.toJSON().architecture).toBe('transformer');
    });

    it('should preserve attention weights after load', async () => {
      const original = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, {
        numLayers: 1,
        numHeads: 2
      });

      await original.train(testCorpus, 2);

      const originalJson = original.toJSON();

      original.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      const loadedJson = loaded!.toJSON();

      expect(loadedJson.transformer.attentionWeights).toEqual(originalJson.transformer.attentionWeights);
      expect(loadedJson.transformer.ffWeights1).toEqual(originalJson.transformer.ffWeights1);
      expect(loadedJson.transformer.ffWeights2).toEqual(originalJson.transformer.ffWeights2);
    });

    it('should produce identical forward pass after load', async () => {
      const original = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, {
        numLayers: 1,
        numHeads: 2
      });

      await original.train(testCorpus, 10);

      original.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();

      // Test that logits are identical after load
      const testContext = [1, 2, 3]; // BOS, word1, word2
      const originalLogits = await original.getLogitsForContext(testContext);
      const loadedLogits = await loaded!.getLogitsForContext(testContext);

      // Verify logits are identical
      expect(loadedLogits.length).toBe(originalLogits.length);
      for (let i = 0; i < originalLogits.length; i++) {
        expect(loadedLogits[i]).toBeCloseTo(originalLogits[i], 10);
      }
    });

    it('should allow continued training after load', async () => {
      const original = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42, undefined, {
        numLayers: 1,
        numHeads: 2
      });

      await original.train(testCorpus, 2);

      original.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      expect(loaded).not.toBeNull();

      const lossBefore = loaded!.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

      await loaded!.train(testCorpus, 2);

      const lossAfter = loaded!.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

      expect(lossAfter).toBeLessThanOrEqual(lossBefore);
    });

    it('should reject non-transformer architecture on load', () => {
      // Save a ProNeuralLM model
      const feedforward = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);
      feedforward.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      // Try to load as TransformerLM
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      expect(loaded).toBeNull();
    });
  });

  describe('RNG state preservation', () => {
    it('should preserve RNG state for deterministic generation', async () => {
      const original = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

      await original.train(testCorpus, 3);

      const originalJson = original.toJSON();

      original.saveToLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
      const loaded = TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);

      const loadedJson = loaded!.toJSON();

      expect(loadedJson.rngSeed).toBe(originalJson.rngSeed);
      expect(loadedJson.rngState).toBe(originalJson.rngState);
    });
  });
});

describe('Cross-architecture compatibility', () => {
  let vocab: string[];

  beforeEach(() => {
    storage.clear();
    vocab = createVocab(testCorpus);
  });

  it('should distinguish between architectures on load', async () => {
    const feedforward = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);
    const transformer = new TransformerLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

    await feedforward.train(testCorpus, 2);
    await transformer.train(testCorpus, 2);

    feedforward.saveToLocalStorage('model-ff');
    transformer.saveToLocalStorage('model-tf');

    // Load correctly
    const loadedFF = ProNeuralLM.loadFromLocalStorage('model-ff');
    const loadedTF = TransformerLM.loadFromLocalStorage('model-tf');

    expect(loadedFF).not.toBeNull();
    expect(loadedTF).not.toBeNull();

    // Cross-load should fail or return correct type
    const wrongTF = TransformerLM.loadFromLocalStorage('model-ff');
    expect(wrongTF).toBeNull();
  });
});

describe('Export data size and efficiency', () => {
  let vocab: string[];

  beforeEach(() => {
    storage.clear();
    vocab = createVocab(testCorpus);
  });

  it('should produce reasonable JSON size for small models', async () => {
    const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

    await model.train(testCorpus, 2);

    const json = JSON.stringify(model.toJSON());
    const sizeKB = json.length / 1024;

    // Small model should be under 100KB
    expect(sizeKB).toBeLessThan(100);
  });

  it('should maintain consistent size across multiple serializations', async () => {
    const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'momentum', 0.9, 0, 42);

    await model.train(testCorpus, 2);

    const json1 = JSON.stringify(model.toJSON());
    const json2 = JSON.stringify(model.toJSON());

    expect(json1.length).toBe(json2.length);
  });
});
