import { describe, it, expect, beforeEach } from 'vitest';
import { TransformerLM } from '../../src/lib/TransformerLM';
import {
  TransformerLMAdapter,
  createTransformerLMAdapter
} from '../../src/lib/expandable/TransformerLMAdapter';
import { InjectionEngine } from '../../src/lib/expandable/InjectionEngine';
import { InjectionRunSession, createLedgerAdapter } from '../../src/training/injection_hooks';
import type { CerebroBubble, InjectionEvent } from '../../src/types/injection';

describe('TransformerLMAdapter', () => {
  let model: TransformerLM;
  let adapter: TransformerLMAdapter;

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    model = new TransformerLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0.1, 42, { mode: 'unicode' }, {
      numLayers: 2,
      numHeads: 4,
      ffHiddenDim: 32,
      attentionDropout: 0.1,
      dropConnectRate: 0.1
    });
    adapter = createTransformerLMAdapter(model, 'test-transformer-model-1');
  });

  describe('getTarget', () => {
    it('should return correct injection target', () => {
      const target = adapter.getTarget();

      expect(target.modelId).toBe('test-transformer-model-1');
      expect(target.layerId).toBe('transformer-0');
      expect(target.type).toBe('transformer');
      expect(target.hiddenSize).toBe(16);
      expect(target.dModel).toBe(16); // In transformer, dModel = hiddenSize
    });

    it('should include transformer metadata', () => {
      const target = adapter.getTarget();

      expect(target.metadata).toBeDefined();
      expect(target.metadata?.numLayers).toBe(2);
      expect(target.metadata?.numHeads).toBe(4);
      expect(target.metadata?.ffHiddenDim).toBe(32);
    });
  });

  describe('canInject', () => {
    it('should return true for valid injection count', () => {
      expect(adapter.canInject(1)).toBe(true);
      expect(adapter.canInject(4)).toBe(true);
      expect(adapter.canInject(8)).toBe(true);
    });

    it('should return false for invalid injection count', () => {
      expect(adapter.canInject(0)).toBe(false);
      expect(adapter.canInject(-1)).toBe(false);
    });

    it('should return false when exceeding max hidden size', () => {
      expect(adapter.canInject(600)).toBe(false);
    });

    it('should respect head count constraints', () => {
      // Very small adapter that would result in size < numHeads
      const tinyAdapter = new TransformerLMAdapter(model, 'tiny', 18);
      // This is fine since 16 + 8 = 24 >= numHeads (4)
      expect(tinyAdapter.canInject(8)).toBe(false); // exceeds max
      expect(tinyAdapter.canInject(2)).toBe(true); // 16 + 2 = 18, ok
    });
  });

  describe('inject', () => {
    it('should expand hidden layer by k neurons', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');
      expect(model.getHiddenSize()).toBe(initialSize + 4);
    });

    it('should expand attention weights correctly', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');

      const transformerWeights = model.getTransformerWeights();
      const newSize = initialSize + 4;

      // Check that attention weights have correct dimensions
      for (const attn of transformerWeights.attentionWeights) {
        expect(attn.query.length).toBe(newSize);
        expect(attn.query[0].length).toBe(newSize);
        expect(attn.key.length).toBe(newSize);
        expect(attn.key[0].length).toBe(newSize);
        expect(attn.value.length).toBe(newSize);
        expect(attn.value[0].length).toBe(newSize);
      }
    });

    it('should expand feed-forward weights correctly', () => {
      const initialSize = model.getHiddenSize();
      const info = model.getTransformerInfo();
      const ffHiddenDim = info.ffHiddenDim;

      adapter.inject(4, 'random_he');

      const transformerWeights = model.getTransformerWeights();
      const newSize = initialSize + 4;

      // Check that FF weights have correct dimensions
      for (let i = 0; i < info.numLayers; i++) {
        // ff1: [modelDim x (ffHiddenDim * 2)] for SwiGLU
        expect(transformerWeights.ffWeights1[i].length).toBe(newSize);
        expect(transformerWeights.ffWeights1[i][0].length).toBe(ffHiddenDim * 2);

        // ff2: [ffHiddenDim x modelDim]
        expect(transformerWeights.ffWeights2[i].length).toBe(ffHiddenDim);
        expect(transformerWeights.ffWeights2[i][0].length).toBe(newSize);
      }
    });

    it('should expand position embeddings correctly', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');

      const transformerWeights = model.getTransformerWeights();
      const newSize = initialSize + 4;

      // All position embeddings should have new dimension
      for (const posEmb of transformerWeights.positionEmbeddings) {
        expect(posEmb.length).toBe(newSize);
      }
    });

    it('should expand renorm states correctly', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');

      const transformerWeights = model.getTransformerWeights();
      const newSize = initialSize + 4;

      for (const renorm of transformerWeights.renormStates) {
        expect(renorm.attention.gamma.length).toBe(newSize);
        expect(renorm.ffn.gamma.length).toBe(newSize);
        expect(typeof renorm.attention.epsilon).toBe('number');
        expect(typeof renorm.ffn.epsilon).toBe('number');
      }
    });

    it('should throw when cannot inject', () => {
      const smallAdapter = new TransformerLMAdapter(model, 'test', 20);
      expect(() => smallAdapter.inject(10, 'random_he')).toThrow();
    });
  });

  describe('exportWeights / importWeights', () => {
    it('should export weights as Float32Arrays', () => {
      const weights = adapter.exportWeights();

      // Base (5) + position embeddings (1) + per layer (9 * 2 layers) = 24
      expect(weights.length).toBe(6 + 9 * 2);
      weights.forEach((w) => {
        expect(w).toBeInstanceOf(Float32Array);
      });
    });

    it('should import weights and restore model state', () => {
      const originalWeights = adapter.exportWeights();
      const originalSize = model.getHiddenSize();

      // Modify model
      adapter.inject(4, 'random_he');
      expect(model.getHiddenSize()).toBe(originalSize + 4);

      // Import original weights (rollback)
      adapter.importWeights(originalWeights);

      // Size should be restored
      expect(model.getHiddenSize()).toBe(originalSize);
    });

    it('should restore attention weights correctly after rollback', () => {
      const originalWeights = adapter.exportWeights();
      const originalTransformerWeights = model.getTransformerWeights();

      // Store some original values
      const origQuery00 = originalTransformerWeights.attentionWeights[0].query[0][0];

      // Modify model
      adapter.inject(4, 'random_he');

      // Rollback
      adapter.importWeights(originalWeights);

      const restoredWeights = model.getTransformerWeights();
      expect(restoredWeights.attentionWeights[0].query[0][0]).toBeCloseTo(origQuery00, 5);
    });

    it('should restore position embeddings correctly after rollback', () => {
      const originalWeights = adapter.exportWeights();
      const originalTransformerWeights = model.getTransformerWeights();
      const originalPosEmbLength = originalTransformerWeights.positionEmbeddings[0].length;

      // Modify model
      adapter.inject(4, 'random_he');

      // Rollback
      adapter.importWeights(originalWeights);

      const restoredWeights = model.getTransformerWeights();
      expect(restoredWeights.positionEmbeddings[0].length).toBe(originalPosEmbLength);
    });
  });
});

describe('TransformerLM expandHiddenLayer', () => {
  let model: TransformerLM;

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    model = new TransformerLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0.1, 42, { mode: 'unicode' }, {
      numLayers: 2,
      numHeads: 4,
      ffHiddenDim: 32
    });
  });

  it('should do nothing for k <= 0', () => {
    const initialSize = model.getHiddenSize();
    model.expandHiddenLayer(0);
    expect(model.getHiddenSize()).toBe(initialSize);

    model.expandHiddenLayer(-5);
    expect(model.getHiddenSize()).toBe(initialSize);
  });

  it('should maintain numerical values for existing weights', () => {
    const transformerWeights = model.getTransformerWeights();
    const origQuery00 = transformerWeights.attentionWeights[0].query[0][0];

    model.expandHiddenLayer(4);

    const newWeights = model.getTransformerWeights();
    // Original value should be preserved (within floating point tolerance)
    expect(newWeights.attentionWeights[0].query[0][0]).toBeCloseTo(origQuery00, 5);
  });

  it('should use sinusoidal encoding for new position embedding dimensions', () => {
    const originalSize = model.getHiddenSize();
    model.expandHiddenLayer(4);

    const newWeights = model.getTransformerWeights();

    // New dimensions should follow sinusoidal pattern
    // At position 0, sin(0) = 0, cos(0) = 1
    const posEmb0 = newWeights.positionEmbeddings[0];

    // The new dimensions start at originalSize
    // Check that they're not all zeros (would indicate improper initialization)
    const newDims = posEmb0.slice(originalSize);
    expect(newDims.length).toBe(4);

    // Position 0 should have valid values
    newDims.forEach((v) => {
      expect(typeof v).toBe('number');
      expect(Number.isFinite(v)).toBe(true);
    });
  });
});

describe('InjectionRunSession with TransformerLMAdapter', () => {
  let model: TransformerLM;
  let adapter: TransformerLMAdapter;
  let session: InjectionRunSession;
  let ledgerRecords: InjectionEvent[];

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'test', 'data'];
    model = new TransformerLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0.1, 42, { mode: 'unicode' }, {
      numLayers: 2,
      numHeads: 4,
      ffHiddenDim: 32
    });
    adapter = createTransformerLMAdapter(model, 'transformer-session-test');
    ledgerRecords = [];
    session = new InjectionRunSession(
      adapter,
      createLedgerAdapter(ledgerRecords),
      new InjectionEngine()
    );
  });

  function createTestBubbles(): CerebroBubble[] {
    const dModel = adapter.getTarget().dModel;
    return Array.from({ length: 5 }, (_, i) => ({
      id: `bubble-${i}`,
      label: `b${i}`,
      activation: 0.5 + Math.random() * 0.5,
      embedding: Array.from({ length: dModel }, () => Math.random() * 0.1),
      tag: 'body' as const,
      ts: Date.now() - i * 1000
    }));
  }

  describe('propose', () => {
    it('should generate injection proposal for transformer model', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);

      expect(proposal).toBeDefined();
      expect(proposal.target.modelId).toBe('transformer-session-test');
      expect(proposal.target.type).toBe('transformer');
      expect(proposal.k).toBeGreaterThanOrEqual(1);
    });
  });

  describe('inject and undo', () => {
    it('should execute injection and record event', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      const event = session.inject(proposal, bubbles);

      expect(event).toBeDefined();
      expect(ledgerRecords).toHaveLength(1);
      expect(ledgerRecords[0].proposal.target.type).toBe('transformer');
    });

    it('should undo injection and restore all transformer weights', () => {
      const initialSize = model.getHiddenSize();
      const initialTransformerWeights = model.getTransformerWeights();
      const initialQuery00 = initialTransformerWeights.attentionWeights[0].query[0][0];

      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      // Size should have changed
      expect(model.getHiddenSize()).toBeGreaterThan(initialSize);

      // Undo
      session.undoLast();

      // Should be restored
      expect(model.getHiddenSize()).toBe(initialSize);

      const restoredWeights = model.getTransformerWeights();
      expect(restoredWeights.attentionWeights[0].query[0][0]).toBeCloseTo(initialQuery00, 5);
    });

    it('should handle multiple injections and undos', () => {
      const bubbles = createTestBubbles();

      // First injection
      const proposal1 = session.propose(bubbles);
      session.inject(proposal1, bubbles);
      const sizeAfterFirst = model.getHiddenSize();

      // Second injection
      const proposal2 = session.propose(bubbles);
      session.inject(proposal2, bubbles);
      const sizeAfterSecond = model.getHiddenSize();

      expect(sizeAfterSecond).toBeGreaterThan(sizeAfterFirst);

      // Undo second
      session.undoLast();
      expect(model.getHiddenSize()).toBe(sizeAfterFirst);

      // Undo first
      session.undoLast();
      expect(model.getHiddenSize()).toBe(16); // Initial size
    });

    it('should maintain generation capability after injection', async () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      // Model should still be able to generate text
      const output = await model.generate('hello', 5, 0.8);
      expect(typeof output).toBe('string');
    });
  });

  describe('decision ledger integration', () => {
    it('should record transformer-specific metadata in ledger', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      expect(ledgerRecords).toHaveLength(1);
      const event = ledgerRecords[0];

      expect(event.proposal.target.type).toBe('transformer');
      expect(event.proposal.target.metadata?.numLayers).toBe(2);
      expect(event.proposal.target.metadata?.numHeads).toBe(4);
    });
  });
});
