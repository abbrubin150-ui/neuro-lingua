import { describe, it, expect, beforeEach } from 'vitest';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import {
  AdvancedNeuralLMAdapter,
  createAdvancedNeuralLMAdapter
} from '../../src/lib/expandable/AdvancedNeuralLMAdapter';
import { InjectionEngine } from '../../src/lib/expandable/InjectionEngine';
import { InjectionRunSession, createLedgerAdapter } from '../../src/training/injection_hooks';
import type { CerebroBubble, InjectionEvent } from '../../src/types/injection';

describe('AdvancedNeuralLMAdapter', () => {
  let model: AdvancedNeuralLM;
  let adapter: AdvancedNeuralLMAdapter;

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    model = new AdvancedNeuralLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0, 42, undefined, {
      useLayerNorm: false,
      activation: 'gelu'
    });
    adapter = createAdvancedNeuralLMAdapter(model, 'test-advanced-model-1');
  });

  describe('getTarget', () => {
    it('should return correct injection target', () => {
      const target = adapter.getTarget();

      expect(target.modelId).toBe('test-advanced-model-1');
      expect(target.layerId).toBe('hidden-0');
      expect(target.type).toBe('ffn');
      expect(target.hiddenSize).toBe(16);
      expect(target.dModel).toBe(8 * 3); // vocabSize * contextSize
    });
  });

  describe('canInject', () => {
    it('should return true for valid injection count', () => {
      expect(adapter.canInject(1)).toBe(true);
      expect(adapter.canInject(8)).toBe(true);
    });

    it('should return false for invalid injection count', () => {
      expect(adapter.canInject(0)).toBe(false);
      expect(adapter.canInject(-1)).toBe(false);
    });

    it('should return false when exceeding max hidden size', () => {
      expect(adapter.canInject(600)).toBe(false);
    });
  });

  describe('inject', () => {
    it('should expand hidden layer by k neurons', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');
      expect(model.getHiddenSize()).toBe(initialSize + 4);
    });

    it('should throw when cannot inject', () => {
      const smallAdapter = new AdvancedNeuralLMAdapter(model, 'test', 20);
      expect(() => smallAdapter.inject(10, 'random_he')).toThrow();
    });
  });

  describe('exportWeights / importWeights', () => {
    it('should export weights as Float32Arrays', () => {
      const weights = adapter.exportWeights();

      expect(weights.length).toBeGreaterThanOrEqual(5);
      weights.forEach((w) => {
        expect(w).toBeInstanceOf(Float32Array);
      });
    });

    it('should import weights and restore model state', () => {
      const originalWeights = adapter.exportWeights();

      // Modify model
      adapter.inject(2, 'random_he');

      const modifiedSize = model.getHiddenSize();
      expect(modifiedSize).toBeGreaterThan(16);

      // Import original weights (rollback)
      adapter.importWeights(originalWeights);

      // Size should be restored
      expect(model.getHiddenSize()).toBe(16);
    });
  });
});

describe('AdvancedNeuralLMAdapter with LayerNorm', () => {
  let model: AdvancedNeuralLM;
  let adapter: AdvancedNeuralLMAdapter;

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    model = new AdvancedNeuralLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0, 42, undefined, {
      useLayerNorm: true,
      activation: 'gelu'
    });
    adapter = createAdvancedNeuralLMAdapter(model, 'test-layernorm-model');
  });

  describe('layer norm expansion', () => {
    it('should export layer norm parameters when enabled', () => {
      const advWeights = model.getAdvancedWeights();

      expect(advWeights).not.toBeNull();
      expect(advWeights!.layerNormGamma.length).toBe(16);
      expect(advWeights!.layerNormBeta.length).toBe(16);
    });

    it('should expand layer norm parameters on injection', () => {
      const initialSize = model.getHiddenSize();
      adapter.inject(4, 'random_he');

      const advWeights = model.getAdvancedWeights();
      expect(advWeights).not.toBeNull();
      expect(advWeights!.layerNormGamma.length).toBe(initialSize + 4);
      expect(advWeights!.layerNormBeta.length).toBe(initialSize + 4);

      // New gamma values should be 1.0
      const lastGammaValues = advWeights!.layerNormGamma.slice(-4);
      lastGammaValues.forEach((v) => expect(v).toBe(1.0));

      // New beta values should be 0.0
      const lastBetaValues = advWeights!.layerNormBeta.slice(-4);
      lastBetaValues.forEach((v) => expect(v).toBe(0.0));
    });

    it('should export and import layer norm parameters', () => {
      const originalWeights = adapter.exportWeights();

      // Should have 7 arrays (5 base + 2 layer norm)
      expect(originalWeights.length).toBe(7);

      // Modify model
      adapter.inject(2, 'random_he');

      // Rollback
      adapter.importWeights(originalWeights);

      const advWeights = model.getAdvancedWeights();
      expect(advWeights!.layerNormGamma.length).toBe(16);
      expect(advWeights!.layerNormBeta.length).toBe(16);
    });
  });
});

describe('InjectionRunSession with AdvancedNeuralLMAdapter', () => {
  let model: AdvancedNeuralLM;
  let adapter: AdvancedNeuralLMAdapter;
  let session: InjectionRunSession;
  let ledgerRecords: InjectionEvent[];

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'test', 'data'];
    model = new AdvancedNeuralLM(vocab, 16, 0.08, 3, 'adam', 0.9, 0, 42, undefined, {
      useLayerNorm: true
    });
    adapter = createAdvancedNeuralLMAdapter(model, 'advanced-session-test');
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
    it('should generate injection proposal for advanced model', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);

      expect(proposal).toBeDefined();
      expect(proposal.target.modelId).toBe('advanced-session-test');
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
    });

    it('should undo injection and restore layer norm', () => {
      const initialSize = model.getHiddenSize();
      const initialLNSize = model.getAdvancedWeights()!.layerNormGamma.length;

      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      // Size should have changed
      expect(model.getHiddenSize()).toBeGreaterThan(initialSize);

      // Undo
      session.undoLast();

      // Should be restored
      expect(model.getHiddenSize()).toBe(initialSize);
      expect(model.getAdvancedWeights()!.layerNormGamma.length).toBe(initialLNSize);
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
  });
});
