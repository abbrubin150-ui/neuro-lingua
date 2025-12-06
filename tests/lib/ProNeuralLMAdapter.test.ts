import { describe, it, expect, beforeEach } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { ProNeuralLMAdapter, createProNeuralLMAdapter } from '../../src/lib/expandable/ProNeuralLMAdapter';
import { InjectionEngine } from '../../src/lib/expandable/InjectionEngine';
import { InjectionRunSession, createLedgerAdapter } from '../../src/training/injection_hooks';
import type { CerebroBubble, InjectionEvent } from '../../src/types/injection';
import { createDecisionEntryFromInjection } from '../../src/types/project';

describe('ProNeuralLMAdapter', () => {
  let model: ProNeuralLM;
  let adapter: ProNeuralLMAdapter;

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    model = new ProNeuralLM(vocab, 16, 0.08, 3, 'momentum', 0.9, 0, 42);
    adapter = createProNeuralLMAdapter(model, 'test-model-1');
  });

  describe('getTarget', () => {
    it('should return correct injection target', () => {
      const target = adapter.getTarget();

      expect(target.modelId).toBe('test-model-1');
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
      // Default max is 512
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
      const smallAdapter = new ProNeuralLMAdapter(model, 'test', 20);
      // Model already has 16 hidden units, trying to add 10 would exceed 20
      expect(() => smallAdapter.inject(10, 'random_he')).toThrow();
    });
  });

  describe('exportWeights / importWeights', () => {
    it('should export weights as Float32Arrays', () => {
      const weights = adapter.exportWeights();

      expect(weights).toHaveLength(5);
      weights.forEach((w) => {
        expect(w).toBeInstanceOf(Float32Array);
      });
    });

    it('should import weights and restore model state', () => {
      const originalWeights = adapter.exportWeights();

      // Modify model
      adapter.inject(2, 'random_he');

      // Snapshot after injection
      const modifiedSize = model.getHiddenSize();
      expect(modifiedSize).toBeGreaterThan(16);

      // Import original weights (rollback)
      adapter.importWeights(originalWeights);

      // Size should be restored
      expect(model.getHiddenSize()).toBe(16);
    });
  });
});

describe('InjectionRunSession with ProNeuralLMAdapter', () => {
  let model: ProNeuralLM;
  let adapter: ProNeuralLMAdapter;
  let session: InjectionRunSession;
  let ledgerRecords: InjectionEvent[];

  beforeEach(() => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'test', 'data'];
    model = new ProNeuralLM(vocab, 16, 0.08, 3, 'momentum', 0.9, 0, 42);
    adapter = createProNeuralLMAdapter(model, 'session-test-model');
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
    it('should generate injection proposal', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);

      expect(proposal).toBeDefined();
      expect(proposal.target.modelId).toBe('session-test-model');
      expect(proposal.k).toBeGreaterThanOrEqual(1);
      expect(['residual_eig', 'random_he']).toContain(proposal.method);
    });
  });

  describe('inject', () => {
    it('should execute injection and record event', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      const event = session.inject(proposal, bubbles);

      expect(event).toBeDefined();
      expect(event.proposal).toBe(proposal);
      expect(typeof event.accepted).toBe('boolean');
      expect(ledgerRecords).toHaveLength(1);
      expect(ledgerRecords[0]).toBe(event);
    });

    it('should track injection history', () => {
      const bubbles = createTestBubbles();

      expect(session.history).toHaveLength(0);

      const proposal1 = session.propose(bubbles);
      session.inject(proposal1, bubbles);

      expect(session.history).toHaveLength(1);

      const proposal2 = session.propose(bubbles);
      session.inject(proposal2, bubbles);

      expect(session.history).toHaveLength(2);
    });
  });

  describe('undoLast', () => {
    it('should undo last injection', () => {
      const bubbles = createTestBubbles();
      const initialSize = model.getHiddenSize();

      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      // Verify injection happened (size may have changed)
      expect(model.getHiddenSize()).toBeGreaterThanOrEqual(initialSize);

      const undone = session.undoLast();

      expect(undone).toBeDefined();
      // After undo, size should be back to initial
      expect(model.getHiddenSize()).toBe(initialSize);
    });

    it('should remove event from ledger on undo', () => {
      const bubbles = createTestBubbles();
      const proposal = session.propose(bubbles);
      session.inject(proposal, bubbles);

      expect(ledgerRecords).toHaveLength(1);

      session.undoLast();

      expect(ledgerRecords).toHaveLength(0);
    });
  });
});

describe('createDecisionEntryFromInjection', () => {
  it('should create decision entry from accepted injection event', () => {
    const event: InjectionEvent = {
      proposal: {
        target: {
          modelId: 'test-model',
          layerId: 'hidden-0',
          type: 'ffn',
          dModel: 24,
          hiddenSize: 16
        },
        k: 4,
        method: 'residual_eig',
        epsilon: 0.05,
        minGain: 0.01,
        orthPenalty: 0.1,
        createdAt: new Date().toISOString()
      },
      accepted: true,
      metricsPre: { meanResidual: 0.15, tracePerp: 0.08 },
      metricsPost: { meanResidual: 0.10, tracePerp: 0.05, estimatedGain: 0.03 },
      delta: { meanResidual: -0.05, tracePerp: -0.03, estimatedGain: 0.02 },
      seed: 12345,
      runId: 'run-1'
    };

    const entry = createDecisionEntryFromInjection('proj-1', 'run-1', event);

    expect(entry.projectId).toBe('proj-1');
    expect(entry.affectedRunIds).toContain('run-1');
    expect(entry.category).toBe('cerebro-injection');
    expect(entry.witness).toBe('cerebro-engine');
    expect(entry.decision).toContain('Injected 4 neurons');
    expect(entry.kpi).toContain('residual');
  });

  it('should create decision entry from rejected injection event', () => {
    const event: InjectionEvent = {
      proposal: {
        target: {
          modelId: 'test-model',
          layerId: 'hidden-0',
          type: 'ffn',
          dModel: 24,
          hiddenSize: 16
        },
        k: 2,
        method: 'random_he',
        epsilon: 0.05,
        minGain: 0.01,
        orthPenalty: 0.1,
        createdAt: new Date().toISOString()
      },
      accepted: false,
      metricsPre: { meanResidual: 0.02, tracePerp: 0.01, estimatedGain: 0.005 },
      metricsPost: { meanResidual: 0.02, tracePerp: 0.01, estimatedGain: 0.004 },
      delta: { meanResidual: 0, tracePerp: 0 },
      seed: 67890,
      runId: 'run-2'
    };

    const entry = createDecisionEntryFromInjection('proj-1', 'run-2', event);

    expect(entry.decision).toContain('Rejected');
    expect(entry.affectedRunIds).toContain('run-2');
  });
});
