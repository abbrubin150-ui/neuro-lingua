/**
 * Tests for trace export round-trip functionality
 * Verifies Σ-SIG metadata integrity (IMMEDIATE_ACTIONS requirement)
 */

import { describe, it, expect } from 'vitest';
import {
  createTraceExport,
  validateTraceExport,
  generateTraceFilename,
  type TraceExport
} from '../../src/lib/traceExport';
import { createRun, createDecisionLedger, type TrainingConfig } from '../../src/types/project';

// Helper to create a minimal TrainingConfig
function createTestConfig(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    architecture: 'feedforward',
    hiddenSize: 64,
    epochs: 20,
    learningRate: 0.08,
    optimizer: 'momentum',
    momentum: 0.9,
    dropout: 0.1,
    contextSize: 3,
    seed: 42,
    useAdvanced: false,
    useGPU: false,
    tokenizerConfig: {
      mode: 'unicode',
      pattern: "[^\\p{L}\\d\\s'-]"
    },
    ...overrides
  };
}

describe('Trace Export Round-Trip (Σ-SIG compliance)', () => {
  describe('createTraceExport', () => {
    it('should create trace with model data only', () => {
      const modelData = {
        weights: { embedding: [[1, 2], [3, 4]], hidden: [[5, 6]] },
        config: { hiddenSize: 64 },
        tokenizer: { vocab: ['a', 'b', 'c'] }
      };

      const trace = createTraceExport(modelData);

      expect(trace.modelWeights).toEqual(modelData.weights);
      expect(trace.config).toEqual(modelData.config);
      expect(trace.tokenizer).toEqual(modelData.tokenizer);
      expect(trace.exportedAt).toBeGreaterThan(0);
      expect(trace.exportedBy).toBe('local-user');
      expect(trace.version).toBe('3.2.4');
    });

    it('should include project metadata when run is provided', () => {
      const modelData = {
        weights: {},
        config: {},
        tokenizer: {}
      };

      const ledger = createDecisionLedger('Test rationale for model training');
      const config = createTestConfig();
      const run = createRun('proj-123', 'Test Run', config, 'test corpus', ledger);

      const trace = createTraceExport(modelData, run);

      expect(trace.projectMeta).toBeDefined();
      expect(trace.projectMeta?.projectId).toBe(run.projectId);
      expect(trace.projectMeta?.runId).toBe(run.id);
      expect(trace.projectMeta?.runName).toBe('Test Run');
    });

    it('should include decision ledger from run', () => {
      const modelData = {
        weights: {},
        config: {},
        tokenizer: {}
      };

      const ledger = createDecisionLedger('Comprehensive testing needed', 'reviewer-001');
      ledger.expiry = '2025-12-31';
      ledger.rollback = 'archive';

      const config = createTestConfig();
      const run = createRun('proj-123', 'Test Run', config, 'test corpus', ledger);

      const trace = createTraceExport(modelData, run);

      expect(trace.decisionLedger).toBeDefined();
      expect(trace.decisionLedger?.rationale).toBe('Comprehensive testing needed');
      expect(trace.decisionLedger?.witness).toBe('reviewer-001');
      expect(trace.decisionLedger?.expiry).toBe('2025-12-31');
      expect(trace.decisionLedger?.rollback).toBe('archive');
      expect(trace.decisionLedger?.createdAt).toBeGreaterThan(0);
    });

    it('should include training trace with history', () => {
      const modelData = {
        weights: {},
        config: {},
        tokenizer: {}
      };

      const trainingHistory = [
        { loss: 3.5, accuracy: 0.3, timestamp: Date.now() - 3000 },
        { loss: 2.8, accuracy: 0.45, timestamp: Date.now() - 2000 },
        { loss: 2.1, accuracy: 0.6, timestamp: Date.now() - 1000 }
      ];

      const finalStats = {
        loss: 2.1,
        acc: 0.6,
        ppl: 8.2
      };

      const corpus = 'the quick brown fox jumps over the lazy dog';

      const trace = createTraceExport(modelData, undefined, trainingHistory, finalStats, corpus);

      expect(trace.trainingTrace).toBeDefined();
      expect(trace.trainingTrace?.epochs).toBe(3);
      expect(trace.trainingTrace?.finalLoss).toBe(2.1);
      expect(trace.trainingTrace?.finalAccuracy).toBe(0.6);
      expect(trace.trainingTrace?.finalPerplexity).toBe(8.2);
      expect(trace.trainingTrace?.trainLoss).toEqual([3.5, 2.8, 2.1]);
      expect(trace.trainingTrace?.trainAccuracy).toEqual([0.3, 0.45, 0.6]);
      expect(trace.trainingTrace?.timestamps).toHaveLength(3);
      expect(trace.trainingTrace?.sha256_corpus).toBeTruthy();
      expect(trace.trainingTrace?.sha256_corpus).not.toBe('00000000');
    });

    it('should include scenario scores when available', () => {
      const modelData = {
        weights: {},
        config: {},
        tokenizer: {}
      };

      const ledger = createDecisionLedger('Test with scenarios');
      const config = createTestConfig();
      const run = createRun('proj-123', 'Scenario Run', config, 'test corpus', ledger);

      // Add scenario results to run
      run.results = {
        finalLoss: 2.0,
        finalAccuracy: 0.7,
        finalPerplexity: 7.4,
        trainingHistory: [
          { loss: 3.0, accuracy: 0.5, timestamp: Date.now() }
        ],
        scenarioResults: [
          { scenarioId: 'scenario-1', response: 'Generated text 1', score: 0.85, timestamp: Date.now() },
          { scenarioId: 'scenario-2', response: 'Generated text 2', score: 0.92, timestamp: Date.now() }
        ]
      };

      const trainingHistory = [
        { loss: 2.0, accuracy: 0.7, timestamp: Date.now() }
      ];

      const finalStats = { loss: 2.0, acc: 0.7, ppl: 7.4 };

      const trace = createTraceExport(modelData, run, trainingHistory, finalStats, 'corpus');

      expect(trace.trainingTrace?.scenariosScores).toBeDefined();
      expect(trace.trainingTrace?.scenariosScores?.['scenario-1']).toBe(0.85);
      expect(trace.trainingTrace?.scenariosScores?.['scenario-2']).toBe(0.92);
    });

    it('should create complete trace with all Σ-SIG metadata', () => {
      const modelData = {
        weights: { embedding: [[1, 2]], hidden: [[3, 4]] },
        config: { hiddenSize: 64, learningRate: 0.1 },
        tokenizer: { vocab: ['a', 'b', 'c'], mode: 'unicode' }
      };

      const ledger = createDecisionLedger('Full integration test', 'ai-assistant');
      ledger.expiry = '2026-01-01';

      const config = createTestConfig({ hiddenSize: 128 });
      const run = createRun('proj-456', 'Integration Run', config, 'large corpus', ledger);

      run.results = {
        finalLoss: 1.5,
        finalAccuracy: 0.8,
        finalPerplexity: 4.5,
        trainingHistory: [
          { loss: 3.0, accuracy: 0.5, timestamp: Date.now() - 2000 },
          { loss: 1.5, accuracy: 0.8, timestamp: Date.now() }
        ],
        scenarioResults: [
          { scenarioId: 'test-scenario', response: 'output', score: 0.9, timestamp: Date.now() }
        ]
      };

      const trainingHistory = [
        { loss: 3.0, accuracy: 0.5, timestamp: Date.now() - 2000 },
        { loss: 1.5, accuracy: 0.8, timestamp: Date.now() }
      ];

      const finalStats = { loss: 1.5, acc: 0.8, ppl: 4.5 };
      const corpus = 'comprehensive training corpus for testing';

      const trace = createTraceExport(modelData, run, trainingHistory, finalStats, corpus, '3.2.4');

      // Verify all sections are present
      expect(trace.modelWeights).toBeDefined();
      expect(trace.config).toBeDefined();
      expect(trace.tokenizer).toBeDefined();
      expect(trace.projectMeta).toBeDefined();
      expect(trace.decisionLedger).toBeDefined();
      expect(trace.trainingTrace).toBeDefined();

      // Verify metadata
      expect(trace.version).toBe('3.2.4');
      expect(trace.exportedBy).toBe('local-user');
      expect(trace.exportedAt).toBeGreaterThan(0);
    });
  });

  describe('Round-trip integrity (IMMEDIATE_ACTIONS requirement)', () => {
    it('should preserve all Σ-SIG metadata through JSON serialization', () => {
      // Create comprehensive trace
      const modelData = {
        weights: { embedding: [[1, 2, 3]], hidden: [[4, 5, 6]], output: [[7, 8]] },
        config: { hiddenSize: 64, epochs: 20, learningRate: 0.08 },
        tokenizer: { vocab: ['<PAD>', '<BOS>', 'the', 'quick'], mode: 'unicode' }
      };

      const ledger = createDecisionLedger('Round-trip test rationale', 'test-witness');
      ledger.expiry = '2025-12-31';
      ledger.rollback = 'archive';

      const config = createTestConfig();
      const run = createRun('project-789', 'Round-trip Run', config, 'test corpus data', ledger);

      run.results = {
        finalLoss: 1.8,
        finalAccuracy: 0.75,
        finalPerplexity: 6.0,
        trainingHistory: [
          { loss: 3.2, accuracy: 0.4, timestamp: 1000 },
          { loss: 1.8, accuracy: 0.75, timestamp: 2000 }
        ],
        scenarioResults: [
          { scenarioId: 's1', response: 'response 1', score: 0.88, timestamp: 1000 },
          { scenarioId: 's2', response: 'response 2', score: 0.93, timestamp: 2000 }
        ]
      };

      const trainingHistory = [
        { loss: 3.2, accuracy: 0.4, timestamp: 1000 },
        { loss: 1.8, accuracy: 0.75, timestamp: 2000 }
      ];

      const finalStats = { loss: 1.8, acc: 0.75, ppl: 6.0 };
      const corpus = 'test corpus for round-trip validation';

      // Create trace
      const originalTrace = createTraceExport(
        modelData,
        run,
        trainingHistory,
        finalStats,
        corpus,
        '3.2.4'
      );

      // Serialize to JSON (simulating export)
      const jsonString = JSON.stringify(originalTrace);

      // Parse back (simulating import)
      const importedTrace = JSON.parse(jsonString) as TraceExport;

      // Validate imported trace
      expect(validateTraceExport(importedTrace)).toBe(true);

      // Verify all fields preserved exactly
      expect(importedTrace.modelWeights).toEqual(originalTrace.modelWeights);
      expect(importedTrace.config).toEqual(originalTrace.config);
      expect(importedTrace.tokenizer).toEqual(originalTrace.tokenizer);
      expect(importedTrace.version).toBe(originalTrace.version);
      expect(importedTrace.exportedBy).toBe(originalTrace.exportedBy);
      expect(importedTrace.exportedAt).toBe(originalTrace.exportedAt);

      // Verify project metadata preserved
      expect(importedTrace.projectMeta).toEqual(originalTrace.projectMeta);
      expect(importedTrace.projectMeta?.projectId).toBe('project-789');
      expect(importedTrace.projectMeta?.runName).toBe('Round-trip Run');

      // Verify decision ledger preserved (Σ-SIG compliance)
      expect(importedTrace.decisionLedger).toEqual(originalTrace.decisionLedger);
      expect(importedTrace.decisionLedger?.rationale).toBe('Round-trip test rationale');
      expect(importedTrace.decisionLedger?.witness).toBe('test-witness');
      expect(importedTrace.decisionLedger?.expiry).toBe('2025-12-31');
      expect(importedTrace.decisionLedger?.rollback).toBe('archive');

      // Verify training trace preserved
      expect(importedTrace.trainingTrace).toEqual(originalTrace.trainingTrace);
      expect(importedTrace.trainingTrace?.epochs).toBe(2);
      expect(importedTrace.trainingTrace?.finalLoss).toBe(1.8);
      expect(importedTrace.trainingTrace?.finalAccuracy).toBe(0.75);
      expect(importedTrace.trainingTrace?.trainLoss).toEqual([3.2, 1.8]);
      expect(importedTrace.trainingTrace?.trainAccuracy).toEqual([0.4, 0.75]);

      // Verify scenario scores preserved
      expect(importedTrace.trainingTrace?.scenariosScores).toBeDefined();
      expect(importedTrace.trainingTrace?.scenariosScores?.['s1']).toBe(0.88);
      expect(importedTrace.trainingTrace?.scenariosScores?.['s2']).toBe(0.93);

      // Verify corpus checksum preserved
      expect(importedTrace.trainingTrace?.sha256_corpus).toBe(
        originalTrace.trainingTrace?.sha256_corpus
      );
    });

    it('should validate complete trace export correctly', () => {
      const validTrace: TraceExport = {
        modelWeights: { embedding: [] },
        config: { hiddenSize: 64 },
        tokenizer: { vocab: [] },
        exportedAt: Date.now(),
        exportedBy: 'test-user',
        version: '3.2.4',
        projectMeta: {
          projectId: 'p1',
          projectName: 'Project 1',
          runId: 'r1',
          runName: 'Run 1'
        },
        decisionLedger: {
          rationale: 'Test',
          witness: 'witness',
          expiry: null,
          rollback: 'keep',
          createdAt: Date.now()
        },
        trainingTrace: {
          epochs: 10,
          finalLoss: 2.0,
          finalAccuracy: 0.7,
          finalPerplexity: 7.4,
          trainLoss: [3.0, 2.0],
          trainAccuracy: [0.5, 0.7],
          timestamps: [1000, 2000],
          sha256_corpus: 'abc123'
        }
      };

      expect(validateTraceExport(validTrace)).toBe(true);
    });

    it('should reject invalid trace exports', () => {
      // Missing required fields
      const invalidTrace1 = {
        config: {},
        tokenizer: {},
        exportedAt: Date.now(),
        exportedBy: 'user',
        version: '3.2.4'
        // Missing modelWeights
      };

      expect(validateTraceExport(invalidTrace1)).toBe(false);

      // Invalid field types
      const invalidTrace2 = {
        modelWeights: {},
        config: {},
        tokenizer: {},
        exportedAt: 'not a number', // Should be number
        exportedBy: 'user',
        version: '3.2.4'
      };

      expect(validateTraceExport(invalidTrace2)).toBe(false);

      // Invalid decision ledger
      const invalidTrace3 = {
        modelWeights: {},
        config: {},
        tokenizer: {},
        exportedAt: Date.now(),
        exportedBy: 'user',
        version: '3.2.4',
        decisionLedger: {
          rationale: 'test',
          witness: 123, // Should be string
          createdAt: Date.now()
        }
      };

      expect(validateTraceExport(invalidTrace3)).toBe(false);
    });
  });

  describe('generateTraceFilename', () => {
    it('should generate valid filename with version', () => {
      const filename = generateTraceFilename('3.2.4');

      expect(filename).toMatch(/^neuro-lingua-v324-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.json$/);
    });

    it('should include run name in filename', () => {
      const filename = generateTraceFilename('3.2.4', 'My Test Run');

      expect(filename).toContain('-my-test-run-');
    });

    it('should include hash in filename', () => {
      const filename = generateTraceFilename('3.2.4', undefined, 'abc123def456');

      expect(filename).toContain('-abc123de');
      expect(filename).toMatch(/\.json$/);
    });

    it('should handle all parameters', () => {
      const filename = generateTraceFilename('3.2.4', 'Test Run', 'hash12345');

      expect(filename).toContain('v324');
      expect(filename).toContain('test-run');
      expect(filename).toContain('hash1234');
      expect(filename).toMatch(/\.json$/);
    });
  });
});
