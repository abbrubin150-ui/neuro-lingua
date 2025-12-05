/**
 * Tests for diff utilities
 */

import { describe, it, expect } from 'vitest';
import {
  createFieldDiff,
  diffHyperparameters,
  diffMetrics,
  determineImprovementDirection,
  findSignificantChanges,
  countChanges,
  generateRunDiff,
  formatFieldDiff
} from '../../src/lib/diffUtils';
import { createRun, createDecisionLedger } from '../../src/types/project';
import type { Run, TrainingConfig } from '../../src/types/project';

// Helper to create a minimal TrainingConfig
function createMinimalConfig(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
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

// Helper to create a test run
function createTestRun(
  name: string,
  configOverrides: Partial<TrainingConfig> = {},
  resultsOverrides: Partial<Run['results']> = {}
): Run {
  const config = createMinimalConfig(configOverrides);
  const run = createRun('test-project', name, config, 'test corpus', createDecisionLedger('test'));

  if (Object.keys(resultsOverrides).length > 0) {
    run.results = {
      finalLoss: 2.5,
      finalAccuracy: 0.4,
      finalPerplexity: 12.0,
      trainingHistory: [],
      ...resultsOverrides
    };
  }

  return run;
}

describe('createFieldDiff', () => {
  it('should detect unchanged values', () => {
    const diff = createFieldDiff('test', 10, 10);
    expect(diff.type).toBe('unchanged');
    expect(diff.oldValue).toBe(10);
    expect(diff.newValue).toBe(10);
  });

  it('should detect changed values', () => {
    const diff = createFieldDiff('test', 10, 20);
    expect(diff.type).toBe('changed');
    expect(diff.oldValue).toBe(10);
    expect(diff.newValue).toBe(20);
  });

  it('should detect added values', () => {
    const diff = createFieldDiff('test', undefined, 20);
    expect(diff.type).toBe('added');
    expect(diff.oldValue).toBeUndefined();
    expect(diff.newValue).toBe(20);
  });

  it('should detect removed values', () => {
    const diff = createFieldDiff('test', 20, undefined);
    expect(diff.type).toBe('removed');
    expect(diff.oldValue).toBe(20);
    expect(diff.newValue).toBeUndefined();
  });

  it('should calculate percent change for numbers', () => {
    const diff = createFieldDiff('test', 100, 150);
    expect(diff.percentChange).toBe(50);
  });

  it('should calculate negative percent change', () => {
    const diff = createFieldDiff('test', 100, 75);
    expect(diff.percentChange).toBe(-25);
  });

  it('should handle change from zero', () => {
    const diff = createFieldDiff('test', 0, 50);
    expect(diff.percentChange).toBe(100);
  });
});

describe('diffHyperparameters', () => {
  it('should detect no changes when configs are identical', () => {
    const run1 = createTestRun('run1');
    const run2 = createTestRun('run2');

    const diff = diffHyperparameters(run1, run2);

    expect(diff.hiddenSize.type).toBe('unchanged');
    expect(diff.learningRate.type).toBe('unchanged');
    expect(diff.optimizer.type).toBe('unchanged');
  });

  it('should detect changed hyperparameters', () => {
    const run1 = createTestRun('run1', { hiddenSize: 64, learningRate: 0.08 });
    const run2 = createTestRun('run2', { hiddenSize: 128, learningRate: 0.1 });

    const diff = diffHyperparameters(run1, run2);

    expect(diff.hiddenSize.type).toBe('changed');
    expect(diff.hiddenSize.oldValue).toBe(64);
    expect(diff.hiddenSize.newValue).toBe(128);
    expect(diff.hiddenSize.percentChange).toBe(100);

    expect(diff.learningRate.type).toBe('changed');
    expect(diff.learningRate.oldValue).toBe(0.08);
    expect(diff.learningRate.newValue).toBe(0.1);
  });

  it('should detect optimizer change', () => {
    const run1 = createTestRun('run1', { optimizer: 'momentum' });
    const run2 = createTestRun('run2', { optimizer: 'adam' });

    const diff = diffHyperparameters(run1, run2);

    expect(diff.optimizer.type).toBe('changed');
    expect(diff.optimizer.oldValue).toBe('momentum');
    expect(diff.optimizer.newValue).toBe('adam');
  });

  it('should include advanced features when present', () => {
    const run1 = createTestRun('run1', { useAdvanced: true, activation: 'relu' });
    const run2 = createTestRun('run2', { useAdvanced: true, activation: 'gelu' });

    const diff = diffHyperparameters(run1, run2);

    expect(diff.activation).toBeDefined();
    expect(diff.activation?.type).toBe('changed');
    expect(diff.activation?.oldValue).toBe('relu');
    expect(diff.activation?.newValue).toBe('gelu');
  });

  it('should include transformer features when present', () => {
    const run1 = createTestRun('run1', { architecture: 'transformer', numHeads: 4 });
    const run2 = createTestRun('run2', { architecture: 'transformer', numHeads: 8 });

    const diff = diffHyperparameters(run1, run2);

    expect(diff.numHeads).toBeDefined();
    expect(diff.numHeads?.type).toBe('changed');
    expect(diff.numHeads?.oldValue).toBe(4);
    expect(diff.numHeads?.newValue).toBe(8);
  });
});

describe('diffMetrics', () => {
  it('should compare final metrics', () => {
    const run1 = createTestRun('run1', {}, {
      finalLoss: 2.5,
      finalAccuracy: 0.4,
      finalPerplexity: 12.0
    });

    const run2 = createTestRun('run2', {}, {
      finalLoss: 1.8,
      finalAccuracy: 0.6,
      finalPerplexity: 6.0
    });

    const diff = diffMetrics(run1, run2);

    expect(diff.finalLoss.type).toBe('changed');
    expect(diff.finalLoss.oldValue).toBe(2.5);
    expect(diff.finalLoss.newValue).toBe(1.8);

    expect(diff.finalAccuracy.type).toBe('changed');
    expect(diff.finalAccuracy.oldValue).toBe(0.4);
    expect(diff.finalAccuracy.newValue).toBe(0.6);

    expect(diff.finalPerplexity.type).toBe('changed');
    expect(diff.finalPerplexity.oldValue).toBe(12.0);
    expect(diff.finalPerplexity.newValue).toBe(6.0);
  });

  it('should handle missing results', () => {
    const run1 = createTestRun('run1');
    const run2 = createTestRun('run2');

    const diff = diffMetrics(run1, run2);

    expect(diff.finalLoss.type).toBe('unchanged');
    expect(diff.finalLoss.oldValue).toBeUndefined();
    expect(diff.finalLoss.newValue).toBeUndefined();
  });

  it('should calculate training time when timestamps available', () => {
    const run1 = createTestRun('run1');
    run1.startedAt = 1000;
    run1.completedAt = 6000; // 5 seconds

    const run2 = createTestRun('run2');
    run2.startedAt = 2000;
    run2.completedAt = 5000; // 3 seconds

    const diff = diffMetrics(run1, run2);

    expect(diff.trainingTime).toBeDefined();
    expect(diff.trainingTime?.oldValue).toBe(5000);
    expect(diff.trainingTime?.newValue).toBe(3000);
  });
});

describe('determineImprovementDirection', () => {
  it('should detect improvement when loss decreases', () => {
    const run1 = createTestRun('run1', {}, { finalLoss: 2.5, finalAccuracy: 0.5, finalPerplexity: 10 });
    const run2 = createTestRun('run2', {}, { finalLoss: 1.5, finalAccuracy: 0.7, finalPerplexity: 5 });

    const diff = diffMetrics(run1, run2);
    const direction = determineImprovementDirection(diff);

    expect(direction).toBe('better');
  });

  it('should detect regression when loss increases', () => {
    const run1 = createTestRun('run1', {}, { finalLoss: 1.5, finalAccuracy: 0.7, finalPerplexity: 5 });
    const run2 = createTestRun('run2', {}, { finalLoss: 2.5, finalAccuracy: 0.5, finalPerplexity: 10 });

    const diff = diffMetrics(run1, run2);
    const direction = determineImprovementDirection(diff);

    expect(direction).toBe('worse');
  });

  it('should detect mixed results', () => {
    const run1 = createTestRun('run1', {}, { finalLoss: 2.0, finalAccuracy: 0.5, finalPerplexity: 8 });
    const run2 = createTestRun('run2', {}, { finalLoss: 1.5, finalAccuracy: 0.4, finalPerplexity: 6 });

    const diff = diffMetrics(run1, run2);
    const direction = determineImprovementDirection(diff);

    expect(direction).toBe('mixed');
  });

  it('should return unknown when no metrics changed', () => {
    const run1 = createTestRun('run1');
    const run2 = createTestRun('run2');

    const diff = diffMetrics(run1, run2);
    const direction = determineImprovementDirection(diff);

    expect(direction).toBe('unknown');
  });
});

describe('findSignificantChanges', () => {
  it('should identify significant numeric changes (>10%)', () => {
    const run1 = createTestRun('run1', { hiddenSize: 64, learningRate: 0.08 });
    const run2 = createTestRun('run2', { hiddenSize: 128, learningRate: 0.1 });

    const diff = diffHyperparameters(run1, run2);
    const significant = findSignificantChanges(diff);

    expect(significant.length).toBeGreaterThan(0);
    expect(significant.some(s => s.includes('hiddenSize'))).toBe(true);
    expect(significant.some(s => s.includes('100.0%'))).toBe(true); // 64 -> 128 is 100% increase
  });

  it('should identify important categorical changes', () => {
    const run1 = createTestRun('run1', { optimizer: 'momentum', activation: 'relu' });
    const run2 = createTestRun('run2', { optimizer: 'adam', activation: 'gelu' });

    const diff = diffHyperparameters(run1, run2);
    const significant = findSignificantChanges(diff);

    expect(significant.some(s => s.includes('optimizer'))).toBe(true);
    expect(significant.some(s => s.includes('activation'))).toBe(true);
  });

  it('should not include small changes (<10%)', () => {
    const run1 = createTestRun('run1', { learningRate: 0.08 });
    const run2 = createTestRun('run2', { learningRate: 0.085 }); // Only 6.25% increase

    const diff = diffHyperparameters(run1, run2);
    const significant = findSignificantChanges(diff);

    expect(significant.some(s => s.includes('learningRate'))).toBe(false);
  });
});

describe('countChanges', () => {
  it('should count all changed fields', () => {
    const run1 = createTestRun('run1', { hiddenSize: 64, learningRate: 0.08, dropout: 0.1 });
    const run2 = createTestRun('run2', { hiddenSize: 128, learningRate: 0.1, dropout: 0.2 });

    const diff = diffHyperparameters(run1, run2);
    const count = countChanges(diff);

    expect(count).toBe(3); // hiddenSize, learningRate, dropout
  });

  it('should return 0 when no changes', () => {
    const run1 = createTestRun('run1');
    const run2 = createTestRun('run2');

    const diff = diffHyperparameters(run1, run2);
    const count = countChanges(diff);

    expect(count).toBe(0);
  });
});

describe('generateRunDiff', () => {
  it('should generate complete diff with summary', () => {
    const run1 = createTestRun('run1', { hiddenSize: 64 }, { finalLoss: 2.5 });
    const run2 = createTestRun('run2', { hiddenSize: 128 }, { finalLoss: 1.5 });

    const runDiff = generateRunDiff(run1, run2);

    expect(runDiff.baseRun).toBe(run1);
    expect(runDiff.compareRun).toBe(run2);
    expect(runDiff.hyperparameters).toBeDefined();
    expect(runDiff.metrics).toBeDefined();
    expect(runDiff.summary.totalChanges).toBeGreaterThan(0);
    expect(runDiff.summary.improvementDirection).toBe('better');
    expect(runDiff.computedAt).toBeGreaterThan(0);
  });

  it('should detect corpus changes', () => {
    const run1 = createTestRun('run1');
    const run2 = createTestRun('run2');
    run2.corpus = 'different corpus';
    run2.corpusChecksum = 'different-checksum';

    const runDiff = generateRunDiff(run1, run2);

    expect(runDiff.corpusChanged).toBe(true);
  });
});

describe('formatFieldDiff', () => {
  it('should format unchanged values', () => {
    const diff = createFieldDiff('test', 10, 10);
    const formatted = formatFieldDiff(diff);
    expect(formatted).toBe('10');
  });

  it('should format added values', () => {
    const diff = createFieldDiff('test', undefined, 20);
    const formatted = formatFieldDiff(diff);
    expect(formatted).toBe('➕ 20');
  });

  it('should format removed values', () => {
    const diff = createFieldDiff('test', 20, undefined);
    const formatted = formatFieldDiff(diff);
    expect(formatted).toBe('➖ 20');
  });

  it('should format changed values with percent', () => {
    const diff = createFieldDiff('test', 100, 150);
    const formatted = formatFieldDiff(diff);
    expect(formatted).toContain('100 → 150');
    expect(formatted).toContain('+50.0%');
  });

  it('should format categorical changes', () => {
    const diff = createFieldDiff('test', 'momentum', 'adam');
    const formatted = formatFieldDiff(diff);
    expect(formatted).toBe('momentum → adam');
  });
});
