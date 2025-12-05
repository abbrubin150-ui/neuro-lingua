/**
 * Tests for export utilities
 */

import { describe, it, expect } from 'vitest';
import {
  createProjectExport,
  runToCSVRow,
  arrayToCSV,
  exportRunsToCSV,
  importProjectExport,
  generateExportSummary
} from '../../src/lib/exportUtils';
import {
  createProject,
  createRun,
  createDecisionLedger,
  createExperimentComparison,
  createDecisionEntry
} from '../../src/types/project';
import type { TrainingConfig } from '../../src/types/project';
import { EXPORT_SCHEMA_VERSION } from '../../src/types/experiment';

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

describe('createProjectExport', () => {
  it('should create export with correct schema version', () => {
    const project = createProject('Test Project', 'Test description');
    const exportData = createProjectExport([project], [], [], []);

    expect(exportData.schemaVersion).toBe(EXPORT_SCHEMA_VERSION);
    expect(exportData.metadata).toBeDefined();
    expect(exportData.metadata.source).toBe('neuro-lingua-domestica');
    expect(exportData.metadata.exportedAt).toBeGreaterThan(0);
  });

  it('should include all provided data', () => {
    const project = createProject('Test Project', 'Test description');
    const config = createMinimalConfig();
    const run = createRun(project.id, 'Test Run', config, 'test corpus', createDecisionLedger('test'));
    const comparison = createExperimentComparison(project.id, 'Test Comparison', 'desc', [run.id, 'run2']);
    const decision = createDecisionEntry(
      project.id,
      'Problem',
      ['alt1', 'alt2'],
      'Decision',
      'Loss',
      [run.id]
    );

    const exportData = createProjectExport([project], [run], [comparison], [decision]);

    expect(exportData.projects).toHaveLength(1);
    expect(exportData.runs).toHaveLength(1);
    expect(exportData.comparisons).toHaveLength(1);
    expect(exportData.decisions).toHaveLength(1);
  });

  it('should handle empty arrays', () => {
    const exportData = createProjectExport([], [], [], []);

    expect(exportData.projects).toHaveLength(0);
    expect(exportData.runs).toHaveLength(0);
    expect(exportData.comparisons).toHaveLength(0);
    expect(exportData.decisions).toHaveLength(0);
  });
});

describe('runToCSVRow', () => {
  it('should convert run to CSV row', () => {
    const config = createMinimalConfig({ hiddenSize: 128, learningRate: 0.1 });
    const run = createRun('proj1', 'Test Run', config, 'test corpus', createDecisionLedger('test rationale'));

    run.results = {
      finalLoss: 2.5,
      finalAccuracy: 0.6,
      finalPerplexity: 12.0,
      trainingHistory: []
    };
    run.completedAt = Date.now();

    const csvRow = runToCSVRow(run, 'Test Project');

    expect(csvRow.runId).toBe(run.id);
    expect(csvRow.runName).toBe('Test Run');
    expect(csvRow.projectName).toBe('Test Project');
    expect(csvRow.architecture).toBe('feedforward');
    expect(csvRow.hiddenSize).toBe(128);
    expect(csvRow.learningRate).toBe(0.1);
    expect(csvRow.finalLoss).toBe(2.5);
    expect(csvRow.finalAccuracy).toBe(0.6);
    expect(csvRow.corpusLength).toBe('test corpus'.length);
    expect(csvRow.decisionRationale).toBe('test rationale');
  });

  it('should handle missing results', () => {
    const config = createMinimalConfig();
    const run = createRun('proj1', 'Test Run', config, 'test corpus', createDecisionLedger('test'));

    const csvRow = runToCSVRow(run, 'Test Project');

    expect(csvRow.finalLoss).toBe(0);
    expect(csvRow.finalAccuracy).toBe(0);
    expect(csvRow.finalPerplexity).toBe(0);
    expect(csvRow.completedAt).toBe('');
  });
});

describe('arrayToCSV', () => {
  it('should convert array to CSV string', () => {
    const data = [
      { name: 'Alice', age: 30, active: true },
      { name: 'Bob', age: 25, active: false }
    ];

    const csv = arrayToCSV(data);
    const lines = csv.split('\n');

    expect(lines).toHaveLength(3); // header + 2 rows
    expect(lines[0]).toBe('name,age,active');
    expect(lines[1]).toBe('Alice,30,true');
    expect(lines[2]).toBe('Bob,25,false');
  });

  it('should handle values with commas', () => {
    const data = [
      { name: 'Alice, Smith', age: 30, city: 'New York, NY' }
    ];

    const csv = arrayToCSV(data);
    const lines = csv.split('\n');

    expect(lines[1]).toContain('"Alice, Smith"');
    expect(lines[1]).toContain('"New York, NY"');
  });

  it('should escape quotes', () => {
    const data = [
      { name: 'Alice "Ace" Smith', age: 30 }
    ];

    const csv = arrayToCSV(data);

    expect(csv).toContain('Alice ""Ace"" Smith');
  });

  it('should handle empty values', () => {
    const data = [
      { name: 'Alice', age: 30, email: '' }
    ];

    const csv = arrayToCSV(data);
    const lines = csv.split('\n');

    expect(lines[1]).toBe('Alice,30,');
  });

  it('should return empty string for empty array', () => {
    const csv = arrayToCSV([]);
    expect(csv).toBe('');
  });
});

describe('exportRunsToCSV', () => {
  it('should export multiple runs to CSV', () => {
    const project = createProject('Test Project', 'Test description');
    const config1 = createMinimalConfig({ hiddenSize: 64 });
    const config2 = createMinimalConfig({ hiddenSize: 128 });

    const run1 = createRun(project.id, 'Run 1', config1, 'corpus 1', createDecisionLedger('test'));
    const run2 = createRun(project.id, 'Run 2', config2, 'corpus 2', createDecisionLedger('test'));

    run1.results = {
      finalLoss: 2.5,
      finalAccuracy: 0.6,
      finalPerplexity: 12.0,
      trainingHistory: []
    };

    run2.results = {
      finalLoss: 1.8,
      finalAccuracy: 0.7,
      finalPerplexity: 6.0,
      trainingHistory: []
    };

    const csv = exportRunsToCSV([run1, run2], [project]);
    const lines = csv.split('\n');

    expect(lines.length).toBeGreaterThan(2); // header + 2 runs
    expect(lines[0]).toContain('runId');
    expect(lines[0]).toContain('hiddenSize');
    expect(lines[0]).toContain('finalLoss');
    expect(lines[1]).toContain('Run 1');
    expect(lines[2]).toContain('Run 2');
  });

  it('should handle unknown project', () => {
    const config = createMinimalConfig();
    const run = createRun('unknown-proj', 'Run 1', config, 'corpus', createDecisionLedger('test'));

    const csv = exportRunsToCSV([run], []);
    const lines = csv.split('\n');

    expect(lines[1]).toContain('Unknown');
  });
});

describe('importProjectExport', () => {
  it('should import valid export', () => {
    const project = createProject('Test Project', 'Test description');
    const exportData = createProjectExport([project], [], [], []);
    const json = JSON.stringify(exportData);

    const imported = importProjectExport(json);

    expect(imported.schemaVersion).toBe(EXPORT_SCHEMA_VERSION);
    expect(imported.projects).toHaveLength(1);
    expect(imported.projects[0].name).toBe('Test Project');
  });

  it('should reject invalid JSON', () => {
    expect(() => {
      importProjectExport('invalid json {{{');
    }).toThrow('Invalid JSON format');
  });

  it('should reject export without schema version', () => {
    const invalidExport = {
      metadata: {},
      projects: [],
      runs: []
    };

    expect(() => {
      importProjectExport(JSON.stringify(invalidExport));
    }).toThrow('Missing schema version');
  });

  it('should reject incompatible schema version', () => {
    const invalidExport = {
      schemaVersion: '999.0.0',
      metadata: { exportedAt: Date.now(), exportedBy: 'test', source: 'test', version: '1.0.0' },
      projects: [],
      runs: [],
      comparisons: [],
      decisions: []
    };

    expect(() => {
      importProjectExport(JSON.stringify(invalidExport));
    }).toThrow('Incompatible schema version');
  });

  it('should accept compatible minor/patch versions', () => {
    const project = createProject('Test', 'desc');
    const exportData = createProjectExport([project], [], [], []);
    exportData.schemaVersion = '1.5.3'; // Same major version

    const json = JSON.stringify(exportData);
    const imported = importProjectExport(json);

    expect(imported.projects).toHaveLength(1);
  });
});

describe('generateExportSummary', () => {
  it('should generate summary with counts', () => {
    const project = createProject('Test Project', 'Test description');
    const config = createMinimalConfig();
    const run1 = createRun(project.id, 'Run 1', config, 'corpus', createDecisionLedger('test'));
    const run2 = createRun(project.id, 'Run 2', config, 'corpus', createDecisionLedger('test'));

    run1.status = 'completed';
    run2.status = 'pending';

    const comparison = createExperimentComparison(project.id, 'Comp 1', 'desc', [run1.id, run2.id]);
    const decision = createDecisionEntry(
      project.id,
      'Problem',
      ['alt1'],
      'Decision',
      'Loss',
      [run1.id]
    );

    const exportData = createProjectExport([project], [run1, run2], [comparison], [decision]);
    const summary = generateExportSummary(exportData);

    expect(summary).toContain('**Projects:** 1');
    expect(summary).toContain('**Runs:** 2');
    expect(summary).toContain('1 completed');
    expect(summary).toContain('**Comparisons:** 1');
    expect(summary).toContain('**Decisions:** 1');
    expect(summary).toContain('Test Project');
  });

  it('should handle empty export', () => {
    const exportData = createProjectExport([], [], [], []);
    const summary = generateExportSummary(exportData);

    expect(summary).toContain('**Projects:** 0');
    expect(summary).toContain('**Runs:** 0');
    expect(summary).toContain('**Comparisons:** 0');
    expect(summary).toContain('**Decisions:** 0');
  });
});
