/**
 * Projects/Runs/Scenarios Integration Tests
 *
 * These tests verify the complete governance flow including:
 * - Project creation and management
 * - Run lifecycle (pending -> running -> completed)
 * - Scenario evaluation and scoring
 * - Decision Ledger enforcement
 * - Export/Import with full traceability
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  createProject,
  createRun,
  createScenario,
  createDecisionLedger,
  computeExecutionStatus,
  type Project,
  type DecisionLedger,
  type TrainingConfig
} from '../../src/types/project';
import { createExperimentComparison, createDecisionEntry } from '../../src/types/project';
import {
  createProjectExport,
  importProjectExport,
  generateExportSummary,
  exportRunsToCSV
} from '../../src/lib/exportUtils';
import { createTraceExport, validateTraceExport } from '../../src/lib/traceExport';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

// Test corpus
const testCorpus = 'hello world neural network machine learning ai models';

// Create vocabulary from corpus
function createVocab(corpus: string): string[] {
  const tokens = ProNeuralLM.tokenizeText(corpus);
  return ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ...Array.from(new Set(tokens))];
}

// Helper to create a minimal TrainingConfig
function createMinimalConfig(overrides: Partial<TrainingConfig> = {}): TrainingConfig {
  return {
    architecture: 'feedforward',
    hiddenSize: 16,
    epochs: 10,
    learningRate: 0.05,
    optimizer: 'momentum',
    momentum: 0.9,
    dropout: 0.1,
    contextSize: 2,
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

describe('Project Lifecycle', () => {
  describe('Project Creation', () => {
    it('should create project with unique ID', () => {
      const project1 = createProject('Project A', 'First project');
      const project2 = createProject('Project B', 'Second project');

      expect(project1.id).toBeDefined();
      expect(project2.id).toBeDefined();
      expect(project1.id).not.toBe(project2.id);
    });

    it('should set correct defaults', () => {
      const project = createProject('Test Project', 'Description');

      expect(project.name).toBe('Test Project');
      expect(project.description).toBe('Description');
      expect(project.language).toBe('en');
      expect(project.defaultArchitecture).toBe('feedforward');
      expect(project.corpusType).toBe('plain-text');
      expect(project.scenarios).toEqual([]);
      expect(project.runIds).toEqual([]);
      expect(project.createdAt).toBeGreaterThan(0);
      expect(project.updatedAt).toBeGreaterThan(0);
    });

    it('should support Hebrew language', () => {
      const project = createProject('פרויקט עברית', 'תיאור בעברית', 'he');

      expect(project.language).toBe('he');
      expect(project.name).toBe('פרויקט עברית');
    });

    it('should support mixed language', () => {
      const project = createProject('Mixed Project', 'Description', 'mixed');

      expect(project.language).toBe('mixed');
    });
  });

  describe('Scenario Management', () => {
    it('should create scenario with unique ID', () => {
      const scenario1 = createScenario('Test 1', 'hello');
      const scenario2 = createScenario('Test 2', 'world');

      expect(scenario1.id).not.toBe(scenario2.id);
    });

    it('should track expected responses', () => {
      const scenario = createScenario('Greeting', 'hello', 'world');

      expect(scenario.prompt).toBe('hello');
      expect(scenario.expectedResponse).toBe('world');
    });

    it('should add scenarios to project', () => {
      const project = createProject('Test', 'desc');
      const scenario1 = createScenario('Scenario 1', 'prompt 1');
      const scenario2 = createScenario('Scenario 2', 'prompt 2');

      project.scenarios.push(scenario1, scenario2);

      expect(project.scenarios).toHaveLength(2);
      expect(project.scenarios[0].name).toBe('Scenario 1');
    });
  });
});

describe('Run Lifecycle', () => {
  let project: Project;
  let config: TrainingConfig;

  beforeEach(() => {
    project = createProject('Test Project', 'Test Description');
    config = createMinimalConfig();
  });

  describe('Run Creation', () => {
    it('should create run with pending status', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      expect(run.status).toBe('pending');
      expect(run.projectId).toBe(project.id);
      expect(run.name).toBe('Run 1');
    });

    it('should generate corpus checksum', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      expect(run.corpusChecksum).toBeDefined();
      expect(run.corpusChecksum.length).toBeGreaterThan(0);

      // Same corpus should produce same checksum
      const run2 = createRun(project.id, 'Run 2', config, testCorpus, ledger);
      expect(run2.corpusChecksum).toBe(run.corpusChecksum);
    });

    it('should snapshot configuration at creation time', () => {
      const ledger = createDecisionLedger('Test rationale');
      const originalLR = config.learningRate;
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      // Verify config was captured at creation time
      expect(run.config.learningRate).toBe(originalLR);
      expect(run.config.architecture).toBe('feedforward');
      expect(run.config.hiddenSize).toBe(16);
    });
  });

  describe('Run Status Transitions', () => {
    it('should transition from pending to running', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      expect(run.status).toBe('pending');

      run.status = 'running';
      run.startedAt = Date.now();

      expect(run.status).toBe('running');
      expect(run.startedAt).toBeGreaterThan(0);
    });

    it('should transition from running to completed with results', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      run.status = 'running';
      run.startedAt = Date.now() - 1000;

      run.status = 'completed';
      run.completedAt = Date.now();
      run.results = {
        finalLoss: 2.5,
        finalAccuracy: 0.65,
        finalPerplexity: 12.0,
        trainingHistory: [
          { loss: 4.0, accuracy: 0.3, timestamp: run.startedAt! },
          { loss: 2.5, accuracy: 0.65, timestamp: run.completedAt }
        ]
      };

      expect(run.status).toBe('completed');
      expect(run.completedAt).toBeGreaterThan(run.startedAt!);
      expect(run.results.finalLoss).toBe(2.5);
    });

    it('should handle failed runs', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      run.status = 'running';
      run.startedAt = Date.now();

      run.status = 'failed';

      expect(run.status).toBe('failed');
      expect(run.results).toBeUndefined();
    });

    it('should handle stopped runs', () => {
      const ledger = createDecisionLedger('Test rationale');
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);

      run.status = 'running';
      run.status = 'stopped';

      expect(run.status).toBe('stopped');
    });
  });
});

describe('Decision Ledger Governance', () => {
  describe('Execution Status Computation', () => {
    it('should EXECUTE with valid ledger', () => {
      const ledger = createDecisionLedger('Valid rationale', 'test-witness');

      expect(computeExecutionStatus(ledger)).toBe('EXECUTE');
    });

    it('should HOLD when expired', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger = createDecisionLedger('Valid rationale', 'test-witness', pastDate.toISOString());

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });

    it('should ESCALATE when missing rationale', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        witness: 'test-witness',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should ESCALATE when missing witness', () => {
      const ledger: DecisionLedger = {
        rationale: 'Valid rationale',
        witness: '',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('ESCALATE');
    });

    it('should prioritize HOLD over ESCALATE', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger: DecisionLedger = {
        rationale: '', // Would cause ESCALATE
        witness: '',
        expiry: pastDate.toISOString(), // But expired -> HOLD
        rollback: 'keep',
        createdAt: Date.now()
      };

      expect(computeExecutionStatus(ledger)).toBe('HOLD');
    });
  });

  describe('Training Blocking', () => {
    it('should block training on HOLD status', () => {
      const pastDate = new Date();
      pastDate.setDate(pastDate.getDate() - 1);

      const ledger = createDecisionLedger('Expired decision', 'witness', pastDate.toISOString());
      const config = createMinimalConfig();
      const run = createRun('proj-1', 'Run', config, testCorpus, ledger);

      const status = computeExecutionStatus(run.decisionLedger);

      expect(status).toBe('HOLD');
      expect(status).not.toBe('EXECUTE');
    });

    it('should block training on ESCALATE status', () => {
      const ledger: DecisionLedger = {
        rationale: '',
        witness: 'witness',
        expiry: null,
        rollback: 'keep',
        createdAt: Date.now()
      };
      const config = createMinimalConfig();
      const run = createRun('proj-1', 'Run', config, testCorpus, ledger);

      const status = computeExecutionStatus(run.decisionLedger);

      expect(status).toBe('ESCALATE');
      expect(status).not.toBe('EXECUTE');
    });
  });
});

describe('Scenario Evaluation', () => {
  it('should store scenario results on completed run', () => {
    const project = createProject('Test', 'desc');
    const scenario1 = createScenario('Greeting', 'hello', 'world');
    const scenario2 = createScenario('Farewell', 'goodbye', 'see you');
    project.scenarios.push(scenario1, scenario2);

    const ledger = createDecisionLedger('Test scenarios');
    const config = createMinimalConfig();
    const run = createRun(project.id, 'Run', config, testCorpus, ledger);

    run.status = 'completed';
    run.results = {
      finalLoss: 2.0,
      finalAccuracy: 0.7,
      finalPerplexity: 7.4,
      trainingHistory: [],
      scenarioResults: [
        {
          scenarioId: scenario1.id,
          response: 'hello world',
          score: 0.85,
          timestamp: Date.now()
        },
        {
          scenarioId: scenario2.id,
          response: 'goodbye friend',
          score: 0.6,
          timestamp: Date.now()
        }
      ]
    };

    expect(run.results.scenarioResults).toHaveLength(2);
    expect(run.results.scenarioResults![0].score).toBe(0.85);
    expect(run.results.scenarioResults![1].score).toBe(0.6);
  });

  it('should update scenario last score', () => {
    const scenario = createScenario('Test', 'prompt', 'expected');

    expect(scenario.lastScore).toBeUndefined();

    scenario.lastScore = 0.9;
    scenario.lastRunAt = Date.now();

    expect(scenario.lastScore).toBe(0.9);
    expect(scenario.lastRunAt).toBeGreaterThan(0);
  });
});

describe('Export/Import Integration', () => {
  describe('Full Project Export', () => {
    it('should export complete project with runs and scenarios', () => {
      const project = createProject('Export Test', 'Testing export');
      const scenario = createScenario('Test Scenario', 'hello');
      project.scenarios.push(scenario);

      const ledger = createDecisionLedger('Export test run');
      const config = createMinimalConfig();
      const run1 = createRun(project.id, 'Export Run 1', config, testCorpus, ledger);
      run1.status = 'completed';
      run1.results = {
        finalLoss: 2.0,
        finalAccuracy: 0.7,
        finalPerplexity: 7.4,
        trainingHistory: [{ loss: 2.0, accuracy: 0.7, timestamp: Date.now() }],
        scenarioResults: [{ scenarioId: scenario.id, response: 'hello world', score: 0.8, timestamp: Date.now() }]
      };
      project.runIds.push(run1.id);

      const run2 = createRun(project.id, 'Export Run 2', config, testCorpus, ledger);
      run2.status = 'completed';
      run2.results = {
        finalLoss: 1.8,
        finalAccuracy: 0.75,
        finalPerplexity: 6.0,
        trainingHistory: [{ loss: 1.8, accuracy: 0.75, timestamp: Date.now() }]
      };
      project.runIds.push(run2.id);

      const comparison = createExperimentComparison(project.id, 'Test Comparison', 'desc', [run1.id, run2.id]);

      const decision = createDecisionEntry(project.id, 'Problem', ['alt1'], 'Decision', 'loss', [run1.id]);

      const exportData = createProjectExport([project], [run1, run2], [comparison], [decision]);

      expect(exportData.projects).toHaveLength(1);
      expect(exportData.runs).toHaveLength(2);
      expect(exportData.comparisons).toHaveLength(1);
      expect(exportData.decisions).toHaveLength(1);
      expect(exportData.schemaVersion).toBeDefined();
      expect(exportData.metadata.source).toBe('neuro-lingua-domestica');
    });

    it('should round-trip export/import preserving all data', () => {
      const project = createProject('Round Trip Test', 'Testing round trip');
      const scenario = createScenario('Scenario', 'prompt', 'expected');
      project.scenarios.push(scenario);

      const ledger = createDecisionLedger('Round trip run', 'witness');
      const config = createMinimalConfig({ learningRate: 0.1, hiddenSize: 32 });
      const run = createRun(project.id, 'Run', config, testCorpus, ledger);
      run.status = 'completed';
      run.results = {
        finalLoss: 1.8,
        finalAccuracy: 0.75,
        finalPerplexity: 6.0,
        trainingHistory: [
          { loss: 3.0, accuracy: 0.5, timestamp: Date.now() - 1000 },
          { loss: 1.8, accuracy: 0.75, timestamp: Date.now() }
        ],
        scenarioResults: [{ scenarioId: scenario.id, response: 'response', score: 0.9, timestamp: Date.now() }]
      };
      project.runIds.push(run.id);

      const exportData = createProjectExport([project], [run], [], []);
      const json = JSON.stringify(exportData);
      const imported = importProjectExport(json);

      // Verify project preserved
      expect(imported.projects).toHaveLength(1);
      expect(imported.projects[0].name).toBe('Round Trip Test');
      expect(imported.projects[0].scenarios).toHaveLength(1);
      expect(imported.projects[0].scenarios[0].prompt).toBe('prompt');

      // Verify run preserved
      expect(imported.runs).toHaveLength(1);
      expect(imported.runs[0].config.learningRate).toBe(0.1);
      expect(imported.runs[0].config.hiddenSize).toBe(32);
      expect(imported.runs[0].decisionLedger.rationale).toBe('Round trip run');

      // Verify results preserved
      expect(imported.runs[0].results?.finalLoss).toBe(1.8);
      expect(imported.runs[0].results?.scenarioResults).toHaveLength(1);
    });
  });

  describe('CSV Export', () => {
    it('should export runs to CSV with all fields', () => {
      const project = createProject('CSV Test', 'desc');
      const ledger = createDecisionLedger('CSV test');
      const config = createMinimalConfig({ hiddenSize: 64, learningRate: 0.05 });
      const run = createRun(project.id, 'Run 1', config, testCorpus, ledger);
      run.status = 'completed';
      run.completedAt = Date.now();
      run.results = {
        finalLoss: 2.5,
        finalAccuracy: 0.6,
        finalPerplexity: 12.0,
        trainingHistory: []
      };

      const csv = exportRunsToCSV([run], [project]);
      const lines = csv.split('\n');

      // Header should include all fields
      expect(lines[0]).toContain('runId');
      expect(lines[0]).toContain('hiddenSize');
      expect(lines[0]).toContain('finalLoss');
      expect(lines[0]).toContain('decisionRationale');

      // Data row should have correct values
      expect(lines[1]).toContain('Run 1');
      expect(lines[1]).toContain('64');
      expect(lines[1]).toContain('CSV Test');
    });
  });

  describe('Trace Export', () => {
    it('should create valid trace with all Σ-SIG metadata', () => {
      const ledger = createDecisionLedger('Trace test', 'witness');
      const config = createMinimalConfig();
      const run = createRun('proj-1', 'Trace Run', config, testCorpus, ledger);

      run.results = {
        finalLoss: 2.0,
        finalAccuracy: 0.7,
        finalPerplexity: 7.4,
        trainingHistory: [
          { loss: 3.0, accuracy: 0.5, timestamp: Date.now() - 1000 },
          { loss: 2.0, accuracy: 0.7, timestamp: Date.now() }
        ],
        scenarioResults: [{ scenarioId: 's1', response: 'output', score: 0.85, timestamp: Date.now() }]
      };

      const modelData = {
        weights: { embedding: [[1, 2]], hidden: [[3, 4]] },
        config: { hiddenSize: 16 },
        tokenizer: { vocab: ['a', 'b'] }
      };

      const trainingHistory = run.results.trainingHistory;
      const finalStats = {
        loss: run.results.finalLoss,
        acc: run.results.finalAccuracy,
        ppl: run.results.finalPerplexity
      };

      const trace = createTraceExport(modelData, run, trainingHistory, finalStats, testCorpus);

      expect(validateTraceExport(trace)).toBe(true);
      expect(trace.decisionLedger?.rationale).toBe('Trace test');
      expect(trace.decisionLedger?.witness).toBe('witness');
      expect(trace.trainingTrace?.epochs).toBe(2);
      expect(trace.trainingTrace?.finalLoss).toBe(2.0);
      expect(trace.trainingTrace?.scenariosScores?.['s1']).toBe(0.85);
    });

    it('should preserve trace through JSON round-trip', () => {
      const ledger = createDecisionLedger('JSON test', 'witness');
      const config = createMinimalConfig();
      const run = createRun('proj-1', 'Run', config, testCorpus, ledger);

      run.results = {
        finalLoss: 1.5,
        finalAccuracy: 0.8,
        finalPerplexity: 4.5,
        trainingHistory: [{ loss: 1.5, accuracy: 0.8, timestamp: Date.now() }]
      };

      const modelData = {
        weights: { test: [[1, 2, 3]] },
        config: {},
        tokenizer: {}
      };

      const trace = createTraceExport(
        modelData,
        run,
        run.results.trainingHistory,
        { loss: 1.5, acc: 0.8, ppl: 4.5 },
        testCorpus
      );

      const json = JSON.stringify(trace);
      const parsed = JSON.parse(json);

      expect(validateTraceExport(parsed)).toBe(true);
      expect(parsed.decisionLedger.rationale).toBe('JSON test');
    });
  });

  describe('Export Summary', () => {
    it('should generate human-readable summary', () => {
      const project = createProject('Summary Test', 'desc');
      const ledger = createDecisionLedger('test');
      const config = createMinimalConfig();

      const run1 = createRun(project.id, 'Run 1', config, testCorpus, ledger);
      run1.status = 'completed';

      const run2 = createRun(project.id, 'Run 2', config, testCorpus, ledger);
      run2.status = 'pending';

      project.runIds.push(run1.id, run2.id);

      const comparison = createExperimentComparison(project.id, 'Comparison', 'desc', [run1.id, run2.id]);

      const exportData = createProjectExport([project], [run1, run2], [comparison], []);
      const summary = generateExportSummary(exportData);

      expect(summary).toContain('**Projects:** 1');
      expect(summary).toContain('**Runs:** 2');
      expect(summary).toContain('1 completed');
      expect(summary).toContain('**Comparisons:** 1');
      expect(summary).toContain('Summary Test');
    });
  });
});

describe('End-to-End Training Flow', () => {
  it('should complete full training lifecycle with model data', async () => {
    const vocab = createVocab(testCorpus);

    // Create project
    const project = createProject('E2E Test', 'End-to-end test');
    const scenario = createScenario('Hello Test', 'hello', 'world');
    project.scenarios.push(scenario);

    // Create run
    const ledger = createDecisionLedger('E2E training test', 'test-user');
    const config = createMinimalConfig({ epochs: 3, hiddenSize: 8 });
    const run = createRun(project.id, 'Training Run', config, testCorpus, ledger);
    project.runIds.push(run.id);

    // Verify execution is allowed
    expect(computeExecutionStatus(run.decisionLedger)).toBe('EXECUTE');

    // Start training
    run.status = 'running';
    run.startedAt = Date.now();

    // Create and train model
    const model = new ProNeuralLM(vocab, config.hiddenSize, config.learningRate, config.contextSize);

    const trainingHistory: Array<{ loss: number; accuracy: number; timestamp: number }> = [];

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      const result = await model.train(testCorpus, 1);
      trainingHistory.push({
        loss: result.loss,
        accuracy: result.accuracy,
        timestamp: Date.now()
      });
    }

    // Evaluate scenario
    const scenarioResponse = await model.generate(scenario.prompt, 5);
    const scenarioScore = scenario.expectedResponse && scenarioResponse.includes(scenario.expectedResponse) ? 1.0 : 0.5;

    // Complete run
    run.status = 'completed';
    run.completedAt = Date.now();
    run.results = {
      finalLoss: trainingHistory[trainingHistory.length - 1].loss,
      finalAccuracy: trainingHistory[trainingHistory.length - 1].accuracy,
      finalPerplexity: Math.exp(trainingHistory[trainingHistory.length - 1].loss),
      trainingHistory,
      scenarioResults: [
        {
          scenarioId: scenario.id,
          response: scenarioResponse,
          score: scenarioScore,
          timestamp: Date.now()
        }
      ]
    };

    // Save model data
    run.modelData = model.toJSON();

    // Verify run completed successfully
    expect(run.status).toBe('completed');
    expect(run.results.finalLoss).toBeLessThan(10);
    expect(run.results.trainingHistory).toHaveLength(3);
    expect(run.results.scenarioResults).toHaveLength(1);
    expect(run.modelData).toBeDefined();

    // Export and verify
    const exportData = createProjectExport([project], [run], [], []);
    expect(exportData.runs[0].modelData).toBeDefined();

    // Verify model can be restored from export
    const importedExport = importProjectExport(JSON.stringify(exportData));
    expect(importedExport.runs[0].modelData).toBeDefined();
  });
});
