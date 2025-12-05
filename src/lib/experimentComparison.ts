/**
 * Experiment Comparison Data Layer
 * Functions for comparing runs, computing diffs, and exporting data
 * Part of Σ-SIG Experiment Explorer - Milestone 1: Data Layer
 */

import type {
  Run,
  Project,
  TrainingConfig
} from '../types/project';

import type {
  ExperimentComparison,
  DecisionEntry,
  FieldDiff,
  DiffType,
  HyperparameterDiff,
  MetricsDiff,
  RunDiff,
  MultiRunComparison,
  ProjectExport,
  RunCSVRow,
  ComparisonCSVRow,
  DecisionCSVRow,
  EXPORT_SCHEMA_VERSION
} from '../types/experiment';

const EXPORT_SCHEMA_VERSION_CONST = '1.0.0';

/**
 * Compute field difference between two values
 */
function computeFieldDiff<T>(
  field: string,
  oldValue: T | undefined,
  newValue: T | undefined
): FieldDiff<T> {
  const hasOld = oldValue !== undefined;
  const hasNew = newValue !== undefined;

  let type: DiffType;
  if (!hasOld && hasNew) {
    type = 'added';
  } else if (hasOld && !hasNew) {
    type = 'removed';
  } else if (oldValue !== newValue) {
    type = 'changed';
  } else {
    type = 'unchanged';
  }

  const diff: FieldDiff<T> = {
    field,
    type,
    oldValue: hasOld ? oldValue : undefined,
    newValue: hasNew ? newValue : undefined
  };

  // Calculate percent change for numbers
  if (
    type === 'changed' &&
    typeof oldValue === 'number' &&
    typeof newValue === 'number' &&
    oldValue !== 0
  ) {
    diff.percentChange = ((newValue - oldValue) / oldValue) * 100;
  }

  return diff;
}

/**
 * Compute hyperparameter differences between two training configs
 */
export function computeHyperparameterDiff(
  baseConfig: TrainingConfig,
  compareConfig: TrainingConfig
): HyperparameterDiff {
  const diff: HyperparameterDiff = {
    architecture: computeFieldDiff('architecture', baseConfig.architecture, compareConfig.architecture),
    hiddenSize: computeFieldDiff('hiddenSize', baseConfig.hiddenSize, compareConfig.hiddenSize),
    epochs: computeFieldDiff('epochs', baseConfig.epochs, compareConfig.epochs),
    learningRate: computeFieldDiff('learningRate', baseConfig.learningRate, compareConfig.learningRate),
    optimizer: computeFieldDiff('optimizer', baseConfig.optimizer, compareConfig.optimizer),
    momentum: computeFieldDiff('momentum', baseConfig.momentum, compareConfig.momentum),
    dropout: computeFieldDiff('dropout', baseConfig.dropout, compareConfig.dropout),
    contextSize: computeFieldDiff('contextSize', baseConfig.contextSize, compareConfig.contextSize),
    seed: computeFieldDiff('seed', baseConfig.seed, compareConfig.seed),
    useAdvanced: computeFieldDiff('useAdvanced', baseConfig.useAdvanced, compareConfig.useAdvanced),
    useGPU: computeFieldDiff('useGPU', baseConfig.useGPU, compareConfig.useGPU)
  };

  // Add optional fields if present in either config
  if (baseConfig.activation || compareConfig.activation) {
    diff.activation = computeFieldDiff('activation', baseConfig.activation, compareConfig.activation);
  }
  if (baseConfig.initialization || compareConfig.initialization) {
    diff.initialization = computeFieldDiff('initialization', baseConfig.initialization, compareConfig.initialization);
  }
  if (baseConfig.lrSchedule || compareConfig.lrSchedule) {
    diff.lrSchedule = computeFieldDiff('lrSchedule', baseConfig.lrSchedule, compareConfig.lrSchedule);
  }
  if (baseConfig.weightDecay !== undefined || compareConfig.weightDecay !== undefined) {
    diff.weightDecay = computeFieldDiff('weightDecay', baseConfig.weightDecay, compareConfig.weightDecay);
  }
  if (baseConfig.useLayerNorm !== undefined || compareConfig.useLayerNorm !== undefined) {
    diff.useLayerNorm = computeFieldDiff('useLayerNorm', baseConfig.useLayerNorm, compareConfig.useLayerNorm);
  }

  // Transformer-specific fields
  if (baseConfig.numHeads !== undefined || compareConfig.numHeads !== undefined) {
    diff.numHeads = computeFieldDiff('numHeads', baseConfig.numHeads, compareConfig.numHeads);
  }
  if (baseConfig.numLayers !== undefined || compareConfig.numLayers !== undefined) {
    diff.numLayers = computeFieldDiff('numLayers', baseConfig.numLayers, compareConfig.numLayers);
  }

  return diff;
}

/**
 * Compute metrics differences between two runs
 */
export function computeMetricsDiff(
  baseRun: Run,
  compareRun: Run
): MetricsDiff {
  const baseResults = baseRun.results;
  const compareResults = compareRun.results;

  const diff: MetricsDiff = {
    finalLoss: computeFieldDiff('finalLoss', baseResults?.finalLoss, compareResults?.finalLoss),
    finalAccuracy: computeFieldDiff('finalAccuracy', baseResults?.finalAccuracy, compareResults?.finalAccuracy),
    finalPerplexity: computeFieldDiff('finalPerplexity', baseResults?.finalPerplexity, compareResults?.finalPerplexity)
  };

  // Add training time if both completed
  if (baseRun.completedAt && baseRun.startedAt && compareRun.completedAt && compareRun.startedAt) {
    const baseTime = baseRun.completedAt - baseRun.startedAt;
    const compareTime = compareRun.completedAt - compareRun.startedAt;
    diff.trainingTime = computeFieldDiff('trainingTime', baseTime, compareTime);
  }

  // Add model size if available (modelData serialized length)
  if (baseRun.modelData && compareRun.modelData) {
    const baseSize = JSON.stringify(baseRun.modelData).length;
    const compareSize = JSON.stringify(compareRun.modelData).length;
    diff.modelSize = computeFieldDiff('modelSize', baseSize, compareSize);
  }

  return diff;
}

/**
 * Determine improvement direction based on metrics
 */
function determineImprovementDirection(metrics: MetricsDiff): 'better' | 'worse' | 'mixed' | 'unknown' {
  const changes: ('better' | 'worse' | 'neutral')[] = [];

  // Loss: lower is better
  if (metrics.finalLoss.type === 'changed' && metrics.finalLoss.percentChange !== undefined) {
    changes.push(metrics.finalLoss.percentChange < 0 ? 'better' : 'worse');
  }

  // Accuracy: higher is better
  if (metrics.finalAccuracy.type === 'changed' && metrics.finalAccuracy.percentChange !== undefined) {
    changes.push(metrics.finalAccuracy.percentChange > 0 ? 'better' : 'worse');
  }

  // Perplexity: lower is better
  if (metrics.finalPerplexity.type === 'changed' && metrics.finalPerplexity.percentChange !== undefined) {
    changes.push(metrics.finalPerplexity.percentChange < 0 ? 'better' : 'worse');
  }

  if (changes.length === 0) return 'unknown';

  const betterCount = changes.filter((c) => c === 'better').length;
  const worseCount = changes.filter((c) => c === 'worse').length;

  if (betterCount > 0 && worseCount === 0) return 'better';
  if (worseCount > 0 && betterCount === 0) return 'worse';
  if (betterCount > 0 && worseCount > 0) return 'mixed';

  return 'unknown';
}

/**
 * Identify significant changes (>10% change in numeric values)
 */
function identifySignificantChanges(
  hyperparameters: HyperparameterDiff,
  metrics: MetricsDiff
): string[] {
  const significant: string[] = [];
  const threshold = 10; // 10% change threshold

  // Check hyperparameters
  Object.entries(hyperparameters).forEach(([key, diff]) => {
    if (diff.percentChange !== undefined && Math.abs(diff.percentChange) > threshold) {
      significant.push(`${key}: ${diff.percentChange > 0 ? '+' : ''}${diff.percentChange.toFixed(1)}%`);
    } else if (diff.type === 'changed' && diff.percentChange === undefined) {
      // Non-numeric change (e.g., architecture, optimizer)
      significant.push(`${key}: ${diff.oldValue} → ${diff.newValue}`);
    }
  });

  // Check metrics
  Object.entries(metrics).forEach(([key, diff]) => {
    if (diff.percentChange !== undefined && Math.abs(diff.percentChange) > threshold) {
      significant.push(`${key}: ${diff.percentChange > 0 ? '+' : ''}${diff.percentChange.toFixed(1)}%`);
    }
  });

  return significant;
}

/**
 * Compute complete run difference
 */
export function computeRunDiff(baseRun: Run, compareRun: Run): RunDiff {
  const hyperparameters = computeHyperparameterDiff(baseRun.config, compareRun.config);
  const metrics = computeMetricsDiff(baseRun, compareRun);
  const corpusChanged = baseRun.corpusChecksum !== compareRun.corpusChecksum;

  const allDiffs = { ...hyperparameters, ...metrics };
  const totalChanges = Object.values(allDiffs).filter(
    (diff) => diff && diff.type === 'changed'
  ).length;

  const significantChanges = identifySignificantChanges(hyperparameters, metrics);
  const improvementDirection = determineImprovementDirection(metrics);

  return {
    baseRun,
    compareRun,
    hyperparameters,
    metrics,
    corpusChanged,
    summary: {
      totalChanges,
      significantChanges,
      improvementDirection
    },
    computedAt: Date.now()
  };
}

/**
 * Compute multi-run comparison (2-3 runs)
 */
export function computeMultiRunComparison(
  comparison: ExperimentComparison,
  runs: Run[]
): MultiRunComparison {
  if (runs.length < 2 || runs.length > 3) {
    throw new Error('Multi-run comparison requires 2-3 runs');
  }

  // First run is the baseline
  const baseRun = runs[0];
  const diffs: RunDiff[] = runs.slice(1).map((run) => computeRunDiff(baseRun, run));

  return {
    comparison,
    diffs,
    runs
  };
}

/**
 * Export project data to JSON
 */
export function exportProjectToJSON(
  projects: Project[],
  runs: Run[],
  comparisons: ExperimentComparison[],
  decisions: DecisionEntry[]
): ProjectExport {
  return {
    schemaVersion: EXPORT_SCHEMA_VERSION_CONST,
    metadata: {
      exportedAt: Date.now(),
      exportedBy: 'local-user',
      source: 'neuro-lingua-domestica',
      version: '3.3.0' // TODO: Get from package.json
    },
    projects,
    runs,
    comparisons,
    decisions
  };
}

/**
 * Download JSON export
 */
export function downloadProjectJSON(exportData: ProjectExport, filename: string = 'neuro-lingua-export.json') {
  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/**
 * Convert runs to CSV rows
 */
export function convertRunsToCSVRows(runs: Run[], projects: Map<string, Project>): RunCSVRow[] {
  return runs.map((run) => {
    const project = projects.get(run.projectId);
    return {
      runId: run.id,
      runName: run.name,
      projectId: run.projectId,
      projectName: project?.name || 'Unknown',
      status: run.status,
      createdAt: new Date(run.createdAt).toISOString(),
      completedAt: run.completedAt ? new Date(run.completedAt).toISOString() : '',
      architecture: run.config.architecture,
      hiddenSize: run.config.hiddenSize,
      epochs: run.config.epochs,
      learningRate: run.config.learningRate,
      optimizer: run.config.optimizer,
      momentum: run.config.momentum,
      dropout: run.config.dropout,
      contextSize: run.config.contextSize,
      useGPU: run.config.useGPU,
      finalLoss: run.results?.finalLoss || 0,
      finalAccuracy: run.results?.finalAccuracy || 0,
      finalPerplexity: run.results?.finalPerplexity || 0,
      corpusLength: run.corpus.length,
      corpusChecksum: run.corpusChecksum,
      decisionRationale: run.decisionLedger.rationale,
      decisionWitness: run.decisionLedger.witness
    };
  });
}

/**
 * Convert comparisons to CSV rows
 */
export function convertComparisonsToCSVRows(
  comparisons: ExperimentComparison[],
  projects: Map<string, Project>
): ComparisonCSVRow[] {
  return comparisons.map((comp) => {
    const project = projects.get(comp.projectId);
    return {
      comparisonId: comp.id,
      comparisonName: comp.name,
      projectName: project?.name || 'Unknown',
      runCount: comp.runIds.length,
      runIds: comp.runIds.join(', '),
      createdAt: new Date(comp.createdAt).toISOString(),
      notes: comp.notes || ''
    };
  });
}

/**
 * Convert decisions to CSV rows
 */
export function convertDecisionsToCSVRows(
  decisions: DecisionEntry[],
  projects: Map<string, Project>
): DecisionCSVRow[] {
  return decisions.map((decision) => {
    const project = projects.get(decision.projectId);
    return {
      decisionId: decision.id,
      projectName: project?.name || 'Unknown',
      problem: decision.problem,
      decision: decision.decision,
      kpi: decision.kpi,
      alternatives: decision.alternatives.join(', '),
      affectedRuns: decision.affectedRunIds.join(', '),
      category: decision.category || '',
      witness: decision.witness,
      createdAt: new Date(decision.createdAt).toISOString()
    };
  });
}

/**
 * Convert CSV rows to CSV string
 */
function rowsToCSVString<T extends Record<string, string | number | boolean>>(
  rows: T[],
  headers: string[]
): string {
  if (rows.length === 0) return '';

  // CSV header
  const csvLines = [headers.join(',')];

  // CSV rows
  rows.forEach((row) => {
    const values = headers.map((header) => {
      const value = row[header];
      if (value === undefined || value === null) return '';

      // Escape values containing commas, quotes, or newlines
      const stringValue = String(value);
      if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
        return `"${stringValue.replace(/"/g, '""')}"`;
      }
      return stringValue;
    });
    csvLines.push(values.join(','));
  });

  return csvLines.join('\n');
}

/**
 * Export runs to CSV
 */
export function exportRunsToCSV(runs: Run[], projects: Map<string, Project>): string {
  const rows = convertRunsToCSVRows(runs, projects);
  const headers = [
    'runId',
    'runName',
    'projectId',
    'projectName',
    'status',
    'createdAt',
    'completedAt',
    'architecture',
    'hiddenSize',
    'epochs',
    'learningRate',
    'optimizer',
    'momentum',
    'dropout',
    'contextSize',
    'useGPU',
    'finalLoss',
    'finalAccuracy',
    'finalPerplexity',
    'corpusLength',
    'corpusChecksum',
    'decisionRationale',
    'decisionWitness'
  ];
  return rowsToCSVString(rows, headers);
}

/**
 * Export comparisons to CSV
 */
export function exportComparisonsToCSV(
  comparisons: ExperimentComparison[],
  projects: Map<string, Project>
): string {
  const rows = convertComparisonsToCSVRows(comparisons, projects);
  const headers = [
    'comparisonId',
    'comparisonName',
    'projectName',
    'runCount',
    'runIds',
    'createdAt',
    'notes'
  ];
  return rowsToCSVString(rows, headers);
}

/**
 * Export decisions to CSV
 */
export function exportDecisionsToCSV(
  decisions: DecisionEntry[],
  projects: Map<string, Project>
): string {
  const rows = convertDecisionsToCSVRows(decisions, projects);
  const headers = [
    'decisionId',
    'projectName',
    'problem',
    'decision',
    'kpi',
    'alternatives',
    'affectedRuns',
    'category',
    'witness',
    'createdAt'
  ];
  return rowsToCSVString(rows, headers);
}

/**
 * Download CSV file
 */
export function downloadCSV(csvContent: string, filename: string) {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
