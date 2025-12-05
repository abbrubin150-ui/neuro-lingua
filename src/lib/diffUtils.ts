/**
 * Diff Utilities for Run Comparison
 * Generates structured diffs between training runs
 * Part of Σ-SIG Experiment Explorer
 */

import type { Run } from '../types/project';
import type {
  FieldDiff,
  DiffType,
  HyperparameterDiff,
  MetricsDiff,
  RunDiff
} from '../types/experiment';

/**
 * Create a field diff for any value type
 */
export function createFieldDiff<T>(
  field: string,
  oldValue: T | undefined,
  newValue: T | undefined
): FieldDiff<T> {
  let type: DiffType = 'unchanged';
  let percentChange: number | undefined;

  if (oldValue === undefined && newValue !== undefined) {
    type = 'added';
  } else if (oldValue !== undefined && newValue === undefined) {
    type = 'removed';
  } else if (oldValue !== newValue) {
    type = 'changed';

    // Calculate percent change for numbers
    if (typeof oldValue === 'number' && typeof newValue === 'number') {
      if (oldValue !== 0) {
        percentChange = ((newValue - oldValue) / Math.abs(oldValue)) * 100;
      } else if (newValue !== 0) {
        percentChange = 100; // Changed from 0 to non-zero
      }
    }
  }

  return {
    field,
    type,
    oldValue,
    newValue,
    percentChange
  };
}

/**
 * Compare hyperparameters between two runs
 */
export function diffHyperparameters(baseRun: Run, compareRun: Run): HyperparameterDiff {
  const baseConfig = baseRun.config;
  const compareConfig = compareRun.config;

  const diff: HyperparameterDiff = {
    architecture: createFieldDiff('architecture', baseConfig.architecture, compareConfig.architecture),
    hiddenSize: createFieldDiff('hiddenSize', baseConfig.hiddenSize, compareConfig.hiddenSize),
    epochs: createFieldDiff('epochs', baseConfig.epochs, compareConfig.epochs),
    learningRate: createFieldDiff('learningRate', baseConfig.learningRate, compareConfig.learningRate),
    optimizer: createFieldDiff('optimizer', baseConfig.optimizer, compareConfig.optimizer),
    momentum: createFieldDiff('momentum', baseConfig.momentum, compareConfig.momentum),
    dropout: createFieldDiff('dropout', baseConfig.dropout, compareConfig.dropout),
    contextSize: createFieldDiff('contextSize', baseConfig.contextSize, compareConfig.contextSize),
    seed: createFieldDiff('seed', baseConfig.seed, compareConfig.seed),
    useAdvanced: createFieldDiff('useAdvanced', baseConfig.useAdvanced, compareConfig.useAdvanced),
    useGPU: createFieldDiff('useGPU', baseConfig.useGPU, compareConfig.useGPU)
  };

  // Add advanced features if present
  if (baseConfig.activation || compareConfig.activation) {
    diff.activation = createFieldDiff('activation', baseConfig.activation, compareConfig.activation);
  }
  if (baseConfig.initialization || compareConfig.initialization) {
    diff.initialization = createFieldDiff('initialization', baseConfig.initialization, compareConfig.initialization);
  }
  if (baseConfig.lrSchedule || compareConfig.lrSchedule) {
    diff.lrSchedule = createFieldDiff('lrSchedule', baseConfig.lrSchedule, compareConfig.lrSchedule);
  }
  if (baseConfig.weightDecay !== undefined || compareConfig.weightDecay !== undefined) {
    diff.weightDecay = createFieldDiff('weightDecay', baseConfig.weightDecay, compareConfig.weightDecay);
  }
  if (baseConfig.useLayerNorm !== undefined || compareConfig.useLayerNorm !== undefined) {
    diff.useLayerNorm = createFieldDiff('useLayerNorm', baseConfig.useLayerNorm, compareConfig.useLayerNorm);
  }

  // Add transformer features if present
  if (baseConfig.numHeads !== undefined || compareConfig.numHeads !== undefined) {
    diff.numHeads = createFieldDiff('numHeads', baseConfig.numHeads, compareConfig.numHeads);
  }
  if (baseConfig.numLayers !== undefined || compareConfig.numLayers !== undefined) {
    diff.numLayers = createFieldDiff('numLayers', baseConfig.numLayers, compareConfig.numLayers);
  }

  return diff;
}

/**
 * Compare metrics between two runs
 */
export function diffMetrics(baseRun: Run, compareRun: Run): MetricsDiff {
  const baseResults = baseRun.results;
  const compareResults = compareRun.results;

  const diff: MetricsDiff = {
    finalLoss: createFieldDiff(
      'finalLoss',
      baseResults?.finalLoss,
      compareResults?.finalLoss
    ),
    finalAccuracy: createFieldDiff(
      'finalAccuracy',
      baseResults?.finalAccuracy,
      compareResults?.finalAccuracy
    ),
    finalPerplexity: createFieldDiff(
      'finalPerplexity',
      baseResults?.finalPerplexity,
      compareResults?.finalPerplexity
    )
  };

  // Calculate training time if both runs have timestamps
  if (baseRun.startedAt && baseRun.completedAt && compareRun.startedAt && compareRun.completedAt) {
    const baseTime = baseRun.completedAt - baseRun.startedAt;
    const compareTime = compareRun.completedAt - compareRun.startedAt;
    diff.trainingTime = createFieldDiff('trainingTime', baseTime, compareTime);
  }

  return diff;
}

/**
 * Determine improvement direction based on metric changes
 */
export function determineImprovementDirection(
  metricsDiff: MetricsDiff
): 'better' | 'worse' | 'mixed' | 'unknown' {
  const changes: ('better' | 'worse' | 'unchanged')[] = [];

  // Lower loss is better
  if (metricsDiff.finalLoss.type === 'changed' && metricsDiff.finalLoss.newValue !== undefined && metricsDiff.finalLoss.oldValue !== undefined) {
    changes.push(metricsDiff.finalLoss.newValue < metricsDiff.finalLoss.oldValue ? 'better' : 'worse');
  }

  // Higher accuracy is better
  if (metricsDiff.finalAccuracy.type === 'changed' && metricsDiff.finalAccuracy.newValue !== undefined && metricsDiff.finalAccuracy.oldValue !== undefined) {
    changes.push(metricsDiff.finalAccuracy.newValue > metricsDiff.finalAccuracy.oldValue ? 'better' : 'worse');
  }

  // Lower perplexity is better
  if (metricsDiff.finalPerplexity.type === 'changed' && metricsDiff.finalPerplexity.newValue !== undefined && metricsDiff.finalPerplexity.oldValue !== undefined) {
    changes.push(metricsDiff.finalPerplexity.newValue < metricsDiff.finalPerplexity.oldValue ? 'better' : 'worse');
  }

  if (changes.length === 0) {
    return 'unknown';
  }

  const betterCount = changes.filter(c => c === 'better').length;
  const worseCount = changes.filter(c => c === 'worse').length;

  if (betterCount > 0 && worseCount === 0) return 'better';
  if (worseCount > 0 && betterCount === 0) return 'worse';
  if (betterCount > 0 && worseCount > 0) return 'mixed';
  return 'unknown';
}

/**
 * Find significant changes (>10% or important categorical changes)
 */
export function findSignificantChanges(diff: HyperparameterDiff): string[] {
  const significant: string[] = [];
  const THRESHOLD = 10; // 10% change threshold

  // Check all fields in hyperparameters
  const fields = Object.entries(diff) as [string, FieldDiff<unknown>][];

  for (const [key, fieldDiff] of fields) {
    if (fieldDiff.type === 'changed') {
      // Numeric changes > 10%
      if (fieldDiff.percentChange !== undefined && Math.abs(fieldDiff.percentChange) > THRESHOLD) {
        significant.push(`${key}: ${fieldDiff.percentChange > 0 ? '+' : ''}${fieldDiff.percentChange.toFixed(1)}%`);
      }
      // Important categorical changes
      else if (['architecture', 'optimizer', 'activation', 'lrSchedule', 'initialization'].includes(key)) {
        significant.push(`${key}: ${fieldDiff.oldValue} → ${fieldDiff.newValue}`);
      }
    } else if (fieldDiff.type === 'added') {
      significant.push(`${key}: added (${fieldDiff.newValue})`);
    } else if (fieldDiff.type === 'removed') {
      significant.push(`${key}: removed`);
    }
  }

  return significant;
}

/**
 * Count total changes in hyperparameters
 */
export function countChanges(diff: HyperparameterDiff): number {
  const fields = Object.values(diff) as FieldDiff<unknown>[];
  return fields.filter(f => f.type !== 'unchanged').length;
}

/**
 * Generate a complete RunDiff between two runs
 */
export function generateRunDiff(baseRun: Run, compareRun: Run): RunDiff {
  const hyperparameterDiff = diffHyperparameters(baseRun, compareRun);
  const metricsDiff = diffMetrics(baseRun, compareRun);
  const corpusChanged = baseRun.corpusChecksum !== compareRun.corpusChecksum;

  const totalChanges = countChanges(hyperparameterDiff);
  const significantChanges = findSignificantChanges(hyperparameterDiff);
  const improvementDirection = determineImprovementDirection(metricsDiff);

  return {
    baseRun,
    compareRun,
    hyperparameters: hyperparameterDiff,
    metrics: metricsDiff,
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
 * Generate multiple diffs for multi-run comparison
 * Each run is compared against the base (first) run
 */
export function generateMultiRunDiffs(runs: Run[]): RunDiff[] {
  if (runs.length < 2) {
    throw new Error('At least 2 runs required for comparison');
  }

  const baseRun = runs[0];
  const diffs: RunDiff[] = [];

  for (let i = 1; i < runs.length; i++) {
    diffs.push(generateRunDiff(baseRun, runs[i]));
  }

  return diffs;
}

/**
 * Format a field diff for display
 */
export function formatFieldDiff<T>(diff: FieldDiff<T>): string {
  switch (diff.type) {
    case 'unchanged':
      return String(diff.oldValue);
    case 'added':
      return `➕ ${diff.newValue}`;
    case 'removed':
      return `➖ ${diff.oldValue}`;
    case 'changed':
      if (diff.percentChange !== undefined) {
        const sign = diff.percentChange > 0 ? '+' : '';
        return `${diff.oldValue} → ${diff.newValue} (${sign}${diff.percentChange.toFixed(1)}%)`;
      }
      return `${diff.oldValue} → ${diff.newValue}`;
    default:
      return '';
  }
}
