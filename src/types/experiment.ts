/**
 * Experiment Explorer Types
 * Types for run comparison, structured decisions, and data export
 * Part of Σ-SIG Experiment Explorer epic
 */

import type { Run, Project, Scenario } from './project';

/**
 * Version identifier for export schema
 */
export const EXPORT_SCHEMA_VERSION = '1.0.0';

/**
 * Comparison Scenario - Saved configuration for comparing runs
 */
export interface ExperimentComparison {
  /** Unique identifier */
  id: string;
  /** Comparison name */
  name: string;
  /** Description of what's being compared */
  description: string;
  /** Project this comparison belongs to */
  projectId: string;
  /** Run IDs to compare (2-3 runs) */
  runIds: string[];
  /** When this comparison was created */
  createdAt: number;
  /** Last time this comparison was viewed */
  lastViewedAt?: number;
  /** User notes about this comparison */
  notes?: string;
}

/**
 * Decision Entry - Structured decision tracking (Decision Ledger 2.0)
 */
export interface DecisionEntry {
  /** Unique identifier */
  id: string;
  /** Project this decision belongs to */
  projectId: string;
  /** Problem statement - what issue prompted this decision */
  problem: string;
  /** List of alternatives that were considered */
  alternatives: string[];
  /** The chosen solution/approach */
  decision: string;
  /** Key Performance Indicator used to make the decision */
  kpi: string;
  /** Run IDs that implement this decision */
  affectedRunIds: string[];
  /** Who made the decision */
  witness: string;
  /** When this decision was made */
  createdAt: number;
  /** Category for filtering (e.g., 'compression', 'optimizer', 'architecture') */
  category?: string;
  /** Additional notes or rationale */
  notes?: string;
}

/**
 * Difference type - indicates how a value changed
 */
export type DiffType = 'added' | 'removed' | 'changed' | 'unchanged';

/**
 * Generic field difference
 */
export interface FieldDiff<T> {
  field: string;
  type: DiffType;
  oldValue?: T;
  newValue?: T;
  percentChange?: number; // For numeric values
}

/**
 * Hyperparameter difference between two runs
 */
export interface HyperparameterDiff {
  architecture: FieldDiff<string>;
  hiddenSize: FieldDiff<number>;
  epochs: FieldDiff<number>;
  learningRate: FieldDiff<number>;
  optimizer: FieldDiff<string>;
  momentum: FieldDiff<number>;
  dropout: FieldDiff<number>;
  contextSize: FieldDiff<number>;
  seed: FieldDiff<number>;
  useAdvanced: FieldDiff<boolean>;
  useGPU: FieldDiff<boolean>;
  // Advanced features (only if present)
  activation?: FieldDiff<string>;
  initialization?: FieldDiff<string>;
  lrSchedule?: FieldDiff<string>;
  weightDecay?: FieldDiff<number>;
  useLayerNorm?: FieldDiff<boolean>;
  // Transformer features (only if present)
  numHeads?: FieldDiff<number>;
  numLayers?: FieldDiff<number>;
}

/**
 * Metrics difference between two runs
 */
export interface MetricsDiff {
  finalLoss: FieldDiff<number>;
  finalAccuracy: FieldDiff<number>;
  finalPerplexity: FieldDiff<number>;
  trainingTime?: FieldDiff<number>; // If available
  modelSize?: FieldDiff<number>; // If available
}

/**
 * Complete run comparison result
 */
export interface RunDiff {
  /** Source run (baseline) */
  baseRun: Run;
  /** Target run (comparison) */
  compareRun: Run;
  /** Hyperparameter differences */
  hyperparameters: HyperparameterDiff;
  /** Metrics differences */
  metrics: MetricsDiff;
  /** Whether corpus changed */
  corpusChanged: boolean;
  /** Summary of changes */
  summary: {
    totalChanges: number;
    significantChanges: string[]; // List of fields with >10% change
    improvementDirection: 'better' | 'worse' | 'mixed' | 'unknown';
  };
  /** Timestamp when diff was computed */
  computedAt: number;
}

/**
 * Multi-run comparison (2-3 runs)
 */
export interface MultiRunComparison {
  /** Comparison metadata */
  comparison: ExperimentComparison;
  /** Individual run diffs (comparing each to the first run) */
  diffs: RunDiff[];
  /** Runs being compared */
  runs: Run[];
}

/**
 * Export data structure for projects
 */
export interface ProjectExport {
  /** Schema version for compatibility */
  schemaVersion: string;
  /** Export metadata */
  metadata: {
    exportedAt: number;
    exportedBy: string;
    source: 'neuro-lingua-domestica';
    version: string;
  };
  /** Projects included in export */
  projects: Project[];
  /** Runs included in export */
  runs: Run[];
  /** Comparisons included in export */
  comparisons: ExperimentComparison[];
  /** Decisions included in export */
  decisions: DecisionEntry[];
}

/**
 * CSV export row for runs (flattened structure)
 */
export interface RunCSVRow {
  // Identifiers
  runId: string;
  runName: string;
  projectId: string;
  projectName: string;
  status: string;
  createdAt: string;
  completedAt: string;
  // Config
  architecture: string;
  hiddenSize: number;
  epochs: number;
  learningRate: number;
  optimizer: string;
  momentum: number;
  dropout: number;
  contextSize: number;
  useGPU: boolean;
  // Results
  finalLoss: number;
  finalAccuracy: number;
  finalPerplexity: number;
  // Corpus
  corpusLength: number;
  corpusChecksum: string;
  // Decision
  decisionRationale: string;
  decisionWitness: string;
  // Index signature for CSV conversion
  [key: string]: string | number | boolean;
}

/**
 * CSV export row for comparisons
 */
export interface ComparisonCSVRow {
  comparisonId: string;
  comparisonName: string;
  projectName: string;
  runCount: number;
  runIds: string;
  createdAt: string;
  notes: string;
  // Index signature for CSV conversion
  [key: string]: string | number;
}

/**
 * CSV export row for decisions
 */
export interface DecisionCSVRow {
  decisionId: string;
  projectName: string;
  problem: string;
  decision: string;
  kpi: string;
  alternatives: string; // comma-separated
  affectedRuns: string; // comma-separated
  category: string;
  witness: string;
  createdAt: string;
  // Index signature for CSV conversion
  [key: string]: string;
}
