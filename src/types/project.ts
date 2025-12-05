/**
 * Project and Run management types for Neuro-Lingua DOMESTICA
 * Implements Σ-SIG compliant experiment tracking
 */

import type { Optimizer, TokenizerConfig } from '../lib/ProNeuralLM';
import type { ActivationFunction, LRSchedule, InitializationScheme } from '../lib/AdvancedNeuralLM';
import type { Architecture } from '../components/TrainingPanel';

/**
 * Decision Ledger - Tracks rationale and governance for each run
 * Based on Σ-SIG / EXACT1 framework
 */
export interface DecisionLedger {
  /** Why this training run was initiated */
  rationale: string;
  /** Who authorized/initiated the training */
  witness: string;
  /** Expiry date for this model (ISO 8601 format) */
  expiry: string | null;
  /** What to do after expiry */
  rollback: 'keep' | 'delete-after-expiry' | 'archive';
  /** Timestamp when ledger was created */
  createdAt: number;
}

/**
 * Execution status based on Decision Ledger
 */
export type ExecutionStatus = 'EXECUTE' | 'HOLD' | 'ESCALATE';

/**
 * Scenario for testing model behavior
 */
export interface Scenario {
  /** Unique identifier */
  id: string;
  /** Scenario name */
  name: string;
  /** Input prompt */
  prompt: string;
  /** Expected response (optional) */
  expectedResponse?: string;
  /** Last score (0-1, optional) */
  lastScore?: number;
  /** Timestamp of last run */
  lastRunAt?: number;
}

/**
 * Scenario result from a specific run
 */
export interface ScenarioResult {
  scenarioId: string;
  response: string;
  score: number;
  timestamp: number;
}

/**
 * Complete snapshot of training configuration
 */
export interface TrainingConfig {
  // Architecture
  architecture: Architecture;

  // Core hyperparameters
  hiddenSize: number;
  epochs: number;
  learningRate: number;
  optimizer: Optimizer;
  momentum: number;
  dropout: number;
  contextSize: number;
  seed: number;

  // Tokenizer
  tokenizerConfig: TokenizerConfig;

  // Advanced features (optional)
  useAdvanced: boolean;
  useGPU: boolean;
  activation?: ActivationFunction;
  leakyReluAlpha?: number;
  eluAlpha?: number;
  initialization?: InitializationScheme;
  lrSchedule?: LRSchedule;
  lrMin?: number;
  lrDecayRate?: number;
  warmupEpochs?: number;
  weightDecay?: number;
  gradientClipNorm?: number;
  useLayerNorm?: boolean;

  // Transformer-specific
  numHeads?: number;
  numLayers?: number;
  ffHiddenDim?: number;
  attentionDropout?: number;
  dropConnectRate?: number;
}

/**
 * Training Run - A single training execution with frozen configuration
 */
export interface Run {
  /** Unique identifier */
  id: string;
  /** Parent project ID */
  projectId: string;
  /** Run name/label */
  name: string;
  /** Snapshot of all training configuration */
  config: TrainingConfig;
  /** Corpus text used for training */
  corpus: string;
  /** SHA256 checksum of corpus (for traceability) */
  corpusChecksum: string;
  /** Decision ledger for this run */
  decisionLedger: DecisionLedger;
  /** Training results */
  results?: {
    finalLoss: number;
    finalAccuracy: number;
    finalPerplexity: number;
    trainingHistory: Array<{
      loss: number;
      accuracy: number;
      timestamp: number;
    }>;
    scenarioResults?: ScenarioResult[];
  };
  /** Model weights (JSON serialized) */
  modelData?: unknown;
  /** When this run was created */
  createdAt: number;
  /** When training started */
  startedAt?: number;
  /** When training completed */
  completedAt?: number;
  /** Current status */
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
}

/**
 * Corpus type classification
 */
export type CorpusType = 'plain-text' | 'dialogue-embedded';

/**
 * Project - Container for multiple training runs
 */
export interface Project {
  /** Unique identifier */
  id: string;
  /** Project name */
  name: string;
  /** Project description */
  description: string;
  /** Primary language of corpus */
  language: 'en' | 'he' | 'mixed';
  /** Default architecture for this project */
  defaultArchitecture: Architecture;
  /** Type of corpus */
  corpusType: CorpusType;
  /** Test scenarios for this project */
  scenarios: Scenario[];
  /** List of run IDs belonging to this project */
  runIds: string[];
  /** When this project was created */
  createdAt: number;
  /** Last modified timestamp */
  updatedAt: number;
  /** Project tags for organization */
  tags?: string[];
}

/**
 * Helper to compute execution status from Decision Ledger
 */
export function computeExecutionStatus(ledger: DecisionLedger): ExecutionStatus {
  // Check expiry
  if (ledger.expiry) {
    const expiryDate = new Date(ledger.expiry);
    const now = new Date();
    if (now > expiryDate) {
      return 'HOLD';
    }
  }

  // Check rationale exists
  if (!ledger.rationale || ledger.rationale.trim().length === 0) {
    return 'ESCALATE';
  }

  // Check witness exists
  if (!ledger.witness || ledger.witness.trim().length === 0) {
    return 'ESCALATE';
  }

  return 'EXECUTE';
}

/**
 * Generate SHA256-like checksum for corpus text
 * Note: This is a simple hash, not cryptographic SHA256
 */
export function generateCorpusChecksum(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  // Convert to hex and pad
  return Math.abs(hash).toString(16).padStart(8, '0');
}

/**
 * Create a new project with defaults
 */
export function createProject(
  name: string,
  description: string,
  language: 'en' | 'he' | 'mixed' = 'en',
  defaultArchitecture: Architecture = 'feedforward'
): Project {
  return {
    id: `proj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    name,
    description,
    language,
    defaultArchitecture,
    corpusType: 'plain-text',
    scenarios: [],
    runIds: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
    tags: []
  };
}

/**
 * Create a new scenario
 */
export function createScenario(name: string, prompt: string, expectedResponse?: string): Scenario {
  return {
    id: `scenario_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    name,
    prompt,
    expectedResponse
  };
}

/**
 * Create a new decision ledger
 */
export function createDecisionLedger(
  rationale: string,
  witness: string = 'local-user',
  expiry: string | null = null,
  rollback: 'keep' | 'delete-after-expiry' | 'archive' = 'keep'
): DecisionLedger {
  return {
    rationale,
    witness,
    expiry,
    rollback,
    createdAt: Date.now()
  };
}

/**
 * Create a new run
 */
export function createRun(
  projectId: string,
  name: string,
  config: TrainingConfig,
  corpus: string,
  decisionLedger: DecisionLedger
): Run {
  return {
    id: `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    projectId,
    name,
    config,
    corpus,
    corpusChecksum: generateCorpusChecksum(corpus),
    decisionLedger,
    createdAt: Date.now(),
    status: 'pending'
  };
}

/**
 * Create a new experiment comparison
 */
export function createExperimentComparison(
  projectId: string,
  name: string,
  description: string,
  runIds: string[]
): import('../types/experiment').ExperimentComparison {
  if (runIds.length < 2 || runIds.length > 3) {
    throw new Error('Comparison requires 2-3 runs');
  }

  return {
    id: `comp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    projectId,
    name,
    description,
    runIds,
    createdAt: Date.now()
  };
}

/**
 * Create a new decision entry (Decision Ledger 2.0)
 */
export function createDecisionEntry(
  projectId: string,
  problem: string,
  alternatives: string[],
  decision: string,
  kpi: string,
  affectedRunIds: string[],
  witness: string = 'local-user',
  category?: string
): import('../types/experiment').DecisionEntry {
  return {
    id: `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    projectId,
    problem,
    alternatives,
    decision,
    kpi,
    affectedRunIds,
    witness,
    createdAt: Date.now(),
    category
  };
}
