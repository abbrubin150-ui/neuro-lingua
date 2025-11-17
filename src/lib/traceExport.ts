/**
 * Trace Export - Enhanced model export with full traceability
 * Implements Σ-SIG compliant audit trail
 */

import type { Run, DecisionLedger, ScenarioResult } from '../types/project';
import { generateCorpusChecksum } from '../types/project';

/**
 * Enhanced model export format with full trace
 */
export interface TraceExport {
  // Model data (original format)
  modelWeights: unknown;
  config: unknown;
  tokenizer: unknown;

  // Project metadata
  projectMeta?: {
    projectId: string;
    projectName: string;
    runId: string;
    runName: string;
  };

  // Decision Ledger (Σ-SIG compliance)
  decisionLedger?: {
    rationale: string;
    witness: string;
    expiry: string | null;
    rollback: 'keep' | 'delete-after-expiry' | 'archive';
    createdAt: number;
  };

  // Training trace
  trainingTrace?: {
    epochs: number;
    finalLoss: number;
    finalAccuracy: number;
    finalPerplexity: number;
    trainLoss: number[];
    trainAccuracy: number[];
    timestamps: number[];
    tokensPerSecAvg?: number;
    scenariosScores?: Record<string, number>;
    sha256_corpus: string;
  };

  // Metadata
  exportedAt: number;
  exportedBy: string;
  version: string;
}

/**
 * Create enhanced trace export from model and run data
 */
export function createTraceExport(
  modelJSON: unknown,
  run?: Run,
  trainingHistory?: Array<{ loss: number; accuracy: number; timestamp: number }>,
  finalStats?: { loss: number; acc: number; ppl: number },
  corpus?: string,
  version: string = '3.2.4'
): TraceExport {
  const modelData = modelJSON as Record<string, unknown>;

  const trace: TraceExport = {
    modelWeights: modelData.weights || modelData,
    config: modelData.config || {},
    tokenizer: modelData.tokenizer || {},
    exportedAt: Date.now(),
    exportedBy: 'local-user',
    version
  };

  // Add project metadata if run exists
  if (run) {
    trace.projectMeta = {
      projectId: run.projectId,
      projectName: run.name,
      runId: run.id,
      runName: run.name
    };

    // Add decision ledger
    trace.decisionLedger = {
      rationale: run.decisionLedger.rationale,
      witness: run.decisionLedger.witness,
      expiry: run.decisionLedger.expiry,
      rollback: run.decisionLedger.rollback,
      createdAt: run.decisionLedger.createdAt
    };
  }

  // Add training trace if available
  if (trainingHistory && finalStats) {
    const trainLoss = trainingHistory.map((h) => h.loss);
    const trainAccuracy = trainingHistory.map((h) => h.accuracy);
    const timestamps = trainingHistory.map((h) => h.timestamp);

    trace.trainingTrace = {
      epochs: trainingHistory.length,
      finalLoss: finalStats.loss,
      finalAccuracy: finalStats.acc,
      finalPerplexity: finalStats.ppl,
      trainLoss,
      trainAccuracy,
      timestamps,
      sha256_corpus: corpus ? generateCorpusChecksum(corpus) : '00000000'
    };

    // Add scenario scores if available from run
    if (run?.results?.scenarioResults) {
      const scenariosScores: Record<string, number> = {};
      run.results.scenarioResults.forEach((result) => {
        scenariosScores[result.scenarioId] = result.score;
      });
      trace.trainingTrace.scenariosScores = scenariosScores;
    }
  }

  return trace;
}

/**
 * Generate filename for trace export
 */
export function generateTraceFilename(
  version: string,
  runName?: string,
  hash?: string
): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const versionSlug = version.replace(/\./g, '');
  const nameSlug = runName ? `-${runName.toLowerCase().replace(/\s+/g, '-')}` : '';
  const hashSlug = hash ? `-${hash.slice(0, 8)}` : '';

  return `neuro-lingua-v${versionSlug}${nameSlug}-${timestamp}${hashSlug}.json`;
}

/**
 * Validate trace export structure
 */
export function validateTraceExport(data: unknown): data is TraceExport {
  if (typeof data !== 'object' || data === null) return false;

  const trace = data as Partial<TraceExport>;

  // Required fields
  if (!trace.modelWeights || !trace.config || !trace.tokenizer) return false;
  if (typeof trace.exportedAt !== 'number') return false;
  if (typeof trace.exportedBy !== 'string') return false;
  if (typeof trace.version !== 'string') return false;

  // Optional but validated if present
  if (trace.decisionLedger) {
    const ledger = trace.decisionLedger;
    if (typeof ledger.rationale !== 'string') return false;
    if (typeof ledger.witness !== 'string') return false;
    if (typeof ledger.createdAt !== 'number') return false;
  }

  if (trace.trainingTrace) {
    const tt = trace.trainingTrace;
    if (typeof tt.epochs !== 'number') return false;
    if (typeof tt.finalLoss !== 'number') return false;
    if (typeof tt.finalAccuracy !== 'number') return false;
    if (!Array.isArray(tt.trainLoss)) return false;
    if (!Array.isArray(tt.trainAccuracy)) return false;
  }

  return true;
}
