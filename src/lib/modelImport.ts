/**
 * Parsing of model files selected by the user (Import button).
 *
 * Three formats are accepted:
 * - package:  a compact model package produced by the model factory or the Export button
 *             (`packModel`, src/lib/modelPackage.ts); its checksum is verified
 * - standard: the object produced by `model.toJSON()`
 * - trace:    the object produced by `createTraceExport()`, whose `modelWeights` field holds the
 *             `model.toJSON()` output together with project / ledger / training metadata
 *
 * The result always contains the flat `toJSON()` object (what the model loaders expect) and the
 * detected architecture.
 */

import { isModelPackage, unpackModel, type ModelCard } from './modelPackageCodec';

export type ImportedArchitecture = 'feedforward' | 'advanced' | 'transformer';

export interface ParsedModelImport {
  /** Flat serialized model, as produced by `model.toJSON()`. */
  modelData: Record<string, unknown>;
  architecture: ImportedArchitecture;
  format: 'standard' | 'trace' | 'package';
  /** Model card carried by a model package. */
  card?: ModelCard;
  /** Metadata carried by trace exports. */
  trace?: {
    projectMeta?: { projectName?: string; runName?: string };
    decisionLedger?: { rationale?: string };
    trainingTrace?: { epochs?: number; finalLoss?: number; finalAccuracy?: number };
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function detectArchitecture(...candidates: unknown[]): ImportedArchitecture {
  for (const candidate of candidates) {
    if (candidate === 'transformer' || candidate === 'advanced') return candidate;
  }
  return 'feedforward';
}

function assertSerializedModel(model: Record<string, unknown>): void {
  const problems: string[] = [];
  if (!Array.isArray(model.vocab) || model.vocab.length === 0) problems.push('vocab');
  if (typeof model.hiddenSize !== 'number') problems.push('hiddenSize');
  if (!Array.isArray(model.embedding)) problems.push('embedding');
  if (!Array.isArray(model.wOutput)) problems.push('wOutput');
  if (problems.length > 0) {
    throw new Error(`Not a serialized model: missing or invalid ${problems.join(', ')}`);
  }
}

/**
 * Normalise an imported JSON value into a flat serialized model plus its architecture.
 * Throws an Error with a human-readable message when the file is not a model export.
 */
export function parseModelImport(data: unknown): ParsedModelImport {
  if (!isRecord(data)) {
    throw new Error('Model file must contain a JSON object');
  }

  if (isModelPackage(data)) {
    const modelData = unpackModel(data);
    assertSerializedModel(modelData);
    return {
      modelData,
      architecture: data.architecture,
      format: 'package',
      card: data.card
    };
  }

  const looksLikeTrace = 'modelWeights' in data && ('exportedAt' in data || 'config' in data);
  if (looksLikeTrace) {
    const weights = data.modelWeights;
    if (!isRecord(weights)) {
      throw new Error('Trace export has no serialized model in "modelWeights"');
    }
    // Very old exports nested the model one level deeper ({ weights: {...} }).
    const modelData = isRecord(weights.weights) ? weights.weights : weights;
    assertSerializedModel(modelData);
    const config = isRecord(data.config) ? data.config : {};
    return {
      modelData,
      architecture: detectArchitecture(
        modelData.architecture,
        data.architecture,
        config.architecture
      ),
      format: 'trace',
      trace: {
        projectMeta: isRecord(data.projectMeta)
          ? (data.projectMeta as { projectName?: string; runName?: string })
          : undefined,
        decisionLedger: isRecord(data.decisionLedger)
          ? (data.decisionLedger as { rationale?: string })
          : undefined,
        trainingTrace: isRecord(data.trainingTrace)
          ? (data.trainingTrace as { epochs?: number; finalLoss?: number; finalAccuracy?: number })
          : undefined
      }
    };
  }

  assertSerializedModel(data);
  const config = isRecord(data.config) ? data.config : {};
  return {
    modelData: data,
    architecture: detectArchitecture(data.architecture, config.architecture),
    format: 'standard'
  };
}
