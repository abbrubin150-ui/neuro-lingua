/**
 * Model packages: the codec (src/lib/modelPackageCodec.ts) plus loaders that restore the right
 * model class.
 */

import { ProNeuralLM } from './ProNeuralLM';
import { AdvancedNeuralLM } from './AdvancedNeuralLM';
import { TransformerLM } from './TransformerLM';
import { unpackModel, type ModelPackage } from './modelPackageCodec';

export * from './modelPackageCodec';

export type ModelArchitecture = 'feedforward' | 'advanced' | 'transformer';

/** Architecture of a model instance. */
export function modelArchitecture(model: ProNeuralLM): ModelArchitecture {
  if (model instanceof TransformerLM) return 'transformer';
  if (model instanceof AdvancedNeuralLM) return 'advanced';
  return 'feedforward';
}

/**
 * Restore a model of the right class from its `toJSON()` object. The architecture is read from the
 * object unless given (trace exports can record it outside the weights).
 */
export function restoreModel(
  json: Record<string, unknown>,
  architecture: unknown = json.architecture
): ProNeuralLM {
  if (architecture === 'transformer') {
    return TransformerLM.fromJSON({ ...json, architecture: 'transformer' });
  }
  if (architecture === 'advanced') return AdvancedNeuralLM.fromJSON(json);
  return ProNeuralLM.fromJSON(json);
}

/** Restore a ready-to-use model of the right class from a package. */
export function loadModelFromPackage(pkg: ModelPackage): ProNeuralLM {
  return restoreModel(unpackModel(pkg), pkg.architecture);
}
