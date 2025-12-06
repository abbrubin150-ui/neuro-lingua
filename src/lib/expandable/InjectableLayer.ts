import type { InjectionTarget } from '../../types/injection';

export interface InjectableLayer {
  getTarget(): InjectionTarget;
  canInject(k: number): boolean;
  inject(k: number, init: 'random_he' | 'residual_eig'): void;
  exportWeights(): Float32Array[];
  importWeights(weights: Float32Array[]): void;
}

export interface InjectionSnapshot {
  weights: Float32Array[];
  capturedAt: number;
}

export function cloneWeights(weights: Float32Array[]): Float32Array[] {
  return weights.map((w) => new Float32Array(w));
}
