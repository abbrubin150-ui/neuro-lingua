import type {
  CerebroBubble,
  InjectionEvent,
  InjectionProposal,
  InjectionTarget
} from '../../types/injection';
import type { InjectableLayer } from './InjectableLayer';
import {
  analyseResiduals,
  estimateEnergyGain,
  gramSchmidt,
  orthogonalProjector,
  projectCovariance,
  reshapeWeightMatrix,
  suggestKFromEnergy,
  topEigenvectors,
  trace,
  weightedCovariance
} from './injection_math';

export interface InjectionDiagnostics {
  meanResidual: number;
  tracePerp: number;
  estimatedGain: number;
  suggestedK: number;
}

interface InjectionEngineOptions {
  epsilon?: number;
  minGain?: number;
  orthPenalty?: number;
}

export class InjectionEngine {
  private readonly epsilon: number;

  private readonly minGain: number;

  private readonly orthPenalty: number;

  constructor(options: InjectionEngineOptions = {}) {
    this.epsilon = options.epsilon ?? 0.05;
    this.minGain = options.minGain ?? 0.01;
    this.orthPenalty = options.orthPenalty ?? 0.1;
  }

  diagnose(bubbles: CerebroBubble[], layer: InjectableLayer): InjectionDiagnostics {
    const target = layer.getTarget();
    const [weight] = layer.exportWeights();
    const basisVectors = reshapeWeightMatrix(weight ?? new Float32Array(), target.dModel);
    const basis = gramSchmidt(basisVectors);

    const residuals = analyseResiduals(bubbles, basis);
    const covariance = weightedCovariance(bubbles, target.dModel);
    const projector = orthogonalProjector(basis, target.dModel);
    const residualCovariance = projectCovariance(covariance, projector);

    const tracePerp = trace(residualCovariance);
    const estimatedGain = estimateEnergyGain(tracePerp, this.orthPenalty);
    const suggestedK = suggestKFromEnergy(residuals.meanEnergy, tracePerp, target.hiddenSize);

    return {
      meanResidual: residuals.meanEnergy,
      tracePerp,
      estimatedGain,
      suggestedK
    };
  }

  propose(diag: InjectionDiagnostics, target: InjectionTarget): InjectionProposal {
    return {
      target,
      k: Math.max(1, Math.min(target.hiddenSize, diag.suggestedK || 1)),
      method: diag.meanResidual > this.epsilon ? 'residual_eig' : 'random_he',
      epsilon: this.epsilon,
      minGain: this.minGain,
      orthPenalty: this.orthPenalty,
      createdAt: new Date().toISOString()
    };
  }

  execute(proposal: InjectionProposal, layer: InjectableLayer, bubbles: CerebroBubble[] = []): InjectionEvent {
    const preDiagnostics = this.diagnose(bubbles, layer);
    const seed = Math.floor(Math.random() * 1_000_000);
    const initMethod = proposal.method === 'svd_local' ? 'residual_eig' : proposal.method;

    if (!layer.canInject(proposal.k)) {
      return {
        proposal,
        accepted: false,
        metricsPre: { meanResidual: preDiagnostics.meanResidual, tracePerp: preDiagnostics.tracePerp },
        metricsPost: {},
        delta: {},
        seed,
        runId: proposal.target.modelId
      };
    }

    const snapshot = layer.exportWeights();
    try {
      layer.inject(proposal.k, initMethod);
    } catch (err) {
      // rollback
      layer.importWeights(snapshot);
      return {
        proposal,
        accepted: false,
        metricsPre: { meanResidual: preDiagnostics.meanResidual, tracePerp: preDiagnostics.tracePerp },
        metricsPost: {},
        delta: {},
        seed,
        runId: proposal.target.modelId
      };
    }

    const postDiagnostics = this.diagnose(bubbles, layer);

    const delta = {
      meanResidual: postDiagnostics.meanResidual - preDiagnostics.meanResidual,
      tracePerp: postDiagnostics.tracePerp - preDiagnostics.tracePerp,
      estimatedGain: postDiagnostics.estimatedGain - preDiagnostics.estimatedGain
    };

    return {
      proposal,
      accepted: postDiagnostics.estimatedGain >= proposal.minGain,
      metricsPre: {
        meanResidual: preDiagnostics.meanResidual,
        tracePerp: preDiagnostics.tracePerp,
        estimatedGain: preDiagnostics.estimatedGain
      },
      metricsPost: {
        meanResidual: postDiagnostics.meanResidual,
        tracePerp: postDiagnostics.tracePerp,
        estimatedGain: postDiagnostics.estimatedGain
      },
      delta,
      seed,
      runId: proposal.target.modelId
    };
  }

  materialiseVectors(
    bubbles: CerebroBubble[],
    layer: InjectableLayer,
    components: number
  ): number[][] {
    const target = layer.getTarget();
    const [weight] = layer.exportWeights();
    const basisVectors = reshapeWeightMatrix(weight ?? new Float32Array(), target.dModel);
    const basis = gramSchmidt(basisVectors);
    const covariance = weightedCovariance(bubbles, target.dModel);
    const projector = orthogonalProjector(basis, target.dModel);
    const residualCovariance = projectCovariance(covariance, projector);

    return topEigenvectors(residualCovariance, components);
  }
}
