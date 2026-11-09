/**
 * Edge Learning diagnostics computation module
 *
 * This module provides statistical analysis of model learning efficiency
 * using information-theoretic measures (Fisher Information, Entropy, etc.)
 *
 * Note: The Python-based implementation is only available in Node.js environments.
 * In browser environments, simulated diagnostics are computed.
 *
 * Usage:
 *   const diagnostics = computeSimulatedEdgeLearningDiagnostics(modelSize, trainingLosses);
 *   console.log(diagnostics.fisherInformation);
 */

export interface EdgeLearningDiagnostics {
  /** Mean diagonal of the empirical Fisher information of the output layer. */
  fisherInformation: number;
  /** Mean predictive entropy H(p(·|context)) in nats. */
  entropy: number;
  /** Inverse of the mean Fisher diagonal (asymptotic per-parameter variance, one sample). */
  estimatorCovariance: number;
  /** Diagonal-approximation Cramér-Rao bound 1 / (N · F̄) for N training windows. */
  cramerRaoBound: number;
  /** Only produced by the deprecated heuristic below; not a measured quantity. */
  efficiency?: number;
  /** Variance of the per-token cross-entropy loss. */
  variance: number;
  /** Number of windows the estimates were computed on. */
  samples?: number;
  timestamp: number;
  status: 'success' | 'error';
  error?: string;
}

/** Minimal model interface needed for the diagnostics. */
export interface DiagnosableModel {
  collectRepresentations(
    text: string,
    maxSamples?: number
  ): Promise<{ representations: number[][]; labels: number[]; probabilities: number[][] }>;
  getVocabSize(): number;
}

/**
 * Edge-learning diagnostics measured on the model itself.
 *
 * For each sampled training window with read-out h, predicted distribution p and target y, the
 * gradient of the cross-entropy with respect to the output layer is (h, 1) ⊗ (p − e_y), so the
 * empirical Fisher trace of that layer is exactly mean_i (‖h_i‖² + 1)·‖p_i − e_{y_i}‖². Reported:
 * its mean diagonal F̄, the inverse 1/F̄, the diagonal Cramér-Rao bound 1/(N·F̄) for the N training
 * windows, the mean predictive entropy and the variance of the per-token loss.
 */
export async function computeEdgeLearningDiagnostics(
  model: DiagnosableModel,
  text: string,
  options: { maxSamples?: number; trainingWindows?: number } = {}
): Promise<EdgeLearningDiagnostics> {
  const timestamp = Date.now();
  try {
    const { representations, labels, probabilities } = await model.collectRepresentations(
      text,
      options.maxSamples ?? 256
    );
    const n = labels.length;
    if (n === 0) {
      return {
        fisherInformation: 0,
        entropy: 0,
        estimatorCovariance: 0,
        cramerRaoBound: 0,
        variance: 0,
        samples: 0,
        timestamp,
        status: 'error',
        error: 'No training windows available'
      };
    }
    const V = model.getVocabSize();
    const H = representations[0]?.length ?? 0;
    let fisherTrace = 0;
    let entropySum = 0;
    const losses: number[] = [];
    for (let i = 0; i < n; i++) {
      const p = probabilities[i];
      const y = labels[i];
      let g2 = 0;
      let h = 0;
      for (let j = 0; j < p.length; j++) {
        const g = p[j] - (j === y ? 1 : 0);
        g2 += g * g;
        if (p[j] > 0) h -= p[j] * Math.log(p[j]);
      }
      let r2 = 0;
      for (const v of representations[i]) r2 += v * v;
      fisherTrace += (r2 + 1) * g2;
      entropySum += h;
      losses.push(-Math.log(Math.max(p[y], 1e-300)));
    }
    fisherTrace /= n;
    const meanFisher = fisherTrace / Math.max(1, H * V + V);
    const meanLoss = losses.reduce((a, b) => a + b, 0) / n;
    const variance = losses.reduce((a, b) => a + (b - meanLoss) ** 2, 0) / Math.max(1, n - 1);
    const windows = options.trainingWindows ?? n;
    return {
      fisherInformation: meanFisher,
      entropy: entropySum / n,
      estimatorCovariance: meanFisher > 0 ? 1 / meanFisher : Infinity,
      cramerRaoBound: meanFisher > 0 ? 1 / (windows * meanFisher) : Infinity,
      variance,
      samples: n,
      timestamp,
      status: 'success'
    };
  } catch (error) {
    return {
      fisherInformation: 0,
      entropy: 0,
      estimatorCovariance: 0,
      cramerRaoBound: 0,
      variance: 0,
      timestamp,
      status: 'error',
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * @deprecated Heuristic numbers derived from the loss curve (e.g. "Fisher" = final loss); they are
 * not information-theoretic measurements. Use computeEdgeLearningDiagnostics().
 *
 * Compute simulated Edge Learning diagnostics from model training history
 *
 * This provides a lightweight approximation of information-theoretic measures
 * based on training loss progression and model size. It's suitable for
 * browser-based training where full Python analysis is not available.
 *
 * @param modelSize - Number of model parameters
 * @param trainingLosses - Array of loss values from training
 * @returns Edge learning diagnostics object
 */
export function computeSimulatedEdgeLearningDiagnostics(
  modelSize: number,
  trainingLosses: number[]
): EdgeLearningDiagnostics {
  const timestamp = Date.now();

  try {
    if (!trainingLosses || trainingLosses.length === 0) {
      return {
        fisherInformation: 0,
        entropy: 0,
        estimatorCovariance: 0,
        cramerRaoBound: 0,
        efficiency: 0,
        variance: 0,
        timestamp,
        status: 'error',
        error: 'No training history available'
      };
    }

    // Calculate metrics from training history
    const finalLoss = trainingLosses[trainingLosses.length - 1] ?? 1e-6;
    const initialLoss = trainingLosses[0] ?? 1;
    const convergenceRate = initialLoss > 0 ? 1 - finalLoss / initialLoss : 0;
    const nSamples = trainingLosses.length;

    // Estimate Fisher information (depends on loss curvature)
    // Higher final loss → higher Fisher information (more curvature)
    const fisherInformation = Math.max(0.1, Math.abs(finalLoss));

    // Estimate entropy based on model size and training convergence
    // Entropy scales with model complexity
    const entropy = Math.log(Math.max(modelSize, 1)) * (1 + convergenceRate);

    // Estimator covariance (inverse of Fisher information)
    // This represents the variance of parameter estimates
    const estimatorCovariance = 1 / (fisherInformation + 1e-8);

    // Cramér-Rao bound (theoretical minimum variance for unbiased estimator)
    // Lower bound on variance of any unbiased estimator
    const cramerRaoBound = estimatorCovariance / Math.max(nSamples, 1);

    // Efficiency: ratio of achieved variance to theoretical minimum
    // Efficiency = 1 means optimal, < 1 means sub-optimal
    const achievedVariance = finalLoss;
    const efficiency = cramerRaoBound > 0 ? Math.min(1, cramerRaoBound / achievedVariance) : 0;

    // Variance (from final training loss)
    const variance = finalLoss;

    return {
      fisherInformation: Math.round(fisherInformation * 10000) / 10000,
      entropy: Math.round(entropy * 10000) / 10000,
      estimatorCovariance: Math.round(estimatorCovariance * 10000) / 10000,
      cramerRaoBound: Math.round(cramerRaoBound * 10000000) / 10000000,
      efficiency: Math.round(efficiency * 10000) / 10000,
      variance: Math.round(variance * 10000) / 10000,
      timestamp,
      status: 'success'
    };
  } catch (error) {
    return {
      fisherInformation: 0,
      entropy: 0,
      estimatorCovariance: 0,
      cramerRaoBound: 0,
      efficiency: 0,
      variance: 0,
      timestamp,
      status: 'error',
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Information-theoretic interpretation of Edge Learning metrics
 *
 * Fisher Information: Measures how "peaked" the loss surface is around the optimum.
 * Higher values indicate a sharper minimum, suggesting the model found a
 * distinctive solution.
 *
 * Entropy: Quantifies the uncertainty in model parameters. Higher entropy
 * suggests more distributed parameter values.
 *
 * Estimator Covariance: Represents the variance of parameter estimates.
 * Derived as the inverse of Fisher information.
 *
 * Cramér-Rao Bound (CRB): The theoretical lower bound on the variance of
 * any unbiased estimator. This is a fundamental limit from information theory.
 *
 * Efficiency: The ratio of actual variance to the CRB. Values close to 1.0
 * indicate the model is learning efficiently relative to information-theoretic limits.
 * Lower values suggest the model could potentially learn more from the data.
 *
 * @see https://en.wikipedia.org/wiki/Fisher_information
 * @see https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound
 */
export const EDGE_LEARNING_INFO = {
  fisherInformation:
    'Mean diagonal of the empirical Fisher information of the output layer, computed exactly from per-window gradients (h,1)⊗(p−e_y).',
  entropy: 'Mean entropy of the predicted next-token distribution, in nats.',
  estimatorCovariance:
    'Inverse of the mean Fisher diagonal: asymptotic variance of one parameter estimated from a single window (diagonal approximation).',
  cramerRaoBound:
    'Diagonal Cramér-Rao bound 1/(N·F̄) on the variance of an unbiased output-layer parameter estimate from the N training windows.',
  efficiency: 'Not measured.',
  variance: 'Variance of the per-token cross-entropy loss across training windows.'
};
