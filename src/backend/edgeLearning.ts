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
  fisherInformation: number;
  entropy: number;
  estimatorCovariance: number;
  cramerRaoBound: number;
  efficiency: number;
  variance: number;
  timestamp: number;
  status: 'success' | 'error';
  error?: string;
}

/**
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
    'Measures curvature of loss surface. Higher = sharper minimum, more distinctive solution.',
  entropy: 'Uncertainty in model parameters. Higher = more distributed weight values.',
  estimatorCovariance: 'Variance of parameter estimates. Inverse of Fisher information.',
  cramerRaoBound: 'Theoretical minimum variance for unbiased estimators (information limit).',
  efficiency: 'Ratio of actual to theoretical minimum variance. Higher = more efficient learning.',
  variance: 'Actual variance (from final training loss). Lower = better fit.'
};
