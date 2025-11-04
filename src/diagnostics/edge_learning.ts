/**
 * Edge Learning Diagnostics
 *
 * TypeScript port of the On-the-Edge learning heuristic.
 * Computes efficiency bounds and edge-of-efficiency diagnostics for neural models.
 *
 * Based on the Python implementation in symmetry_coupling/on_the_edge_learning.py
 */

/**
 * Edge learning diagnostics results
 */
export interface EdgeDiagnostics {
  // Efficiency measures
  efficiencyProduct: number; // E_p: Fisher info × estimator covariance
  efficiencyBound: number; // How close to theoretical bound (1.0 = optimal)

  // Information measures
  fisherInformation: number; // I(θ): How much information about parameter
  entropy: number; // H(θ): Entropy of the distribution

  // Sample complexity
  effectiveSampleSize: number; // Effective n for current model
  optimalSampleSize: number; // Optimal n for edge-of-efficiency

  // Edge status
  onEdge: boolean; // Whether model is on the edge of efficiency
  edgeDistance: number; // Distance from edge (0 = on edge)

  // Recommendations
  recommendation: string; // Human-readable guidance
}

/**
 * Configuration for edge learning diagnostics
 */
export interface EdgeConfig {
  thetaMin?: number; // Minimum parameter value
  thetaMax?: number; // Maximum parameter value
  delta?: number; // Edge band width (±δ around 1.0)
  alpha?: number; // Smoothing parameter
  epsilonMin?: number; // Minimum Fisher information
}

const DEFAULT_EDGE_CONFIG: Required<EdgeConfig> = {
  thetaMin: -5.0,
  thetaMax: 5.0,
  delta: 0.01,
  alpha: 10.0,
  epsilonMin: 1e-6
};

/**
 * Compute Fisher Information for a given parameter
 *
 * Fisher information quantifies how much information
 * a random variable carries about an unknown parameter.
 */
function computeFisherInformation(theta: number, epsilonMin: number): number {
  const value = 1.0 / (1.0 + theta * theta);
  return Math.max(value, epsilonMin);
}

/**
 * Compute entropy for a given parameter
 */
function computeEntropy(theta: number): number {
  return -Math.log(1.0 + theta * theta);
}

/**
 * Compute estimator covariance (variance + bias²)
 */
function computeEstimatorCovariance(theta: number, n: number, epsilonMin: number): number {
  const fisher = computeFisherInformation(theta, epsilonMin);
  const variance = 1.0 / (n * fisher);
  const bias = (0.1 * theta) / Math.sqrt(Math.max(n, 1.0));
  return variance + bias * bias;
}

/**
 * Compute efficiency product E_p = n × Fisher × Covariance
 *
 * This is the key metric: should be close to 1.0 for optimal efficiency.
 */
function computeEfficiencyProduct(theta: number, n: number, epsilonMin: number): number {
  const fisher = computeFisherInformation(theta, epsilonMin);
  const covariance = computeEstimatorCovariance(theta, n, epsilonMin);
  return n * fisher * covariance;
}

/**
 * Check if efficiency is on the edge (E_p ≈ 1 ± δ)
 */
function isOnEdge(efficiencyProduct: number, delta: number): boolean {
  return Math.abs(efficiencyProduct - 1.0) <= delta;
}

/**
 * Compute edge distance (how far from optimal)
 */
function computeEdgeDistance(efficiencyProduct: number, delta: number): number {
  const deviation = Math.abs(efficiencyProduct - 1.0);
  return Math.max(0, deviation - delta);
}

/**
 * Estimate effective sample size from model metrics
 */
function estimateEffectiveSampleSize(
  vocabSize: number,
  paramCount: number,
  accuracy: number
): number {
  // Rough heuristic: effective n ≈ vocab × accuracy factor
  // Better models extract more information per sample
  const accuracyFactor = Math.max(0.1, accuracy);
  return vocabSize * accuracyFactor * Math.log(paramCount + 1);
}

/**
 * Estimate theta (model parameter) from loss
 */
function estimateThetaFromLoss(loss: number): number {
  // Map loss to theta range: higher loss → higher |theta|
  // This is a simple heuristic; could be refined
  return Math.tanh(loss - 1.0) * 2.0;
}

/**
 * Generate recommendations based on edge diagnostics
 */
function generateRecommendation(diagnostics: EdgeDiagnostics): string {
  const ep = diagnostics.efficiencyProduct;

  if (diagnostics.onEdge) {
    return '✅ Model is on the edge of efficiency! Training is optimal.';
  }

  if (ep < 0.8) {
    return '⚠️ Under-efficient: Model needs more training or larger capacity.';
  }

  if (ep > 1.2) {
    return '⚠️ Over-efficient: Model may be overfitting. Consider regularization.';
  }

  return '📊 Model efficiency is reasonable but not optimal.';
}

/**
 * Compute edge learning diagnostics for a trained model
 */
export function computeEdgeDiagnostics(
  vocabSize: number,
  paramCount: number,
  loss: number,
  accuracy: number,
  config: EdgeConfig = {}
): EdgeDiagnostics {
  const cfg = { ...DEFAULT_EDGE_CONFIG, ...config };

  // Estimate model state
  const theta = estimateThetaFromLoss(loss);
  const n = estimateEffectiveSampleSize(vocabSize, paramCount, accuracy);

  // Compute core metrics
  const fisherInformation = computeFisherInformation(theta, cfg.epsilonMin);
  const entropy = computeEntropy(theta);
  const efficiencyProduct = computeEfficiencyProduct(theta, n, cfg.epsilonMin);
  const edgeDistance = computeEdgeDistance(efficiencyProduct, cfg.delta);
  const onEdge = isOnEdge(efficiencyProduct, cfg.delta);

  // Compute optimal sample size (where E_p ≈ 1.0)
  const covariance = computeEstimatorCovariance(theta, n, cfg.epsilonMin);
  const optimalSampleSize = Math.round(1.0 / (fisherInformation * covariance));

  const diagnostics: EdgeDiagnostics = {
    efficiencyProduct,
    efficiencyBound: 1.0 / Math.max(efficiencyProduct, 0.01), // How close to bound
    fisherInformation,
    entropy,
    effectiveSampleSize: Math.round(n),
    optimalSampleSize,
    onEdge,
    edgeDistance,
    recommendation: '' // Will be filled below
  };

  diagnostics.recommendation = generateRecommendation(diagnostics);

  return diagnostics;
}

/**
 * Format edge diagnostics for display
 */
export function formatEdgeDiagnostics(diagnostics: EdgeDiagnostics): string {
  const lines = [
    '📊 Edge Learning Diagnostics',
    '─'.repeat(40),
    `Efficiency Product: ${diagnostics.efficiencyProduct.toFixed(3)} ${diagnostics.onEdge ? '✓' : ''}`,
    `Fisher Information: ${diagnostics.fisherInformation.toFixed(4)}`,
    `Entropy: ${diagnostics.entropy.toFixed(3)}`,
    `Effective Sample Size: ${diagnostics.effectiveSampleSize}`,
    `Optimal Sample Size: ${diagnostics.optimalSampleSize}`,
    `Edge Status: ${diagnostics.onEdge ? 'ON EDGE ✓' : `OFF by ${diagnostics.edgeDistance.toFixed(3)}`}`,
    '─'.repeat(40),
    diagnostics.recommendation
  ];

  return lines.join('\n');
}
