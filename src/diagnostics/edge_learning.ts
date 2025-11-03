/**
 * Edge Learning Diagnostics
 *
 * Simplified TypeScript implementation of edge-of-efficiency metrics
 * for neural language models. Based on the On-the-Edge learning heuristic.
 *
 * This provides browser-compatible computation of:
 * - Fisher Information
 * - Efficiency Bounds
 * - Cramer-Rao Bounds
 * - Edge Detection
 */

export interface EdgeMetrics {
  fisherInformation: number;
  cramérRaoBound: number;
  efficiency: number;
  onEdge: boolean;
  edgeScore: number;
  sampleComplexity: number;
}

export interface EdgeDiagnosticsConfig {
  epsilon?: number; // Minimum Fisher information
  delta?: number; // Edge tolerance (±δ around efficiency = 1)
  alpha?: number; // Smoothing parameter for edge detection
}

const DEFAULT_CONFIG: Required<EdgeDiagnosticsConfig> = {
  epsilon: 1e-6,
  delta: 0.01,
  alpha: 10.0
};

/**
 * Compute Fisher Information for a simple probabilistic model
 *
 * For a language model, this approximates the sensitivity of the model
 * to parameter changes at the current configuration.
 *
 * @param modelParams Model parameter statistics (variance, entropy, etc.)
 * @returns Fisher information (scalar approximation)
 */
function computeFisherInformation(modelParams: {
  variance: number;
  entropy: number;
  perplexity: number;
}): number {
  const { variance, entropy, perplexity } = modelParams;

  // Simple heuristic: Fisher ∝ 1/variance for location parameters
  // Combined with model uncertainty (perplexity)
  const varianceComponent = variance > 0 ? 1 / variance : 0;

  // Entropy component (higher entropy = more uncertainty = lower Fisher)
  const entropyComponent = entropy > 0 ? 1 / Math.exp(entropy) : 1;

  // Perplexity component (lower is better)
  const perplexityComponent = perplexity > 1 ? 1 / Math.log(perplexity) : 1;

  // Combine components
  const fisher = varianceComponent * entropyComponent * perplexityComponent;

  return Math.max(fisher, DEFAULT_CONFIG.epsilon);
}

/**
 * Compute Cramér-Rao Lower Bound
 *
 * The CRLB provides a lower bound on the variance of any unbiased estimator.
 * For a model with n samples and Fisher information I:
 *   Var(θ̂) ≥ 1 / (n · I(θ))
 *
 * @param fisherInfo Fisher information
 * @param nSamples Number of training samples
 * @returns Cramér-Rao bound
 */
function computeCramérRaoBound(fisherInfo: number, nSamples: number): number {
  if (nSamples <= 0 || fisherInfo <= 0) {
    return Infinity;
  }
  return 1 / (nSamples * fisherInfo);
}

/**
 * Compute efficiency: how close the estimator is to the Cramér-Rao bound
 *
 * Efficiency E = (CRLB) / (Actual Variance)
 * E = 1 means the estimator is optimal (achieves the bound)
 * E < 1 means sub-optimal
 * E > 1 is impossible for unbiased estimators
 *
 * @param cramérRaoBound Cramér-Rao lower bound
 * @param actualVariance Actual estimator variance
 * @returns Efficiency ratio
 */
function computeEfficiency(cramérRaoBound: number, actualVariance: number): number {
  if (actualVariance <= 0) {
    return 0;
  }
  if (!isFinite(cramérRaoBound)) {
    return 0;
  }
  return cramérRaoBound / actualVariance;
}

/**
 * Detect if the model is "on the edge" of efficiency
 *
 * A model is on the edge if its efficiency is close to 1 (within δ tolerance).
 * This suggests the model is near-optimal given the sample complexity.
 *
 * @param efficiency Efficiency ratio
 * @param config Diagnostics configuration
 * @returns Edge detection result
 */
function detectEdge(
  efficiency: number,
  config: Required<EdgeDiagnosticsConfig>
): { onEdge: boolean; score: number } {
  const { delta, alpha } = config;

  // Deviation from perfect efficiency
  const deviation = Math.abs(efficiency - 1.0);

  // Smoothed gate function (sigmoid-like)
  const smooth = Math.exp(-alpha * deviation * deviation);

  // Binary decision: within tolerance band?
  const onEdge = deviation <= delta;

  // Continuous score: how close to the edge (0 to 1)
  const score = smooth;

  return { onEdge, score };
}

/**
 * Estimate sample complexity needed to reach efficiency = 1
 *
 * If current efficiency < 1, estimate how many more samples are needed
 * to reach the Cramér-Rao bound.
 *
 * @param currentN Current number of samples
 * @param efficiency Current efficiency
 * @returns Estimated sample complexity for optimal efficiency
 */
function estimateSampleComplexity(currentN: number, efficiency: number): number {
  if (efficiency >= 1.0) {
    // Already optimal
    return currentN;
  }

  if (efficiency <= 0) {
    return Infinity;
  }

  // If efficiency = CRLB / Var, and CRLB ∝ 1/n, then:
  // To achieve E = 1, we need n_new = n_current / E_current
  const estimatedN = currentN / efficiency;

  return Math.ceil(estimatedN);
}

/**
 * Compute comprehensive edge learning diagnostics for a trained model
 *
 * @param modelStats Model training statistics
 * @param config Optional configuration
 * @returns Edge learning metrics
 */
export function computeEdgeDiagnostics(
  modelStats: {
    loss: number;
    accuracy: number;
    perplexity: number;
    vocabSize: number;
    paramCount: number;
    trainingSize: number;
    trainingHistory: { loss: number; accuracy: number }[];
  },
  config: EdgeDiagnosticsConfig = {}
): EdgeMetrics {
  const fullConfig = { ...DEFAULT_CONFIG, ...config };

  // Compute model parameter statistics
  const history = modelStats.trainingHistory;
  const recentLosses = history.slice(-10).map((h) => h.loss);

  // Estimate variance from recent training history
  const meanLoss = recentLosses.reduce((sum, l) => sum + l, 0) / recentLosses.length;
  const variance =
    recentLosses.reduce((sum, l) => sum + Math.pow(l - meanLoss, 2), 0) / recentLosses.length;

  // Entropy approximation from perplexity
  // H ≈ log(perplexity)
  const entropy = Math.log(modelStats.perplexity);

  // Compute Fisher information
  const fisherInfo = computeFisherInformation({
    variance: variance > 0 ? variance : 1e-6,
    entropy,
    perplexity: modelStats.perplexity
  });

  // Compute Cramér-Rao bound
  const cramérRaoBound = computeCramérRaoBound(fisherInfo, modelStats.trainingSize);

  // Compute efficiency
  const efficiency = computeEfficiency(cramérRaoBound, variance);

  // Detect edge
  const { onEdge, score: edgeScore } = detectEdge(efficiency, fullConfig);

  // Estimate sample complexity
  const sampleComplexity = estimateSampleComplexity(modelStats.trainingSize, efficiency);

  return {
    fisherInformation: fisherInfo,
    cramérRaoBound,
    efficiency,
    onEdge,
    edgeScore,
    sampleComplexity
  };
}

/**
 * Format edge metrics for display
 */
export function formatEdgeMetrics(metrics: EdgeMetrics): string {
  const lines: string[] = [
    '🔬 Edge Learning Diagnostics',
    '',
    `Fisher Information: ${metrics.fisherInformation.toExponential(3)}`,
    `Cramér-Rao Bound: ${metrics.cramérRaoBound.toExponential(3)}`,
    `Efficiency: ${metrics.efficiency.toFixed(4)} ${metrics.efficiency >= 0.95 ? '✅' : '⚠️'}`,
    `Edge Score: ${metrics.edgeScore.toFixed(4)}`,
    `On Edge: ${metrics.onEdge ? '✅ YES' : '❌ NO'}`,
    ''
  ];

  if (!metrics.onEdge && metrics.efficiency < 1.0) {
    const additional = metrics.sampleComplexity - metrics.sampleComplexity;
    lines.push(`💡 Estimated samples needed for optimal efficiency: ${metrics.sampleComplexity}`);
    if (isFinite(additional) && additional > 0) {
      lines.push(`   (Additional ${additional} samples from current)`);
    }
  }

  if (metrics.onEdge) {
    lines.push('✨ Model is on the edge of efficiency!');
    lines.push('   This suggests near-optimal use of training data.');
  }

  return lines.join('\n');
}
