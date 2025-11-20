/**
 * Information Bottleneck Loss Implementation
 *
 * The Information Bottleneck (IB) principle finds optimal representations Z that:
 * 1. Compress input X: minimize I(X;Z)
 * 2. Preserve relevant information for prediction Y: maximize I(Z;Y)
 *
 * Loss: L_IB = -I(Z;Y) + β·I(X;Z)
 *
 * where β controls the compression-prediction trade-off:
 * - β → 0: maximum compression (minimal information retained)
 * - β → ∞: maximum prediction (all information retained)
 *
 * Reference: "The Information Bottleneck Method" (Tishby et al., 1999)
 */

import { stableSoftmax } from '../lib/MathUtils';

/**
 * Configuration for Information Bottleneck loss
 */
export interface InformationBottleneckConfig {
  /** Beta parameter controlling compression-prediction trade-off */
  beta: number;
  /** Number of bins for mutual information estimation (default: 50) */
  numBins?: number;
  /** Epsilon for numerical stability (default: 1e-10) */
  epsilon?: number;
}

/**
 * Beta annealing schedule types
 */
export type BetaSchedule = 'constant' | 'linear' | 'exponential' | 'cosine';

/**
 * Information-theoretic metrics computed during training
 */
export interface InformationMetrics {
  /** Mutual information I(X;Z) - compression term */
  compressionMI: number;
  /** Mutual information I(Z;Y) - prediction term */
  predictionMI: number;
  /** Combined IB loss value */
  ibLoss: number;
  /** Current beta value */
  beta: number;
  /** Entropy H(Z) */
  representationEntropy: number;
  /** Conditional entropy H(Z|X) */
  conditionalEntropy: number;
}

/**
 * Estimate mutual information using histogram-based method
 *
 * I(X;Y) = H(X) + H(Y) - H(X,Y)
 * where H(X) = -Σ p(x) log p(x) is entropy
 *
 * @param x - First variable samples (flattened representation)
 * @param y - Second variable samples (flattened representation)
 * @param numBins - Number of histogram bins for discretization
 * @param epsilon - Small constant for numerical stability
 * @returns Estimated mutual information in nats
 */
export function estimateMutualInformation(
  x: number[],
  y: number[],
  numBins = 50,
  epsilon = 1e-10
): number {
  if (x.length !== y.length || x.length === 0) {
    return 0;
  }

  // Discretize continuous values into bins
  const xMin = Math.min(...x);
  const xMax = Math.max(...x);
  const yMin = Math.min(...y);
  const yMax = Math.max(...y);

  const xRange = xMax - xMin + epsilon;
  const yRange = yMax - yMin + epsilon;

  // Create joint histogram
  const jointCounts = new Map<string, number>();
  const xCounts = new Map<number, number>();
  const yCounts = new Map<number, number>();

  for (let i = 0; i < x.length; i++) {
    const xBin = Math.floor(((x[i] - xMin) / xRange) * (numBins - 1));
    const yBin = Math.floor(((y[i] - yMin) / yRange) * (numBins - 1));

    const jointKey = `${xBin},${yBin}`;
    jointCounts.set(jointKey, (jointCounts.get(jointKey) ?? 0) + 1);
    xCounts.set(xBin, (xCounts.get(xBin) ?? 0) + 1);
    yCounts.set(yBin, (yCounts.get(yBin) ?? 0) + 1);
  }

  const n = x.length;

  // Compute entropies
  let hX = 0; // H(X)
  for (const count of xCounts.values()) {
    const p = count / n;
    hX -= p * Math.log(p + epsilon);
  }

  let hY = 0; // H(Y)
  for (const count of yCounts.values()) {
    const p = count / n;
    hY -= p * Math.log(p + epsilon);
  }

  let hXY = 0; // H(X,Y)
  for (const count of jointCounts.values()) {
    const p = count / n;
    hXY -= p * Math.log(p + epsilon);
  }

  // I(X;Y) = H(X) + H(Y) - H(X,Y)
  return Math.max(0, hX + hY - hXY);
}

/**
 * Compute entropy H(Z) of representation
 *
 * @param activations - Hidden layer activations [batchSize, hiddenDim]
 * @param numBins - Number of bins for discretization
 * @param epsilon - Numerical stability constant
 * @returns Entropy in nats
 */
export function computeRepresentationEntropy(
  activations: number[][],
  numBins = 50,
  epsilon = 1e-10
): number {
  if (activations.length === 0) return 0;

  // Flatten activations
  const flattened = activations.flat();
  if (flattened.length === 0) return 0;

  // Discretize into histogram
  const minVal = Math.min(...flattened);
  const maxVal = Math.max(...flattened);
  const range = maxVal - minVal + epsilon;

  const counts = new Array(numBins).fill(0);
  for (const val of flattened) {
    const bin = Math.min(Math.floor(((val - minVal) / range) * numBins), numBins - 1);
    counts[bin]++;
  }

  // Compute entropy H(Z) = -Σ p(z) log p(z)
  const n = flattened.length;
  let entropy = 0;
  for (const count of counts) {
    if (count > 0) {
      const p = count / n;
      entropy -= p * Math.log(p + epsilon);
    }
  }

  return entropy;
}

/**
 * Compute Information Bottleneck loss
 *
 * L_IB = -I(Z;Y) + β·I(X;Z)
 *
 * @param inputs - Input embeddings [batchSize, inputDim]
 * @param hiddenActivations - Hidden layer activations Z [batchSize, hiddenDim]
 * @param targetLogits - Output logits [batchSize, vocabSize]
 * @param targetIndices - True target class indices [batchSize]
 * @param config - IB configuration including beta
 * @returns Information-theoretic metrics including IB loss
 */
export function informationBottleneckLoss(
  inputs: number[][],
  hiddenActivations: number[][],
  targetLogits: number[][],
  targetIndices: number[],
  config: InformationBottleneckConfig
): InformationMetrics {
  const { beta, numBins = 50, epsilon = 1e-10 } = config;

  if (
    inputs.length !== hiddenActivations.length ||
    inputs.length !== targetLogits.length ||
    inputs.length !== targetIndices.length
  ) {
    throw new Error('Batch size mismatch in IB loss computation');
  }

  if (inputs.length === 0) {
    return {
      compressionMI: 0,
      predictionMI: 0,
      ibLoss: 0,
      beta,
      representationEntropy: 0,
      conditionalEntropy: 0
    };
  }

  // Flatten arrays for MI estimation
  const flatInputs = inputs.flat();
  const flatHidden = hiddenActivations.flat();

  // Convert target logits to probabilities (for I(Z;Y) estimation)
  const targetProbs = targetLogits.map((logits) => stableSoftmax(logits));

  // Get predicted probability for true class
  const flatTargetProbs = targetIndices.map((idx, i) => targetProbs[i][idx]);

  // Estimate I(X;Z) - compression term
  const compressionMI = estimateMutualInformation(flatInputs, flatHidden, numBins, epsilon);

  // Estimate I(Z;Y) - prediction term
  // Use hidden activations and target probabilities
  const predictionMI = estimateMutualInformation(flatHidden, flatTargetProbs, numBins, epsilon);

  // Compute H(Z)
  const representationEntropy = computeRepresentationEntropy(hiddenActivations, numBins, epsilon);

  // H(Z|X) ≈ H(Z) - I(X;Z)
  const conditionalEntropy = Math.max(0, representationEntropy - compressionMI);

  // Information Bottleneck loss: minimize -I(Z;Y) + β·I(X;Z)
  // Equivalent to: maximize I(Z;Y) - β·I(X;Z)
  const ibLoss = -predictionMI + beta * compressionMI;

  return {
    compressionMI,
    predictionMI,
    ibLoss,
    beta,
    representationEntropy,
    conditionalEntropy
  };
}

/**
 * Get beta value according to annealing schedule
 *
 * @param schedule - Type of annealing schedule
 * @param epoch - Current training epoch (0-indexed)
 * @param totalEpochs - Total number of training epochs
 * @param betaStart - Initial beta value
 * @param betaEnd - Final beta value
 * @returns Current beta value
 */
export function getBetaSchedule(
  schedule: BetaSchedule,
  epoch: number,
  totalEpochs: number,
  betaStart: number,
  betaEnd: number
): number {
  if (totalEpochs <= 1) return betaStart;

  const progress = Math.min(1, epoch / (totalEpochs - 1));

  switch (schedule) {
    case 'constant':
      return betaStart;

    case 'linear':
      // Linear interpolation from betaStart to betaEnd
      return betaStart + (betaEnd - betaStart) * progress;

    case 'exponential':
      // Exponential interpolation
      // beta(t) = betaStart * (betaEnd/betaStart)^progress
      if (betaStart <= 0) return betaEnd * progress;
      return betaStart * Math.pow(betaEnd / betaStart, progress);

    case 'cosine':
      // Cosine annealing from betaStart to betaEnd
      // beta(t) = betaEnd + (betaStart - betaEnd) * (1 + cos(π*progress)) / 2
      return betaEnd + ((betaStart - betaEnd) * (1 + Math.cos(Math.PI * progress))) / 2;

    default:
      return betaStart;
  }
}

/**
 * Standard cross-entropy loss for comparison
 *
 * @param logits - Model output logits [vocabSize]
 * @param targetIndex - True class index
 * @returns Cross-entropy loss
 */
export function standardCrossEntropyLoss(logits: number[], targetIndex: number): number {
  const probs = stableSoftmax(logits);
  const targetProb = probs[targetIndex];
  return -Math.log(Math.max(targetProb, 1e-10));
}

/**
 * Hybrid loss combining cross-entropy and information bottleneck
 *
 * L = (1-α)·CE + α·IB
 *
 * @param ceLoss - Standard cross-entropy loss
 * @param ibLoss - Information bottleneck loss
 * @param alpha - Weight for IB term (0 = pure CE, 1 = pure IB)
 * @returns Combined loss
 */
export function hybridIBLoss(ceLoss: number, ibLoss: number, alpha: number): number {
  return (1 - alpha) * ceLoss + alpha * ibLoss;
}
