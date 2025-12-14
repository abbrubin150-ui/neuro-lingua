/**
 * Neural Tangent Kernel (NTK) Analysis
 *
 * The NTK characterizes the training dynamics of neural networks in the
 * infinite-width limit. For a network f(x; θ), the NTK is defined as:
 *
 *   Θ(x, x') = ∇_θ f(x; θ)^T ∇_θ f(x'; θ)
 *
 * Key properties:
 * - In the infinite-width limit, NTK remains approximately constant during training
 * - Network output evolves as kernel regression with NTK
 * - Eigenvalue spectrum of NTK determines learnability and convergence speed
 *
 * This module provides:
 * - NTK computation for finite networks
 * - Eigenvalue analysis for trainability assessment
 * - Lazy training regime detection
 * - Feature learning vs kernel regime analysis
 *
 * References:
 * - Jacot et al. (2018) "Neural Tangent Kernel: Convergence and Generalization
 *   in Neural Networks"
 * - Lee et al. (2019) "Wide Neural Networks of Any Depth Evolve as Linear Models
 *   Under Gradient Descent"
 * - Arora et al. (2019) "On Exact Computation with an Infinitely Wide Neural Net"
 *
 * @module math/ntk_analysis
 */

// ============================================================================
// Types and Interfaces
// ============================================================================

export interface NTKMatrix {
  /** NTK matrix Θ[i,j] = ⟨∇f(x_i), ∇f(x_j)⟩ */
  kernel: number[][];
  /** Eigenvalues of NTK (sorted descending) */
  eigenvalues: number[];
  /** Condition number (λ_max / λ_min) */
  conditionNumber: number;
  /** Effective rank (trace / λ_max) */
  effectiveRank: number;
  /** Trace of NTK */
  trace: number;
}

export interface NTKDynamics {
  /** Current NTK snapshot */
  ntk: NTKMatrix;
  /** Change from initial NTK (Frobenius norm) */
  ntkChange: number;
  /** Relative change (change / initial norm) */
  relativeChange: number;
  /** Indicates lazy training regime (small NTK change) */
  lazyRegime: boolean;
  /** Feature learning indicator (large NTK change) */
  featureLearning: boolean;
}

export interface TrainabilityAnalysis {
  /** Minimum eigenvalue of NTK */
  lambdaMin: number;
  /** Maximum eigenvalue of NTK */
  lambdaMax: number;
  /** Trainability score (0-1, higher is better) */
  trainabilityScore: number;
  /** Expected convergence rate O(exp(-λ_min * t)) */
  convergenceRate: number;
  /** Network is trainable (λ_min > threshold) */
  isTrainable: boolean;
  /** Warnings and recommendations */
  warnings: string[];
}

export interface SpectrumAnalysis {
  /** Eigenvalue spectrum */
  spectrum: number[];
  /** Spectral decay rate (fitted power law exponent) */
  decayRate: number;
  /** Number of significant eigenvalues (90% of trace) */
  significantCount: number;
  /** Entropy of normalized eigenvalue distribution */
  spectralEntropy: number;
  /** Gap ratio (λ_1 / λ_2) */
  gapRatio: number;
}

// ============================================================================
// Gradient Computation Utilities
// ============================================================================

/**
 * Compute numerical gradient of a scalar function
 *
 * @param f Function to differentiate
 * @param x Point at which to compute gradient
 * @param h Step size for finite difference (default: 1e-5)
 * @returns Gradient vector
 */
export function numericalGradient(
  f: (x: number[]) => number,
  x: number[],
  h: number = 1e-5
): number[] {
  const n = x.length;
  const gradient = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const xPlus = [...x];
    const xMinus = [...x];
    xPlus[i] += h;
    xMinus[i] -= h;

    // Central difference
    gradient[i] = (f(xPlus) - f(xMinus)) / (2 * h);
  }

  return gradient;
}

/**
 * Compute Jacobian of a vector function
 *
 * @param f Vector function f: R^n -> R^m
 * @param x Point at which to compute Jacobian
 * @param h Step size
 * @returns Jacobian matrix [m x n]
 */
export function numericalJacobian(
  f: (x: number[]) => number[],
  x: number[],
  h: number = 1e-5
): number[][] {
  const y = f(x);
  const m = y.length;
  const n = x.length;
  const jacobian: number[][] = [];

  for (let i = 0; i < m; i++) {
    jacobian.push(new Array(n).fill(0));
  }

  for (let j = 0; j < n; j++) {
    const xPlus = [...x];
    const xMinus = [...x];
    xPlus[j] += h;
    xMinus[j] -= h;

    const fPlus = f(xPlus);
    const fMinus = f(xMinus);

    for (let i = 0; i < m; i++) {
      jacobian[i][j] = (fPlus[i] - fMinus[i]) / (2 * h);
    }
  }

  return jacobian;
}

// ============================================================================
// NTK Computation
// ============================================================================

/**
 * Compute Neural Tangent Kernel for a network
 *
 * Given a network f(x; θ) and data points {x_1, ..., x_n}, compute:
 *   Θ[i,j] = ∇_θ f(x_i; θ)^T ∇_θ f(x_j; θ)
 *
 * @param networkGradients Array of gradient vectors [∇_θ f(x_i)]
 * @returns NTK matrix analysis
 */
export function computeNTK(networkGradients: number[][]): NTKMatrix {
  const n = networkGradients.length;

  if (n === 0) {
    return {
      kernel: [],
      eigenvalues: [],
      conditionNumber: 1,
      effectiveRank: 0,
      trace: 0
    };
  }

  // Compute kernel matrix: Θ[i,j] = ⟨g_i, g_j⟩
  const kernel: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      let dot = 0;
      const gi = networkGradients[i];
      const gj = networkGradients[j];
      for (let k = 0; k < gi.length; k++) {
        dot += gi[k] * gj[k];
      }
      row.push(dot);
    }
    kernel.push(row);
  }

  // Compute eigenvalues using power iteration
  const eigenvalues = computeEigenvalues(kernel, Math.min(n, 20));

  // Compute statistics
  const trace = eigenvalues.reduce((a, b) => a + b, 0);
  const lambdaMax = eigenvalues[0] || 1;
  const lambdaMin = Math.max(eigenvalues[eigenvalues.length - 1] || 1e-10, 1e-10);
  const conditionNumber = lambdaMax / lambdaMin;
  const effectiveRank = trace / lambdaMax;

  return {
    kernel,
    eigenvalues,
    conditionNumber,
    effectiveRank,
    trace
  };
}

/**
 * Compute top eigenvalues using power iteration with deflation
 */
function computeEigenvalues(matrix: number[][], k: number): number[] {
  const n = matrix.length;
  if (n === 0) return [];

  const eigenvalues: number[] = [];
  const currentMatrix = matrix.map((row) => [...row]);

  for (let e = 0; e < k && e < n; e++) {
    // Power iteration for largest eigenvalue
    let v = new Array(n).fill(1 / Math.sqrt(n));
    let eigenvalue = 0;

    for (let iter = 0; iter < 100; iter++) {
      // Compute Av
      const Av = matrixVectorMultiply(currentMatrix, v);

      // Compute eigenvalue estimate (Rayleigh quotient)
      eigenvalue = dotProduct(v, Av);

      // Normalize
      const norm = Math.sqrt(dotProduct(Av, Av));
      if (norm < 1e-10) break;

      const newV = Av.map((x) => x / norm);

      // Check convergence
      const diff = Math.sqrt(newV.reduce((s, x, i) => s + (x - v[i]) ** 2, 0));
      v = newV;
      if (diff < 1e-8) break;
    }

    if (eigenvalue > 1e-10) {
      eigenvalues.push(eigenvalue);

      // Deflate: A := A - λ * v * v^T
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          currentMatrix[i][j] -= eigenvalue * v[i] * v[j];
        }
      }
    }
  }

  return eigenvalues.sort((a, b) => b - a);
}

function matrixVectorMultiply(A: number[][], v: number[]): number[] {
  return A.map((row) => dotProduct(row, v));
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// ============================================================================
// NTK Dynamics Analysis
// ============================================================================

/**
 * Analyze NTK dynamics during training
 *
 * @param initialGradients Gradients at initialization
 * @param currentGradients Gradients at current step
 * @param lazyThreshold Threshold for lazy training regime (default: 0.1)
 * @returns NTK dynamics analysis
 */
export function analyzeNTKDynamics(
  initialGradients: number[][],
  currentGradients: number[][],
  lazyThreshold: number = 0.1
): NTKDynamics {
  const initialNTK = computeNTK(initialGradients);
  const currentNTK = computeNTK(currentGradients);

  // Compute Frobenius norm of NTK change
  let changeNormSq = 0;
  let initialNormSq = 0;
  const n = initialNTK.kernel.length;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const diff = currentNTK.kernel[i][j] - initialNTK.kernel[i][j];
      changeNormSq += diff * diff;
      initialNormSq += initialNTK.kernel[i][j] * initialNTK.kernel[i][j];
    }
  }

  const ntkChange = Math.sqrt(changeNormSq);
  const initialNorm = Math.sqrt(initialNormSq);
  const relativeChange = initialNorm > 0 ? ntkChange / initialNorm : 0;

  const lazyRegime = relativeChange < lazyThreshold;
  const featureLearning = relativeChange > 2 * lazyThreshold;

  return {
    ntk: currentNTK,
    ntkChange,
    relativeChange,
    lazyRegime,
    featureLearning
  };
}

// ============================================================================
// Trainability Analysis
// ============================================================================

/**
 * Analyze network trainability based on NTK spectrum
 *
 * A network is considered trainable if:
 * 1. λ_min > 0 (NTK is positive definite)
 * 2. Condition number is not too large
 * 3. Spectral decay is not too fast
 *
 * @param ntk NTK matrix analysis
 * @param trainabilityThreshold Minimum λ_min for trainability (default: 1e-6)
 * @returns Trainability analysis
 */
export function analyzeTrainability(
  ntk: NTKMatrix,
  trainabilityThreshold: number = 1e-6
): TrainabilityAnalysis {
  const warnings: string[] = [];

  const lambdaMin = ntk.eigenvalues[ntk.eigenvalues.length - 1] || 0;
  const lambdaMax = ntk.eigenvalues[0] || 1;

  // Check basic trainability
  const isTrainable = lambdaMin > trainabilityThreshold;

  if (!isTrainable) {
    warnings.push(`Minimum eigenvalue ${lambdaMin.toExponential(2)} below threshold`);
  }

  // Condition number check
  if (ntk.conditionNumber > 1e6) {
    warnings.push(
      `High condition number ${ntk.conditionNumber.toExponential(2)} may slow convergence`
    );
  }

  // Effective rank check
  if (ntk.effectiveRank < ntk.eigenvalues.length / 10) {
    warnings.push(`Low effective rank suggests limited expressivity`);
  }

  // Convergence rate: f(t) ≈ f(0) exp(-λ_min * t)
  const convergenceRate = Math.max(lambdaMin, 1e-10);

  // Trainability score: combines multiple factors
  let trainabilityScore =
    Math.min(1, Math.log10(lambdaMin + 1e-10) / Math.log10(trainabilityThreshold)) * 0.4 +
    Math.min(1, 6 / Math.log10(Math.max(ntk.conditionNumber, 1) + 1)) * 0.3 +
    Math.min(1, ntk.effectiveRank / ntk.eigenvalues.length) * 0.3;

  trainabilityScore = Math.max(0, Math.min(1, trainabilityScore));

  return {
    lambdaMin,
    lambdaMax,
    trainabilityScore,
    convergenceRate,
    isTrainable,
    warnings
  };
}

// ============================================================================
// Spectrum Analysis
// ============================================================================

/**
 * Analyze the eigenvalue spectrum of NTK
 *
 * @param eigenvalues Sorted eigenvalues (descending)
 * @returns Spectrum analysis
 */
export function analyzeSpectrum(eigenvalues: number[]): SpectrumAnalysis {
  if (eigenvalues.length === 0) {
    return {
      spectrum: [],
      decayRate: 0,
      significantCount: 0,
      spectralEntropy: 0,
      gapRatio: 1
    };
  }

  const spectrum = [...eigenvalues];
  const total = spectrum.reduce((a, b) => a + b, 0);

  // Count significant eigenvalues (90% of trace)
  let cumulative = 0;
  let significantCount = 0;
  for (const lambda of spectrum) {
    cumulative += lambda;
    significantCount++;
    if (cumulative >= 0.9 * total) break;
  }

  // Fit power law decay: λ_k ≈ λ_1 * k^(-α)
  // log(λ_k) ≈ log(λ_1) - α * log(k)
  let decayRate = 0;
  if (spectrum.length > 2) {
    const logLambdas: number[] = [];
    const logIndices: number[] = [];

    for (let i = 0; i < spectrum.length; i++) {
      if (spectrum[i] > 1e-10) {
        logLambdas.push(Math.log(spectrum[i]));
        logIndices.push(Math.log(i + 1));
      }
    }

    if (logIndices.length > 2) {
      // Simple linear regression
      const n = logIndices.length;
      const sumX = logIndices.reduce((a, b) => a + b, 0);
      const sumY = logLambdas.reduce((a, b) => a + b, 0);
      const sumXY = logIndices.reduce((s, x, i) => s + x * logLambdas[i], 0);
      const sumXX = logIndices.reduce((s, x) => s + x * x, 0);

      decayRate = -(n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
      decayRate = Math.max(0, decayRate); // Ensure non-negative
    }
  }

  // Spectral entropy
  let spectralEntropy = 0;
  if (total > 0) {
    for (const lambda of spectrum) {
      if (lambda > 0) {
        const p = lambda / total;
        spectralEntropy -= p * Math.log(p);
      }
    }
  }

  // Gap ratio
  const gapRatio = spectrum.length > 1 && spectrum[1] > 0 ? spectrum[0] / spectrum[1] : 1;

  return {
    spectrum,
    decayRate,
    significantCount,
    spectralEntropy,
    gapRatio
  };
}

// ============================================================================
// NTK-based Prediction
// ============================================================================

/**
 * Predict training dynamics using NTK theory
 *
 * In the lazy training regime, the network output evolves as:
 *   f(x, t) = f(x, 0) + Θ(x, X) * Θ(X, X)^{-1} * (I - exp(-η * Θ(X, X) * t)) * (y - f(X, 0))
 *
 * @param ntk NTK matrix Θ(X, X)
 * @param initialOutputs f(X, 0) at training points
 * @param targets y at training points
 * @param learningRate η
 * @param steps Number of gradient descent steps t
 * @returns Predicted outputs after t steps
 */
export function predictNTKDynamics(
  ntk: NTKMatrix,
  initialOutputs: number[],
  targets: number[],
  learningRate: number,
  steps: number
): { predictedOutputs: number[]; expectedLoss: number } {
  const n = initialOutputs.length;

  if (n === 0 || n !== targets.length) {
    return { predictedOutputs: [], expectedLoss: 0 };
  }

  // Compute residual: r_0 = y - f(X, 0)
  const residual = targets.map((y, i) => y - initialOutputs[i]);

  // Compute exponential decay: (I - exp(-η * Θ * t))
  // Using eigendecomposition: exp(-η * Θ * t) ≈ Σ_i exp(-η * λ_i * t) * v_i * v_i^T
  // Simplified: assume diagonal approximation for small t

  // For each eigenvalue, compute decay factor
  const decayFactors = ntk.eigenvalues.map(
    (lambda) => 1 - Math.exp(-learningRate * lambda * steps)
  );

  // Approximate: Θ^{-1} * (I - exp(-η*Θ*t)) ≈ diag(decay_i / λ_i)
  // Simplified prediction using effective learning
  const effectiveLR = decayFactors.reduce((a, b) => a + b, 0) / ntk.eigenvalues.length;

  const predictedOutputs = initialOutputs.map((f0, i) => f0 + effectiveLR * residual[i]);

  // Expected loss: ||y - f(t)||^2
  const expectedLoss = predictedOutputs.reduce((s, f, i) => s + (targets[i] - f) ** 2, 0) / n;

  return { predictedOutputs, expectedLoss };
}

// ============================================================================
// Width-Depth Analysis
// ============================================================================

export interface WidthDepthAnalysis {
  /** Estimated effective width */
  effectiveWidth: number;
  /** NTK alignment with target function */
  alignment: number;
  /** Generalization gap estimate */
  generalizationGap: number;
  /** Regime: 'kernel' (lazy) or 'feature' (rich) */
  regime: 'kernel' | 'feature' | 'transition';
}

/**
 * Analyze width-depth trade-offs using NTK theory
 *
 * @param ntk NTK matrix
 * @param targets Target values
 * @param ntkChange Relative change in NTK during training
 */
export function analyzeWidthDepth(
  ntk: NTKMatrix,
  targets: number[],
  ntkChange: number
): WidthDepthAnalysis {
  // Effective width: proportional to trace / n
  const n = targets.length;
  const effectiveWidth = ntk.trace / n;

  // Alignment: how well NTK can fit targets
  // Higher alignment → easier to fit
  let alignment = 0;
  if (ntk.kernel.length > 0) {
    const targetNorm = Math.sqrt(targets.reduce((s, t) => s + t * t, 0));
    if (targetNorm > 0) {
      // Compute y^T Θ y / (||y||^2 * trace(Θ))
      let yThetaY = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          yThetaY += targets[i] * ntk.kernel[i][j] * targets[j];
        }
      }
      alignment = yThetaY / (targetNorm * targetNorm * ntk.trace);
    }
  }

  // Generalization gap estimate: based on effective rank
  // Lower effective rank → higher generalization gap
  const generalizationGap = 1 - ntk.effectiveRank / n;

  // Determine regime
  let regime: 'kernel' | 'feature' | 'transition';
  if (ntkChange < 0.1) {
    regime = 'kernel';
  } else if (ntkChange > 0.5) {
    regime = 'feature';
  } else {
    regime = 'transition';
  }

  return {
    effectiveWidth,
    alignment,
    generalizationGap,
    regime
  };
}

// ============================================================================
// Automatic Alerts
// ============================================================================

export interface NTKAlert {
  severity: 'info' | 'warning' | 'critical';
  message: string;
  recommendation: string;
}

/**
 * Generate alerts based on NTK analysis
 */
export function generateNTKAlerts(
  trainability: TrainabilityAnalysis,
  dynamics?: NTKDynamics,
  spectrum?: SpectrumAnalysis
): NTKAlert[] {
  const alerts: NTKAlert[] = [];

  // Trainability alerts
  if (!trainability.isTrainable) {
    alerts.push({
      severity: 'critical',
      message: `Network may not be trainable: λ_min = ${trainability.lambdaMin.toExponential(2)}`,
      recommendation: 'Increase network width or use better initialization'
    });
  }

  if (trainability.trainabilityScore < 0.3) {
    alerts.push({
      severity: 'warning',
      message: `Low trainability score: ${(trainability.trainabilityScore * 100).toFixed(1)}%`,
      recommendation: 'Consider adjusting architecture or learning rate'
    });
  }

  // Dynamics alerts
  if (dynamics) {
    if (dynamics.featureLearning) {
      alerts.push({
        severity: 'info',
        message: 'Feature learning regime detected (NTK changing significantly)',
        recommendation: 'Good for representation learning; may need more epochs'
      });
    }

    if (dynamics.lazyRegime) {
      alerts.push({
        severity: 'info',
        message: 'Lazy training regime detected (NTK approximately constant)',
        recommendation: 'Training behaves like kernel regression'
      });
    }
  }

  // Spectrum alerts
  if (spectrum) {
    if (spectrum.decayRate > 2) {
      alerts.push({
        severity: 'warning',
        message: `Fast spectral decay (rate ${spectrum.decayRate.toFixed(2)})`,
        recommendation: 'Only a few directions contribute; may limit expressivity'
      });
    }

    if (spectrum.gapRatio > 100) {
      alerts.push({
        severity: 'warning',
        message: `Large spectral gap (λ_1/λ_2 = ${spectrum.gapRatio.toFixed(1)})`,
        recommendation: 'Network may be dominated by a single principal direction'
      });
    }
  }

  return alerts;
}

// ============================================================================
// Utility: NTK Summary
// ============================================================================

export interface NTKSummary {
  ntk: NTKMatrix;
  trainability: TrainabilityAnalysis;
  spectrum: SpectrumAnalysis;
  alerts: NTKAlert[];
}

/**
 * Compute comprehensive NTK summary
 *
 * @param networkGradients Gradient vectors for each data point
 * @returns Complete NTK analysis summary
 */
export function computeNTKSummary(networkGradients: number[][]): NTKSummary {
  const ntk = computeNTK(networkGradients);
  const trainability = analyzeTrainability(ntk);
  const spectrum = analyzeSpectrum(ntk.eigenvalues);
  const alerts = generateNTKAlerts(trainability, undefined, spectrum);

  return {
    ntk,
    trainability,
    spectrum,
    alerts
  };
}
