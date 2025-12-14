/**
 * Convergence Analysis for Optimization Algorithms
 *
 * This module provides formal convergence theorems and analysis for:
 * - Sophia optimizer (second-order stochastic optimization)
 * - Lion optimizer (sign-based momentum)
 * - General stochastic gradient methods
 *
 * Mathematical Framework:
 * - Lipschitz gradient conditions
 * - Strong convexity analysis
 * - Non-convex convergence guarantees
 * - Rate of convergence bounds
 *
 * References:
 * - Liu et al. (2023) "Sophia: A Scalable Second-order Optimizer"
 * - Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
 * - Bottou et al. (2018) "Optimization Methods for Large-Scale Machine Learning"
 */

/**
 * Assumptions required for convergence analysis
 */
export interface ConvergenceAssumptions {
  /** L-smoothness: ||∇f(x) - ∇f(y)|| ≤ L||x - y|| */
  lipschitzGradient: boolean;
  /** Lipschitz constant for gradient */
  lipschitzConstant?: number;

  /** M-Lipschitz Hessian: ||∇²f(x) - ∇²f(y)|| ≤ M||x - y|| */
  lipschitzHessian: boolean;
  /** Lipschitz constant for Hessian */
  hessianLipschitz?: number;

  /** μ-strong convexity: f(y) ≥ f(x) + ∇f(x)·(y-x) + (μ/2)||y-x||² */
  strongConvexity: boolean;
  /** Strong convexity parameter */
  strongConvexityParameter?: number;

  /** Bounded stochastic gradient variance */
  boundedVariance: boolean;
  /** Variance bound σ² */
  varianceBound?: number;

  /** Bounded stochastic gradient: ||g_t|| ≤ G */
  boundedGradient: boolean;
  /** Gradient bound */
  gradientBound?: number;

  /** Bounded iterates: ||θ_t|| ≤ D */
  boundedIterates: boolean;
  /** Iterate bound */
  iterateBound?: number;
}

/**
 * Convergence rate classification
 */
export type ConvergenceRate =
  | 'sublinear' // O(1/√T) or O(1/T)
  | 'linear'    // O(ρ^T) where 0 < ρ < 1
  | 'quadratic' // O(ρ^(2^T))
  | 'superlinear'; // Faster than linear

/**
 * Convergence theorem result
 */
export interface ConvergenceTheorem {
  /** Name of the theorem */
  name: string;
  /** Required assumptions */
  assumptions: ConvergenceAssumptions;
  /** Type of convergence rate */
  rate: ConvergenceRate;
  /** Mathematical expression for convergence bound */
  boundExpression: string;
  /** Proof sketch or key steps */
  proofSketch: string;
  /** Reference to academic paper */
  reference?: string;
  /** Practical implications */
  implications: string[];
}

/**
 * Sophia Optimizer Convergence Analysis
 *
 * Theorem: Under L-smooth, bounded variance assumptions, Sophia achieves:
 *
 * E[||∇f(θ_T)||²] ≤ O(1/T) + O(σ²/T)
 *
 * for non-convex objectives, with implicit learning rate adaptation
 * through diagonal Hessian preconditioning.
 */
export function sophiaConvergenceTheorem(): ConvergenceTheorem {
  const assumptions: ConvergenceAssumptions = {
    lipschitzGradient: true,
    lipschitzConstant: undefined, // L
    lipschitzHessian: true,
    hessianLipschitz: undefined, // M
    strongConvexity: false,
    boundedVariance: true,
    varianceBound: undefined, // σ²
    boundedGradient: true,
    gradientBound: undefined, // G
    boundedIterates: false
  };

  const proofSketch = `
    1. Define Lyapunov function V_t = f(θ_t) - f*

    2. Sophia update: θ_{t+1} = θ_t - η × clip(m_t / h_t, -ρ, ρ)
       where m_t is momentum, h_t is Hessian diagonal estimate

    3. By L-smoothness:
       f(θ_{t+1}) ≤ f(θ_t) + ∇f(θ_t)·(θ_{t+1} - θ_t) + (L/2)||θ_{t+1} - θ_t||²

    4. Taking expectation over stochastic gradients:
       E[f(θ_{t+1})] ≤ E[f(θ_t)] - η(1 - ηL/2h_min) E[||∇f(θ_t)||²/h_t] + O(σ²η²)

    5. The Hessian preconditioning adapts η per-dimension:
       - Large curvature → smaller effective LR
       - Small curvature → larger effective LR

    6. Clipping bound ρ ensures stability: max update magnitude = ηρ

    7. Summing over T iterations and rearranging:
       (1/T) Σ E[||∇f(θ_t)||²] ≤ O((f(θ_0) - f*)/(ηT)) + O(σ²η)

    8. Optimal η ~ 1/√T gives O(1/√T) rate
       With variance reduction: O(1/T)
  `;

  return {
    name: 'Sophia Non-Convex Convergence',
    assumptions,
    rate: 'sublinear',
    boundExpression: 'E[||∇f(θ_T)||²] ≤ (f(θ_0) - f*)/(ηT) + Lσ²η',
    proofSketch,
    reference: 'Liu et al. (2023) "Sophia: A Scalable Stochastic Second-order Optimizer"',
    implications: [
      'Convergence to stationary point guaranteed for non-convex objectives',
      'Hessian diagonal provides dimension-wise adaptation (vs global LR)',
      'Clipping bound ρ trades off convergence speed vs stability',
      '2× faster than Adam empirically due to curvature information'
    ]
  };
}

/**
 * Lion Optimizer Convergence Analysis
 *
 * Theorem: Under L-smooth, bounded gradient assumptions, Lion achieves:
 *
 * (1/T) Σ E[||∇f(θ_t)||] ≤ O(1/√T)
 *
 * with constant step sizes. The sign operation normalizes gradients
 * implicitly, providing robustness to gradient scaling.
 */
export function lionConvergenceTheorem(): ConvergenceTheorem {
  const assumptions: ConvergenceAssumptions = {
    lipschitzGradient: true,
    lipschitzConstant: undefined, // L
    lipschitzHessian: false,
    strongConvexity: false,
    boundedVariance: true,
    varianceBound: undefined, // σ²
    boundedGradient: true,
    gradientBound: undefined, // G
    boundedIterates: true,
    iterateBound: undefined // D
  };

  const proofSketch = `
    1. Lion update: θ_{t+1} = θ_t - η × sign(β₁m_t + (1-β₁)g_t)
       where m_t is momentum, g_t is stochastic gradient

    2. Key insight: sign(·) normalizes gradient to unit ∞-norm
       Update magnitude = η√d (in 2-norm)

    3. Define potential function:
       Φ_t = f(θ_t) + (η/2)||m_t||_∞

    4. By L-smoothness and bounded gradients:
       E[Φ_{t+1}] ≤ E[Φ_t] - (η/√d) E[||∇f(θ_t)||_1] + O(η²L√d)

    5. Since ||∇f||_1 ≥ ||∇f||_2 / √d:
       E[Φ_{t+1}] ≤ E[Φ_t] - (η/d) E[||∇f(θ_t)||] + O(η²L√d)

    6. Summing and using bounded iterates:
       (1/T) Σ E[||∇f(θ_t)||] ≤ O(d(f(θ_0) - f*)/ηT) + O(ηL√d)

    7. The sign operation provides:
       - Implicit gradient clipping (magnitude = 1 per dimension)
       - Scale invariance to gradient magnitude
       - Robustness to outlier gradients

    8. Memory efficiency: Only 1 momentum buffer (vs 2 for Adam)
  `;

  return {
    name: 'Lion Non-Convex Convergence',
    assumptions,
    rate: 'sublinear',
    boundExpression: '(1/T) Σ E[||∇f(θ_t)||] ≤ O((f₀ - f*)/√T)',
    proofSketch,
    reference: 'Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"',
    implications: [
      'Sign normalization provides implicit gradient clipping',
      '50% memory savings vs Adam (1 buffer vs 2)',
      'Lower effective learning rate needed (vs Adam)',
      'Better for large-batch training due to sign averaging'
    ]
  };
}

/**
 * Strongly Convex SGD Convergence (for comparison)
 */
export function sgdStronglyConvexTheorem(): ConvergenceTheorem {
  const assumptions: ConvergenceAssumptions = {
    lipschitzGradient: true,
    lipschitzConstant: undefined, // L
    lipschitzHessian: false,
    strongConvexity: true,
    strongConvexityParameter: undefined, // μ
    boundedVariance: true,
    varianceBound: undefined, // σ²
    boundedGradient: false,
    boundedIterates: false
  };

  const proofSketch = `
    1. SGD update: θ_{t+1} = θ_t - η_t g_t

    2. By μ-strong convexity:
       f(θ*) ≥ f(θ_t) + ∇f(θ_t)·(θ* - θ_t) + (μ/2)||θ* - θ_t||²

    3. Rearranging and taking expectation:
       E[||θ_{t+1} - θ*||²] ≤ (1 - μη_t)E[||θ_t - θ*||²] + η_t²σ²

    4. With decreasing step size η_t = 2/(μ(t+1)):
       E[||θ_T - θ*||²] = O(σ²/(μ²T))

    5. Condition number κ = L/μ determines practical convergence:
       - Small κ: Fast convergence
       - Large κ: Slow convergence, preconditioning helps
  `;

  return {
    name: 'SGD Strongly Convex Convergence',
    assumptions,
    rate: 'linear',
    boundExpression: 'E[||θ_T - θ*||²] ≤ O(σ²/(μ²T))',
    proofSketch,
    reference: 'Bottou et al. (2018) "Optimization Methods for Large-Scale ML"',
    implications: [
      'Linear convergence to optimum for strongly convex objectives',
      'Rate depends on condition number κ = L/μ',
      'Variance σ² determines the noise floor',
      'Decreasing step sizes required for exact convergence'
    ]
  };
}

/**
 * Practical convergence analysis given optimizer configuration
 */
export interface PracticalConvergenceAnalysis {
  /** Estimated iteration count to reach tolerance */
  estimatedIterations: number;
  /** Effective learning rate accounting for preconditioning */
  effectiveLearningRate: number;
  /** Expected gradient norm after T iterations */
  expectedGradientNorm: number;
  /** Stability assessment */
  stabilityScore: number; // 0-1, higher is more stable
  /** Recommendations for improved convergence */
  recommendations: string[];
}

/**
 * Configuration for convergence analysis
 */
export interface OptimizerAnalysisConfig {
  optimizer: 'sophia' | 'lion' | 'adam' | 'sgd';
  learningRate: number;
  beta1?: number;
  beta2?: number;
  weightDecay?: number;
  clipBound?: number; // ρ for Sophia
  estimatedLipschitz?: number;
  estimatedVariance?: number;
  parameterCount: number;
}

/**
 * Analyze practical convergence properties
 */
export function analyzeConvergence(
  config: OptimizerAnalysisConfig,
  targetGradientNorm: number,
  initialLoss: number
): PracticalConvergenceAnalysis {
  const {
    optimizer,
    learningRate,
    beta1 = 0.9,
    beta2 = 0.99,
    weightDecay = 0.01,
    clipBound = 1.0,
    estimatedLipschitz = 1.0,
    estimatedVariance = 0.01,
    parameterCount
  } = config;

  let effectiveLR = learningRate;
  let stabilityScore = 1.0;
  const recommendations: string[] = [];

  // Optimizer-specific adjustments
  switch (optimizer) {
    case 'sophia':
      // Sophia: LR is scaled by inverse Hessian diagonal
      // Effective LR ~ lr * sqrt(d) / h_avg
      effectiveLR = learningRate * Math.sqrt(parameterCount);
      stabilityScore = Math.min(1.0, clipBound / (2 * learningRate * estimatedLipschitz));

      if (learningRate > 1e-3) {
        recommendations.push('Sophia typically uses lr ~ 1e-4; consider reducing');
      }
      if (clipBound > 2.0) {
        recommendations.push('High clip bound may cause instability');
      }
      break;

    case 'lion':
      // Lion: Sign operation normalizes updates
      // Effective LR per dimension = lr
      effectiveLR = learningRate * Math.sqrt(parameterCount);
      stabilityScore = Math.min(1.0, 1 / (learningRate * estimatedLipschitz));

      if (learningRate > 3e-4) {
        recommendations.push('Lion uses lower LR than Adam; try 1e-4 to 3e-4');
      }
      break;

    case 'adam':
      // Adam: Adaptive LR via second moment
      effectiveLR = learningRate / Math.sqrt(1 - beta2);
      stabilityScore = Math.min(1.0, 1 / (effectiveLR * estimatedLipschitz));

      if (beta1 < 0.8) {
        recommendations.push('Low β₁ may cause oscillations');
      }
      break;

    case 'sgd':
      effectiveLR = learningRate;
      stabilityScore = Math.min(1.0, 2 / (learningRate * estimatedLipschitz));

      if (learningRate > 2 / estimatedLipschitz) {
        recommendations.push('Learning rate may exceed stability bound 2/L');
      }
      break;
  }

  // Estimate iterations needed
  // Using approximate bound: ||∇f||² ~ (f₀ - f*) / (η * T)
  const lossGap = initialLoss; // Assume f* ≈ 0 for simplicity
  const targetNormSq = targetGradientNorm * targetGradientNorm;

  // T ~ (f₀ - f*) / (η × ||∇f||²_target)
  let estimatedIterations = Math.ceil(lossGap / (effectiveLR * targetNormSq));

  // Add variance contribution
  const varianceContribution = estimatedVariance / (learningRate * targetNormSq);
  estimatedIterations = Math.max(estimatedIterations, Math.ceil(varianceContribution));

  // Cap at reasonable maximum
  estimatedIterations = Math.min(estimatedIterations, 1e6);

  // Compute expected gradient norm after T iterations
  const expectedGradientNorm = Math.sqrt(
    lossGap / (effectiveLR * estimatedIterations) + estimatedVariance * learningRate
  );

  // Weight decay check
  if (weightDecay > 0.1) {
    recommendations.push('High weight decay may over-regularize');
  }

  // Stability warning
  if (stabilityScore < 0.5) {
    recommendations.push('Low stability score; consider reducing learning rate');
  }

  return {
    estimatedIterations,
    effectiveLearningRate: effectiveLR,
    expectedGradientNorm,
    stabilityScore,
    recommendations
  };
}

/**
 * Verify optimizer satisfies convergence conditions
 */
export interface ConditionVerification {
  condition: string;
  satisfied: boolean;
  value?: number;
  threshold?: number;
  explanation: string;
}

/**
 * Check convergence conditions for an optimizer setup
 */
export function verifyConvergenceConditions(
  config: OptimizerAnalysisConfig,
  lipschitzEstimate?: number,
  varianceEstimate?: number
): ConditionVerification[] {
  const checks: ConditionVerification[] = [];

  const L = lipschitzEstimate ?? 1.0;
  const sigma2 = varianceEstimate ?? 0.01;

  // Learning rate stability check
  const lrStabilityBound = 2 / L;
  checks.push({
    condition: 'Learning rate stability (η ≤ 2/L)',
    satisfied: config.learningRate <= lrStabilityBound,
    value: config.learningRate,
    threshold: lrStabilityBound,
    explanation: `For L-smooth functions, step size must be ≤ 2/L for gradient descent stability`
  });

  // Momentum stability
  if (config.beta1 !== undefined) {
    const momentumBound = 1 / (1 + config.learningRate * L);
    checks.push({
      condition: 'Momentum stability (β₁ reasonable)',
      satisfied: config.beta1 >= momentumBound && config.beta1 < 1,
      value: config.beta1,
      threshold: momentumBound,
      explanation: `Momentum parameter should be in [${momentumBound.toFixed(3)}, 1) for stability`
    });
  }

  // Variance-to-LR ratio
  const varianceRatio = sigma2 / config.learningRate;
  checks.push({
    condition: 'Variance-to-LR ratio (σ²/η manageable)',
    satisfied: varianceRatio < 100,
    value: varianceRatio,
    threshold: 100,
    explanation: 'High variance relative to LR leads to noisy convergence'
  });

  // Sophia-specific checks
  if (config.optimizer === 'sophia') {
    const rho = config.clipBound ?? 1.0;
    checks.push({
      condition: 'Sophia clip bound (ρ ≥ expected gradient ratio)',
      satisfied: rho >= 0.5 && rho <= 5.0,
      value: rho,
      explanation: 'Clip bound too small limits convergence; too large causes instability'
    });
  }

  // Lion-specific checks
  if (config.optimizer === 'lion') {
    // Sign operation means effective update magnitude is constant
    const signLR = config.learningRate * Math.sqrt(config.parameterCount);
    checks.push({
      condition: 'Lion effective LR (η√d reasonable)',
      satisfied: signLR < 1.0,
      value: signLR,
      threshold: 1.0,
      explanation: 'Lion update magnitude scales with √d; total should be < 1'
    });
  }

  return checks;
}

/**
 * Spectral analysis for quasi-Hessian in Sophia
 */
export interface QuasiHessianAnalysis {
  /** Estimated condition number of preconditioner */
  conditionNumber: number;
  /** Maximum eigenvalue estimate */
  maxEigenvalue: number;
  /** Minimum eigenvalue estimate */
  minEigenvalue: number;
  /** Effective dimension (trace/maxEig) */
  effectiveDimension: number;
  /** Recommendations for adjustment */
  recommendations: string[];
}

/**
 * Analyze Sophia's diagonal Hessian estimate
 */
export function analyzeQuasiHessian(hessianDiagonal: number[]): QuasiHessianAnalysis {
  if (hessianDiagonal.length === 0) {
    return {
      conditionNumber: 1,
      maxEigenvalue: 0,
      minEigenvalue: 0,
      effectiveDimension: 0,
      recommendations: ['Empty Hessian diagonal']
    };
  }

  // For diagonal matrix, eigenvalues = diagonal entries
  const sorted = [...hessianDiagonal].sort((a, b) => b - a);
  const maxEig = sorted[0];
  const minEig = Math.max(sorted[sorted.length - 1], 1e-10);

  const conditionNumber = maxEig / minEig;
  const trace = hessianDiagonal.reduce((a, b) => a + b, 0);
  const effectiveDimension = trace / maxEig;

  const recommendations: string[] = [];

  if (conditionNumber > 1e6) {
    recommendations.push('High condition number: consider regularizing small curvatures');
  }

  if (effectiveDimension < hessianDiagonal.length / 10) {
    recommendations.push('Low effective dimension: curvature concentrated in few directions');
  }

  // Check for very small values (near-zero curvature)
  const smallCount = hessianDiagonal.filter(h => h < 1e-8).length;
  if (smallCount > hessianDiagonal.length * 0.1) {
    recommendations.push(`${smallCount} dimensions have near-zero curvature; may cause large updates`);
  }

  return {
    conditionNumber,
    maxEigenvalue: maxEig,
    minEigenvalue: minEig,
    effectiveDimension,
    recommendations
  };
}

/**
 * Lyapunov analysis for optimization dynamics
 *
 * For the discrete dynamical system θ_{t+1} = θ_t - η∇f(θ_t),
 * we analyze stability via the linearized system around optimum.
 */
export interface OptimizationLyapunov {
  /** Spectral radius of update operator */
  spectralRadius: number;
  /** Lyapunov exponent (negative = stable) */
  lyapunovExponent: number;
  /** Stable under current learning rate */
  stable: boolean;
  /** Critical learning rate (max stable) */
  criticalLR: number;
  /** Stability margin */
  margin: number;
}

/**
 * Analyze Lyapunov stability of optimization
 */
export function analyzeOptimizationStability(
  hessianEigenvalues: number[],
  learningRate: number,
  momentum = 0
): OptimizationLyapunov {
  if (hessianEigenvalues.length === 0) {
    return {
      spectralRadius: 0,
      lyapunovExponent: 0,
      stable: true,
      criticalLR: Infinity,
      margin: 1
    };
  }

  const maxEig = Math.max(...hessianEigenvalues);
  const minEig = Math.min(...hessianEigenvalues.filter(e => e > 0));

  // For gradient descent: I - ηH has eigenvalues 1 - ηλ_i
  // Stable when |1 - ηλ| < 1 for all eigenvalues
  // => 0 < η < 2/λ_max

  let spectralRadius: number;
  if (momentum === 0) {
    // Simple GD: ρ = max(|1 - ηλ_max|, |1 - ηλ_min|)
    spectralRadius = Math.max(
      Math.abs(1 - learningRate * maxEig),
      Math.abs(1 - learningRate * minEig)
    );
  } else {
    // Heavy ball momentum: more complex eigenvalue analysis
    // ρ ≈ max(√β, |1 - ηλ|) approximately
    spectralRadius = Math.max(
      Math.sqrt(momentum),
      Math.abs(1 - learningRate * maxEig)
    );
  }

  const criticalLR = 2 / maxEig;
  const stable = spectralRadius < 1;
  const lyapunovExponent = Math.log(spectralRadius);
  const margin = criticalLR - learningRate;

  return {
    spectralRadius,
    lyapunovExponent,
    stable,
    criticalLR,
    margin
  };
}
