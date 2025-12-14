/**
 * Mathematical Analysis Module
 *
 * This module provides rigorous mathematical analysis tools for the
 * Neuro-Lingua neural language model, including:
 *
 * - Numerical stability operations (Kahan summation, stable norms)
 * - Convergence theorems for optimizers (Sophia, Lion)
 * - Information-theoretic analysis (KSG, rate-distortion)
 * - Statistical sampling analysis (entropy tests, Mirostat)
 * - Spectral graph analysis (attention patterns)
 * - Approximation theory bounds (Wedin, Nyström)
 *
 * All algorithms include mathematical guarantees and error bounds.
 */

// Numerical stability operations
export {
  kahanSum,
  neumaierSum,
  pairwiseSum,
  stableFrobeniusNorm,
  stableSpectralNorm,
  stableMatrixNorm,
  precisionErrorBound,
  estimateConditionNumber,
  checkNumericalStability,
  stableVariance,
  initParallelVariance,
  updateParallelVariance,
  mergeParallelVariance,
  getParallelVariance,
  type CompensatedSumResult,
  type MatrixNormOptions,
  type MatrixNormResult,
  type PrecisionErrorBound,
  type StabilityCheck,
  type ParallelVarianceState
} from './numerics';

// Convergence analysis for optimizers
export {
  sophiaConvergenceTheorem,
  lionConvergenceTheorem,
  sgdStronglyConvexTheorem,
  analyzeConvergence,
  verifyConvergenceConditions,
  analyzeQuasiHessian,
  analyzeOptimizationStability,
  type ConvergenceAssumptions,
  type ConvergenceRate,
  type ConvergenceTheorem,
  type PracticalConvergenceAnalysis,
  type OptimizerAnalysisConfig,
  type ConditionVerification,
  type QuasiHessianAnalysis,
  type OptimizationLyapunov
} from './convergence';

// Advanced information theory
export {
  ksgMutualInformation,
  knnEntropy,
  estimateMutualInformationAdvanced,
  computeRateDistortionCurve,
  computeInformationPlane,
  estimateFisherInformation,
  type MIEstimator,
  type MIEstimatorConfig,
  type MIEstimateResult,
  type RateDistortionPoint,
  type RateDistortionConfig,
  type InformationPlanePoint,
  type FisherInformation
} from './information_theory';

// Statistical sampling analysis
export {
  computeEntropy,
  computeCrossEntropy,
  computeKLDivergence,
  computeSurprise,
  entropyDistributionTest,
  analyzeMirostatConvergence,
  analyzeSamplingQuality,
  analyzeTemperatureCalibration,
  analyzeNucleusThreshold,
  type EntropyDistributionTest,
  type MirostatConvergenceAnalysis,
  type SamplingQualityMetrics,
  type TemperatureCalibration,
  type NucleusThresholdAnalysis
} from './sampling_analysis';

// Spectral graph analysis
export {
  computeLaplacian,
  analyzeAttentionGraph,
  checkExpanderProperties,
  analyzeSparseAttentionPattern,
  generateSparsePattern,
  type AttentionGraphAnalysis,
  type ExpanderProperties,
  type SparseAttentionPattern
} from './spectral_graph';

// Approximation theory
export {
  wedinBound,
  nystromApproximation,
  analyzeLowRankApproximation,
  randomizedSVD,
  type WedinBound,
  type NystromApproximation,
  type LowRankAnalysis,
  type RandomizedSVD
} from './approximation';

// Re-export existing modules with explicit exports to avoid conflicts
// analysis.ts exports
export {
  spectralRadius,
  analyzeLyapunov,
  ANALYSIS_ASSUMPTIONS,
  type Matrix,
  type SpectralRadiusOptions,
  type SpectralRadiusResult,
  type LyapunovAnalysisOptions,
  type LyapunovAnalysisResult
} from './analysis';

// statistics.ts exports (Vector type only, Matrix already exported from analysis)
export {
  empiricalFisherFromGradients,
  fisherHessianStatistics,
  fisherDiagonalScaling,
  fisherQuadraticForm,
  type Vector,
  type FisherStatistics,
  type FisherFromGradientsOptions
} from './statistics';
