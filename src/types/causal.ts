/**
 * Type definitions for the Probabilistic Dynamic Causal Inference System
 *
 * This module implements a causal inference framework based on a Directed Acyclic Graph (DAG)
 * for dynamic data, accounting for temporal dependencies, unmeasured confounders, and
 * selection mechanisms.
 *
 * Mathematical foundation:
 * - X_{it}: Measured feature vector
 * - Y_{it}: Continuous unobserved change
 * - Ỹ_{it} = Q_{θt}(Y_{it}): Quantized observed change
 * - Z_{it} ∈ {A, B}: Reinforcement policy
 * - U_i: Unmeasured confounders (student-level)
 * - θ_t: Quantization parameters
 *
 * Reference: Rubin (1974), Pearl (2009), Tishby et al. (1999)
 */

// ============================================================================
// Core Variable Types
// ============================================================================

/**
 * Policy type for reinforcement selection
 * A: Control/baseline policy
 * B: Treatment/intervention policy
 */
export type Policy = 'A' | 'B';

/**
 * Quantization bin identifier
 * Represents discrete levels after quantization
 */
export type BinIndex = number;

/**
 * Measured feature vector X_{it} for student i at time t
 */
export interface FeatureVector {
  /** Unique student identifier */
  studentId: string;
  /** Time step (e.g., day in academic year) */
  timeStep: number;
  /** Feature values (academic metrics, behavioral indicators, etc.) */
  values: number[];
  /** Feature names for interpretability */
  featureNames?: string[];
}

/**
 * Continuous outcome Y_{it} - unobserved true change
 */
export interface ContinuousOutcome {
  studentId: string;
  timeStep: number;
  /** The true continuous value (available in historical data) */
  value: number;
}

/**
 * Quantized outcome Ỹ_{it} = Q_{θt}(Y_{it})
 */
export interface QuantizedOutcome {
  studentId: string;
  timeStep: number;
  /** Bin index after quantization */
  binIndex: BinIndex;
  /** Original bin boundaries for interpretability */
  binBoundaries?: [number, number];
}

/**
 * Policy selection Z_{it}
 */
export interface PolicySelection {
  studentId: string;
  timeStep: number;
  /** Selected policy */
  policy: Policy;
  /** Propensity score P(Z=B|X,Y_{t-1},θ_{t-1}) */
  propensityScore?: number;
}

/**
 * Unmeasured confounders U_i (student-level, time-invariant)
 * These are latent variables that affect both outcomes and selection
 */
export interface UnmeasuredConfounder {
  studentId: string;
  /** Latent factor values (estimated or simulated) */
  values: number[];
  /** Dimension of the confounder space */
  dimension: number;
}

/**
 * Classroom/teacher group effect V_{C_i}
 * Hierarchical random effect shared among students in same group
 */
export interface GroupEffect {
  groupId: string;
  /** Random effect values */
  values: number[];
  /** Students in this group */
  studentIds: string[];
}

// ============================================================================
// Quantization Types
// ============================================================================

/**
 * Quantization function type
 */
export type QuantizationMethod = 'uniform' | 'entropy' | 'adaptive' | 'symmetric';

/**
 * Quantization parameters θ_t
 */
export interface QuantizationParams {
  /** Time step */
  timeStep: number;
  /** Number of bins K */
  numBins: number;
  /** Bin boundaries τ_1, ..., τ_{K-1} */
  boundaries: number[];
  /** Quantization method used */
  method: QuantizationMethod;
  /** Whether bins are symmetric around zero */
  symmetric: boolean;
}

/**
 * Dequantization mapping: E[Y|Ỹ=k,θ]
 */
export interface DequantizationMapping {
  /** Bin index */
  binIndex: BinIndex;
  /** Expected value given bin */
  expectedValue: number;
  /** Variance within bin */
  variance: number;
  /** Number of samples used for estimation */
  sampleCount: number;
}

// ============================================================================
// Model Configuration
// ============================================================================

/**
 * Configuration for the dynamic outcome model
 * Y_{it} = β_0 + β_Z Z_{it} + β_X^T X_{it} + β_U^T U_i + ρ Y_{i,t-1} + ε_{it}
 */
export interface OutcomeModelConfig {
  /** Intercept β_0 */
  intercept: number;
  /** Treatment effect coefficient β_Z */
  treatmentCoefficient: number;
  /** Feature coefficients β_X */
  featureCoefficients: number[];
  /** Confounder coefficients β_U */
  confounderCoefficients: number[];
  /** Autoregressive coefficient ρ ∈ (-1, 1) */
  autoregCoefficient: number;
  /** Noise standard deviation σ */
  noiseStd: number;
  /** Group effect coefficients β_V (optional) */
  groupCoefficients?: number[];
}

/**
 * Configuration for the selection model
 * P(Z=B|X,Y_{t-1},θ_{t-1},U) = σ(γ_0 + γ_X^T X + γ_Y Y_{t-1} + γ_θ^T θ + γ_U^T U)
 */
export interface SelectionModelConfig {
  /** Intercept γ_0 */
  intercept: number;
  /** Feature coefficients γ_X */
  featureCoefficients: number[];
  /** Previous outcome coefficient γ_Y */
  previousOutcomeCoefficient: number;
  /** Quantization parameter coefficients γ_θ */
  quantizationCoefficients: number[];
  /** Confounder coefficients γ_U */
  confounderCoefficients: number[];
}

/**
 * Configuration for unmeasured confounders
 * U_i ~ N(μ_U, Σ_U)
 */
export interface ConfounderConfig {
  /** Dimension of confounder space */
  dimension: number;
  /** Mean vector μ_U */
  mean: number[];
  /** Covariance matrix Σ_U (stored as flattened array, row-major) */
  covariance: number[];
}

/**
 * Main configuration for the causal inference engine
 */
export interface CausalModelConfig {
  /** Outcome model configuration */
  outcomeModel: OutcomeModelConfig;
  /** Selection model configuration */
  selectionModel: SelectionModelConfig;
  /** Confounder configuration */
  confounderConfig: ConfounderConfig;
  /** Initial quantization parameters */
  initialQuantization: QuantizationParams;
  /** Number of students N */
  numStudents: number;
  /** Number of time steps T */
  numTimeSteps: number;
  /** Feature dimension */
  featureDimension: number;
  /** Random seed for reproducibility */
  seed?: number;
}

// ============================================================================
// Causal Inference Types
// ============================================================================

/**
 * Average Treatment Effect (ATE)
 * τ = E[Y(B) - Y(A)]
 */
export interface AverageTreatmentEffect {
  /** Point estimate of ATE */
  estimate: number;
  /** Standard error */
  standardError: number;
  /** 95% confidence interval */
  confidenceInterval: [number, number];
  /** Number of observations used */
  numObservations: number;
  /** Estimation method used */
  method: 'naive' | 'ipw' | 'aipw' | 'doubly_robust';
}

/**
 * Propensity score model estimate
 */
export interface PropensityEstimate {
  studentId: string;
  timeStep: number;
  /** P(Z=B|X,Y_{t-1},θ_{t-1}) */
  score: number;
  /** Features used for estimation */
  features: number[];
}

/**
 * Outcome model prediction
 */
export interface OutcomePrediction {
  studentId: string;
  timeStep: number;
  policy: Policy;
  /** E[Y|Z,X,Y_{t-1}] */
  prediction: number;
  /** Prediction variance */
  variance: number;
}

/**
 * AIPW (Augmented Inverse Propensity Weighted) estimator components
 */
export interface AIPWComponents {
  /** IPW component */
  ipwTerm: number;
  /** Outcome model augmentation */
  augmentationTerm: number;
  /** Combined AIPW estimate */
  combined: number;
  /** Influence function value (for variance estimation) */
  influenceFunction: number;
}

// ============================================================================
// Identifiability Types
// ============================================================================

/**
 * Identifiability conditions check result
 */
export interface IdentifiabilityResult {
  /** Overall identifiability status */
  identifiable: boolean;
  /** Unconfoundedness condition satisfied */
  unconfoundednessPartial: boolean;
  /** Positivity condition satisfied */
  positivity: boolean;
  /** Consistency condition satisfied */
  consistency: boolean;
  /** Transportability condition satisfied */
  transportability: boolean;
  /** Quantization invertibility condition */
  quantizationInvertible: boolean;
  /** Detailed diagnostics */
  diagnostics: IdentifiabilityDiagnostics;
}

/**
 * Detailed identifiability diagnostics
 */
export interface IdentifiabilityDiagnostics {
  /** Minimum propensity score (for positivity) */
  minPropensity: number;
  /** Maximum propensity score (for positivity) */
  maxPropensity: number;
  /** Quantization resolution relative to effect size */
  quantizationResolution: number;
  /** Sign preservation check */
  signPreserved: boolean;
  /** Warnings and notes */
  warnings: string[];
}

// ============================================================================
// Algorithm State Types
// ============================================================================

/**
 * Offline learning phase results
 */
export interface OfflinePhaseResult {
  /** Learned propensity score model parameters */
  propensityParams: number[];
  /** Learned outcome model parameters */
  outcomeParams: number[];
  /** Calibrated dequantization mappings */
  dequantizationMappings: DequantizationMapping[];
  /** Historical ATE estimate (potentially biased) */
  historicalATE: AverageTreatmentEffect;
  /** Optimized initial quantization */
  optimizedQuantization: QuantizationParams;
  /** Fisher information matrix (for power analysis) */
  fisherInformation?: number[][];
}

/**
 * Online selection phase state
 */
export interface OnlinePhaseState {
  /** Current time step */
  currentTimeStep: number;
  /** Current quantization parameters */
  currentQuantization: QuantizationParams;
  /** Accumulated observations */
  observations: OnlineObservation[];
  /** Adaptive propensity scores */
  adaptivePropensities: PropensityEstimate[];
  /** Running ATE estimate */
  runningATE: AverageTreatmentEffect;
}

/**
 * Single online observation
 */
export interface OnlineObservation {
  studentId: string;
  timeStep: number;
  features: number[];
  policy: Policy;
  quantizedOutcome: BinIndex;
  propensityScore: number;
  dequantizedOutcome: number;
}

// ============================================================================
// Statistical Testing Types
// ============================================================================

/**
 * Hypothesis test result
 * H_0: τ = 0 vs H_1: τ ≠ 0
 */
export interface HypothesisTestResult {
  /** Test statistic */
  testStatistic: number;
  /** P-value (two-sided) */
  pValue: number;
  /** Reject H_0 at significance level */
  reject: boolean;
  /** Significance level used */
  significanceLevel: number;
  /** Degrees of freedom (if applicable) */
  degreesOfFreedom?: number;
  /** Confidence interval for effect */
  confidenceInterval: [number, number];
  /** Achieved statistical power (estimated) */
  achievedPower?: number;
}

/**
 * Power analysis result
 */
export interface PowerAnalysisResult {
  /** Target effect size */
  effectSize: number;
  /** Required sample size for target power */
  requiredSampleSize: number;
  /** Achieved power with current sample */
  achievedPower: number;
  /** Minimum detectable effect */
  minimumDetectableEffect: number;
}

// ============================================================================
// Bias Verification Types
// ============================================================================

/**
 * Neutrality axiom verification
 * Algorithm should be invariant to A↔B and Y→-Y swap
 */
export interface NeutralityVerification {
  /** Neutrality axiom satisfied */
  neutral: boolean;
  /** Effect estimate under original labeling */
  originalEstimate: number;
  /** Effect estimate under swapped labeling */
  swappedEstimate: number;
  /** Difference (should be close to negation) */
  symmetryError: number;
  /** Tolerance threshold */
  tolerance: number;
}

/**
 * Differential privacy-inspired neutrality constraint
 */
export interface DifferentialNeutralityCheck {
  /** ε-differential neutrality satisfied */
  satisfied: boolean;
  /** Achieved ε value */
  epsilon: number;
  /** Target ε threshold */
  targetEpsilon: number;
  /** Noise level σ_ε added to θ_t */
  noiseLevel: number;
}

/**
 * Causal fairness metric
 * Δ = |E[τ̂|H_0,θ] - 0|
 */
export interface FairnessMetric {
  /** Fairness constraint satisfied */
  fair: boolean;
  /** Achieved Δ value */
  delta: number;
  /** Target δ threshold */
  targetDelta: number;
  /** Null distribution samples used */
  numNullSamples: number;
}

/**
 * Comprehensive bias verification result
 */
export interface BiasVerificationResult {
  /** Overall bias-free determination */
  unbiased: boolean;
  /** Neutrality verification */
  neutrality: NeutralityVerification;
  /** Differential neutrality check */
  differentialNeutrality: DifferentialNeutralityCheck;
  /** Causal fairness metric */
  fairness: FairnessMetric;
  /** Type I error rate under simulation */
  typeIErrorRate: number;
  /** Sensitivity analysis summary */
  sensitivitySummary: SensitivityAnalysis;
  /** Recommendations for improving unbiasedness */
  recommendations: string[];
}

/**
 * Sensitivity analysis result
 */
export interface SensitivityAnalysis {
  /** θ perturbation tested */
  perturbationSize: number;
  /** ATE change under perturbation */
  ateChange: number;
  /** Exceeds threshold */
  sensitive: boolean;
  /** Threshold used */
  threshold: number;
}

// ============================================================================
// Historical Data Types
// ============================================================================

/**
 * Complete historical record for a student at a time point
 */
export interface HistoricalRecord {
  studentId: string;
  timeStep: number;
  features: FeatureVector;
  continuousOutcome: ContinuousOutcome;
  quantizedOutcome: QuantizedOutcome;
  policySelection: PolicySelection;
  quantizationParams: QuantizationParams;
  groupId?: string;
}

/**
 * Historical dataset for offline learning
 */
export interface HistoricalDataset {
  /** All historical records */
  records: HistoricalRecord[];
  /** Number of unique students */
  numStudents: number;
  /** Number of time steps */
  numTimeSteps: number;
  /** Feature dimension */
  featureDimension: number;
  /** Group structure (if hierarchical) */
  groups?: GroupEffect[];
  /** Metadata */
  metadata: {
    collectionPeriod: string;
    dataSource: string;
    notes?: string;
  };
}

// ============================================================================
// Engine State Types
// ============================================================================

/**
 * Complete state of the causal inference engine
 */
export interface CausalEngineState {
  /** Current phase */
  phase: 'uninitialized' | 'offline_learning' | 'online_selection' | 'testing' | 'completed';
  /** Configuration */
  config: CausalModelConfig;
  /** Offline phase results (after offline learning) */
  offlineResults?: OfflinePhaseResult;
  /** Online phase state (during online selection) */
  onlineState?: OnlinePhaseState;
  /** Final test results (after testing) */
  testResults?: HypothesisTestResult;
  /** Bias verification (after testing) */
  biasVerification?: BiasVerificationResult;
  /** Audit log */
  auditLog: AuditLogEntry[];
}

/**
 * Audit log entry for traceability
 */
export interface AuditLogEntry {
  timestamp: number;
  action: string;
  details: Record<string, unknown>;
  phase: CausalEngineState['phase'];
}

// ============================================================================
// Ledger Types (for Σ-SIG Compliance)
// ============================================================================

/**
 * Decision ledger entry for causal analysis
 */
export interface CausalLedgerEntry {
  id: string;
  timestamp: number;
  /** Decision type */
  type: 'quantization_update' | 'policy_selection' | 'ate_estimate' | 'bias_check';
  /** Decision details */
  details: {
    previousState?: unknown;
    newState?: unknown;
    rationale: string;
    alternatives?: string[];
  };
  /** Linked student/time context */
  context: {
    studentIds?: string[];
    timeStep?: number;
  };
  /** Verification status */
  verified: boolean;
}

/**
 * Complete causal analysis ledger
 */
export interface CausalAnalysisLedger {
  entries: CausalLedgerEntry[];
  /** Summary statistics */
  summary: {
    totalDecisions: number;
    quantizationUpdates: number;
    policySelections: number;
    ateEstimates: number;
    biasChecks: number;
  };
}
