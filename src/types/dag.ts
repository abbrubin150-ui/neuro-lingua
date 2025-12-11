/**
 * DAG (Directed Acyclic Graph) Types for Causal Inference
 *
 * This module provides explicit DAG representation for causal models,
 * including temporal dependencies, unmeasured confounders, and
 * selection mechanisms.
 *
 * DAG Structure:
 * - X_{it}: Measured features (observed)
 * - Y_{it}: Continuous outcome
 * - Y_{i,t-k}: Temporal dependencies (k lags)
 * - U_i: Unmeasured confounders (latent)
 * - Z_{it}: Policy selection (treatment)
 * - theta_t: Quantization parameters
 *
 * Reference: Pearl (2009), Spirtes et al. (2000)
 *
 * @module dag
 */

// ============================================================================
// Node Types
// ============================================================================

/**
 * Type of node in the causal DAG
 */
export type CausalNodeType =
  | 'observed'      // X: Measured features
  | 'outcome'       // Y: Target outcome
  | 'treatment'     // Z: Policy/intervention
  | 'confounder'    // U: Unmeasured confounder
  | 'instrument'    // IV: Instrumental variable
  | 'mediator'      // M: Mediating variable
  | 'collider'      // C: Collider node
  | 'temporal'      // Y_{t-k}: Lagged outcome
  | 'quantization'  // theta: Quantization parameters
  | 'selection';    // S: Selection indicator

/**
 * Single node in the causal DAG
 */
export interface CausalNode {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Node type */
  type: CausalNodeType;
  /** Whether node is observed */
  observed: boolean;
  /** Time index (for temporal nodes) */
  timeIndex?: number;
  /** Lag index (for Y_{t-k}) */
  lagIndex?: number;
  /** Entity index (for hierarchical models) */
  entityId?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Edge in the causal DAG
 */
export interface CausalEdge {
  /** Unique identifier */
  id: string;
  /** Source node ID */
  from: string;
  /** Target node ID */
  to: string;
  /** Edge type */
  type: 'causal' | 'confounding' | 'selection' | 'temporal' | 'bidirected';
  /** Structural coefficient (if known) */
  coefficient?: number;
  /** Whether edge is identified */
  identified: boolean;
  /** Time lag (for temporal edges) */
  lag?: number;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

// ============================================================================
// DAG Structure
// ============================================================================

/**
 * Complete causal DAG specification
 */
export interface CausalDAG {
  /** Unique identifier */
  id: string;
  /** Display name */
  name: string;
  /** All nodes */
  nodes: CausalNode[];
  /** All edges */
  edges: CausalEdge[];
  /** Treatment variable ID */
  treatmentId: string;
  /** Outcome variable ID */
  outcomeId: string;
  /** Confounder IDs */
  confounderIds: string[];
  /** Maximum temporal lag */
  maxLag: number;
  /** Whether DAG satisfies identifiability conditions */
  identifiable: boolean;
  /** Metadata */
  metadata: {
    createdAt: number;
    updatedAt: number;
    description?: string;
    assumptions?: string[];
  };
}

// ============================================================================
// Temporal Dependencies
// ============================================================================

/**
 * Temporal lag specification
 */
export interface TemporalLag {
  /** Source variable */
  variable: string;
  /** Target variable */
  target: string;
  /** Lag index (1 = Y_{t-1}) */
  lag: number;
  /** Autoregressive coefficient */
  coefficient: number;
  /** Standard error of coefficient */
  standardError?: number;
}

/**
 * Complete temporal dependency structure
 */
export interface TemporalDependencies {
  /** Autoregressive lags for outcome */
  outcomeLags: TemporalLag[];
  /** Cross-lagged effects (e.g., X_{t-1} -> Y_t) */
  crossLags: TemporalLag[];
  /** Maximum lag order */
  maxOrder: number;
  /** Whether system is stationary */
  stationary: boolean;
  /** Spectral radius (for stability) */
  spectralRadius?: number;
}

// ============================================================================
// Selection Mechanism
// ============================================================================

/**
 * Selection mechanism specification
 * P(Z|X,Y_{t-1},U,theta) = sigma(g(X,Y_{t-1},U,theta))
 */
export interface SelectionMechanism {
  /** Selection function type */
  type: 'logistic' | 'probit' | 'linear' | 'custom';
  /** Feature coefficients */
  featureCoefficients: number[];
  /** Temporal coefficients (for lagged outcomes) */
  temporalCoefficients: number[];
  /** Confounder coefficients (typically unidentified) */
  confounderCoefficients: number[];
  /** Quantization parameter influence */
  quantizationInfluence: number[];
  /** Intercept */
  intercept: number;
  /** Whether mechanism is known/estimated */
  known: boolean;
}

// ============================================================================
// Unmeasured Confounders
// ============================================================================

/**
 * Unmeasured confounder specification
 */
export interface UnmeasuredConfounderSpec {
  /** Confounder ID */
  id: string;
  /** Dimension */
  dimension: number;
  /** Distribution type */
  distribution: 'normal' | 'uniform' | 'mixture' | 'unknown';
  /** Mean (if known) */
  mean?: number[];
  /** Covariance (if known) */
  covariance?: number[][];
  /** Affected nodes (where U -> V) */
  affectedNodes: string[];
  /** Sensitivity bounds (Rosenbaum gamma) */
  sensitivityBounds?: {
    gamma: number;
    pValueLower: number;
    pValueUpper: number;
  };
}

// ============================================================================
// DAG Operations Results
// ============================================================================

/**
 * D-separation query result
 */
export interface DSeparationResult {
  /** Whether X ⫫ Y | Z */
  separated: boolean;
  /** Set X */
  setX: string[];
  /** Set Y */
  setY: string[];
  /** Conditioning set Z */
  conditioningSet: string[];
  /** Paths that remain open */
  openPaths: string[][];
  /** Paths that are blocked */
  blockedPaths: string[][];
}

/**
 * Backdoor criterion check result
 */
export interface BackdoorCriterionResult {
  /** Whether backdoor criterion is satisfied */
  satisfied: boolean;
  /** Valid adjustment sets */
  adjustmentSets: string[][];
  /** Minimal adjustment set */
  minimalSet: string[];
  /** Confounding paths blocked */
  blockedPaths: string[][];
  /** Confounding paths remaining (if any) */
  openPaths: string[][];
}

/**
 * Frontdoor criterion check result
 */
export interface FrontdoorCriterionResult {
  /** Whether frontdoor criterion is satisfied */
  satisfied: boolean;
  /** Mediator set */
  mediators: string[];
  /** Identification formula */
  identificationFormula?: string;
}

/**
 * Instrumental variable check result
 */
export interface InstrumentCheckResult {
  /** Whether valid instrument */
  valid: boolean;
  /** Instrument node IDs */
  instruments: string[];
  /** Relevance check (Z -> treatment) */
  relevance: boolean;
  /** Exclusion check (Z -> outcome only through treatment) */
  exclusion: boolean;
  /** Independence check (Z ⫫ U) */
  independence: boolean;
}

// ============================================================================
// Identifiability
// ============================================================================

/**
 * Identifiability analysis result
 */
export interface IdentifiabilityAnalysis {
  /** Overall identifiability status */
  identifiable: boolean;
  /** Method used for identification */
  method: 'backdoor' | 'frontdoor' | 'iv' | 'regression' | 'unidentified';
  /** Adjustment formula */
  adjustmentFormula?: string;
  /** Required assumptions */
  assumptions: string[];
  /** Testable implications */
  testableImplications: string[];
  /** Sensitivity parameters */
  sensitivityParams?: {
    name: string;
    range: [number, number];
    description: string;
  }[];
}

// ============================================================================
// Causal Effect Types
// ============================================================================

/**
 * Type of causal effect to estimate
 */
export type CausalEffectType =
  | 'ATE'     // Average Treatment Effect: E[Y(1) - Y(0)]
  | 'ATT'     // Average Treatment on Treated: E[Y(1) - Y(0) | Z=1]
  | 'ATC'     // Average Treatment on Control: E[Y(1) - Y(0) | Z=0]
  | 'CATE'    // Conditional ATE: E[Y(1) - Y(0) | X=x]
  | 'LATE'    // Local ATE (compliers): E[Y(1) - Y(0) | complier]
  | 'NDE'     // Natural Direct Effect
  | 'NIE'     // Natural Indirect Effect
  | 'TE'      // Total Effect
  | 'CDE';    // Controlled Direct Effect

/**
 * Causal query specification
 */
export interface CausalQuery {
  /** Query type */
  type: CausalEffectType;
  /** Treatment variable */
  treatment: string;
  /** Outcome variable */
  outcome: string;
  /** Treatment value for intervention */
  treatmentValue: number;
  /** Control value */
  controlValue: number;
  /** Conditioning variables (for CATE) */
  conditioningVars?: string[];
  /** Mediator (for NDE/NIE/CDE) */
  mediator?: string;
}

// ============================================================================
// DAG Validation
// ============================================================================

/**
 * DAG validation result
 */
export interface DAGValidationResult {
  /** Whether DAG is valid */
  valid: boolean;
  /** Whether graph is acyclic */
  acyclic: boolean;
  /** Whether all required nodes present */
  complete: boolean;
  /** List of issues found */
  issues: {
    severity: 'error' | 'warning' | 'info';
    message: string;
    nodeIds?: string[];
    edgeIds?: string[];
  }[];
  /** Suggestions for improvement */
  suggestions: string[];
}

// ============================================================================
// Temporal DAG Extensions
// ============================================================================

/**
 * Rolled-out temporal DAG (time-expanded)
 */
export interface TemporalDAG extends CausalDAG {
  /** Number of time steps */
  numTimeSteps: number;
  /** Template DAG (single time slice) */
  templateDAG: CausalDAG;
  /** Inter-temporal edges */
  temporalEdges: CausalEdge[];
  /** Stationarity assumption */
  stationary: boolean;
}

/**
 * Time slice in a temporal DAG
 */
export interface TimeSlice {
  /** Time index */
  timeIndex: number;
  /** Nodes at this time */
  nodes: CausalNode[];
  /** Intra-slice edges */
  intraEdges: CausalEdge[];
  /** Edges to next time slice */
  forwardEdges: CausalEdge[];
}

// ============================================================================
// Sensitivity Analysis Extensions
// ============================================================================

/**
 * Rosenbaum bounds for sensitivity analysis
 */
export interface RosenbaumBounds {
  /** Gamma parameter (odds ratio bound) */
  gamma: number;
  /** Upper bound on p-value */
  pValueUpper: number;
  /** Lower bound on p-value */
  pValueLower: number;
  /** Point estimate under gamma */
  pointEstimate: number;
  /** Confidence interval under gamma */
  confidenceInterval: [number, number];
}

/**
 * E-value for sensitivity analysis
 */
export interface EValue {
  /** E-value for point estimate */
  pointEstimate: number;
  /** E-value for confidence interval bound */
  ciLowerBound: number;
  /** Interpretation */
  interpretation: string;
}

/**
 * Manski bounds (nonparametric)
 */
export interface ManskiBounds {
  /** Lower bound on ATE */
  lower: number;
  /** Upper bound on ATE */
  upper: number;
  /** Monotonicity assumption used */
  monotonicity: boolean;
  /** Selection assumption used */
  selectionAssumption?: string;
}

// ============================================================================
// Export all types
// ============================================================================

export type {
  CausalNodeType,
  CausalNode,
  CausalEdge,
  CausalDAG,
  TemporalLag,
  TemporalDependencies,
  SelectionMechanism,
  UnmeasuredConfounderSpec,
  DSeparationResult,
  BackdoorCriterionResult,
  FrontdoorCriterionResult,
  InstrumentCheckResult,
  IdentifiabilityAnalysis,
  CausalEffectType,
  CausalQuery,
  DAGValidationResult,
  TemporalDAG,
  TimeSlice,
  RosenbaumBounds,
  EValue,
  ManskiBounds
};
