/**
 * RHS Kernel Primitives Type Definitions
 *
 * This module defines the type system for the R/H/S (Noise/Coherence/Soleket) kernel,
 * implementing a generative triadic framework for emergent dynamics.
 *
 * Based on the Neuro-Lingua Framework Specification
 */

// ============================================================================
// A. Kernel Primitives (RHS) - A01-A12
// ============================================================================

/**
 * A01 - R/Noise Primitive
 * Defines incoherence as perturbation/variation input
 */
export interface RNoise {
  readonly type: 'R_NOISE';
  /** Perturbation vector - source of variation */
  perturbation: number[];
  /** Variance/intensity of the noise */
  variance: number;
  /** Timestamp of noise generation */
  timestamp: number;
  /** Optional source label for traceability */
  source?: string;
}

/**
 * A02 - H/Coherence Primitive
 * Defines coherence as constraint/structure/holding
 */
export interface HCoherence {
  readonly type: 'H_COHERENCE';
  /** Constraint matrix defining structural boundaries */
  constraints: number[][];
  /** Structure encoding - pattern representation */
  structure: number[];
  /** Holding strength - how firmly constraints are enforced */
  holdingStrength: number;
  /** Active constraint indices */
  activeConstraints: number[];
}

/**
 * A03 - S/Soleket Primitive
 * Defines clearance as transformative mediation (degree-of-freedom removal)
 */
export interface SSoleket {
  readonly type: 'S_SOLEKET';
  /** Mediation function coefficients */
  mediationCoefficients: number[];
  /** Degrees of freedom removed */
  dofRemoved: number;
  /** Transformation matrix */
  transformMatrix: number[][];
  /** Clearance level (0-1) */
  clearanceLevel: number;
}

/**
 * A04 - CycleOperator
 * Implements R→H→S→R′ generative loop
 */
export interface CycleOperator {
  readonly type: 'CYCLE_OPERATOR';
  /** Current phase in the cycle */
  currentPhase: 'R' | 'H' | 'S';
  /** Cycle count */
  cycleCount: number;
  /** Transition parameters */
  transitionParams: {
    rToH: TransitionConfig;
    hToS: TransitionConfig;
    sToR: TransitionConfig;
  };
}

export interface TransitionConfig {
  /** Rate of transition */
  rate: number;
  /** Threshold for transition */
  threshold: number;
  /** Damping factor */
  damping: number;
}

/**
 * A05 - StateSpace_RHS
 * Encodes system state as ⟨R,H,S⟩ manifold
 */
export interface StateSpaceRHS {
  readonly type: 'STATE_SPACE_RHS';
  /** Current R state */
  r: RNoise;
  /** Current H state */
  h: HCoherence;
  /** Current S state */
  s: SSoleket;
  /** Manifold dimension */
  dimension: number;
  /** State history for trajectory analysis */
  history: StateSnapshot[];
}

export interface StateSnapshot {
  timestamp: number;
  r: number[];
  h: number[];
  s: number[];
}

/**
 * A06 - Attractor
 * Represents stable basin emergent in H (or S-level order parameter)
 */
export interface Attractor {
  readonly type: 'ATTRACTOR';
  /** Basin center coordinates */
  center: number[];
  /** Basin radius/extent */
  radius: number;
  /** Stability score (0-1) */
  stability: number;
  /** Emergence level (H or S) */
  emergenceLevel: 'H' | 'S';
  /** Attractor type */
  attractorType: 'point' | 'limit_cycle' | 'strange';
}

/**
 * A07 - ClearanceOperator
 * Executes Soleket: contracts phase space, removes incompatible modes
 */
export interface ClearanceOperator {
  readonly type: 'CLEARANCE_OPERATOR';
  /** Phase space contraction factor */
  contractionFactor: number;
  /** Modes marked for removal */
  incompatibleModes: number[];
  /** Clearance threshold */
  threshold: number;
  /** Active clearing status */
  isClearing: boolean;
}

/**
 * A08 - DownwardConstraint
 * Encodes S_t → {R,H}_{t+1} causal constraint
 */
export interface DownwardConstraint {
  readonly type: 'DOWNWARD_CONSTRAINT';
  /** Source S state index */
  sourceTimeIndex: number;
  /** Target R/H state index (t+1) */
  targetTimeIndex: number;
  /** Constraint strength */
  strength: number;
  /** Affected variables (R, H, or both) */
  affectedVariables: ('R' | 'H')[];
  /** Constraint encoding */
  constraintMatrix: number[][];
}

/**
 * A09 - ClosureCondition
 * True iff measurable macro→micro influence exists
 */
export interface ClosureCondition {
  readonly type: 'CLOSURE_CONDITION';
  /** Whether closure is satisfied */
  isSatisfied: boolean;
  /** Measured influence strength */
  influenceStrength: number;
  /** Evidence for closure */
  evidence: ClosureEvidence[];
}

export interface ClosureEvidence {
  method: 'statistical' | 'causal' | 'interventional';
  value: number;
  confidence: number;
}

/**
 * A10 - TensionField
 * Encodes incompatibility surface between R and H
 */
export interface TensionField {
  readonly type: 'TENSION_FIELD';
  /** Tension values across the field */
  values: number[][];
  /** Peak tension locations */
  peaks: TensionPeak[];
  /** Total tension energy */
  totalEnergy: number;
  /** Gradient of tension */
  gradient: number[][];
}

export interface TensionPeak {
  location: number[];
  magnitude: number;
}

/**
 * A11 - CoherenceScore
 * Maps state to scalar coherence proxy for tracking
 */
export interface CoherenceScore {
  readonly type: 'COHERENCE_SCORE';
  /** Scalar coherence value */
  value: number;
  /** Components contributing to score */
  components: CoherenceComponent[];
  /** Timestamp */
  timestamp: number;
}

export interface CoherenceComponent {
  name: string;
  contribution: number;
  weight: number;
}

/**
 * A12 - ResonanceCheck
 * True iff emergent S behaves as resonant attractor (stability+selectivity)
 */
export interface ResonanceCheck {
  readonly type: 'RESONANCE_CHECK';
  /** Whether resonance is detected */
  isResonant: boolean;
  /** Stability measure */
  stability: number;
  /** Selectivity measure */
  selectivity: number;
  /** Resonance frequency if applicable */
  frequency?: number;
}

// ============================================================================
// B. Generativity Tests (G1-G3) - B01-B08
// ============================================================================

/**
 * B01 - M0_NonGenerativeModelClass
 * Baseline class of non-generative constructions
 */
export interface M0NonGenerativeModelClass {
  readonly type: 'M0_NON_GENERATIVE';
  /** Model class identifier */
  classId: string;
  /** Construction methods in class */
  methods: NonGenerativeMethod[];
}

export type NonGenerativeMethod = 'aggregation' | 'linear_combination' | 'monotone_transform';

/**
 * B02 - AggConstructor
 * Defines Z := X∪Y / additive aggregation baseline
 */
export interface AggConstructor {
  readonly type: 'AGG_CONSTRUCTOR';
  /** Input X */
  inputX: number[];
  /** Input Y */
  inputY: number[];
  /** Output Z = X ∪ Y */
  outputZ: number[];
}

/**
 * B03 - LinCombConstructor
 * Defines Z := aX+bY linear mixture baseline
 */
export interface LinCombConstructor {
  readonly type: 'LIN_COMB_CONSTRUCTOR';
  /** Coefficient a */
  coeffA: number;
  /** Coefficient b */
  coeffB: number;
  /** Input X */
  inputX: number[];
  /** Input Y */
  inputY: number[];
  /** Output Z = aX + bY */
  outputZ: number[];
}

/**
 * B04 - MonotoneTransformConstructor
 * Defines Z := f(X,Y) without topology change baseline
 */
export interface MonotoneTransformConstructor {
  readonly type: 'MONOTONE_TRANSFORM_CONSTRUCTOR';
  /** Transform function identifier */
  transformId: string;
  /** Whether transform preserves topology */
  preservesTopology: boolean;
}

/**
 * B05 - G1_Irreducibility
 * Passes iff Z ∉ M0(X,Y) (model-class exclusion)
 */
export interface G1Irreducibility {
  readonly type: 'G1_IRREDUCIBILITY';
  /** Whether test passes */
  passes: boolean;
  /** Tested against M0 methods */
  testedMethods: NonGenerativeMethod[];
  /** Irreducibility score */
  score: number;
}

/**
 * B06 - G2_Mediation
 * Passes iff ∃π that reduces tension while preserving integrity
 */
export interface G2Mediation {
  readonly type: 'G2_MEDIATION';
  /** Whether test passes */
  passes: boolean;
  /** Tension reduction achieved */
  tensionReduction: number;
  /** Integrity preserved */
  integrityPreserved: boolean;
  /** Mediation path π */
  mediationPath?: number[];
}

/**
 * B07 - G3_DownwardClosure
 * Passes iff Z_t constrains P(X/Y)_{t+1} (macro→micro)
 */
export interface G3DownwardClosure {
  readonly type: 'G3_DOWNWARD_CLOSURE';
  /** Whether test passes */
  passes: boolean;
  /** Constraint strength */
  constraintStrength: number;
  /** Statistical significance */
  pValue: number;
}

/**
 * B08 - G_Status
 * Generative iff G1∧G2∧G3
 */
export interface GStatus {
  readonly type: 'G_STATUS';
  /** G1 result */
  g1: G1Irreducibility;
  /** G2 result */
  g2: G2Mediation;
  /** G3 result */
  g3: G3DownwardClosure;
  /** Final status */
  isGenerative: boolean;
}

// ============================================================================
// C. Evidence Tier (E-tier) - C01-C05
// ============================================================================

export type EvidenceTier = 'E0' | 'E1' | 'E2' | 'E3';

/**
 * C01-C04 - Evidence Labels
 */
export interface EvidenceLabel {
  tier: EvidenceTier;
  description: string;
}

export const EVIDENCE_LABELS: Record<EvidenceTier, EvidenceLabel> = {
  E0: { tier: 'E0', description: 'Interpretive - Structural isomorphism only; no measurement' },
  E1: { tier: 'E1', description: 'Operational - Proxies defined; limited measurement' },
  E2: { tier: 'E2', description: 'Simulated - Computational/agent simulation validates dynamics' },
  E3: { tier: 'E3', description: 'Experimental - Physical/experimental verification of closure' },
};

/**
 * C05 - E_Assignment
 * Assigns E-tier given method/evidence record
 */
export interface EAssignment {
  readonly type: 'E_ASSIGNMENT';
  /** Assigned tier */
  tier: EvidenceTier;
  /** Method used for assignment */
  method: string;
  /** Evidence record */
  evidenceRecord: EvidenceRecord[];
  /** Confidence in assignment */
  confidence: number;
}

export interface EvidenceRecord {
  source: string;
  type: 'measurement' | 'simulation' | 'theory' | 'experiment';
  value: number;
  timestamp: number;
}

// ============================================================================
// D. Proxies/Metrics (Signals) - D01-D33
// ============================================================================

/**
 * D01 - KL_Def
 * Defines D_KL(R||H) as local tension proxy
 */
export interface KLDefinition {
  readonly type: 'KL_DEF';
  /** KL divergence value */
  value: number;
  /** Distribution R */
  distributionR: number[];
  /** Distribution H */
  distributionH: number[];
}

/**
 * D02-D05 - KL Estimation Pipeline
 */
export interface KLEstimator {
  readonly type: 'KL_ESTIMATOR';
  /** Estimated KL divergence */
  estimate: number;
  /** Sample size used */
  sampleSize: number;
  /** Confidence interval */
  confidenceInterval: [number, number];
}

export interface KLWindow {
  /** Window size */
  size: number;
  /** Current position */
  position: number;
  /** Values in window */
  values: number[];
}

export interface KLThresholds {
  /** Low threshold τ_low */
  low: number;
  /** High threshold τ_high */
  high: number;
}

export type KLFailureType = 'sustained_high' | 'sustained_zero' | 'none';

export interface KLFailures {
  readonly type: 'KL_FAILURES';
  /** Failure type detected */
  failureType: KLFailureType;
  /** Duration of anomaly */
  duration: number;
  /** Severity (0-1) */
  severity: number;
}

/**
 * D06-D10 - Free Energy Pipeline
 */
export interface FreeEnergyDefinition {
  readonly type: 'FREE_E_DEF';
  /** Variational free energy value */
  value: number;
  /** Surprisal component */
  surprisal: number;
  /** Complexity component */
  complexity: number;
}

export interface FreeEnergyEstimator {
  readonly type: 'FREE_E_ESTIMATOR';
  /** Estimated free energy */
  estimate: number;
  /** Model residuals used */
  residuals: number[];
}

export interface FreeEnergyWindow {
  /** Window/epoch size */
  epochSize: number;
  /** Values tracked */
  values: number[];
}

export interface FreeEnergyThresholds {
  /** Exploration band lower */
  explorationLower: number;
  /** Exploration band upper */
  explorationUpper: number;
  /** Collapse threshold */
  collapseThreshold: number;
}

export type FreeEnergyFailureType = 'collapse' | 'over_constraint' | 'none';

export interface FreeEnergyFailures {
  readonly type: 'FREE_E_FAILURES';
  /** Failure type */
  failureType: FreeEnergyFailureType;
  /** Detected at epoch */
  detectedAtEpoch: number;
}

/**
 * D11-D15 - Transfer Entropy Pipeline
 */
export interface TEDefinition {
  readonly type: 'TE_DEF';
  /** Transfer entropy S→R */
  value: number;
  /** Source variable */
  source: 'S';
  /** Target variable */
  target: 'R';
}

export interface TEEstimator {
  readonly type: 'TE_ESTIMATOR';
  /** Estimated TE */
  estimate: number;
  /** History embedding used */
  historyEmbedding: number[];
  /** Lag order */
  lagOrder: number;
}

export interface TEWindow {
  /** Lag order */
  lagOrder: number;
  /** Window length */
  windowLength: number;
}

export interface TEThresholds {
  /** Minimum TE for closure */
  minTEForClosure: number;
}

export interface TEFailures {
  readonly type: 'TE_FAILURES';
  /** Whether S is epiphenomenal */
  isEpiphenomenal: boolean;
  /** TE value when flagged */
  teValue: number;
}

/**
 * D16-D20 - Granger Causality Pipeline
 */
export interface GrangerDefinition {
  readonly type: 'GRANGER_DEF';
  /** Granger causality S→R */
  fStatistic: number;
  /** p-value */
  pValue: number;
  /** Direction */
  direction: 'S_to_R' | 'S_to_H';
}

export interface GrangerEstimator {
  readonly type: 'GRANGER_ESTIMATOR';
  /** VAR model coefficients */
  varCoefficients: number[][];
  /** F-statistic */
  fStatistic: number;
  /** Significance */
  isSignificant: boolean;
}

export interface GrangerWindow {
  /** Time horizon */
  horizon: number;
  /** Sampling rate */
  samplingRate: number;
}

export interface GrangerThresholds {
  /** Significance level */
  significanceLevel: number;
  /** Confidence interval width */
  ciWidth: number;
}

export interface GrangerFailures {
  readonly type: 'GRANGER_FAILURES';
  /** Whether macro state is non-causal */
  isNonCausal: boolean;
  /** p-value when flagged */
  pValue: number;
}

/**
 * D21-D25 - Phi (Irreducibility) Pipeline
 */
export interface PhiDefinition {
  readonly type: 'PHI_DEF';
  /** Integrated information Φ */
  value: number;
  /** Excess entropy */
  excessEntropy: number;
}

export interface PhiEstimator {
  readonly type: 'PHI_ESTIMATOR';
  /** Estimated Φ */
  estimate: number;
  /** Partition used */
  partition: number[][];
  /** Shared exclusions */
  sharedExclusions: number;
}

export interface PhiWindow {
  /** Block size for estimation */
  blockSize: number;
}

export interface PhiThresholds {
  /** Minimum Φ to pass G1 */
  minPhiForG1: number;
}

export interface PhiFailures {
  readonly type: 'PHI_FAILURES';
  /** Whether Z is reducible (taxonomy) */
  isReducible: boolean;
  /** Φ value when flagged */
  phiValue: number;
}

/**
 * D26-D31 - Mode Conditions
 */
export type ModeType =
  | 'collapse'
  | 'conflict'
  | 'epiphenomenal'
  | 'ossification'
  | 'explosion'
  | 'model_collapse';

export interface ModeCondition {
  readonly type: 'MODE_CONDITION';
  /** Mode type */
  mode: ModeType;
  /** Whether condition is active */
  isActive: boolean;
  /** Description */
  description: string;
  /** Severity (0-1) */
  severity: number;
}

export const MODE_DESCRIPTIONS: Record<ModeType, string> = {
  collapse: 'Tension→0 via averaging (loss of R variance)',
  conflict: 'Tension high and unmediated (H rejected)',
  epiphenomenal: 'TE≈0; S has no downward influence',
  ossification: 'H dominates; R suppressed; novelty stalls',
  explosion: 'R dominates; coherence cannot form',
  model_collapse: 'Self-training reduces variance',
};

/**
 * D32 - NearMissFlag
 */
export interface NearMissFlag {
  readonly type: 'NEAR_MISS_FLAG';
  /** Whether flagged */
  isFlagged: boolean;
  /** Which G tests failed */
  failedTests: ('G1' | 'G2' | 'G3')[];
  /** Notes for review */
  reviewNotes: string;
}

/**
 * D33 - EvidenceGapFlag
 */
export interface EvidenceGapFlag {
  readonly type: 'EVIDENCE_GAP_FLAG';
  /** Whether flagged */
  isFlagged: boolean;
  /** Current E-tier */
  currentTier: EvidenceTier;
  /** Required upgrade path */
  upgradePath: string;
}

// ============================================================================
// E. Mapping/Convolution/Reporting - E01-E12
// ============================================================================

/**
 * E01 - TargetTriadRecord
 * Stores ⟨X,Y,Z⟩ candidate triad with context
 */
export interface TargetTriadRecord {
  readonly type: 'TARGET_TRIAD_RECORD';
  /** Identifier */
  id: string;
  /** X component */
  x: number[];
  /** Y component */
  y: number[];
  /** Z component (emergent) */
  z: number[];
  /** Context/domain */
  context: string;
  /** Source reference */
  sourceRef?: string;
}

/**
 * E02 - RoleAlignment_phi
 * Maps (X,Y,Z) → (R,H,S) roles
 */
export interface RoleAlignment {
  readonly type: 'ROLE_ALIGNMENT';
  /** Which input maps to R */
  rMapping: 'X' | 'Y' | 'Z';
  /** Which input maps to H */
  hMapping: 'X' | 'Y' | 'Z';
  /** Which input maps to S */
  sMapping: 'X' | 'Y' | 'Z';
  /** Alignment confidence */
  confidence: number;
}

/**
 * E03 - K_DepthScore
 * Rates alignment depth K∈{0,1,2} per mapping
 */
export type KDepth = 0 | 1 | 2;

export interface KDepthScore {
  readonly type: 'K_DEPTH_SCORE';
  /** Depth score */
  depth: KDepth;
  /** Justification */
  justification: string;
}

/**
 * E04 - ConvolutionOperator
 * Slides RHS kernel over triads/domains to populate cells
 */
export interface ConvolutionOperator {
  readonly type: 'CONVOLUTION_OPERATOR';
  /** Kernel being applied */
  kernel: StateSpaceRHS;
  /** Current position */
  position: { row: number; col: number };
  /** Populated cells count */
  populatedCount: number;
}

/**
 * E05 - NearMissDetector
 * Detects near-miss triads (taxonomy vs generative)
 */
export interface NearMissDetector {
  readonly type: 'NEAR_MISS_DETECTOR';
  /** Detection results */
  detections: NearMissDetection[];
}

export interface NearMissDetection {
  triadId: string;
  failedTests: ('G1' | 'G2' | 'G3')[];
  classification: 'taxonomy' | 'near_generative';
}

/**
 * E06 - CanonicalTriadCatalog
 * Registry of accepted triads with sources
 */
export interface CanonicalTriadCatalog {
  readonly type: 'CANONICAL_TRIAD_CATALOG';
  /** All registered triads */
  triads: CatalogEntry[];
  /** Version */
  version: string;
}

export interface CatalogEntry {
  id: string;
  name: string;
  triad: TargetTriadRecord;
  source: SourceCitation;
  acceptedDate: number;
}

/**
 * E07 - SourceCitationPointer
 */
export interface SourceCitation {
  author: string;
  title: string;
  year: number;
  doi?: string;
  url?: string;
}

/**
 * E08 - MappingReport
 * Emits per-cell narrative+Gi+E-tier+K-depth
 */
export interface MappingReport {
  readonly type: 'MAPPING_REPORT';
  /** Cell identifier */
  cellId: string;
  /** Narrative description */
  narrative: string;
  /** G-status */
  gStatus: GStatus;
  /** Evidence tier */
  eTier: EvidenceTier;
  /** K-depth */
  kDepth: KDepth;
  /** Generated timestamp */
  generatedAt: number;
}

/**
 * E09 - MatrixGenerator
 * Generates 14×14 outputs deterministically from specs
 */
export interface MatrixGenerator {
  readonly type: 'MATRIX_GENERATOR';
  /** Row count */
  rows: 14;
  /** Column count */
  cols: 14;
  /** Generation seed for determinism */
  seed: number;
}

/**
 * E10 - CellValidator
 * Ensures each cell has R/H/S + Gi + E-tier fields
 */
export interface CellValidator {
  readonly type: 'CELL_VALIDATOR';
  /** Validation results */
  results: CellValidationResult[];
  /** Overall valid */
  isValid: boolean;
}

export interface CellValidationResult {
  cellId: string;
  hasRHS: boolean;
  hasGStatus: boolean;
  hasETier: boolean;
  hasKDepth: boolean;
  isValid: boolean;
  errors: string[];
}

/**
 * E11 - VersionLedger
 * Tracks versions/changes (merge-only)
 */
export interface VersionLedger {
  readonly type: 'VERSION_LEDGER';
  /** Current version */
  currentVersion: string;
  /** Change history */
  history: LedgerEntry[];
}

export interface LedgerEntry {
  version: string;
  timestamp: number;
  changes: string[];
  author: string;
}

/**
 * E12 - ExportPack
 * Bundles matrix + registry + ledger for distribution
 */
export interface ExportPack {
  readonly type: 'EXPORT_PACK';
  /** Matrix data */
  matrix: MatrixCell[];
  /** Catalog */
  catalog: CanonicalTriadCatalog;
  /** Ledger */
  ledger: VersionLedger;
  /** Export timestamp */
  exportedAt: number;
  /** Export format version */
  formatVersion: string;
}

// ============================================================================
// L. Knowledge Levels (14 Rows) - L01-L14
// ============================================================================

export type KnowledgeLevelId =
  | 'L01'
  | 'L02'
  | 'L03'
  | 'L04'
  | 'L05'
  | 'L06'
  | 'L07'
  | 'L08'
  | 'L09'
  | 'L10'
  | 'L11'
  | 'L12'
  | 'L13'
  | 'L14';

export interface KnowledgeLevel {
  readonly id: KnowledgeLevelId;
  readonly name: string;
  readonly description: string;
  readonly rowIndex: number;
}

export const KNOWLEDGE_LEVELS: Record<KnowledgeLevelId, KnowledgeLevel> = {
  L01: { id: 'L01', name: 'Digital Foundation', description: 'Domain row #1 for convolution/matrix', rowIndex: 0 },
  L02: { id: 'L02', name: 'Boolean Logic', description: 'Domain row #2 for convolution/matrix', rowIndex: 1 },
  L03: { id: 'L03', name: 'Fundamental Questions', description: 'Domain row #3 for convolution/matrix', rowIndex: 2 },
  L04: { id: 'L04', name: 'Governance & Organization', description: 'Domain row #4 for convolution/matrix', rowIndex: 3 },
  L05: { id: 'L05', name: 'Standards & Regulation', description: 'Domain row #5 for convolution/matrix', rowIndex: 4 },
  L06: { id: 'L06', name: 'Execution & Implementation', description: 'Domain row #6 for convolution/matrix', rowIndex: 5 },
  L07: { id: 'L07', name: 'Measurement & Control', description: 'Domain row #7 for convolution/matrix', rowIndex: 6 },
  L08: { id: 'L08', name: 'Monitoring & Response', description: 'Domain row #8 for convolution/matrix', rowIndex: 7 },
  L09: { id: 'L09', name: 'Learning & Improvement', description: 'Domain row #9 for convolution/matrix', rowIndex: 8 },
  L10: { id: 'L10', name: 'Interface & Experience (UX)', description: 'Domain row #10 for convolution/matrix', rowIndex: 9 },
  L11: { id: 'L11', name: 'Human Rights', description: 'Domain row #11 for convolution/matrix', rowIndex: 10 },
  L12: { id: 'L12', name: 'Geopolitics', description: 'Domain row #12 for convolution/matrix', rowIndex: 11 },
  L13: { id: 'L13', name: 'Digital Commons', description: 'Domain row #13 for convolution/matrix', rowIndex: 12 },
  L14: { id: 'L14', name: 'Existential Resilience', description: 'Domain row #14 for convolution/matrix', rowIndex: 13 },
};

// ============================================================================
// T. Canonical Triads (14 Columns) - T01-T14
// ============================================================================

export type CanonicalTriadId =
  | 'T01'
  | 'T02'
  | 'T03'
  | 'T04'
  | 'T05'
  | 'T06'
  | 'T07'
  | 'T08'
  | 'T09'
  | 'T10'
  | 'T11'
  | 'T12'
  | 'T13'
  | 'T14';

export interface CanonicalTriad {
  readonly id: CanonicalTriadId;
  readonly name: string;
  readonly components: [string, string, string];
  readonly description: string;
  readonly colIndex: number;
}

export const CANONICAL_TRIADS: Record<CanonicalTriadId, CanonicalTriad> = {
  T01: { id: 'T01', name: 'Peirce', components: ['Firstness', 'Secondness', 'Thirdness'], description: 'Canonical triad column #1 used as mapping lens', colIndex: 0 },
  T02: { id: 'T02', name: 'Haken', components: ['Fluctuation', 'Order', 'Slaving'], description: 'Canonical triad column #2 used as mapping lens', colIndex: 1 },
  T03: { id: 'T03', name: 'Lacan', components: ['Real', 'Imaginary', 'Symbolic'], description: 'Canonical triad column #3 used as mapping lens', colIndex: 2 },
  T04: { id: 'T04', name: 'Morin', components: ['Disorder', 'Interaction', 'Organization'], description: 'Canonical triad column #4 used as mapping lens', colIndex: 3 },
  T05: { id: 'T05', name: 'Campbell', components: ['Variation', 'Selection', 'Retention'], description: 'Canonical triad column #5 used as mapping lens', colIndex: 4 },
  T06: { id: 'T06', name: 'Hofkirchner', components: ['Cognition', 'Communication', 'Cooperation'], description: 'Canonical triad column #6 used as mapping lens', colIndex: 5 },
  T07: { id: 'T07', name: 'Bruna', components: ['Oscillation', 'Interference', 'Attractor'], description: 'Canonical triad column #7 used as mapping lens', colIndex: 6 },
  T08: { id: 'T08', name: 'Cacella', components: ['Rise', 'Permanence', 'Meta'], description: 'Canonical triad column #8 used as mapping lens', colIndex: 7 },
  T09: { id: 'T09', name: 'Self-Coherence', components: ['Drive', 'Constraint', 'Mediation'], description: 'Canonical triad column #9 used as mapping lens', colIndex: 8 },
  T10: { id: 'T10', name: 'Semiotic Closure', components: ['Signal', 'Code', 'Meaning'], description: 'Canonical triad column #10 used as mapping lens', colIndex: 9 },
  T11: { id: 'T11', name: 'Cybernetic Closure', components: ['Noise', 'Regulation', 'Control-Law'], description: 'Canonical triad column #11 used as mapping lens', colIndex: 10 },
  T12: { id: 'T12', name: 'Evolutionary Learning', components: ['Explore', 'Exploit', 'Policy'], description: 'Canonical triad column #12 used as mapping lens', colIndex: 11 },
  T13: { id: 'T13', name: 'Music-Generative', components: ['Rhythm', 'Harmony', 'Emergence'], description: 'Canonical triad column #13 used as mapping lens', colIndex: 12 },
  T14: { id: 'T14', name: 'RHS Kernel', components: ['R', 'H', 'S'], description: 'Canonical triad column #14 used as mapping lens', colIndex: 13 },
};

// ============================================================================
// F. Matrix (14×14 = 196 Cells)
// ============================================================================

export interface MatrixCell {
  readonly type: 'MATRIX_CELL';
  /** Cell ID (e.g., "F0101") */
  cellId: string;
  /** Row index (0-13) */
  rowIndex: number;
  /** Column index (0-13) */
  colIndex: number;
  /** Knowledge level */
  knowledgeLevel: KnowledgeLevelId;
  /** Canonical triad */
  canonicalTriad: CanonicalTriadId;
  /** R/H/S instantiation */
  rhs: StateSpaceRHS | null;
  /** G-status */
  gStatus: GStatus | null;
  /** Evidence tier */
  eTier: EvidenceTier | null;
  /** K-depth */
  kDepth: KDepth | null;
  /** Narrative */
  narrative: string;
}

/**
 * Generate cell ID from row and column indices
 */
export function generateCellId(row: number, col: number): string {
  const rowStr = String(row + 1).padStart(2, '0');
  const colStr = String(col + 1).padStart(2, '0');
  return `F${rowStr}${colStr}`;
}

/**
 * Parse cell ID to get row and column indices
 */
export function parseCellId(cellId: string): { row: number; col: number } | null {
  const match = cellId.match(/^F(\d{2})(\d{2})$/);
  if (!match) return null;
  return {
    row: parseInt(match[1], 10) - 1,
    col: parseInt(match[2], 10) - 1,
  };
}

// ============================================================================
// G. Governance/Control (Σ-SIG + Mirror) - G01-G30
// ============================================================================

/**
 * G01 - R_AnchorGate
 * Enforces merge-only / no deletion boundary
 */
export interface RAnchorGate {
  readonly type: 'R_ANCHOR_GATE';
  /** Whether gate is active */
  isActive: boolean;
  /** Allowed operations */
  allowedOperations: ('merge' | 'append')[];
  /** Blocked operations */
  blockedOperations: ('delete' | 'overwrite')[];
}

/**
 * G02 - CANON5_Bundle
 * Bundles non-relaxable canon constraints
 */
export interface Canon5Bundle {
  readonly type: 'CANON5_BUNDLE';
  /** C68 - Silence Window */
  c68: C68SilenceWindow;
  /** C76 - Zero Substitute */
  c76: C76ZeroSubstitute;
  /** C79 - Uncertainty Bounds */
  c79: C79UncertaintyBounds;
  /** C80 - External Signature */
  c80: C80ExternalSignature;
  /** C117 - Opt Out No Lock In */
  c117: C117OptOutNoLockIn;
}

/**
 * G03 - C68_SilenceWindow
 */
export interface C68SilenceWindow {
  readonly type: 'C68_SILENCE_WINDOW';
  /** Window duration in ms */
  windowDuration: number;
  /** Presence stamp required */
  presenceStampRequired: boolean;
}

/**
 * G04 - PresenceStampLog
 */
export interface PresenceStampLog {
  readonly type: 'PRESENCE_STAMP_LOG';
  /** Log entries */
  entries: PresenceStamp[];
}

export interface PresenceStamp {
  timestamp: number;
  status: 'present' | 'silent' | 'holding';
  duration: number;
}

/**
 * G05 - C76_ZeroSubstitute
 */
export interface C76ZeroSubstitute {
  readonly type: 'C76_ZERO_SUBSTITUTE';
  /** Whether rule is active */
  isActive: boolean;
  /** If equal moral substitute exists → no trace/commit */
  substituteExists: boolean;
}

/**
 * G06 - C79_UncertaintyBounds
 */
export interface C79UncertaintyBounds {
  readonly type: 'C79_UNCERTAINTY_BOUNDS';
  /** Lower bound */
  lowerBound: number;
  /** Upper bound */
  upperBound: number;
  /** Current uncertainty */
  currentUncertainty: number;
  /** Whether bounds are logged */
  isBoundLogged: boolean;
}

/**
 * G07 - C80_ExternalSignature
 */
export interface C80ExternalSignature {
  readonly type: 'C80_EXTERNAL_SIGNATURE';
  /** External pointer required */
  externalPointerRequired: boolean;
  /** Signature present */
  hasSignature: boolean;
  /** Signature value */
  signature?: string;
}

/**
 * G08 - C117_OptOut_NoLockIn
 */
export interface C117OptOutNoLockIn {
  readonly type: 'C117_OPT_OUT_NO_LOCK_IN';
  /** Opt-out available */
  optOutAvailable: boolean;
  /** No coercive lock-in */
  noCoerciveLockIn: boolean;
  /** Anti-incumbency enforced */
  antiIncumbencyEnforced: boolean;
}

/**
 * G09 - Gate0_NoveltyFalsifiable
 */
export interface Gate0NoveltyFalsifiable {
  readonly type: 'GATE0_NOVELTY_FALSIFIABLE';
  /** Novelty claim present */
  hasNoveltyClaim: boolean;
  /** Claim is falsifiable */
  isFalsifiable: boolean;
  /** Gate passes */
  passes: boolean;
}

/**
 * G10-G13 - Mirror Sidecar System
 */
export interface MirrorSidecar {
  readonly type: 'MIRROR_SIDECAR';
  /** Whether sidecar is active */
  isActive: boolean;
  /** Last validation timestamp */
  lastValidation: number;
  /** Downgrade decisions */
  downgrades: MirrorDowngrade[];
}

export interface MirrorDowngrade {
  reason: string;
  timestamp: number;
  originalDecision: DecisionState;
  downgradedTo: DecisionState;
}

export interface MirrorLipschitzDrift {
  readonly type: 'MIRROR_LIPSCHITZ_DRIFT';
  /** Current drift |s′−s| */
  currentDrift: number;
  /** Epsilon threshold ε_lip */
  epsilonLip: number;
  /** Whether veto triggered */
  vetoTriggered: boolean;
}

export interface MirrorFPBCheck {
  readonly type: 'MIRROR_FPB_CHECK';
  /** False-positive block rate */
  fpbRate: number;
  /** Is rate sane */
  isSane: boolean;
}

export interface MirrorVeto {
  readonly type: 'MIRROR_VETO';
  /** Veto active */
  isActive: boolean;
  /** Veto reason */
  reason: 'drift' | 'fpb' | 'quarantine' | null;
}

/**
 * G14-G18 - Decision System (EXACT1)
 */
export type DecisionState = 'EXECUTE' | 'HOLD' | 'ESCALATE';

export interface EXACT1Primitive {
  readonly type: 'EXACT1_PRIMITIVE';
  /** Current decision state - exactly one must be true */
  currentState: DecisionState;
  /** Timestamp of decision */
  decidedAt: number;
}

export interface NANDOnlyCompiler {
  readonly type: 'NAND_ONLY_COMPILER';
  /** Whether NAND restriction is enforced */
  isEnforced: boolean;
  /** Compiled logic gates */
  compiledGates: number;
}

/**
 * G19 - Stop3_Counter
 */
export interface Stop3Counter {
  readonly type: 'STOP3_COUNTER';
  /** Consecutive violations */
  consecutiveViolations: number;
  /** Max before halt */
  maxViolations: 3;
  /** Is halted */
  isHalted: boolean;
}

/**
 * G20-G21 - Rollback System
 */
export interface RollbackSnapshot {
  readonly type: 'ROLLBACK_SNAPSHOT';
  /** Snapshot ID */
  snapshotId: string;
  /** Timestamp */
  timestamp: number;
  /** State data */
  stateData: unknown;
  /** Is known good */
  isKnownGood: boolean;
}

export interface RollbackTrigger {
  readonly type: 'ROLLBACK_TRIGGER';
  /** Trigger active */
  isActive: boolean;
  /** Trigger reason */
  reason: 'veto' | 'kpi_breach' | null;
  /** Target snapshot */
  targetSnapshotId: string | null;
}

/**
 * G22-G23 - Witness System
 */
export interface WitnessANDGate {
  readonly type: 'WITNESS_AND_GATE';
  /** Required witness count */
  requiredWitnesses: number;
  /** Current witnesses */
  currentWitnesses: string[];
  /** Gate passes */
  passes: boolean;
}

export interface DualWitnessMode {
  readonly type: 'DUAL_WITNESS_MODE';
  /** System witness */
  systemWitness: string;
  /** Role model witness (when available) */
  roleModelWitness: string | null;
  /** Is dual mode active */
  isDualActive: boolean;
}

/**
 * G24-G26 - Ledger System
 */
export interface MergeOnlyMemoryPolicy {
  readonly type: 'MERGE_ONLY_MEMORY_POLICY';
  /** Policy enforced */
  isEnforced: boolean;
  /** Append count */
  appendCount: number;
}

export interface DecisionLedgerRecord {
  readonly type: 'DECISION_LEDGER_RECORD';
  /** Record ID */
  recordId: string;
  /** Intent */
  intent: string;
  /** Decision made */
  decision: DecisionState;
  /** Rationale */
  rationale: string;
  /** Dependencies */
  dependencies: string[];
  /** Timestamp */
  timestamp: number;
}

export interface AuditHash {
  readonly type: 'AUDIT_HASH';
  /** Hash value */
  hash: string;
  /** Algorithm used */
  algorithm: 'sha256' | 'sha384' | 'sha512';
  /** Timestamp */
  timestamp: number;
}

/**
 * G27-G30 - Policy and Control
 */
export interface SafeModeDegrade {
  readonly type: 'SAFE_MODE_DEGRADE';
  /** Is degraded */
  isDegraded: boolean;
  /** Degradation level */
  level: number;
  /** Reason */
  reason: string;
}

export interface RelaxModePolicy {
  readonly type: 'RELAX_MODE_POLICY';
  /** Is relaxed */
  isRelaxed: boolean;
  /** KPI constraint satisfied */
  kpiSatisfied: boolean;
  /** Canon constraints satisfied */
  canonSatisfied: boolean;
}

export interface ProfessorControlInterfaceSchema {
  readonly type: 'PROFESSOR_CONTROL_INTERFACE_SCHEMA';
  /** Schema version */
  schemaVersion: string;
  /** Control parameters */
  parameters: Record<string, unknown>;
}

export interface AttestationBundle {
  readonly type: 'ATTESTATION_BUNDLE';
  /** Verifiable credential placeholders */
  vcPlaceholders: string[];
  /** DID placeholders */
  didPlaceholders: string[];
  /** Proofs */
  proofs: string[];
}

// ============================================================================
// H. Telemetry/KPI/Sensors - H01-H15
// ============================================================================

/**
 * H01-H03 - KPI Definitions
 */
export interface KPIDeltaDIADef {
  readonly type: 'KPI_DELTA_DIA_DEF';
  /** ΔDIA rate - must be ≥0 */
  deltaRate: number;
  /** Is constraint satisfied */
  isSatisfied: boolean;
}

export interface KPIDeltaDIAEstimator {
  readonly type: 'KPI_DELTA_DIA_ESTIMATOR';
  /** Estimated delta */
  estimate: number;
  /** Window used */
  windowSize: number;
  /** Samples */
  samples: number[];
}

export interface KPIEdgeBand {
  readonly type: 'KPI_EDGE_BAND';
  /** Lower edge */
  lower: number;
  /** Upper edge */
  upper: number;
  /** Is in band */
  isInBand: boolean;
}

/**
 * H04-H05 - Sensors
 */
export interface SensorLiveSlope {
  readonly type: 'SENSOR_LIVE_SLOPE';
  /** Current slope */
  slope: number;
  /** Horizon */
  horizon: number;
  /** Is drifting */
  isDrifting: boolean;
}

export interface KPIBEIBreath {
  readonly type: 'KPI_BEI_BREATH';
  /** Pre-breath Φ */
  phiPre: number;
  /** Post-breath Φ */
  phiPost: number;
  /** Invariance |Φ_pre−Φ_post| */
  invariance: number;
  /** Is invariant */
  isInvariant: boolean;
}

/**
 * H06-H08 - LPSF System
 */
export interface LPSFCandidateGen {
  readonly type: 'LPSF_CANDIDATE_GEN';
  /** Number of candidates K */
  k: number;
  /** Generated candidates */
  candidates: unknown[];
}

export interface LPSFThresholds {
  readonly type: 'LPSF_THRESHOLDS';
  /** Expression threshold τ_expr */
  tauExpr: number;
  /** Privacy threshold τ_privacy */
  tauPrivacy: number;
  /** Safety threshold τ_safety */
  tauSafety: number;
}

export interface LPSFSelector {
  readonly type: 'LPSF_SELECTOR';
  /** Selected candidate index */
  selectedIndex: number;
  /** Selection reason */
  reason: string;
  /** EXACT1 satisfied */
  exact1Satisfied: boolean;
}

/**
 * H09-H10 - Trace and Journal
 */
export interface GenesisTraceField {
  readonly type: 'GENESIS_TRACE_FIELD';
  /** Origin tremor (non-executable) */
  originTremor: string;
  /** Is attestation only */
  isAttestationOnly: true;
}

export interface MilnerJournalEntry {
  readonly type: 'MILNER_JOURNAL_ENTRY';
  /** Date */
  date: string;
  /** Status */
  status: 'alive' | 'absent' | 'hint';
  /** Sensor readings */
  sensorReadings: Record<string, number>;
}

/**
 * H11-H15 - Monitoring and Alerts
 */
export interface TelemetryStop3 {
  readonly type: 'TELEMETRY_STOP3';
  /** Weekly delta trends */
  weeklyDeltaTrends: number[];
  /** Is stopped */
  isStopped: boolean;
}

export interface DashboardView {
  readonly type: 'DASHBOARD_VIEW';
  /** KPI summary */
  kpiSummary: Record<string, number>;
  /** Sensor summary */
  sensorSummary: Record<string, number>;
  /** Last updated */
  lastUpdated: number;
}

export interface DriftAlert {
  readonly type: 'DRIFT_ALERT';
  /** Alert active */
  isActive: boolean;
  /** Alert type */
  alertType: 'mirror_veto' | 'kpi_breach';
  /** Severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Message */
  message: string;
}

export interface QuarantineList {
  readonly type: 'QUARANTINE_LIST';
  /** Quarantined items */
  items: QuarantinedItem[];
}

export interface QuarantinedItem {
  id: string;
  type: 'module' | 'memory';
  reason: string;
  quarantinedAt: number;
}

export interface RecoveryLog {
  readonly type: 'RECOVERY_LOG';
  /** Recovery entries */
  entries: RecoveryEntry[];
}

export interface RecoveryEntry {
  timestamp: number;
  action: 'mitigation' | 'rollback' | 'return_to_green';
  details: string;
  success: boolean;
}

// ============================================================================
// I. Autonomous Leisure Product - I01-I20
// ============================================================================

/**
 * I01 - ProductBrief
 */
export interface ProductBrief {
  readonly type: 'PRODUCT_BRIEF';
  /** Name */
  name: string;
  /** Description */
  description: string;
  /** Constraints */
  constraints: string[];
}

/**
 * I02-I05 - Physical Constraints
 */
export interface StationaryConstraint {
  readonly type: 'STATIONARY_CONSTRAINT';
  /** Has moving parts */
  hasMovingParts: false;
  /** Interaction mode */
  interactionMode: 'touch_only';
}

export interface ThermochromicLayer {
  readonly type: 'THERMOCHROMIC_LAYER';
  /** Temperature range for color change */
  tempRange: [number, number];
  /** Color states */
  colorStates: string[];
}

export interface HeatTransferInterface {
  readonly type: 'HEAT_TRANSFER_INTERFACE';
  /** Conductivity */
  conductivity: number;
  /** Material */
  material: string;
}

export interface TouchSafetyEnvelope {
  readonly type: 'TOUCH_SAFETY_ENVELOPE';
  /** Max safe temperature */
  maxSafeTemp: number;
  /** Materials certified safe */
  certifiedMaterials: string[];
}

/**
 * I06-I09 - Energy System
 */
export interface EnergyHarvestModule {
  readonly type: 'ENERGY_HARVEST_MODULE';
  /** Harvest sources */
  sources: ('light' | 'thermal' | 'rf')[];
  /** Efficiency per source */
  efficiency: Record<string, number>;
}

export interface StorageSupercapacitor {
  readonly type: 'STORAGE_SUPERCAPACITOR';
  /** Capacity (Farads) */
  capacity: number;
  /** Max voltage */
  maxVoltage: number;
  /** Current charge level */
  chargeLevel: number;
}

export interface PowerManagement {
  readonly type: 'POWER_MANAGEMENT';
  /** Is regulating */
  isRegulating: boolean;
  /** Protection active */
  protectionActive: boolean;
}

export interface SelfChargingLoop {
  readonly type: 'SELF_CHARGING_LOOP';
  /** Is autonomous */
  isAutonomous: boolean;
  /** Loop stages */
  stages: ('harvest' | 'store' | 'regulate')[];
}

/**
 * I10-I11 - UX System
 */
export interface ColorStateMachine {
  readonly type: 'COLOR_STATE_MACHINE';
  /** Current state */
  currentState: string;
  /** State transitions */
  transitions: Record<string, string[]>;
}

export interface PlayUXLoop {
  readonly type: 'PLAY_UX_LOOP';
  /** Loop active */
  isActive: boolean;
  /** Stages */
  stages: ('touch' | 'color_shift' | 'curiosity' | 'repeat')[];
}

/**
 * I12-I13 - Compliance
 */
export interface MaintenanceFreeConstraint {
  readonly type: 'MAINTENANCE_FREE_CONSTRAINT';
  /** Is sealed */
  isSealed: boolean;
  /** Requires servicing */
  requiresServicing: false;
}

export interface MaterialsCompliance {
  readonly type: 'MATERIALS_COMPLIANCE';
  /** Skin safe */
  isSkinSafe: boolean;
  /** Environmental compliant */
  isEnvironmentalCompliant: boolean;
  /** Certifications */
  certifications: string[];
}

/**
 * I14-I16 - Testing and Safety
 */
export interface PrototypeTestPlan {
  readonly type: 'PROTOTYPE_TEST_PLAN';
  /** Tests to run */
  tests: PrototypeTest[];
}

export interface PrototypeTest {
  name: string;
  type: 'color_delta' | 'response_time' | 'harvest_rate' | 'safety';
  criteria: string;
  passed?: boolean;
}

export interface FailureModesList {
  readonly type: 'FAILURE_MODES_LIST';
  /** Identified failure modes */
  modes: FailureMode[];
}

export interface FailureMode {
  id: string;
  name: string;
  type: 'thermal_drift' | 'fade' | 'leakage' | 'delamination';
  severity: 'low' | 'medium' | 'high';
  mitigation: string;
}

export interface FailSafeBehavior {
  readonly type: 'FAIL_SAFE_BEHAVIOR';
  /** Neutral color on fault */
  neutralColor: string;
  /** Max temperature limit */
  maxTempLimit: number;
}

/**
 * I17-I20 - Production
 */
export interface AssemblySpec {
  readonly type: 'ASSEMBLY_SPEC';
  /** Layer stack */
  layerStack: string[];
  /** Sealing method */
  sealingMethod: string;
  /** Connectors */
  connectors: string[];
}

export interface RecycleDisposalSpec {
  readonly type: 'RECYCLE_DISPOSAL_SPEC';
  /** Separation instructions */
  separationInstructions: string[];
  /** Recyclable components */
  recyclableComponents: string[];
}

export interface UserInstructions {
  readonly type: 'USER_INSTRUCTIONS';
  /** Usage instructions */
  usage: string[];
  /** Warnings */
  warnings: string[];
  /** Framing */
  framing: 'leisure';
}

export interface AcceptanceTests {
  readonly type: 'ACCEPTANCE_TESTS';
  /** Test results */
  results: AcceptanceTestResult[];
  /** MVP ready */
  isMVPReady: boolean;
}

export interface AcceptanceTestResult {
  testId: string;
  passed: boolean;
  details: string;
}
