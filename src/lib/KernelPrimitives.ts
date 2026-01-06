/**
 * RHS Kernel Primitives Implementation
 *
 * This module implements the core R/H/S (Noise/Coherence/Soleket) kernel,
 * providing the foundational primitives for the generative triadic framework.
 *
 * A01-A12: Core kernel primitives
 */

import type {
  RNoise,
  HCoherence,
  SSoleket,
  CycleOperator,
  StateSpaceRHS,
  Attractor,
  ClearanceOperator,
  DownwardConstraint,
  ClosureCondition,
  TensionField,
  CoherenceScore,
  ResonanceCheck,
  TransitionConfig,
  StateSnapshot,
  TensionPeak,
  CoherenceComponent,
  ClosureEvidence,
} from '../types/kernel';

// ============================================================================
// A01 - R/Noise Primitive Factory
// ============================================================================

/**
 * Creates an R/Noise primitive - source of perturbation/variation
 */
export function createRNoise(
  dimension: number,
  variance: number = 1.0,
  source?: string
): RNoise {
  const perturbation = generateGaussianNoise(dimension, variance);
  return {
    type: 'R_NOISE',
    perturbation,
    variance,
    timestamp: Date.now(),
    source,
  };
}

/**
 * Generates Gaussian noise using Box-Muller transform
 */
function generateGaussianNoise(dimension: number, variance: number): number[] {
  const noise: number[] = [];
  const stdDev = Math.sqrt(variance);

  for (let i = 0; i < dimension; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

    noise.push(z0 * stdDev);
    if (i + 1 < dimension) {
      noise.push(z1 * stdDev);
    }
  }

  return noise;
}

/**
 * Adds perturbation to an existing R state
 */
export function perturbRNoise(r: RNoise, additionalVariance: number): RNoise {
  const additionalNoise = generateGaussianNoise(r.perturbation.length, additionalVariance);
  const newPerturbation = r.perturbation.map((v, i) => v + additionalNoise[i]);
  const newVariance = r.variance + additionalVariance;

  return {
    ...r,
    perturbation: newPerturbation,
    variance: newVariance,
    timestamp: Date.now(),
  };
}

/**
 * Computes the energy/intensity of an R state
 */
export function computeREnergy(r: RNoise): number {
  return r.perturbation.reduce((sum, v) => sum + v * v, 0) / r.perturbation.length;
}

// ============================================================================
// A02 - H/Coherence Primitive Factory
// ============================================================================

/**
 * Creates an H/Coherence primitive - constraint/structure/holding
 */
export function createHCoherence(
  dimension: number,
  holdingStrength: number = 1.0,
  constraintDensity: number = 0.5
): HCoherence {
  const constraints = generateConstraintMatrix(dimension, constraintDensity);
  const structure = computeStructureVector(constraints);
  const activeConstraints = identifyActiveConstraints(constraints);

  return {
    type: 'H_COHERENCE',
    constraints,
    structure,
    holdingStrength,
    activeConstraints,
  };
}

/**
 * Generates a sparse constraint matrix
 */
function generateConstraintMatrix(dimension: number, density: number): number[][] {
  const matrix: number[][] = [];

  for (let i = 0; i < dimension; i++) {
    const row: number[] = [];
    for (let j = 0; j < dimension; j++) {
      if (i === j) {
        row.push(1.0); // Diagonal elements are always 1
      } else if (Math.random() < density) {
        row.push((Math.random() - 0.5) * 2); // Random constraint strength
      } else {
        row.push(0);
      }
    }
    matrix.push(row);
  }

  // Make symmetric for stability
  for (let i = 0; i < dimension; i++) {
    for (let j = i + 1; j < dimension; j++) {
      const avg = (matrix[i][j] + matrix[j][i]) / 2;
      matrix[i][j] = avg;
      matrix[j][i] = avg;
    }
  }

  return matrix;
}

/**
 * Computes structure vector from constraint matrix (dominant eigenvector approximation)
 */
function computeStructureVector(constraints: number[][]): number[] {
  const n = constraints.length;
  let v = Array(n).fill(1 / Math.sqrt(n));

  // Power iteration for dominant eigenvector
  for (let iter = 0; iter < 50; iter++) {
    const newV: number[] = Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        newV[i] += constraints[i][j] * v[j];
      }
    }

    // Normalize
    const norm = Math.sqrt(newV.reduce((sum, val) => sum + val * val, 0));
    if (norm > 1e-10) {
      v = newV.map((val) => val / norm);
    }
  }

  return v;
}

/**
 * Identifies active (non-zero) constraints
 */
function identifyActiveConstraints(constraints: number[][]): number[] {
  const active: number[] = [];
  const n = constraints.length;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(constraints[i][j]) > 0.1) {
        active.push(i * n + j); // Linear index
      }
    }
  }

  return active;
}

/**
 * Applies coherence constraints to a vector
 */
export function applyCoherence(h: HCoherence, input: number[]): number[] {
  const n = input.length;
  const output: number[] = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      output[i] += h.constraints[i][j] * input[j] * h.holdingStrength;
    }
  }

  return output;
}

/**
 * Computes the coherence strength of a state relative to H
 */
export function computeCoherenceStrength(h: HCoherence, state: number[]): number {
  const projected = applyCoherence(h, state);
  const dotProduct = state.reduce((sum, v, i) => sum + v * projected[i], 0);
  const stateNorm = Math.sqrt(state.reduce((sum, v) => sum + v * v, 0));
  const projNorm = Math.sqrt(projected.reduce((sum, v) => sum + v * v, 0));

  if (stateNorm < 1e-10 || projNorm < 1e-10) return 0;
  return dotProduct / (stateNorm * projNorm);
}

// ============================================================================
// A03 - S/Soleket Primitive Factory
// ============================================================================

/**
 * Creates an S/Soleket primitive - transformative mediation
 */
export function createSSoleket(
  dimension: number,
  clearanceLevel: number = 0.5
): SSoleket {
  const dofRemoved = Math.floor(dimension * clearanceLevel);
  const mediationCoefficients = computeMediationCoefficients(dimension, dofRemoved);
  const transformMatrix = computeTransformMatrix(dimension, dofRemoved);

  return {
    type: 'S_SOLEKET',
    mediationCoefficients,
    dofRemoved,
    transformMatrix,
    clearanceLevel,
  };
}

/**
 * Computes mediation coefficients for degree-of-freedom removal
 */
function computeMediationCoefficients(dimension: number, dofRemoved: number): number[] {
  const coefficients: number[] = [];
  const keptDof = dimension - dofRemoved;

  for (let i = 0; i < dimension; i++) {
    if (i < keptDof) {
      // Kept degrees of freedom have full weight
      coefficients.push(1.0);
    } else {
      // Removed degrees of freedom decay exponentially
      coefficients.push(Math.exp(-(i - keptDof) / Math.max(1, dofRemoved)));
    }
  }

  return coefficients;
}

/**
 * Computes the transformation matrix for Soleket
 */
function computeTransformMatrix(dimension: number, dofRemoved: number): number[][] {
  const matrix: number[][] = [];
  const keptDof = dimension - dofRemoved;

  for (let i = 0; i < dimension; i++) {
    const row: number[] = [];
    for (let j = 0; j < dimension; j++) {
      if (i === j) {
        // Diagonal: 1 for kept DOF, exponential decay for removed
        row.push(j < keptDof ? 1.0 : Math.exp(-(j - keptDof + 1) / Math.max(1, dofRemoved)));
      } else if (Math.abs(i - j) === 1) {
        // Nearest neighbor coupling for smoothing
        row.push(0.1);
      } else {
        row.push(0);
      }
    }
    matrix.push(row);
  }

  return matrix;
}

/**
 * Applies Soleket transformation to a state
 */
export function applySoleket(s: SSoleket, input: number[]): number[] {
  const n = input.length;
  const output: number[] = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      output[i] += s.transformMatrix[i][j] * input[j] * s.mediationCoefficients[j];
    }
  }

  return output;
}

/**
 * Computes the clearance effect - how much phase space was contracted
 */
export function computeClearanceEffect(s: SSoleket, before: number[], after: number[]): number {
  const beforeVariance = before.reduce((sum, v) => sum + v * v, 0) / before.length;
  const afterVariance = after.reduce((sum, v) => sum + v * v, 0) / after.length;

  if (beforeVariance < 1e-10) return 0;
  return 1 - afterVariance / beforeVariance;
}

// ============================================================================
// A04 - CycleOperator Implementation
// ============================================================================

/**
 * Creates a cycle operator for R→H→S→R′ generative loop
 */
export function createCycleOperator(
  transitionParams?: Partial<CycleOperator['transitionParams']>
): CycleOperator {
  const defaultTransition: TransitionConfig = {
    rate: 0.1,
    threshold: 0.5,
    damping: 0.9,
  };

  return {
    type: 'CYCLE_OPERATOR',
    currentPhase: 'R',
    cycleCount: 0,
    transitionParams: {
      rToH: transitionParams?.rToH ?? { ...defaultTransition },
      hToS: transitionParams?.hToS ?? { ...defaultTransition },
      sToR: transitionParams?.sToR ?? { ...defaultTransition },
    },
  };
}

/**
 * Advances the cycle by one phase
 */
export function advanceCycle(operator: CycleOperator): CycleOperator {
  const nextPhase = {
    R: 'H' as const,
    H: 'S' as const,
    S: 'R' as const,
  };

  const newCycleCount = operator.currentPhase === 'S' ? operator.cycleCount + 1 : operator.cycleCount;

  return {
    ...operator,
    currentPhase: nextPhase[operator.currentPhase],
    cycleCount: newCycleCount,
  };
}

/**
 * Executes a full R→H→S→R′ cycle
 */
export function executeCycle(
  state: StateSpaceRHS,
  operator: CycleOperator
): { state: StateSpaceRHS; operator: CycleOperator } {
  let currentState = state;
  let currentOperator = operator;

  // R → H transition
  const rToHResult = transitionRToH(currentState, currentOperator.transitionParams.rToH);
  currentState = rToHResult;
  currentOperator = advanceCycle(currentOperator);

  // H → S transition
  const hToSResult = transitionHToS(currentState, currentOperator.transitionParams.hToS);
  currentState = hToSResult;
  currentOperator = advanceCycle(currentOperator);

  // S → R′ transition
  const sToRResult = transitionSToR(currentState, currentOperator.transitionParams.sToR);
  currentState = sToRResult;
  currentOperator = advanceCycle(currentOperator);

  return { state: currentState, operator: currentOperator };
}

/**
 * R → H transition: noise encounters coherence
 */
function transitionRToH(state: StateSpaceRHS, config: TransitionConfig): StateSpaceRHS {
  const coherenceApplied = applyCoherence(state.h, state.r.perturbation);
  const damped = coherenceApplied.map((v) => v * config.damping);

  const newH: HCoherence = {
    ...state.h,
    structure: damped,
  };

  return {
    ...state,
    h: newH,
    history: [
      ...state.history,
      {
        timestamp: Date.now(),
        r: state.r.perturbation,
        h: damped,
        s: state.s.mediationCoefficients,
      },
    ],
  };
}

/**
 * H → S transition: coherence produces clearance
 */
function transitionHToS(state: StateSpaceRHS, config: TransitionConfig): StateSpaceRHS {
  const structureEnergy = state.h.structure.reduce((sum, v) => sum + v * v, 0);
  const clearanceLevel = Math.min(1, structureEnergy * config.rate);

  const newS = createSSoleket(state.dimension, clearanceLevel);

  return {
    ...state,
    s: newS,
    history: [
      ...state.history,
      {
        timestamp: Date.now(),
        r: state.r.perturbation,
        h: state.h.structure,
        s: newS.mediationCoefficients,
      },
    ],
  };
}

/**
 * S → R′ transition: clearance transforms noise
 */
function transitionSToR(state: StateSpaceRHS, config: TransitionConfig): StateSpaceRHS {
  const cleared = applySoleket(state.s, state.r.perturbation);
  const newNoise = generateGaussianNoise(state.dimension, state.r.variance * (1 - state.s.clearanceLevel));
  const combined = cleared.map((v, i) => v * config.damping + newNoise[i] * config.rate);

  const newR: RNoise = {
    ...state.r,
    perturbation: combined,
    timestamp: Date.now(),
  };

  return {
    ...state,
    r: newR,
    history: [
      ...state.history,
      {
        timestamp: Date.now(),
        r: combined,
        h: state.h.structure,
        s: state.s.mediationCoefficients,
      },
    ],
  };
}

// ============================================================================
// A05 - StateSpace_RHS Factory
// ============================================================================

/**
 * Creates a state space encoding the ⟨R,H,S⟩ manifold
 */
export function createStateSpaceRHS(
  dimension: number,
  initialVariance: number = 1.0,
  holdingStrength: number = 1.0,
  clearanceLevel: number = 0.5
): StateSpaceRHS {
  const r = createRNoise(dimension, initialVariance);
  const h = createHCoherence(dimension, holdingStrength);
  const s = createSSoleket(dimension, clearanceLevel);

  return {
    type: 'STATE_SPACE_RHS',
    r,
    h,
    s,
    dimension,
    history: [
      {
        timestamp: Date.now(),
        r: r.perturbation,
        h: h.structure,
        s: s.mediationCoefficients,
      },
    ],
  };
}

/**
 * Gets the current state vector (concatenation of R, H, S)
 */
export function getStateVector(state: StateSpaceRHS): number[] {
  return [...state.r.perturbation, ...state.h.structure, ...state.s.mediationCoefficients];
}

/**
 * Computes the manifold distance between two states
 */
export function computeManifoldDistance(state1: StateSpaceRHS, state2: StateSpaceRHS): number {
  const v1 = getStateVector(state1);
  const v2 = getStateVector(state2);

  return Math.sqrt(v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i], 2), 0));
}

// ============================================================================
// A06 - Attractor Detection
// ============================================================================

/**
 * Detects attractors in the state history
 */
export function detectAttractors(state: StateSpaceRHS, minStability: number = 0.7): Attractor[] {
  const attractors: Attractor[] = [];

  if (state.history.length < 10) return attractors;

  // Analyze H-level attractors
  const hAttractor = detectHAttractor(state.history, minStability);
  if (hAttractor) attractors.push(hAttractor);

  // Analyze S-level attractors
  const sAttractor = detectSAttractor(state.history, minStability);
  if (sAttractor) attractors.push(sAttractor);

  return attractors;
}

/**
 * Detects attractors in the H (coherence) trajectory
 */
function detectHAttractor(history: StateSnapshot[], minStability: number): Attractor | null {
  const recent = history.slice(-20);
  if (recent.length < 10) return null;

  // Compute centroid of recent H states
  const dimension = recent[0].h.length;
  const center = Array(dimension).fill(0);

  for (const snapshot of recent) {
    for (let i = 0; i < dimension; i++) {
      center[i] += snapshot.h[i] / recent.length;
    }
  }

  // Compute average distance from center (radius)
  let totalDist = 0;
  for (const snapshot of recent) {
    const dist = Math.sqrt(snapshot.h.reduce((sum, v, i) => sum + Math.pow(v - center[i], 2), 0));
    totalDist += dist;
  }
  const radius = totalDist / recent.length;

  // Compute stability (inverse of variance in distances)
  const distances = recent.map((snapshot) =>
    Math.sqrt(snapshot.h.reduce((sum, v, i) => sum + Math.pow(v - center[i], 2), 0))
  );
  const meanDist = distances.reduce((a, b) => a + b, 0) / distances.length;
  const variance = distances.reduce((sum, d) => sum + Math.pow(d - meanDist, 2), 0) / distances.length;
  const stability = 1 / (1 + variance);

  if (stability < minStability) return null;

  // Determine attractor type based on trajectory pattern
  const attractorType = classifyAttractorType(recent.map((s) => s.h));

  return {
    type: 'ATTRACTOR',
    center,
    radius,
    stability,
    emergenceLevel: 'H',
    attractorType,
  };
}

/**
 * Detects attractors in the S (soleket) trajectory
 */
function detectSAttractor(history: StateSnapshot[], minStability: number): Attractor | null {
  const recent = history.slice(-20);
  if (recent.length < 10) return null;

  const dimension = recent[0].s.length;
  const center = Array(dimension).fill(0);

  for (const snapshot of recent) {
    for (let i = 0; i < dimension; i++) {
      center[i] += snapshot.s[i] / recent.length;
    }
  }

  let totalDist = 0;
  for (const snapshot of recent) {
    const dist = Math.sqrt(snapshot.s.reduce((sum, v, i) => sum + Math.pow(v - center[i], 2), 0));
    totalDist += dist;
  }
  const radius = totalDist / recent.length;

  const distances = recent.map((snapshot) =>
    Math.sqrt(snapshot.s.reduce((sum, v, i) => sum + Math.pow(v - center[i], 2), 0))
  );
  const meanDist = distances.reduce((a, b) => a + b, 0) / distances.length;
  const variance = distances.reduce((sum, d) => sum + Math.pow(d - meanDist, 2), 0) / distances.length;
  const stability = 1 / (1 + variance);

  if (stability < minStability) return null;

  const attractorType = classifyAttractorType(recent.map((s) => s.s));

  return {
    type: 'ATTRACTOR',
    center,
    radius,
    stability,
    emergenceLevel: 'S',
    attractorType,
  };
}

/**
 * Classifies attractor type based on trajectory pattern
 */
function classifyAttractorType(trajectory: number[][]): 'point' | 'limit_cycle' | 'strange' {
  if (trajectory.length < 10) return 'point';

  // Compute return map characteristics
  const distances: number[] = [];
  for (let i = 1; i < trajectory.length; i++) {
    const dist = Math.sqrt(
      trajectory[i].reduce((sum, v, j) => sum + Math.pow(v - trajectory[i - 1][j], 2), 0)
    );
    distances.push(dist);
  }

  const meanDist = distances.reduce((a, b) => a + b, 0) / distances.length;
  const variance = distances.reduce((sum, d) => sum + Math.pow(d - meanDist, 2), 0) / distances.length;
  const cv = Math.sqrt(variance) / (meanDist + 1e-10); // Coefficient of variation

  // Low movement = point attractor
  if (meanDist < 0.1) return 'point';

  // Regular movement = limit cycle
  if (cv < 0.5) return 'limit_cycle';

  // Irregular movement = strange attractor
  return 'strange';
}

// ============================================================================
// A07 - ClearanceOperator Implementation
// ============================================================================

/**
 * Creates a clearance operator for phase space contraction
 */
export function createClearanceOperator(
  contractionFactor: number = 0.5,
  threshold: number = 0.3
): ClearanceOperator {
  return {
    type: 'CLEARANCE_OPERATOR',
    contractionFactor,
    incompatibleModes: [],
    threshold,
    isClearing: false,
  };
}

/**
 * Executes clearance: contracts phase space and removes incompatible modes
 */
export function executeClearance(
  operator: ClearanceOperator,
  state: StateSpaceRHS
): { operator: ClearanceOperator; state: StateSpaceRHS } {
  // Identify incompatible modes based on tension with coherence
  const incompatibleModes = identifyIncompatibleModes(state, operator.threshold);

  // Contract phase space by applying clearance
  const contractedR = contractPhaseSpace(state.r, incompatibleModes, operator.contractionFactor);

  const newOperator: ClearanceOperator = {
    ...operator,
    incompatibleModes,
    isClearing: true,
  };

  const newState: StateSpaceRHS = {
    ...state,
    r: contractedR,
  };

  return { operator: newOperator, state: newState };
}

/**
 * Identifies modes that are incompatible with current coherence structure
 */
function identifyIncompatibleModes(state: StateSpaceRHS, threshold: number): number[] {
  const incompatible: number[] = [];
  const projected = applyCoherence(state.h, state.r.perturbation);

  for (let i = 0; i < state.r.perturbation.length; i++) {
    const residual = Math.abs(state.r.perturbation[i] - projected[i]);
    if (residual > threshold) {
      incompatible.push(i);
    }
  }

  return incompatible;
}

/**
 * Contracts phase space by reducing incompatible modes
 */
function contractPhaseSpace(
  r: RNoise,
  incompatibleModes: number[],
  contractionFactor: number
): RNoise {
  const newPerturbation = [...r.perturbation];

  for (const mode of incompatibleModes) {
    newPerturbation[mode] *= 1 - contractionFactor;
  }

  return {
    ...r,
    perturbation: newPerturbation,
    timestamp: Date.now(),
  };
}

// ============================================================================
// A08 - DownwardConstraint Implementation
// ============================================================================

/**
 * Creates a downward constraint encoding S_t → {R,H}_{t+1}
 */
export function createDownwardConstraint(
  sourceTimeIndex: number,
  strength: number = 0.5,
  affectedVariables: ('R' | 'H')[] = ['R', 'H']
): DownwardConstraint {
  return {
    type: 'DOWNWARD_CONSTRAINT',
    sourceTimeIndex,
    targetTimeIndex: sourceTimeIndex + 1,
    strength,
    affectedVariables,
    constraintMatrix: [],
  };
}

/**
 * Computes the constraint matrix from S state
 */
export function computeDownwardConstraintMatrix(
  constraint: DownwardConstraint,
  sState: SSoleket
): DownwardConstraint {
  // The constraint matrix is derived from the Soleket transformation
  const constraintMatrix = sState.transformMatrix.map((row) =>
    row.map((v) => v * constraint.strength)
  );

  return {
    ...constraint,
    constraintMatrix,
  };
}

/**
 * Applies downward constraint to future state
 */
export function applyDownwardConstraint(
  constraint: DownwardConstraint,
  state: StateSpaceRHS
): StateSpaceRHS {
  let newState = state;

  if (constraint.affectedVariables.includes('R')) {
    const constrainedPerturbation = applyConstraintMatrix(
      constraint.constraintMatrix,
      state.r.perturbation
    );
    newState = {
      ...newState,
      r: { ...newState.r, perturbation: constrainedPerturbation },
    };
  }

  if (constraint.affectedVariables.includes('H')) {
    const constrainedStructure = applyConstraintMatrix(
      constraint.constraintMatrix,
      state.h.structure
    );
    newState = {
      ...newState,
      h: { ...newState.h, structure: constrainedStructure },
    };
  }

  return newState;
}

/**
 * Applies a constraint matrix to a vector
 */
function applyConstraintMatrix(matrix: number[][], vector: number[]): number[] {
  if (matrix.length === 0) return vector;

  const result: number[] = Array(vector.length).fill(0);
  for (let i = 0; i < Math.min(matrix.length, vector.length); i++) {
    for (let j = 0; j < Math.min(matrix[i].length, vector.length); j++) {
      result[i] += matrix[i][j] * vector[j];
    }
  }
  return result;
}

// ============================================================================
// A09 - ClosureCondition Implementation
// ============================================================================

/**
 * Evaluates closure condition: whether macro→micro influence exists
 */
export function evaluateClosureCondition(state: StateSpaceRHS): ClosureCondition {
  const evidence: ClosureEvidence[] = [];

  // Statistical test: correlation between S and future R
  const statisticalEvidence = computeStatisticalClosure(state);
  evidence.push(statisticalEvidence);

  // Causal test: Granger-like causality check
  const causalEvidence = computeCausalClosure(state);
  evidence.push(causalEvidence);

  // Overall influence strength
  const influenceStrength =
    evidence.reduce((sum, e) => sum + e.value * e.confidence, 0) /
    evidence.reduce((sum, e) => sum + e.confidence, 0);

  const isSatisfied = influenceStrength > 0.3 && evidence.some((e) => e.value > 0.5);

  return {
    type: 'CLOSURE_CONDITION',
    isSatisfied,
    influenceStrength,
    evidence,
  };
}

/**
 * Computes statistical evidence for closure
 */
function computeStatisticalClosure(state: StateSpaceRHS): ClosureEvidence {
  if (state.history.length < 5) {
    return { method: 'statistical', value: 0, confidence: 0.1 };
  }

  // Compute correlation between S_t and R_{t+1}
  let correlation = 0;
  let count = 0;

  for (let t = 0; t < state.history.length - 1; t++) {
    const sT = state.history[t].s;
    const rT1 = state.history[t + 1].r;

    const sNorm = Math.sqrt(sT.reduce((sum, v) => sum + v * v, 0));
    const rNorm = Math.sqrt(rT1.reduce((sum, v) => sum + v * v, 0));

    if (sNorm > 1e-10 && rNorm > 1e-10) {
      const dotProduct = sT.reduce((sum, v, i) => sum + v * rT1[i], 0);
      correlation += dotProduct / (sNorm * rNorm);
      count++;
    }
  }

  const avgCorrelation = count > 0 ? Math.abs(correlation / count) : 0;
  const confidence = Math.min(1, count / 10);

  return { method: 'statistical', value: avgCorrelation, confidence };
}

/**
 * Computes causal evidence for closure (simplified Granger-like test)
 */
function computeCausalClosure(state: StateSpaceRHS): ClosureEvidence {
  if (state.history.length < 10) {
    return { method: 'causal', value: 0, confidence: 0.1 };
  }

  // Compute predictive improvement when using S
  let mseWithS = 0;
  let mseWithoutS = 0;

  for (let t = 2; t < state.history.length - 1; t++) {
    const rPredWithS = predictWithS(state.history, t);
    const rPredWithoutS = predictWithoutS(state.history, t);
    const rActual = state.history[t + 1].r;

    mseWithS += rActual.reduce((sum, v, i) => sum + Math.pow(v - rPredWithS[i], 2), 0);
    mseWithoutS += rActual.reduce((sum, v, i) => sum + Math.pow(v - rPredWithoutS[i], 2), 0);
  }

  // Causal value: improvement ratio
  const value = mseWithoutS > 1e-10 ? Math.max(0, 1 - mseWithS / mseWithoutS) : 0;
  const confidence = Math.min(1, (state.history.length - 3) / 20);

  return { method: 'causal', value, confidence };
}

/**
 * Simple prediction using S
 */
function predictWithS(history: StateSnapshot[], t: number): number[] {
  const sT = history[t].s;
  const rT = history[t].r;

  // Simple model: R_{t+1} = α * R_t + β * S_t
  return rT.map((v, i) => 0.5 * v + 0.5 * sT[i]);
}

/**
 * Simple prediction without S
 */
function predictWithoutS(history: StateSnapshot[], t: number): number[] {
  const rT = history[t].r;
  const rTm1 = history[t - 1].r;

  // Simple autoregressive: R_{t+1} = R_t + (R_t - R_{t-1})
  return rT.map((v, i) => v + (v - rTm1[i]) * 0.5);
}

// ============================================================================
// A10 - TensionField Implementation
// ============================================================================

/**
 * Computes the tension field between R and H
 */
export function computeTensionField(state: StateSpaceRHS): TensionField {
  const n = state.dimension;
  const values: number[][] = [];
  const peaks: TensionPeak[] = [];

  // Compute tension at each point
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const rComponent = state.r.perturbation[i] * state.r.perturbation[j];
      const hComponent = state.h.constraints[i][j];
      const tension = Math.abs(rComponent - hComponent);
      row.push(tension);

      // Track peaks
      if (tension > 0.5) {
        peaks.push({ location: [i, j], magnitude: tension });
      }
    }
    values.push(row);
  }

  // Compute gradient
  const gradient = computeTensionGradient(values);

  // Total energy
  const totalEnergy = values.reduce((sum, row) => sum + row.reduce((s, v) => s + v * v, 0), 0);

  return {
    type: 'TENSION_FIELD',
    values,
    peaks: peaks.sort((a, b) => b.magnitude - a.magnitude).slice(0, 10),
    totalEnergy,
    gradient,
  };
}

/**
 * Computes the gradient of the tension field
 */
function computeTensionGradient(values: number[][]): number[][] {
  const n = values.length;
  const gradient: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      // Central difference approximation
      const dxPlus = i < n - 1 ? values[i + 1][j] : values[i][j];
      const dxMinus = i > 0 ? values[i - 1][j] : values[i][j];
      const dyPlus = j < n - 1 ? values[i][j + 1] : values[i][j];
      const dyMinus = j > 0 ? values[i][j - 1] : values[i][j];

      const gradMagnitude = Math.sqrt(
        Math.pow((dxPlus - dxMinus) / 2, 2) + Math.pow((dyPlus - dyMinus) / 2, 2)
      );
      row.push(gradMagnitude);
    }
    gradient.push(row);
  }

  return gradient;
}

/**
 * Finds the direction of maximum tension reduction
 */
export function findTensionReductionDirection(tensionField: TensionField): number[] {
  const n = tensionField.gradient.length;
  const direction: number[] = Array(n).fill(0);

  // Move opposite to gradient (steepest descent)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      direction[i] -= tensionField.gradient[i][j];
    }
  }

  // Normalize
  const norm = Math.sqrt(direction.reduce((sum, v) => sum + v * v, 0));
  if (norm > 1e-10) {
    return direction.map((v) => v / norm);
  }

  return direction;
}

// ============================================================================
// A11 - CoherenceScore Implementation
// ============================================================================

/**
 * Computes coherence score for tracking
 */
export function computeCoherenceScore(state: StateSpaceRHS): CoherenceScore {
  const components: CoherenceComponent[] = [];

  // Structural coherence: alignment with H structure
  const structuralCoherence = computeStructuralCoherence(state);
  components.push({
    name: 'structural',
    contribution: structuralCoherence,
    weight: 0.4,
  });

  // Dynamic coherence: stability over time
  const dynamicCoherence = computeDynamicCoherence(state);
  components.push({
    name: 'dynamic',
    contribution: dynamicCoherence,
    weight: 0.3,
  });

  // Integration coherence: R-H-S coupling
  const integrationCoherence = computeIntegrationCoherence(state);
  components.push({
    name: 'integration',
    contribution: integrationCoherence,
    weight: 0.3,
  });

  // Weighted sum
  const value = components.reduce((sum, c) => sum + c.contribution * c.weight, 0);

  return {
    type: 'COHERENCE_SCORE',
    value,
    components,
    timestamp: Date.now(),
  };
}

/**
 * Computes structural coherence
 */
function computeStructuralCoherence(state: StateSpaceRHS): number {
  return computeCoherenceStrength(state.h, state.r.perturbation);
}

/**
 * Computes dynamic coherence from history
 */
function computeDynamicCoherence(state: StateSpaceRHS): number {
  if (state.history.length < 3) return 0.5;

  // Compute variance in H over recent history
  const recent = state.history.slice(-10);
  const n = recent[0].h.length;

  let totalVariance = 0;
  for (let i = 0; i < n; i++) {
    const values = recent.map((s) => s.h[i]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    totalVariance += variance;
  }

  // High variance = low coherence
  return 1 / (1 + totalVariance);
}

/**
 * Computes integration coherence (R-H-S coupling)
 */
function computeIntegrationCoherence(state: StateSpaceRHS): number {
  // Apply full cycle and measure preservation
  const projected = applyCoherence(state.h, state.r.perturbation);
  const cleared = applySoleket(state.s, projected);

  // Compute similarity between original R and final output
  const rNorm = Math.sqrt(state.r.perturbation.reduce((sum, v) => sum + v * v, 0));
  const cNorm = Math.sqrt(cleared.reduce((sum, v) => sum + v * v, 0));

  if (rNorm < 1e-10 || cNorm < 1e-10) return 0;

  const dotProduct = state.r.perturbation.reduce((sum, v, i) => sum + v * cleared[i], 0);
  return Math.abs(dotProduct / (rNorm * cNorm));
}

// ============================================================================
// A12 - ResonanceCheck Implementation
// ============================================================================

/**
 * Checks if emergent S behaves as resonant attractor
 */
export function checkResonance(state: StateSpaceRHS): ResonanceCheck {
  const attractors = detectAttractors(state, 0.5);
  const sAttractors = attractors.filter((a) => a.emergenceLevel === 'S');

  if (sAttractors.length === 0) {
    return {
      type: 'RESONANCE_CHECK',
      isResonant: false,
      stability: 0,
      selectivity: 0,
    };
  }

  const bestAttractor = sAttractors.reduce((best, curr) =>
    curr.stability > best.stability ? curr : best
  );

  // Compute selectivity: how selective is the attractor basin
  const selectivity = computeSelectivity(state, bestAttractor);

  // Compute resonance frequency if limit cycle
  const frequency =
    bestAttractor.attractorType === 'limit_cycle' ? estimateResonanceFrequency(state) : undefined;

  const isResonant = bestAttractor.stability > 0.7 && selectivity > 0.5;

  return {
    type: 'RESONANCE_CHECK',
    isResonant,
    stability: bestAttractor.stability,
    selectivity,
    frequency,
  };
}

/**
 * Computes selectivity of an attractor
 */
function computeSelectivity(state: StateSpaceRHS, attractor: Attractor): number {
  // Selectivity: how many states converge to this attractor vs escape
  const recent = state.history.slice(-20);

  let converging = 0;
  for (const snapshot of recent) {
    const dist = Math.sqrt(
      snapshot.s.reduce((sum, v, i) => sum + Math.pow(v - attractor.center[i], 2), 0)
    );
    if (dist < attractor.radius * 2) {
      converging++;
    }
  }

  return converging / recent.length;
}

/**
 * Estimates resonance frequency from S trajectory
 */
function estimateResonanceFrequency(state: StateSpaceRHS): number {
  if (state.history.length < 20) return 0;

  const recent = state.history.slice(-50);

  // Simple zero-crossing count on first component
  let crossings = 0;
  const mean = recent.reduce((sum, s) => sum + s.s[0], 0) / recent.length;

  for (let i = 1; i < recent.length; i++) {
    if ((recent[i].s[0] - mean) * (recent[i - 1].s[0] - mean) < 0) {
      crossings++;
    }
  }

  // Frequency estimate
  const avgTimeDelta =
    (recent[recent.length - 1].timestamp - recent[0].timestamp) / (recent.length - 1);
  if (avgTimeDelta > 0) {
    return (crossings / 2 / recent.length) * (1000 / avgTimeDelta); // Hz
  }

  return 0;
}

// ============================================================================
// Utility Exports
// ============================================================================

export {
  generateGaussianNoise,
  generateConstraintMatrix,
  computeStructureVector,
  identifyActiveConstraints,
  computeMediationCoefficients,
  computeTransformMatrix,
};
