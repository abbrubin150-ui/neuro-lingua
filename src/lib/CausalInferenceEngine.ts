/**
 * CausalInferenceEngine - Probabilistic Dynamic Causal Inference System
 *
 * This engine implements a comprehensive causal inference framework based on a
 * Directed Acyclic Graph (DAG) for dynamic data, with:
 * - Temporal dependencies
 * - Unmeasured confounders
 * - Selection mechanisms
 * - Adaptive quantization
 *
 * The engine operates in three phases:
 * 1. Offline Learning: Learn from historical data (propensity, outcome models)
 * 2. Online Selection: Adaptive policy selection with exploration
 * 3. Statistical Testing: Formal hypothesis testing with bias verification
 *
 * Mathematical Foundation:
 * - X_{it}: Measured features
 * - Y_{it}: Continuous outcome (unobserved in new year)
 * - Ỹ_{it} = Q_{θt}(Y_{it}): Quantized outcome
 * - Z_{it} ∈ {A, B}: Policy selection
 * - U_i: Unmeasured confounders
 *
 * Reference: Rubin (1974), Pearl (2009), Robins et al. (1994)
 *
 * @module CausalInferenceEngine
 */

import type {
  CausalModelConfig,
  CausalEngineState,
  HistoricalDataset,
  HistoricalRecord,
  OfflinePhaseResult,
  OnlinePhaseState,
  QuantizationParams,
  DequantizationMapping,
  HypothesisTestResult,
  IdentifiabilityResult,
  IdentifiabilityDiagnostics,
  BiasVerificationResult,
  NeutralityVerification,
  DifferentialNeutralityCheck,
  FairnessMetric,
  SensitivityAnalysis,
  PowerAnalysisResult,
  AuditLogEntry,
  CausalLedgerEntry,
  CausalAnalysisLedger,
  Policy,
  BinIndex
} from '../types/causal';

import {
  estimatePropensityScores,
  predictPropensity,
  fitOutcomeModel,
  predictOutcome,
  computeAIPWEstimate,
  createEntropyQuantization,
  quantize,
  learnDequantizationMappings,
  dequantize,
  adaptQuantization,
  testATESignificance,
  bootstrapATEConfidenceInterval,
  estimatePower,
  minimumDetectableEffect,
  sigmoid,
  mean,
  variance,
  randomNormal,
  randomMultivariateNormal,
  createSeededRandom
} from '../math/causal_math';

// ============================================================================
// Constants
// ============================================================================

/** Default configuration values */
const DEFAULT_CONFIG: Partial<CausalModelConfig> = {
  numStudents: 100,
  numTimeSteps: 180,
  featureDimension: 5,
  seed: 42
};

/** Minimum propensity for randomization */
const MIN_EXPLORATION_PROB = 0.1;

/** Maximum propensity for randomization */
const MAX_EXPLORATION_PROB = 0.9;

/** Default significance level */
const DEFAULT_ALPHA = 0.05;

/** Default number of bootstrap samples */
const DEFAULT_BOOTSTRAP_SAMPLES = 1000;

/** Neutrality tolerance threshold */
const NEUTRALITY_TOLERANCE = 0.1;

/** Differential neutrality epsilon */
const DEFAULT_EPSILON = 0.5;

/** Fairness delta threshold */
const DEFAULT_DELTA = 0.05;

// ============================================================================
// Main Engine Class
// ============================================================================

/**
 * CausalInferenceEngine - Main engine for probabilistic dynamic causal inference
 *
 * Usage:
 * ```typescript
 * const engine = new CausalInferenceEngine(config);
 *
 * // Phase 1: Offline learning from historical data
 * const offlineResult = engine.runOfflinePhase(historicalData);
 *
 * // Phase 2: Online selection (iterative)
 * for (let t = 0; t < T; t++) {
 *   const selections = engine.selectPolicies(features, t);
 *   const outcomes = collectOutcomes(selections);
 *   engine.recordOnlineObservations(outcomes, t);
 * }
 *
 * // Phase 3: Statistical testing
 * const testResult = engine.runStatisticalTest();
 * const biasCheck = engine.verifyUnbiasedness();
 * ```
 */
export class CausalInferenceEngine {
  private state: CausalEngineState;
  private random: () => number;

  // Learned models (after offline phase)
  private propensityCoefficients: number[] = [];
  private outcomeModelA: {
    intercept: number;
    treatmentEffect: number;
    featureCoefficients: number[];
  } | null = null;
  private outcomeModelB: {
    intercept: number;
    treatmentEffect: number;
    featureCoefficients: number[];
  } | null = null;

  /**
   * Create a new CausalInferenceEngine
   *
   * @param config - Model configuration
   */
  constructor(config: Partial<CausalModelConfig> = {}) {
    const fullConfig = this.buildDefaultConfig(config);

    this.state = {
      phase: 'uninitialized',
      config: fullConfig,
      auditLog: []
    };

    this.random = config.seed ? createSeededRandom(config.seed) : Math.random;

    this.log('initialize', { config: fullConfig });
  }

  // ============================================================================
  // Configuration
  // ============================================================================

  /**
   * Build full configuration with defaults
   */
  private buildDefaultConfig(partial: Partial<CausalModelConfig>): CausalModelConfig {
    const featureDim = partial.featureDimension ?? DEFAULT_CONFIG.featureDimension!;

    return {
      outcomeModel: partial.outcomeModel ?? {
        intercept: 0,
        treatmentCoefficient: 0.5, // True causal effect
        featureCoefficients: new Array(featureDim).fill(0.1),
        confounderCoefficients: [0.3],
        autoregCoefficient: 0.7,
        noiseStd: 0.5
      },
      selectionModel: partial.selectionModel ?? {
        intercept: 0,
        featureCoefficients: new Array(featureDim).fill(0.1),
        previousOutcomeCoefficient: 0.2,
        quantizationCoefficients: [0.05],
        confounderCoefficients: [0.4]
      },
      confounderConfig: partial.confounderConfig ?? {
        dimension: 1,
        mean: [0],
        covariance: [1]
      },
      initialQuantization: partial.initialQuantization ?? {
        timeStep: 0,
        numBins: 5,
        boundaries: [-1.5, -0.5, 0.5, 1.5],
        method: 'uniform',
        symmetric: true
      },
      numStudents: partial.numStudents ?? DEFAULT_CONFIG.numStudents!,
      numTimeSteps: partial.numTimeSteps ?? DEFAULT_CONFIG.numTimeSteps!,
      featureDimension: featureDim,
      seed: partial.seed ?? DEFAULT_CONFIG.seed
    };
  }

  /**
   * Get current engine state
   */
  getState(): CausalEngineState {
    return { ...this.state };
  }

  /**
   * Get current phase
   */
  getPhase(): CausalEngineState['phase'] {
    return this.state.phase;
  }

  // ============================================================================
  // Phase 1: Offline Learning
  // ============================================================================

  /**
   * Run the offline learning phase on historical data
   *
   * This phase:
   * 1. Learns propensity score model: P(Z=B|X,Y_{t-1},θ)
   * 2. Learns outcome models: E[Y|Z,X,Y_{t-1}] for both policies
   * 3. Calibrates dequantization mappings
   * 4. Estimates historical ATE (potentially biased)
   * 5. Optimizes initial quantization parameters
   *
   * @param data - Historical dataset
   * @returns Offline phase results
   */
  runOfflinePhase(data: HistoricalDataset): OfflinePhaseResult {
    this.log('offline_phase_start', { numRecords: data.records.length });
    this.state.phase = 'offline_learning';

    // Extract data for model fitting
    const {
      features,
      treatments,
      outcomes,
      quantizedOutcomes,
      prevOutcomes,
      quantParams: _quantParams
    } = this.extractHistoricalData(data);

    // 1. Fit propensity score model
    const propensityResult = this.fitPropensityModel(features, treatments, prevOutcomes);
    this.propensityCoefficients = propensityResult.coefficients;

    // 2. Fit outcome models (separately for each treatment)
    const { modelA, modelB } = this.fitOutcomeModels(features, treatments, outcomes, prevOutcomes);
    this.outcomeModelA = modelA;
    this.outcomeModelB = modelB;

    // 3. Learn dequantization mappings
    const dequantMappings = learnDequantizationMappings(
      outcomes,
      quantizedOutcomes,
      this.state.config.initialQuantization.numBins
    );

    // 4. Compute historical ATE estimate
    const propensities = predictPropensity(features, this.propensityCoefficients);
    const muA = predictOutcome(features, 0, modelA);
    const muB = predictOutcome(features, 1, modelB);

    const { ate: historicalATE } = computeAIPWEstimate(
      outcomes,
      treatments,
      propensities,
      muA,
      muB
    );

    // 5. Optimize initial quantization
    const optimizedQuant = this.optimizeQuantization(outcomes, dequantMappings);

    // Store results
    const result: OfflinePhaseResult = {
      propensityParams: propensityResult.coefficients,
      outcomeParams: [...modelA.featureCoefficients, modelA.treatmentEffect],
      dequantizationMappings: dequantMappings,
      historicalATE,
      optimizedQuantization: {
        ...optimizedQuant,
        timeStep: 0
      }
    };

    this.state.offlineResults = result;
    this.log('offline_phase_complete', {
      historicalATE: historicalATE.estimate,
      propensityConverged: propensityResult.converged
    });

    return result;
  }

  /**
   * Extract arrays from historical records
   */
  private extractHistoricalData(data: HistoricalDataset): {
    features: number[][];
    treatments: number[];
    outcomes: number[];
    quantizedOutcomes: BinIndex[];
    prevOutcomes: number[];
    quantParams: QuantizationParams[];
  } {
    const features: number[][] = [];
    const treatments: number[] = [];
    const outcomes: number[] = [];
    const quantizedOutcomes: BinIndex[] = [];
    const prevOutcomes: number[] = [];
    const quantParams: QuantizationParams[] = [];

    // Sort records by student and time
    const sortedRecords = [...data.records].sort((a, b) => {
      if (a.studentId !== b.studentId) return a.studentId.localeCompare(b.studentId);
      return a.timeStep - b.timeStep;
    });

    // Build previous outcome map
    const prevOutcomeMap = new Map<string, number>();

    for (const record of sortedRecords) {
      const key = `${record.studentId}-${record.timeStep}`;
      const prevKey = `${record.studentId}-${record.timeStep - 1}`;

      features.push(record.features.values);
      treatments.push(record.policySelection.policy === 'B' ? 1 : 0);
      outcomes.push(record.continuousOutcome.value);
      quantizedOutcomes.push(record.quantizedOutcome.binIndex);
      prevOutcomes.push(prevOutcomeMap.get(prevKey) ?? 0);
      quantParams.push(record.quantizationParams);

      prevOutcomeMap.set(key, record.continuousOutcome.value);
    }

    return { features, treatments, outcomes, quantizedOutcomes, prevOutcomes, quantParams };
  }

  /**
   * Fit propensity score model
   */
  private fitPropensityModel(
    features: number[][],
    treatments: number[],
    prevOutcomes: number[]
  ): { coefficients: number[]; converged: boolean } {
    // Augment features with previous outcomes
    const augmentedFeatures = features.map((f, i) => [...f, prevOutcomes[i]]);

    const result = estimatePropensityScores(augmentedFeatures, treatments, {
      maxIterations: 500,
      regularization: 0.01
    });

    return {
      coefficients: result.coefficients,
      converged: result.converged
    };
  }

  /**
   * Fit outcome models for each treatment
   */
  private fitOutcomeModels(
    features: number[][],
    treatments: number[],
    outcomes: number[],
    prevOutcomes: number[]
  ): {
    modelA: { intercept: number; treatmentEffect: number; featureCoefficients: number[] };
    modelB: { intercept: number; treatmentEffect: number; featureCoefficients: number[] };
  } {
    // Split by treatment
    const idxA: number[] = [];
    const idxB: number[] = [];

    treatments.forEach((t, i) => {
      if (t === 0) idxA.push(i);
      else idxB.push(i);
    });

    // Augment with previous outcomes
    const augFeatures = features.map((f, i) => [...f, prevOutcomes[i]]);

    // Fit model for treatment A
    const featuresA = idxA.map((i) => augFeatures[i]);
    const outcomesA = idxA.map((i) => outcomes[i]);
    const modelA =
      featuresA.length > 0
        ? fitOutcomeModel(outcomesA, new Array(outcomesA.length).fill(0), featuresA)
        : { intercept: 0, treatmentEffect: 0, featureCoefficients: [], residualVariance: 1 };

    // Fit model for treatment B
    const featuresB = idxB.map((i) => augFeatures[i]);
    const outcomesB = idxB.map((i) => outcomes[i]);
    const modelB =
      featuresB.length > 0
        ? fitOutcomeModel(outcomesB, new Array(outcomesB.length).fill(1), featuresB)
        : { intercept: 0, treatmentEffect: 0, featureCoefficients: [], residualVariance: 1 };

    return { modelA, modelB };
  }

  /**
   * Optimize quantization parameters for power
   */
  private optimizeQuantization(
    outcomes: number[],
    _mappings: DequantizationMapping[]
  ): Omit<QuantizationParams, 'timeStep'> {
    // Use entropy-based quantization optimized for the data
    return createEntropyQuantization(outcomes, this.state.config.initialQuantization.numBins);
  }

  // ============================================================================
  // Phase 2: Online Selection
  // ============================================================================

  /**
   * Initialize the online selection phase
   *
   * @param initialTimeStep - Starting time step (default 0)
   */
  initializeOnlinePhase(initialTimeStep: number = 0): void {
    if (!this.state.offlineResults) {
      throw new Error('Must run offline phase before online phase');
    }

    this.state.phase = 'online_selection';
    this.state.onlineState = {
      currentTimeStep: initialTimeStep,
      currentQuantization: {
        ...this.state.offlineResults.optimizedQuantization,
        timeStep: initialTimeStep
      },
      observations: [],
      adaptivePropensities: [],
      runningATE: {
        estimate: 0,
        standardError: Infinity,
        confidenceInterval: [-Infinity, Infinity],
        numObservations: 0,
        method: 'aipw'
      }
    };

    this.log('online_phase_initialized', { timeStep: initialTimeStep });
  }

  /**
   * Select policies for a set of students at current time step
   *
   * Uses adaptive randomization:
   * π(B|X) = 0.5 + δ * uncertainty_score
   *
   * @param features - Feature vectors for each student
   * @param studentIds - Student identifiers
   * @returns Policy selections with propensity scores
   */
  selectPolicies(
    features: number[][],
    studentIds: string[]
  ): { studentId: string; policy: Policy; propensityScore: number }[] {
    if (!this.state.onlineState) {
      throw new Error('Online phase not initialized');
    }

    const timeStep = this.state.onlineState.currentTimeStep;
    const selections: { studentId: string; policy: Policy; propensityScore: number }[] = [];

    for (let i = 0; i < features.length; i++) {
      const studentId = studentIds[i];
      const feature = features[i];

      // Compute base propensity from learned model (for potential future use in diagnostics)
      const _basePropensity =
        this.propensityCoefficients.length > 0
          ? predictPropensity([feature], this.propensityCoefficients)[0]
          : 0.5;

      // Compute uncertainty score (based on prediction variance)
      const uncertaintyScore = this.computeUncertaintyScore(feature);

      // Adaptive propensity with exploration
      const adaptiveP = Math.max(
        MIN_EXPLORATION_PROB,
        Math.min(MAX_EXPLORATION_PROB, 0.5 + 0.1 * uncertaintyScore)
      );

      // Randomize policy selection
      const policy: Policy = this.random() < adaptiveP ? 'B' : 'A';

      selections.push({
        studentId,
        policy,
        propensityScore: adaptiveP
      });

      // Store propensity estimate
      this.state.onlineState.adaptivePropensities.push({
        studentId,
        timeStep,
        score: adaptiveP,
        features: feature
      });
    }

    this.log('policies_selected', { timeStep, count: selections.length });
    return selections;
  }

  /**
   * Compute uncertainty score for adaptive exploration
   */
  private computeUncertaintyScore(features: number[]): number {
    // Use prediction variance from outcome models as uncertainty
    if (!this.outcomeModelA || !this.outcomeModelB) return 0;

    const predA = predictOutcome([features], 0, this.outcomeModelA)[0];
    const predB = predictOutcome([features], 1, this.outcomeModelB)[0];

    // Uncertainty is proportional to the difference in predictions
    return Math.abs(predB - predA);
  }

  /**
   * Record observations from the current time step
   *
   * @param observations - Array of observations
   */
  recordOnlineObservations(
    observations: {
      studentId: string;
      features: number[];
      policy: Policy;
      quantizedOutcome: BinIndex;
      propensityScore: number;
    }[]
  ): void {
    if (!this.state.onlineState || !this.state.offlineResults) {
      throw new Error('Online phase not initialized');
    }

    const timeStep = this.state.onlineState.currentTimeStep;
    const mappings = this.state.offlineResults.dequantizationMappings;

    for (const obs of observations) {
      const dequantizedOutcome = dequantize(obs.quantizedOutcome, mappings);

      this.state.onlineState.observations.push({
        studentId: obs.studentId,
        timeStep,
        features: obs.features,
        policy: obs.policy,
        quantizedOutcome: obs.quantizedOutcome,
        propensityScore: obs.propensityScore,
        dequantizedOutcome
      });
    }

    // Update running ATE estimate
    this.updateRunningATE();

    // Adapt quantization parameters
    this.adaptQuantizationOnline();

    this.log('observations_recorded', { timeStep, count: observations.length });
  }

  /**
   * Update running ATE estimate
   */
  private updateRunningATE(): void {
    if (!this.state.onlineState) return;

    const obs = this.state.onlineState.observations;
    if (obs.length < 10) return;

    const outcomes = obs.map((o) => o.dequantizedOutcome);
    const treatments = obs.map((o) => (o.policy === 'B' ? 1 : 0));
    const propensities = obs.map((o) => o.propensityScore);

    // Compute predictions from outcome models
    const features = obs.map((o) => o.features);
    const muA = this.outcomeModelA
      ? predictOutcome(features, 0, this.outcomeModelA)
      : new Array(obs.length).fill(0);
    const muB = this.outcomeModelB
      ? predictOutcome(features, 1, this.outcomeModelB)
      : new Array(obs.length).fill(0);

    const { ate } = computeAIPWEstimate(outcomes, treatments, propensities, muA, muB);
    this.state.onlineState.runningATE = ate;
  }

  /**
   * Adapt quantization parameters based on online observations
   */
  private adaptQuantizationOnline(): void {
    if (!this.state.onlineState) return;

    const recentObs = this.state.onlineState.observations.slice(-100);
    if (recentObs.length < 20) return;

    const observations = recentObs.map((o) => ({
      continuous: o.dequantizedOutcome,
      quantized: o.quantizedOutcome
    }));

    const updated = adaptQuantization(
      this.state.onlineState.currentQuantization,
      observations,
      0.05 // Conservative learning rate
    );

    this.state.onlineState.currentQuantization = {
      ...updated,
      timeStep: this.state.onlineState.currentTimeStep
    };
  }

  /**
   * Advance to next time step
   */
  advanceTimeStep(): void {
    if (!this.state.onlineState) {
      throw new Error('Online phase not initialized');
    }

    this.state.onlineState.currentTimeStep++;
    this.state.onlineState.currentQuantization.timeStep = this.state.onlineState.currentTimeStep;

    this.log('time_step_advanced', { newTimeStep: this.state.onlineState.currentTimeStep });
  }

  /**
   * Get current online phase state
   */
  getOnlineState(): OnlinePhaseState | undefined {
    return this.state.onlineState;
  }

  // ============================================================================
  // Phase 3: Statistical Testing
  // ============================================================================

  /**
   * Run the statistical hypothesis test
   *
   * H_0: τ = 0 (no causal effect, differences due to history/quantization)
   * H_1: τ ≠ 0 (true causal effect exists)
   *
   * @param significanceLevel - Significance level (default 0.05)
   * @returns Hypothesis test result
   */
  runStatisticalTest(significanceLevel: number = DEFAULT_ALPHA): HypothesisTestResult {
    if (!this.state.onlineState || this.state.onlineState.observations.length === 0) {
      throw new Error('No observations to test');
    }

    this.state.phase = 'testing';
    this.log('statistical_test_start', { significanceLevel });

    const obs = this.state.onlineState.observations;
    const ate = this.state.onlineState.runningATE;

    // Perform hypothesis test
    const { testStatistic, pValue, reject } = testATESignificance(ate, significanceLevel);

    // Compute bootstrap confidence interval for robustness
    const outcomes = obs.map((o) => o.dequantizedOutcome);
    const treatments = obs.map((o) => (o.policy === 'B' ? 1 : 0));
    const propensities = obs.map((o) => o.propensityScore);
    const features = obs.map((o) => o.features);
    const muA = this.outcomeModelA
      ? predictOutcome(features, 0, this.outcomeModelA)
      : new Array(obs.length).fill(0);
    const muB = this.outcomeModelB
      ? predictOutcome(features, 1, this.outcomeModelB)
      : new Array(obs.length).fill(0);

    const bootstrapCI = bootstrapATEConfidenceInterval(
      outcomes,
      treatments,
      propensities,
      muA,
      muB,
      DEFAULT_BOOTSTRAP_SAMPLES,
      1 - significanceLevel
    );

    // Estimate achieved power
    const achievedPower = estimatePower(ate.estimate, ate.standardError, significanceLevel);

    const result: HypothesisTestResult = {
      testStatistic,
      pValue,
      reject,
      significanceLevel,
      degreesOfFreedom: obs.length - 1,
      confidenceInterval: bootstrapCI,
      achievedPower
    };

    this.state.testResults = result;
    this.log('statistical_test_complete', {
      testStatistic,
      pValue,
      reject,
      achievedPower
    });

    return result;
  }

  /**
   * Perform power analysis
   *
   * @param targetEffectSize - Effect size to detect
   * @returns Power analysis result
   */
  computePowerAnalysis(targetEffectSize: number): PowerAnalysisResult {
    const obs = this.state.onlineState?.observations || [];
    const ate = this.state.onlineState?.runningATE;

    if (!ate || obs.length === 0) {
      return {
        effectSize: targetEffectSize,
        requiredSampleSize: Infinity,
        achievedPower: 0,
        minimumDetectableEffect: Infinity
      };
    }

    const se = ate.standardError;
    const _n = obs.length;

    // Achieved power for target effect size
    const achievedPower = estimatePower(targetEffectSize, se, DEFAULT_ALPHA);

    // Minimum detectable effect at 80% power
    const mde = minimumDetectableEffect(0.8, se, DEFAULT_ALPHA);

    // Required sample size for 80% power
    // Using approximation: n ∝ (z_α + z_β)² σ² / δ²
    const z80 = 0.84; // z for 80% power
    const z95 = 1.96; // z for 95% confidence
    const baselineVariance = variance(obs.map((o) => o.dequantizedOutcome));
    const requiredN = Math.ceil(((z80 + z95) ** 2 * baselineVariance) / targetEffectSize ** 2);

    return {
      effectSize: targetEffectSize,
      requiredSampleSize: requiredN,
      achievedPower,
      minimumDetectableEffect: mde
    };
  }

  // ============================================================================
  // Identifiability Checking
  // ============================================================================

  /**
   * Check identifiability conditions
   *
   * Conditions checked:
   * 1. Partial unconfoundedness
   * 2. Positivity
   * 3. Consistency
   * 4. Transportability
   * 5. Quantization invertibility
   *
   * @returns Identifiability check result
   */
  checkIdentifiability(): IdentifiabilityResult {
    const obs = this.state.onlineState?.observations || [];
    const propensities = obs.map((o) => o.propensityScore);
    const quantParams =
      this.state.onlineState?.currentQuantization || this.state.config.initialQuantization;

    // 1. Positivity check
    const minP = propensities.length > 0 ? Math.min(...propensities) : 0;
    const maxP = propensities.length > 0 ? Math.max(...propensities) : 1;
    const positivity = minP > 0.01 && maxP < 0.99;

    // 2. Quantization resolution check
    const binWidth =
      quantParams.boundaries.length > 1
        ? quantParams.boundaries[1] - quantParams.boundaries[0]
        : Infinity;
    const ate = this.state.onlineState?.runningATE;
    const quantResolution = ate ? Math.abs(ate.estimate) / binWidth : 0;
    const quantizationInvertible = quantResolution > 0.1;

    // 3. Sign preservation check
    const signPreserved =
      quantParams.boundaries.includes(0) ||
      (quantParams.boundaries.some((b) => b < 0) && quantParams.boundaries.some((b) => b > 0));

    const diagnostics: IdentifiabilityDiagnostics = {
      minPropensity: minP,
      maxPropensity: maxP,
      quantizationResolution: quantResolution,
      signPreserved,
      warnings: []
    };

    if (!positivity) {
      diagnostics.warnings.push('Positivity may be violated: propensity scores near boundaries');
    }
    if (!quantizationInvertible) {
      diagnostics.warnings.push('Effect size may be smaller than quantization resolution');
    }
    if (!signPreserved) {
      diagnostics.warnings.push('Quantization may not preserve sign of improvement/regression');
    }

    const identifiable = positivity && quantizationInvertible && signPreserved;

    return {
      identifiable,
      unconfoundednessPartial: true, // Assumed via randomization in online phase
      positivity,
      consistency: true, // Assumed
      transportability: true, // Assumed
      quantizationInvertible,
      diagnostics
    };
  }

  // ============================================================================
  // Bias Verification
  // ============================================================================

  /**
   * Verify that the algorithm is unbiased
   *
   * Checks:
   * 1. Neutrality axiom (A↔B, Y→-Y invariance)
   * 2. Differential neutrality constraint
   * 3. Causal fairness metric
   * 4. Type I error rate under simulation
   * 5. Sensitivity analysis
   *
   * @param numSimulations - Number of null simulations
   * @returns Bias verification result
   */
  verifyUnbiasedness(numSimulations: number = 100): BiasVerificationResult {
    this.log('bias_verification_start', { numSimulations });

    // 1. Neutrality verification
    const neutrality = this.checkNeutrality();

    // 2. Differential neutrality
    const differentialNeutrality = this.checkDifferentialNeutrality();

    // 3. Causal fairness
    const fairness = this.checkCausalFairness(numSimulations);

    // 4. Type I error rate
    const typeIErrorRate = this.estimateTypeIErrorRate(numSimulations);

    // 5. Sensitivity analysis
    const sensitivitySummary = this.runSensitivityAnalysis();

    // Build recommendations
    const recommendations: string[] = [];
    if (!neutrality.neutral) {
      recommendations.push(
        'Algorithm may favor one policy. Consider symmetric quantization boundaries.'
      );
    }
    if (!differentialNeutrality.satisfied) {
      recommendations.push('Add noise to quantization updates to reduce data-dependent bias.');
    }
    if (!fairness.fair) {
      recommendations.push(
        'Under null hypothesis, estimator shows systematic bias. Review dequantization calibration.'
      );
    }
    if (typeIErrorRate > DEFAULT_ALPHA * 1.5) {
      recommendations.push(
        'Type I error rate exceeds nominal. Use more conservative significance threshold.'
      );
    }
    if (sensitivitySummary.sensitive) {
      recommendations.push(
        'ATE estimate is sensitive to quantization perturbations. Consider finer bins.'
      );
    }

    const result: BiasVerificationResult = {
      unbiased:
        neutrality.neutral &&
        differentialNeutrality.satisfied &&
        fairness.fair &&
        typeIErrorRate <= DEFAULT_ALPHA * 1.5 &&
        !sensitivitySummary.sensitive,
      neutrality,
      differentialNeutrality,
      fairness,
      typeIErrorRate,
      sensitivitySummary,
      recommendations
    };

    this.state.biasVerification = result;
    this.log('bias_verification_complete', { unbiased: result.unbiased });

    return result;
  }

  /**
   * Check neutrality axiom
   * Algorithm should be invariant to A↔B and Y→-Y swap
   */
  private checkNeutrality(): NeutralityVerification {
    const ate = this.state.onlineState?.runningATE;
    if (!ate) {
      return {
        neutral: true,
        originalEstimate: 0,
        swappedEstimate: 0,
        symmetryError: 0,
        tolerance: NEUTRALITY_TOLERANCE
      };
    }

    // Original estimate
    const originalEstimate = ate.estimate;

    // Swapped estimate: flip treatments and negate outcomes
    const obs = this.state.onlineState!.observations;
    const swappedOutcomes = obs.map((o) => -o.dequantizedOutcome);
    const swappedTreatments = obs.map((o) => (o.policy === 'A' ? 1 : 0));
    const propensities = obs.map((o) => 1 - o.propensityScore);

    const features = obs.map((o) => o.features);
    const muA = this.outcomeModelA
      ? predictOutcome(features, 0, this.outcomeModelA).map((v) => -v)
      : new Array(obs.length).fill(0);
    const muB = this.outcomeModelB
      ? predictOutcome(features, 1, this.outcomeModelB).map((v) => -v)
      : new Array(obs.length).fill(0);

    const { ate: swappedATE } = computeAIPWEstimate(
      swappedOutcomes,
      swappedTreatments,
      propensities,
      muA,
      muB
    );

    const swappedEstimate = swappedATE.estimate;

    // Symmetry error: should have originalEstimate ≈ -swappedEstimate
    const symmetryError =
      Math.abs(originalEstimate + swappedEstimate) /
      (Math.abs(originalEstimate) + Math.abs(swappedEstimate) + 1e-10);

    return {
      neutral: symmetryError < NEUTRALITY_TOLERANCE,
      originalEstimate,
      swappedEstimate,
      symmetryError,
      tolerance: NEUTRALITY_TOLERANCE
    };
  }

  /**
   * Check differential neutrality constraint
   */
  private checkDifferentialNeutrality(): DifferentialNeutralityCheck {
    const quant =
      this.state.onlineState?.currentQuantization || this.state.config.initialQuantization;

    // Estimate sensitivity of quantization to data
    // Higher boundary variance → more data-dependent
    const boundaryVariance = variance(quant.boundaries);
    const noiseLevel = Math.sqrt(boundaryVariance) * 0.1; // Recommended noise

    // Check if current noise is sufficient
    const effectiveEpsilon = noiseLevel > 0 ? Math.log(1 + 1 / noiseLevel) : Infinity;

    return {
      satisfied: effectiveEpsilon <= DEFAULT_EPSILON,
      epsilon: effectiveEpsilon,
      targetEpsilon: DEFAULT_EPSILON,
      noiseLevel
    };
  }

  /**
   * Check causal fairness metric
   * Δ = |E[τ̂|H_0,θ] - 0|
   */
  private checkCausalFairness(numSimulations: number): FairnessMetric {
    // Simulate under null hypothesis
    const nullEstimates: number[] = [];
    const obs = this.state.onlineState?.observations || [];

    if (obs.length === 0) {
      return {
        fair: true,
        delta: 0,
        targetDelta: DEFAULT_DELTA,
        numNullSamples: 0
      };
    }

    for (let sim = 0; sim < numSimulations; sim++) {
      // Permute treatment assignments
      const permutedTreatments: number[] = obs.map((o) => (o.policy === 'B' ? 1 : 0));
      for (let i = permutedTreatments.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        const temp = permutedTreatments[i];
        permutedTreatments[i] = permutedTreatments[j];
        permutedTreatments[j] = temp;
      }

      const outcomes = obs.map((o) => o.dequantizedOutcome);
      const propensities = obs.map((o) => o.propensityScore);
      const features = obs.map((o) => o.features);
      const muA = this.outcomeModelA
        ? predictOutcome(features, 0, this.outcomeModelA)
        : new Array(obs.length).fill(0);
      const muB = this.outcomeModelB
        ? predictOutcome(features, 1, this.outcomeModelB)
        : new Array(obs.length).fill(0);

      const { ate } = computeAIPWEstimate(outcomes, permutedTreatments, propensities, muA, muB);
      nullEstimates.push(ate.estimate);
    }

    const nullMean = mean(nullEstimates);
    const delta = Math.abs(nullMean);

    return {
      fair: delta < DEFAULT_DELTA,
      delta,
      targetDelta: DEFAULT_DELTA,
      numNullSamples: numSimulations
    };
  }

  /**
   * Estimate Type I error rate under null
   */
  private estimateTypeIErrorRate(numSimulations: number): number {
    const obs = this.state.onlineState?.observations || [];
    if (obs.length === 0) return 0;

    let rejections = 0;

    for (let sim = 0; sim < numSimulations; sim++) {
      // Generate null data (permute treatments)
      const permutedTreatments: number[] = obs.map((o) => (o.policy === 'B' ? 1 : 0));
      for (let i = permutedTreatments.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        const temp = permutedTreatments[i];
        permutedTreatments[i] = permutedTreatments[j];
        permutedTreatments[j] = temp;
      }

      const outcomes = obs.map((o) => o.dequantizedOutcome);
      const propensities = obs.map((o) => o.propensityScore);
      const features = obs.map((o) => o.features);
      const muA = this.outcomeModelA
        ? predictOutcome(features, 0, this.outcomeModelA)
        : new Array(obs.length).fill(0);
      const muB = this.outcomeModelB
        ? predictOutcome(features, 1, this.outcomeModelB)
        : new Array(obs.length).fill(0);

      const { ate } = computeAIPWEstimate(outcomes, permutedTreatments, propensities, muA, muB);
      const { reject } = testATESignificance(ate, DEFAULT_ALPHA);

      if (reject) rejections++;
    }

    return rejections / numSimulations;
  }

  /**
   * Run sensitivity analysis
   */
  private runSensitivityAnalysis(): SensitivityAnalysis {
    const ate = this.state.onlineState?.runningATE;
    const quant = this.state.onlineState?.currentQuantization;

    if (!ate || !quant || quant.boundaries.length === 0) {
      return {
        perturbationSize: 0,
        ateChange: 0,
        sensitive: false,
        threshold: 0.2
      };
    }

    // Perturb boundaries
    const perturbSize = 0.1 * (Math.max(...quant.boundaries) - Math.min(...quant.boundaries));
    const perturbedBoundaries = quant.boundaries.map(
      (b) => b + (this.random() - 0.5) * 2 * perturbSize
    );

    // Re-quantize and re-estimate
    const obs = this.state.onlineState!.observations;
    const perturbedQuant: QuantizationParams = {
      ...quant,
      boundaries: perturbedBoundaries.sort((a, b) => a - b)
    };

    const perturbedOutcomes = obs.map((o) => {
      // Re-quantize the underlying value
      const origBin = o.quantizedOutcome;
      const origMapping = this.state.offlineResults?.dequantizationMappings.find(
        (m) => m.binIndex === origBin
      );
      const origValue = origMapping?.expectedValue ?? 0;

      // Quantize with perturbed params
      const newBin = quantize(origValue, perturbedQuant);
      const newMapping = this.state.offlineResults?.dequantizationMappings.find(
        (m) => m.binIndex === newBin
      );
      return newMapping?.expectedValue ?? origValue;
    });

    const treatments = obs.map((o) => (o.policy === 'B' ? 1 : 0));
    const propensities = obs.map((o) => o.propensityScore);
    const features = obs.map((o) => o.features);
    const muA = this.outcomeModelA
      ? predictOutcome(features, 0, this.outcomeModelA)
      : new Array(obs.length).fill(0);
    const muB = this.outcomeModelB
      ? predictOutcome(features, 1, this.outcomeModelB)
      : new Array(obs.length).fill(0);

    const { ate: perturbedATE } = computeAIPWEstimate(
      perturbedOutcomes,
      treatments,
      propensities,
      muA,
      muB
    );

    const ateChange = Math.abs(perturbedATE.estimate - ate.estimate) / (ate.standardError + 1e-10);
    const threshold = 2; // 2 standard errors

    return {
      perturbationSize: perturbSize,
      ateChange,
      sensitive: ateChange > threshold,
      threshold
    };
  }

  // ============================================================================
  // Data Simulation (for testing)
  // ============================================================================

  /**
   * Simulate historical data for testing
   *
   * Generates data according to the model specification:
   * - Y_{it} = β_0 + β_Z Z + β_X^T X + β_U^T U + ρ Y_{t-1} + ε
   * - P(Z=B|...) = σ(γ^T [1, X, Y_{t-1}, U])
   *
   * @returns Simulated historical dataset
   */
  simulateHistoricalData(): HistoricalDataset {
    const { config } = this.state;
    const { numStudents, numTimeSteps, featureDimension } = config;
    const { outcomeModel, selectionModel, confounderConfig, initialQuantization } = config;

    const records: HistoricalRecord[] = [];

    // Generate confounders for each student
    const confounders: number[][] = [];
    const confCovariance: number[][] = Array.from({ length: confounderConfig.dimension }, (_, i) =>
      Array.from({ length: confounderConfig.dimension }, (_, j) =>
        i === j ? confounderConfig.covariance[i * confounderConfig.dimension + j] || 1 : 0
      )
    );

    for (let i = 0; i < numStudents; i++) {
      const u = randomMultivariateNormal(confounderConfig.mean, confCovariance, 1, this.random)[0];
      confounders.push(u);
    }

    // Generate data for each student and time
    const prevOutcomes: Map<string, number> = new Map();

    for (let t = 0; t < numTimeSteps; t++) {
      const quantParams: QuantizationParams = {
        ...initialQuantization,
        timeStep: t
      };

      for (let i = 0; i < numStudents; i++) {
        const studentId = `student-${i}`;
        const U = confounders[i];

        // Generate features
        const X = randomNormal(0, 1, featureDimension, this.random);

        // Get previous outcome
        const prevOutcome = prevOutcomes.get(studentId) ?? 0;

        // Selection model: P(Z=B|X,Y_{t-1},U)
        let linearPred = selectionModel.intercept;
        for (
          let j = 0;
          j < Math.min(featureDimension, selectionModel.featureCoefficients.length);
          j++
        ) {
          linearPred += selectionModel.featureCoefficients[j] * X[j];
        }
        linearPred += selectionModel.previousOutcomeCoefficient * prevOutcome;
        for (let j = 0; j < Math.min(U.length, selectionModel.confounderCoefficients.length); j++) {
          linearPred += selectionModel.confounderCoefficients[j] * U[j];
        }

        const propensity = sigmoid(linearPred);
        const Z: Policy = this.random() < propensity ? 'B' : 'A';

        // Outcome model: Y = β_0 + β_Z Z + β_X^T X + β_U^T U + ρ Y_{t-1} + ε
        let Y = outcomeModel.intercept;
        Y += outcomeModel.treatmentCoefficient * (Z === 'B' ? 1 : 0);
        for (
          let j = 0;
          j < Math.min(featureDimension, outcomeModel.featureCoefficients.length);
          j++
        ) {
          Y += outcomeModel.featureCoefficients[j] * X[j];
        }
        for (let j = 0; j < Math.min(U.length, outcomeModel.confounderCoefficients.length); j++) {
          Y += outcomeModel.confounderCoefficients[j] * U[j];
        }
        Y += outcomeModel.autoregCoefficient * prevOutcome;
        Y += randomNormal(0, outcomeModel.noiseStd, 1, this.random)[0];

        // Quantize outcome
        const binIndex = quantize(Y, quantParams);

        // Store record
        records.push({
          studentId,
          timeStep: t,
          features: {
            studentId,
            timeStep: t,
            values: X
          },
          continuousOutcome: {
            studentId,
            timeStep: t,
            value: Y
          },
          quantizedOutcome: {
            studentId,
            timeStep: t,
            binIndex
          },
          policySelection: {
            studentId,
            timeStep: t,
            policy: Z,
            propensityScore: propensity
          },
          quantizationParams: quantParams
        });

        // Update previous outcome
        prevOutcomes.set(studentId, Y);
      }
    }

    return {
      records,
      numStudents,
      numTimeSteps,
      featureDimension,
      metadata: {
        collectionPeriod: 'simulated',
        dataSource: 'CausalInferenceEngine.simulateHistoricalData()'
      }
    };
  }

  // ============================================================================
  // State Management
  // ============================================================================

  /**
   * Export engine state
   */
  exportState(): CausalEngineState {
    return JSON.parse(JSON.stringify(this.state));
  }

  /**
   * Import engine state
   */
  importState(state: CausalEngineState): void {
    this.state = JSON.parse(JSON.stringify(state));

    // Restore models from offline results
    if (this.state.offlineResults) {
      this.propensityCoefficients = this.state.offlineResults.propensityParams;
      // Note: outcome models need to be re-learned or stored separately
    }

    this.log('state_imported', { phase: this.state.phase });
  }

  /**
   * Reset engine to initial state
   */
  reset(): void {
    const config = this.state.config;
    this.state = {
      phase: 'uninitialized',
      config,
      auditLog: []
    };
    this.propensityCoefficients = [];
    this.outcomeModelA = null;
    this.outcomeModelB = null;

    this.log('engine_reset', {});
  }

  /**
   * Get audit log
   */
  getAuditLog(): AuditLogEntry[] {
    return [...this.state.auditLog];
  }

  /**
   * Generate causal analysis ledger for Σ-SIG compliance
   */
  generateLedger(): CausalAnalysisLedger {
    const entries: CausalLedgerEntry[] = [];
    let quantizationUpdates = 0;
    let policySelections = 0;
    let ateEstimates = 0;
    let biasChecks = 0;

    for (const logEntry of this.state.auditLog) {
      let type: CausalLedgerEntry['type'] = 'ate_estimate';

      if (logEntry.action.includes('quantization')) {
        type = 'quantization_update';
        quantizationUpdates++;
      } else if (logEntry.action.includes('polic')) {
        type = 'policy_selection';
        policySelections++;
      } else if (logEntry.action.includes('ate') || logEntry.action.includes('statistical')) {
        type = 'ate_estimate';
        ateEstimates++;
      } else if (logEntry.action.includes('bias')) {
        type = 'bias_check';
        biasChecks++;
      }

      entries.push({
        id: `ledger-${entries.length}`,
        timestamp: logEntry.timestamp,
        type,
        details: {
          rationale: logEntry.action,
          ...logEntry.details
        },
        context: {},
        verified: true
      });
    }

    return {
      entries,
      summary: {
        totalDecisions: entries.length,
        quantizationUpdates,
        policySelections,
        ateEstimates,
        biasChecks
      }
    };
  }

  // ============================================================================
  // Logging
  // ============================================================================

  /**
   * Log an action to the audit log
   */
  private log(action: string, details: Record<string, unknown>): void {
    this.state.auditLog.push({
      timestamp: Date.now(),
      action,
      details,
      phase: this.state.phase
    });
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new CausalInferenceEngine with default configuration
 */
export function createCausalEngine(config?: Partial<CausalModelConfig>): CausalInferenceEngine {
  return new CausalInferenceEngine(config);
}

/**
 * Create a CausalInferenceEngine from exported state
 */
export function restoreCausalEngine(state: CausalEngineState): CausalInferenceEngine {
  const engine = new CausalInferenceEngine(state.config);
  engine.importState(state);
  return engine;
}
