/**
 * Statistical Analysis for Sampling Methods
 *
 * This module provides rigorous statistical tests and analysis for:
 * - Typical Sampling entropy distribution tests
 * - Mirostat v2 convergence analysis
 * - Perplexity consistency metrics
 * - Sampling quality diagnostics
 *
 * Mathematical Framework:
 * - Entropy-based typicality measures
 * - Markov chain convergence theory
 * - Statistical hypothesis testing
 * - Adaptive sampling theory
 *
 * References:
 * - Meister et al. (2020) "If Beam Search is the Answer, What was the Question?"
 * - Basu et al. (2021) "Mirostat: A Neural Text Decoding Algorithm"
 * - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration"
 */

import { kahanSum, neumaierSum, stableVariance } from './numerics';

/**
 * Entropy distribution test result
 */
export interface EntropyDistributionTest {
  /** P-value of the test */
  pValue: number;
  /** Whether the test passes at the significance level */
  passesTest: boolean;
  /** Test statistic value */
  testStatistic: number;
  /** Observed entropy statistics */
  observedEntropy: {
    mean: number;
    variance: number;
    skewness: number;
    kurtosis: number;
  };
  /** Expected entropy under null hypothesis */
  expectedEntropy: number;
  /** Degrees of freedom */
  degreesOfFreedom: number;
  /** Diagnostics and interpretation */
  interpretation: string;
}

/**
 * Compute entropy of a probability distribution
 * H(p) = -Σ p_i log(p_i)
 */
export function computeEntropy(probs: number[], epsilon = 1e-10): number {
  let entropy = 0;
  for (const p of probs) {
    if (p > epsilon) {
      entropy -= p * Math.log(p);
    }
  }
  return entropy;
}

/**
 * Compute cross-entropy H(p, q) = -Σ p_i log(q_i)
 */
export function computeCrossEntropy(
  trueProbs: number[],
  modelProbs: number[],
  epsilon = 1e-10
): number {
  if (trueProbs.length !== modelProbs.length) {
    throw new Error('Probability distributions must have same length');
  }

  let crossEntropy = 0;
  for (let i = 0; i < trueProbs.length; i++) {
    if (trueProbs[i] > epsilon) {
      crossEntropy -= trueProbs[i] * Math.log(Math.max(modelProbs[i], epsilon));
    }
  }
  return crossEntropy;
}

/**
 * Compute KL divergence D_KL(P || Q)
 */
export function computeKLDivergence(
  P: number[],
  Q: number[],
  epsilon = 1e-10
): number {
  if (P.length !== Q.length) {
    throw new Error('Distributions must have same length');
  }

  let kl = 0;
  for (let i = 0; i < P.length; i++) {
    if (P[i] > epsilon) {
      kl += P[i] * Math.log(P[i] / Math.max(Q[i], epsilon));
    }
  }
  return kl;
}

/**
 * Compute surprise (information content) of a token
 * I(token) = -log(p(token))
 */
export function computeSurprise(probability: number, epsilon = 1e-10): number {
  return -Math.log(Math.max(probability, epsilon));
}

/**
 * Chi-squared distribution CDF approximation
 * Using Wilson-Hilferty transformation
 */
function chiSquaredCDF(x: number, df: number): number {
  if (df <= 0) return 0;
  if (x <= 0) return 0;

  // Wilson-Hilferty approximation
  const term = Math.pow(x / df, 1 / 3);
  const z = (term - (1 - 2 / (9 * df))) / Math.sqrt(2 / (9 * df));

  // Standard normal CDF approximation
  return standardNormalCDF(z);
}

/**
 * Standard normal CDF approximation (Abramowitz & Stegun)
 */
function standardNormalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);

  return 0.5 * (1.0 + sign * y);
}

/**
 * Test whether sampled tokens follow expected entropy distribution.
 *
 * For typical sampling, tokens should have surprise close to the
 * model's expected entropy. This test checks if the observed
 * surprise distribution matches the theoretical prediction.
 *
 * Null hypothesis: Samples follow the typical set distribution
 *
 * @param samples - Array of (token, probability) pairs
 * @param significanceLevel - α for hypothesis test (default: 0.05)
 * @returns Test result with p-value and diagnostics
 */
export function entropyDistributionTest(
  samples: Array<{ token: number; probability: number }>,
  significanceLevel = 0.05
): EntropyDistributionTest {
  if (samples.length < 10) {
    return {
      pValue: 1,
      passesTest: true,
      testStatistic: 0,
      observedEntropy: { mean: 0, variance: 0, skewness: 0, kurtosis: 0 },
      expectedEntropy: 0,
      degreesOfFreedom: 0,
      interpretation: 'Insufficient samples for reliable test (n < 10)'
    };
  }

  // Compute surprise for each sample
  const surprises = samples.map(s => computeSurprise(s.probability));

  // Compute entropy statistics
  const stats = stableVariance(surprises);
  const n = surprises.length;

  // Compute skewness and kurtosis
  let m3 = 0, m4 = 0;
  for (const s of surprises) {
    const diff = s - stats.mean;
    m3 += diff * diff * diff;
    m4 += diff * diff * diff * diff;
  }
  m3 /= n;
  m4 /= n;

  const skewness = stats.standardDeviation > 0
    ? m3 / Math.pow(stats.standardDeviation, 3)
    : 0;
  const kurtosis = stats.standardDeviation > 0
    ? m4 / Math.pow(stats.standardDeviation, 4) - 3
    : 0;

  // Expected entropy: for typical sampling, E[surprise] ≈ H(p)
  // Under the typical set theorem, samples should concentrate around H
  const expectedEntropy = stats.mean; // Use sample mean as estimate

  // Chi-squared goodness of fit test
  // H0: surprises ~ Gamma(k, θ) where k*θ = H (entropy)
  // This is approximate; exact distribution depends on the model

  // Use Jarque-Bera test for normality of surprises
  // JB = n/6 * (S^2 + K^2/4)
  const jbStatistic = (n / 6) * (skewness * skewness + kurtosis * kurtosis / 4);

  // JB ~ χ²(2) under null
  const pValue = 1 - chiSquaredCDF(jbStatistic, 2);

  const passesTest = pValue > significanceLevel;

  let interpretation: string;
  if (passesTest) {
    interpretation = 'Surprise distribution is consistent with typical sampling. ' +
      `Mean surprise ${stats.mean.toFixed(3)} nats, std ${stats.standardDeviation.toFixed(3)}.`;
  } else {
    if (Math.abs(skewness) > 1) {
      interpretation = `Distribution is ${skewness > 0 ? 'right' : 'left'}-skewed. ` +
        'May indicate non-typical token selection.';
    } else if (Math.abs(kurtosis) > 2) {
      interpretation = `Distribution has ${kurtosis > 0 ? 'heavy' : 'light'} tails. ` +
        'May indicate inconsistent sampling behavior.';
    } else {
      interpretation = 'Surprise distribution deviates from expected typical set behavior.';
    }
  }

  return {
    pValue,
    passesTest,
    testStatistic: jbStatistic,
    observedEntropy: {
      mean: stats.mean,
      variance: stats.variance,
      skewness,
      kurtosis
    },
    expectedEntropy,
    degreesOfFreedom: 2,
    interpretation
  };
}

/**
 * Mirostat convergence analysis result
 */
export interface MirostatConvergenceAnalysis {
  /** Convergence rate (how fast μ adapts) */
  convergenceRate: number;
  /** Estimated steady-state μ value */
  steadyStateMu: number;
  /** 95% confidence interval for μ */
  muConfidenceInterval: [number, number];
  /** Perplexity consistency (coefficient of variation) */
  perplexityCV: number;
  /** Whether convergence is stable */
  stable: boolean;
  /** Estimated time constant (samples to reach 63% of final μ) */
  timeConstant: number;
  /** Mixing time (samples until stationary) */
  mixingTime: number;
  /** Recommendations */
  recommendations: string[];
}

/**
 * Analyze Mirostat v2 convergence properties.
 *
 * Mirostat maintains target perplexity by adapting μ (threshold):
 *   μ_{t+1} = μ_t + η * (surprise_t - τ)
 *
 * This analyzes the convergence behavior of the adaptive system.
 *
 * @param muHistory - History of μ values
 * @param surpriseHistory - History of observed surprises
 * @param targetEntropy - Target entropy τ
 * @param learningRate - Adaptation rate η
 * @returns Convergence analysis
 */
export function analyzeMirostatConvergence(
  muHistory: number[],
  surpriseHistory: number[],
  targetEntropy: number,
  learningRate: number
): MirostatConvergenceAnalysis {
  const recommendations: string[] = [];

  if (muHistory.length < 20) {
    return {
      convergenceRate: 0,
      steadyStateMu: muHistory[muHistory.length - 1] ?? targetEntropy,
      muConfidenceInterval: [0, 0],
      perplexityCV: 0,
      stable: false,
      timeConstant: Infinity,
      mixingTime: Infinity,
      recommendations: ['Need at least 20 samples for convergence analysis']
    };
  }

  const n = muHistory.length;

  // Estimate steady-state μ using last 25% of samples
  const steadyStateStart = Math.floor(n * 0.75);
  const steadyStateSamples = muHistory.slice(steadyStateStart);
  const muStats = stableVariance(steadyStateSamples);
  const steadyStateMu = muStats.mean;

  // Confidence interval for μ
  const seμ = muStats.standardDeviation / Math.sqrt(steadyStateSamples.length);
  const muConfidenceInterval: [number, number] = [
    steadyStateMu - 1.96 * seμ,
    steadyStateMu + 1.96 * seμ
  ];

  // Perplexity from surprises: PPL = exp(mean surprise)
  const surpriseStats = stableVariance(surpriseHistory);
  const perplexityCV = surpriseStats.standardDeviation / (surpriseStats.mean + 1e-10);

  // Analyze convergence rate
  // The system μ_{t+1} = μ_t + η(s_t - τ) is a random walk with drift
  // Convergence rate depends on η and variance of surprises

  // Estimate autocorrelation of μ to find mixing time
  const muDemeaned = muHistory.map(m => m - muStats.mean);
  let acf1 = 0;
  for (let i = 1; i < n; i++) {
    acf1 += muDemeaned[i] * muDemeaned[i - 1];
  }
  acf1 /= (n - 1) * muStats.variance;

  // Time constant: τ = -1/log(ρ) where ρ is lag-1 autocorrelation
  const timeConstant = acf1 > 0 && acf1 < 1
    ? -1 / Math.log(acf1)
    : 50; // Default if undefined

  // Mixing time: approximately 2-3 time constants
  const mixingTime = Math.ceil(3 * timeConstant);

  // Convergence rate: how fast deviations decay
  const convergenceRate = 1 - Math.abs(acf1);

  // Stability check
  const stable = perplexityCV < 0.5 && convergenceRate > 0.1;

  // Generate recommendations
  if (perplexityCV > 0.3) {
    recommendations.push('High perplexity variation. Consider reducing target entropy or learning rate.');
  }

  if (convergenceRate < 0.05) {
    recommendations.push('Slow convergence. Consider increasing learning rate η.');
  }

  if (muStats.standardDeviation > targetEntropy * 0.5) {
    recommendations.push('Large μ fluctuations. Model may be poorly calibrated.');
  }

  const avgSurprise = surpriseStats.mean;
  if (Math.abs(avgSurprise - targetEntropy) > targetEntropy * 0.1) {
    recommendations.push(`Observed entropy (${avgSurprise.toFixed(2)}) differs from target (${targetEntropy}).`);
  }

  if (recommendations.length === 0) {
    recommendations.push('Mirostat appears well-calibrated.');
  }

  return {
    convergenceRate,
    steadyStateMu,
    muConfidenceInterval,
    perplexityCV,
    stable,
    timeConstant,
    mixingTime,
    recommendations
  };
}

/**
 * Sampling quality metrics
 */
export interface SamplingQualityMetrics {
  /** Average entropy of sampled tokens */
  averageEntropy: number;
  /** Entropy of the empirical distribution */
  empiricalEntropy: number;
  /** Diversity score (unique tokens / total) */
  diversityScore: number;
  /** Repetition rate */
  repetitionRate: number;
  /** Self-BLEU (n-gram overlap) estimate */
  selfBleuEstimate: number;
  /** Coherence score (based on consecutive token similarity) */
  coherenceScore: number;
  /** Overall quality score (0-100) */
  overallScore: number;
  /** Quality grade */
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
}

/**
 * Analyze sampling quality from generated sequences.
 *
 * @param tokens - Generated token indices
 * @param probabilities - Probabilities of each token
 * @returns Quality metrics
 */
export function analyzeSamplingQuality(
  tokens: number[],
  probabilities: number[]
): SamplingQualityMetrics {
  if (tokens.length === 0) {
    return {
      averageEntropy: 0,
      empiricalEntropy: 0,
      diversityScore: 0,
      repetitionRate: 0,
      selfBleuEstimate: 0,
      coherenceScore: 0,
      overallScore: 0,
      grade: 'F'
    };
  }

  // Average entropy from token probabilities
  const surprises = probabilities.map(p => computeSurprise(p));
  const averageEntropy = kahanSum(surprises) / surprises.length;

  // Empirical entropy (entropy of token frequency distribution)
  const tokenCounts = new Map<number, number>();
  for (const token of tokens) {
    tokenCounts.set(token, (tokenCounts.get(token) ?? 0) + 1);
  }

  const n = tokens.length;
  let empiricalEntropy = 0;
  for (const count of tokenCounts.values()) {
    const p = count / n;
    empiricalEntropy -= p * Math.log(p);
  }

  // Diversity: unique tokens / total tokens
  const diversityScore = tokenCounts.size / n;

  // Repetition rate: consecutive repeats
  let repeatCount = 0;
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === tokens[i - 1]) {
      repeatCount++;
    }
  }
  const repetitionRate = repeatCount / (tokens.length - 1);

  // Self-BLEU estimate (bigram overlap)
  const bigrams = new Set<string>();
  let bigramRepeat = 0;
  for (let i = 1; i < tokens.length; i++) {
    const bigram = `${tokens[i - 1]}_${tokens[i]}`;
    if (bigrams.has(bigram)) {
      bigramRepeat++;
    }
    bigrams.add(bigram);
  }
  const selfBleuEstimate = 1 - bigramRepeat / (tokens.length - 1);

  // Coherence: how smooth is the probability sequence
  let probVariance = 0;
  for (let i = 1; i < probabilities.length; i++) {
    const diff = probabilities[i] - probabilities[i - 1];
    probVariance += diff * diff;
  }
  probVariance /= (probabilities.length - 1);
  const coherenceScore = 1 / (1 + probVariance * 10);

  // Overall score (weighted average)
  const overallScore = Math.min(100, Math.max(0,
    25 * (1 - repetitionRate) +          // Low repetition
    25 * diversityScore +                 // High diversity
    25 * selfBleuEstimate +               // Low self-BLEU
    25 * coherenceScore                   // High coherence
  ) * 100);

  // Grade based on overall score
  let grade: 'A' | 'B' | 'C' | 'D' | 'F';
  if (overallScore >= 90) grade = 'A';
  else if (overallScore >= 80) grade = 'B';
  else if (overallScore >= 70) grade = 'C';
  else if (overallScore >= 60) grade = 'D';
  else grade = 'F';

  return {
    averageEntropy,
    empiricalEntropy,
    diversityScore,
    repetitionRate,
    selfBleuEstimate,
    coherenceScore,
    overallScore,
    grade
  };
}

/**
 * Temperature calibration analysis
 */
export interface TemperatureCalibration {
  /** Optimal temperature for target entropy */
  optimalTemperature: number;
  /** Current vs target entropy deviation */
  entropyDeviation: number;
  /** Temperature sensitivity (d(entropy)/d(temp)) */
  sensitivity: number;
  /** Calibration recommendation */
  recommendation: string;
}

/**
 * Analyze temperature calibration for a model.
 *
 * @param logits - Sample of logit vectors
 * @param currentTemperature - Current temperature setting
 * @param targetEntropy - Desired output entropy
 * @returns Calibration analysis
 */
export function analyzeTemperatureCalibration(
  logits: number[][],
  currentTemperature: number,
  targetEntropy: number
): TemperatureCalibration {
  if (logits.length === 0) {
    return {
      optimalTemperature: 1.0,
      entropyDeviation: 0,
      sensitivity: 0,
      recommendation: 'No logits provided'
    };
  }

  // Compute entropy at different temperatures
  const temperatures = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0];
  const entropies: number[] = [];

  for (const T of temperatures) {
    let totalEntropy = 0;
    for (const logit of logits) {
      // Softmax with temperature
      const maxLogit = Math.max(...logit);
      const scaledLogits = logit.map(l => (l - maxLogit) / T);
      const expLogits = scaledLogits.map(l => Math.exp(l));
      const sumExp = kahanSum(expLogits);
      const probs = expLogits.map(e => e / sumExp);

      totalEntropy += computeEntropy(probs);
    }
    entropies.push(totalEntropy / logits.length);
  }

  // Find temperature closest to target entropy
  let bestIdx = 0;
  let minDiff = Infinity;
  for (let i = 0; i < temperatures.length; i++) {
    const diff = Math.abs(entropies[i] - targetEntropy);
    if (diff < minDiff) {
      minDiff = diff;
      bestIdx = i;
    }
  }

  // Interpolate for optimal temperature
  let optimalTemperature = temperatures[bestIdx];

  // Estimate sensitivity using finite difference
  const idx1 = Math.max(0, bestIdx - 1);
  const idx2 = Math.min(temperatures.length - 1, bestIdx + 1);
  const sensitivity = (entropies[idx2] - entropies[idx1]) /
    (temperatures[idx2] - temperatures[idx1]);

  // Current entropy
  const currentIdx = temperatures.indexOf(currentTemperature) ?? 2; // Default to T=1
  const currentEntropy = entropies[currentIdx] ?? entropies[2];
  const entropyDeviation = currentEntropy - targetEntropy;

  // Recommendation
  let recommendation: string;
  if (Math.abs(entropyDeviation) < 0.1) {
    recommendation = 'Temperature is well-calibrated.';
  } else if (entropyDeviation > 0) {
    recommendation = `Entropy too high (${currentEntropy.toFixed(2)} > ${targetEntropy}). ` +
      `Consider lowering temperature to ${(currentTemperature * 0.8).toFixed(2)}.`;
  } else {
    recommendation = `Entropy too low (${currentEntropy.toFixed(2)} < ${targetEntropy}). ` +
      `Consider raising temperature to ${(currentTemperature * 1.2).toFixed(2)}.`;
  }

  return {
    optimalTemperature,
    entropyDeviation,
    sensitivity,
    recommendation
  };
}

/**
 * Nucleus (top-p) threshold analysis
 */
export interface NucleusThresholdAnalysis {
  /** Recommended p value for target diversity */
  recommendedP: number;
  /** Average nucleus size at different p values */
  nucleusSizes: Map<number, number>;
  /** Effective vocabulary fraction */
  effectiveVocabFraction: number;
  /** Truncation entropy loss */
  entropyLoss: number;
}

/**
 * Analyze optimal nucleus threshold for top-p sampling.
 *
 * @param logits - Sample of logit vectors
 * @param targetDiversity - Target diversity score
 * @returns Threshold analysis
 */
export function analyzeNucleusThreshold(
  logits: number[][],
  targetDiversity = 0.3
): NucleusThresholdAnalysis {
  const pValues = [0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 0.99];
  const nucleusSizes = new Map<number, number>();

  for (const p of pValues) {
    let totalSize = 0;

    for (const logit of logits) {
      // Softmax
      const maxLogit = Math.max(...logit);
      const expLogits = logit.map(l => Math.exp(l - maxLogit));
      const sumExp = kahanSum(expLogits);
      const probs = expLogits.map(e => e / sumExp);

      // Sort probabilities descending
      const sorted = [...probs].sort((a, b) => b - a);

      // Find nucleus size
      let cumProb = 0;
      let size = 0;
      for (const prob of sorted) {
        cumProb += prob;
        size++;
        if (cumProb >= p) break;
      }

      totalSize += size;
    }

    nucleusSizes.set(p, totalSize / logits.length);
  }

  // Find p that gives target diversity
  let recommendedP = 0.9;
  for (const [p, size] of nucleusSizes) {
    const vocabSize = logits[0]?.length ?? 100;
    if (size / vocabSize >= targetDiversity) {
      recommendedP = p;
      break;
    }
  }

  // Compute effective vocab fraction
  const avgNucleusSize = nucleusSizes.get(recommendedP) ?? 10;
  const vocabSize = logits[0]?.length ?? 100;
  const effectiveVocabFraction = avgNucleusSize / vocabSize;

  // Estimate entropy loss from truncation
  let totalEntropy = 0;
  let truncatedEntropy = 0;

  for (const logit of logits) {
    const maxLogit = Math.max(...logit);
    const expLogits = logit.map(l => Math.exp(l - maxLogit));
    const sumExp = kahanSum(expLogits);
    const probs = expLogits.map(e => e / sumExp);

    totalEntropy += computeEntropy(probs);

    // Truncated entropy
    const sorted = [...probs].sort((a, b) => b - a);
    let cumProb = 0;
    const nucleus: number[] = [];
    for (const prob of sorted) {
      cumProb += prob;
      nucleus.push(prob);
      if (cumProb >= recommendedP) break;
    }

    // Renormalize
    const nucleusSum = kahanSum(nucleus);
    const nucleusNorm = nucleus.map(p => p / nucleusSum);
    truncatedEntropy += computeEntropy(nucleusNorm);
  }

  totalEntropy /= logits.length;
  truncatedEntropy /= logits.length;
  const entropyLoss = totalEntropy - truncatedEntropy;

  return {
    recommendedP,
    nucleusSizes,
    effectiveVocabFraction,
    entropyLoss
  };
}
