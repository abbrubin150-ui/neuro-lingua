import { clip, logSumExp, stableSoftmax } from '../lib/MathUtils';

export type SamplingRng = () => number;

export interface SamplingOptions {
  temperature?: number;
  topK?: number;
  topP?: number;
  minProbability?: number;
  frequencyPenalty?: number; // Penalize tokens based on their frequency (0.0-2.0, default 0)
  presencePenalty?: number; // Penalize tokens that have appeared at all (0.0-2.0, default 0)
  generatedTokens?: number[]; // Previously generated tokens for repetition penalty
  rng?: SamplingRng;
}

export interface MirostatV2State {
  mu: number;
}

export interface MirostatV2Options extends SamplingOptions {
  targetEntropy?: number; // tau in the paper (desired surprise)
  learningRate?: number; // eta update factor
  initialMu?: number; // explicit initial mu (defaults to 2 * tau)
  state?: MirostatV2State;
}

export interface BeamSearchOptions {
  beamWidth: number;
  maxLength: number;
  eosToken?: number;
  temperature?: number;
  lengthPenalty?: number;
  rng?: SamplingRng;
}

export interface BeamSearchState {
  tokens: number[];
  logProbability: number;
  completed: boolean;
}

export interface BeamSearchResult {
  tokens: number[];
  score: number;
  probability: number;
}

export interface ContrastiveSearchOptions {
  topK: number;
  alpha: number;
  maxLength: number;
  eosToken?: number;
  penaltyAlpha?: number;
  rng?: SamplingRng;
}

export interface ContrastiveSearchResult {
  tokens: number[];
  score: number;
}

export type LogitsFn = (prefix: number[]) => number[];
export type EmbeddingFn = (tokenIds: number[]) => number[];

function defaultRng(): number {
  return Math.random();
}

function argMax(values: number[]): number {
  if (values.length === 0) return -1;
  let maxIndex = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[maxIndex]) {
      maxIndex = i;
    }
  }
  return maxIndex;
}

function renormalize(probs: number[]): number[] {
  const sum = probs.reduce((acc, value) => acc + value, 0);
  if (!isFinite(sum) || sum <= 0) {
    const fallback = new Array(probs.length).fill(0);
    const idx = argMax(probs);
    if (idx >= 0) fallback[idx] = 1;
    return fallback;
  }
  return probs.map((value) => value / sum);
}

function applyTopK(probs: number[], k: number): number[] {
  if (k <= 0 || k >= probs.length) {
    return [...probs];
  }
  const indices = probs.map((_, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  const threshold = probs[indices[Math.max(0, k - 1)]];
  const filtered = probs.map((value) => (value >= threshold ? value : 0));
  return renormalize(filtered);
}

function applyTopP(probs: number[], topP: number): number[] {
  if (topP <= 0 || topP >= 1) {
    return [...probs];
  }
  const indexed = probs.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => b.value - a.value);
  let cumulative = 0;
  const keep = new Set<number>();
  for (const item of indexed) {
    keep.add(item.index);
    cumulative += item.value;
    if (cumulative >= topP) break;
  }
  const filtered = probs.map((value, index) => (keep.has(index) ? value : 0));
  return renormalize(filtered);
}

function applyMinProbability(probs: number[], minProbability = 0): number[] {
  if (minProbability <= 0) return [...probs];
  const filtered = probs.map((value) => (value >= minProbability ? value : 0));
  return renormalize(filtered);
}

/**
 * Apply repetition penalty to logits to reduce repetitive output.
 * Uses both frequency penalty (linear based on count) and presence penalty (binary).
 *
 * Frequency penalty: penalty = count * frequencyPenalty
 * Presence penalty: penalty = (count > 0 ? 1 : 0) * presencePenalty
 *
 * Reference: OpenAI API documentation on frequency and presence penalties
 *
 * @param logits - Raw logits from the model
 * @param generatedTokens - Previously generated tokens in the sequence
 * @param frequencyPenalty - Linear penalty based on token frequency (0.0-2.0)
 * @param presencePenalty - Binary penalty for any token presence (0.0-2.0)
 * @returns Modified logits with penalties applied
 */
function applyRepetitionPenalty(
  logits: number[],
  generatedTokens: number[] = [],
  frequencyPenalty: number = 0,
  presencePenalty: number = 0
): number[] {
  if (frequencyPenalty === 0 && presencePenalty === 0) {
    return logits;
  }
  if (generatedTokens.length === 0) {
    return logits;
  }

  // Count token frequencies
  const tokenCounts = new Map<number, number>();
  for (const token of generatedTokens) {
    tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
  }

  // Apply penalties
  const penalized = [...logits];
  for (const [tokenId, count] of tokenCounts.entries()) {
    if (tokenId >= 0 && tokenId < penalized.length) {
      // Frequency penalty: proportional to count
      const freqPenalty = count * frequencyPenalty;
      // Presence penalty: binary (1 if present, 0 otherwise)
      const presPenalty = presencePenalty;
      // Total penalty
      const totalPenalty = freqPenalty + presPenalty;
      // Subtract from logit (higher penalty = lower probability)
      penalized[tokenId] -= totalPenalty;
    }
  }

  return penalized;
}

export function normalizeLogWeights(logWeights: number[]): number[] {
  if (logWeights.length === 0) return [];
  const norm = logSumExp(logWeights);
  return logWeights.map((w) => Math.exp(w - norm));
}

export function sampleCategorical(probs: number[], rng: SamplingRng = defaultRng): number {
  let r = clip(rng(), 0, 0.999999);
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

export function mirostatV2Sample(
  logits: number[],
  options: MirostatV2Options = {}
): { index: number; state: MirostatV2State; surprise: number } {
  if (logits.length === 0) {
    throw new Error('Cannot run Mirostat on empty logits.');
  }

  const {
    targetEntropy = 5,
    learningRate = 0.1,
    initialMu,
    temperature = 1,
    topK = 0,
    topP = 0,
    minProbability = 0,
    frequencyPenalty = 0,
    presencePenalty = 0,
    generatedTokens,
    rng = defaultRng,
    state
  } = options;

  if (!(targetEntropy > 0)) {
    throw new Error('Mirostat targetEntropy must be positive.');
  }

  if (!(learningRate > 0) || learningRate > 1) {
    throw new Error('Mirostat learningRate must be in (0, 1].');
  }

  const baseMu = initialMu ?? targetEntropy * 2;
  if (!(baseMu > 0)) {
    throw new Error('Mirostat initialMu must be positive.');
  }

  const mu = state?.mu ?? baseMu;

  // Apply optional repetition penalties to discourage loops
  const penalized = applyRepetitionPenalty(
    logits,
    generatedTokens,
    frequencyPenalty,
    presencePenalty
  );

  const scaled = penalized.map((value) => value / clip(temperature, 0.05, 5));
  let probs = stableSoftmax(scaled);

  // Respect standard filters before applying Mirostat truncation
  probs = applyTopK(probs, topK);
  probs = applyTopP(probs, topP);
  probs = applyMinProbability(probs, minProbability);

  // Dynamic truncation parameter (k = exp(mu))
  const k = Math.max(1, Math.min(probs.length, Math.round(Math.exp(mu))));

  const ranked = probs.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value);
  const cutoff = new Set(ranked.slice(0, k).map((item) => item.index));
  const truncated = renormalize(probs.map((value, index) => (cutoff.has(index) ? value : 0)));

  const index = sampleCategorical(truncated, rng);
  const prob = clip(truncated[index], 1e-9, 1);
  const surprise = -Math.log(prob);
  const newMu = mu - learningRate * (surprise - targetEntropy);
  return { index, state: { mu: newMu }, surprise };
}

export function temperatureSample(
  logits: number[],
  temperature = 1,
  rng: SamplingRng = defaultRng
): number {
  return sampleFromLogits(logits, { temperature, rng });
}

export function topKSample(logits: number[], topK: number, options: SamplingOptions = {}): number {
  return sampleFromLogits(logits, { ...options, topK });
}

export function nucleusSample(
  logits: number[],
  topP: number,
  options: SamplingOptions = {}
): number {
  return sampleFromLogits(logits, { ...options, topP });
}

/**
 * Typical sampling (locally typical sampling): filters tokens based on information content.
 * This method keeps tokens that have typical information content (close to the expected value),
 * which tends to produce higher quality and more coherent text than top-p sampling.
 *
 * The algorithm:
 * 1. Compute entropy H(p) = -Σ p_i log(p_i) (expected information)
 * 2. For each token, compute information content: -log(p_i)
 * 3. Keep tokens where |H - (-log(p_i))| / H < tau (relative deviation threshold)
 * 4. Sample from the filtered distribution
 *
 * Reference: Meister et al. (2022) "Typical Decoding for Natural Language Generation"
 * https://arxiv.org/abs/2202.00666
 *
 * @param logits - Raw logits from the model
 * @param tau - Threshold for typicality (0.0-1.0, default 0.9). Lower = more filtering.
 * @param options - Additional sampling options
 * @returns Index of the sampled token
 */
export function typicalSample(
  logits: number[],
  tau: number = 0.9,
  options: SamplingOptions = {}
): number {
  if (logits.length === 0) {
    throw new Error('Cannot perform typical sampling on empty logits.');
  }
  if (tau <= 0 || tau > 1) {
    throw new Error('tau must be between 0 and 1 for typical sampling.');
  }

  const { temperature = 1, rng = defaultRng } = options;

  // Apply temperature scaling
  const scaled = logits.map((value) => value / clip(temperature, 0.05, 5));
  const probs = stableSoftmax(scaled);

  // Compute entropy H(p) = -Σ p_i log(p_i)
  let entropy = 0;
  for (const p of probs) {
    if (p > 0) {
      entropy -= p * Math.log(p);
    }
  }

  // Compute information content for each token: -log(p_i)
  const informationContent = probs.map((p) => (p > 0 ? -Math.log(p) : Infinity));

  // Filter tokens based on typicality
  // Keep tokens where the absolute deviation from entropy is small
  const indexed = probs.map((value, index) => ({
    value,
    index,
    deviation: Math.abs(entropy - informationContent[index])
  }));

  // Sort by deviation (most typical first)
  indexed.sort((a, b) => a.deviation - b.deviation);

  // Keep tokens until we reach the tau threshold
  let cumulativeProb = 0;
  const keep = new Set<number>();
  for (const item of indexed) {
    keep.add(item.index);
    cumulativeProb += item.value;
    if (cumulativeProb >= tau) break;
  }

  // Filter probabilities
  const filtered = probs.map((value, index) => (keep.has(index) ? value : 0));
  const renormalized = renormalize(filtered);

  if (renormalized.every((value) => value === 0)) {
    return argMax(probs);
  }

  return sampleCategorical(renormalized, rng);
}

/**
 * Greedy decoding: selects the token with the highest probability (argmax).
 * This is deterministic and always picks the most likely token.
 *
 * @param logits - The raw logits from the model
 * @returns The index of the token with highest logit value
 */
export function greedySample(logits: number[]): number {
  if (logits.length === 0) {
    throw new Error('Cannot perform greedy sampling on empty logits.');
  }
  return argMax(logits);
}

export function sampleFromLogits(logits: number[], options: SamplingOptions = {}): number {
  if (logits.length === 0) {
    throw new Error('Cannot sample from empty logits.');
  }

  const {
    temperature = 1,
    topK = 0,
    topP = 0,
    minProbability = 0,
    frequencyPenalty = 0,
    presencePenalty = 0,
    generatedTokens = [],
    rng = defaultRng
  } = options;

  // Apply repetition penalties first (before temperature scaling)
  const processedLogits = applyRepetitionPenalty(
    logits,
    generatedTokens,
    frequencyPenalty,
    presencePenalty
  );

  // Apply temperature scaling
  const scaled = processedLogits.map((value) => value / clip(temperature, 0.05, 5));

  // Convert to probabilities and apply filters
  let probs = stableSoftmax(scaled);
  probs = applyMinProbability(probs, minProbability);
  probs = applyTopK(probs, topK);
  probs = applyTopP(probs, topP);

  if (probs.every((value) => value === 0)) {
    const index = argMax(logits);
    return index >= 0 ? index : 0;
  }

  return sampleCategorical(probs, rng);
}

export function beamSearch(
  step: LogitsFn,
  startTokens: number[],
  options: BeamSearchOptions
): BeamSearchResult[] {
  const {
    beamWidth,
    maxLength,
    eosToken,
    temperature = 1,
    lengthPenalty = 1,
    rng = defaultRng
  } = options;

  if (beamWidth <= 0) {
    throw new Error('beamWidth must be positive.');
  }
  if (maxLength <= 0) {
    throw new Error('maxLength must be positive.');
  }

  const active: BeamSearchState[] = [
    { tokens: [...startTokens], logProbability: 0, completed: false }
  ];
  const finished: BeamSearchState[] = [];

  while (active.length > 0 && finished.length < beamWidth) {
    const newBeams: BeamSearchState[] = [];
    for (const beam of active) {
      if (beam.completed || beam.tokens.length >= startTokens.length + maxLength) {
        finished.push(beam);
        continue;
      }

      const logits = step(beam.tokens);
      if (logits.length === 0) {
        finished.push({ ...beam, completed: true });
        continue;
      }

      const scaled = logits.map((value) => value / clip(temperature, 0.05, 5));
      const norm = logSumExp(scaled);
      const logProbs = scaled.map((value) => value - norm);

      const ranking = logProbs.map((value, index) => ({
        index,
        score: value + 1e-8 * (rng() - 0.5)
      }));
      ranking.sort((a, b) => b.score - a.score);
      const candidates = ranking.slice(0, beamWidth * 2);

      for (const candidate of candidates) {
        const index = candidate.index;
        const logProb = logProbs[index];
        const tokens = [...beam.tokens, index];
        const completed = eosToken !== undefined && index === eosToken;
        newBeams.push({
          tokens,
          completed,
          logProbability: beam.logProbability + logProb
        });
      }
    }

    if (newBeams.length === 0) break;

    newBeams.sort((a, b) => b.logProbability - a.logProbability);
    active.length = 0;
    for (const beam of newBeams.slice(0, beamWidth)) {
      if (beam.completed) {
        finished.push(beam);
      } else {
        active.push(beam);
      }
    }

    if (active.length === 0) {
      while (finished.length < beamWidth && newBeams.length > finished.length) {
        const beam = newBeams[finished.length];
        finished.push({ ...beam, completed: true });
      }
    }
  }

  const scored = finished.map((beam) => {
    const length = Math.max(1, beam.tokens.length - startTokens.length);
    const score = beam.logProbability / Math.pow(length, lengthPenalty);
    return { ...beam, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const logWeights = scored.map((beam) => beam.logProbability);
  const weights = normalizeLogWeights(logWeights);

  return scored.map((beam, index) => ({
    tokens: beam.tokens.slice(startTokens.length),
    score: beam.score,
    probability: weights[index]
  }));
}

/**
 * Computes cosine similarity between two vectors
 */
function cosineSimilarity(vec1: number[], vec2: number[]): number {
  if (vec1.length !== vec2.length) {
    throw new Error('Vectors must have the same length for cosine similarity.');
  }

  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let i = 0; i < vec1.length; i++) {
    dotProduct += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
  if (denominator === 0) return 0;

  return dotProduct / denominator;
}

/**
 * Contrastive search/decoding: balances model confidence with diversity.
 * This method penalizes tokens that are too similar to previously generated tokens,
 * helping to reduce repetition while maintaining coherence.
 *
 * The scoring function is: score(token) = (1 - alpha) * p(token) - alpha * max(similarity(token, context))
 *
 * @param step - Function that takes token sequence and returns next-token logits
 * @param embeddingFn - Function that converts token IDs to embedding vectors
 * @param startTokens - Initial sequence of tokens
 * @param options - Configuration options for contrastive search
 * @returns Result containing generated tokens and score
 */
export function contrastiveSearch(
  step: LogitsFn,
  embeddingFn: EmbeddingFn,
  startTokens: number[],
  options: ContrastiveSearchOptions
): ContrastiveSearchResult {
  const { topK, alpha, maxLength, eosToken } = options;

  if (topK <= 0) {
    throw new Error('topK must be positive for contrastive search.');
  }
  if (alpha < 0 || alpha > 1) {
    throw new Error('alpha must be between 0 and 1.');
  }
  if (maxLength <= 0) {
    throw new Error('maxLength must be positive.');
  }

  const tokens = [...startTokens];
  let totalScore = 0;

  // Cache embeddings for all generated tokens
  const contextEmbeddings: number[][] = [];

  for (let step_num = 0; step_num < maxLength; step_num++) {
    const logits = step(tokens);
    if (logits.length === 0) break;

    // Convert logits to probabilities
    const probs = stableSoftmax(logits);

    // Get top-k candidates
    const indexed = probs.map((value, index) => ({ value, index }));
    indexed.sort((a, b) => b.value - a.value);
    const topKCandidates = indexed.slice(0, Math.min(topK, indexed.length));

    let bestToken = topKCandidates[0].index;
    let bestScore = -Infinity;

    // Score each candidate
    for (const candidate of topKCandidates) {
      const tokenId = candidate.index;
      const modelProb = candidate.value;

      // Get embedding for candidate token
      const candidateEmbedding = embeddingFn([tokenId]);

      // Compute maximum similarity with context
      let maxSimilarity = 0;
      if (contextEmbeddings.length > 0) {
        for (const contextEmb of contextEmbeddings) {
          const similarity = cosineSimilarity(candidateEmbedding, contextEmb);
          maxSimilarity = Math.max(maxSimilarity, similarity);
        }
      }

      // Contrastive scoring: balance probability and diversity
      const score = (1 - alpha) * modelProb - alpha * maxSimilarity;

      if (score > bestScore) {
        bestScore = score;
        bestToken = tokenId;
      }
    }

    // Add selected token
    tokens.push(bestToken);
    totalScore += bestScore;

    // Cache the embedding
    const selectedEmbedding = embeddingFn([bestToken]);
    contextEmbeddings.push(selectedEmbedding);

    // Check for EOS
    if (eosToken !== undefined && bestToken === eosToken) {
      break;
    }
  }

  return {
    tokens: tokens.slice(startTokens.length),
    score: totalScore
  };
}

export function monteCarloSample<T>(
  logitsFn: () => number[],
  evaluator: (index: number) => T,
  options: SamplingOptions & { trials: number }
): T[] {
  const { trials, ...sampling } = options;
  if (trials <= 0) {
    throw new Error('Monte Carlo sampling requires at least one trial.');
  }
  const results: T[] = [];
  for (let i = 0; i < trials; i++) {
    const logits = logitsFn();
    const index = sampleFromLogits(logits, sampling);
    results.push(evaluator(index));
  }
  return results;
}
