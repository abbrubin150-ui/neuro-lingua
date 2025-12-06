import type { CerebroBubble, CerebroBubbleTag } from '../../types/injection';
import type { ProNeuralLM } from '../ProNeuralLM';

/**
 * Configuration for bubble extraction
 */
export interface BubbleExtractionConfig {
  /** Maximum number of bubbles to extract */
  maxBubbles?: number;
  /** Minimum activation threshold (0-1) */
  minActivation?: number;
  /** Use random sampling instead of top-k */
  randomSample?: boolean;
}

const DEFAULT_CONFIG: Required<BubbleExtractionConfig> = {
  maxBubbles: 24,
  minActivation: 0.1,
  randomSample: false
};

/**
 * Semantic tag assignment based on token characteristics.
 * Uses heuristics to categorize tokens into bubble tags.
 */
function assignTag(token: string): CerebroBubbleTag {
  const lower = token.toLowerCase();

  // Body-related tokens (physical, sensory)
  const bodyPatterns = /^(hand|eye|head|body|arm|leg|foot|heart|face|skin|blood|bone)s?$/;
  if (bodyPatterns.test(lower)) return 'body';

  // Desire/want tokens
  const desirePatterns = /^(want|need|wish|hope|desire|crave|long|yearn|seek)s?$/;
  if (desirePatterns.test(lower)) return 'desire';

  // Risk/danger tokens
  const riskPatterns = /^(risk|danger|threat|fear|harm|loss|fail|error|wrong|bad|pain)s?$/;
  if (riskPatterns.test(lower)) return 'risk';

  // Value/positive tokens
  const valuePatterns = /^(good|value|worth|benefit|gain|profit|success|right|true|love)s?$/;
  if (valuePatterns.test(lower)) return 'value';

  // Action tokens (verbs)
  const actionPatterns =
    /^(go|run|walk|move|do|make|take|give|get|put|come|see|look|find|use|work|try)s?$/;
  if (actionPatterns.test(lower)) return 'action';

  return 'other';
}

/**
 * Calculate activation score for a token based on embedding magnitude.
 * Normalized to 0-1 range.
 */
function calculateActivation(embedding: number[]): number {
  const magnitude = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
  // Normalize using typical embedding magnitudes (assuming ~1.0 average)
  return Math.min(1.0, magnitude / 2.0);
}

/**
 * Extract semantic bubbles from a model's embedding layer.
 *
 * Bubbles represent semantic clusters derived from the learned embeddings.
 * Each bubble contains:
 * - The embedding vector
 * - An activation score (embedding magnitude)
 * - A semantic tag based on token characteristics
 *
 * @param model The neural language model to extract from
 * @param config Extraction configuration
 * @returns Array of CerebroBubble objects
 */
export function extractBubblesFromModel(
  model: ProNeuralLM,
  config: BubbleExtractionConfig = {}
): CerebroBubble[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const weights = model.getWeights();
  const embeddings = weights.embedding;
  const vocabSize = model.getVocabSize();

  // Get vocabulary mapping (use internal method via type assertion)
  const idxToWord = (model as any).idxToWord as Map<number, string>;

  // Calculate activations for all tokens
  const tokenData: { idx: number; token: string; activation: number; embedding: number[] }[] = [];

  for (let i = 0; i < vocabSize; i++) {
    const token = idxToWord.get(i) ?? `token-${i}`;
    const embedding = embeddings[i];
    const activation = calculateActivation(embedding);

    // Skip special tokens and low activation
    if (token.startsWith('<') && token.endsWith('>')) continue;
    if (activation < cfg.minActivation) continue;

    tokenData.push({ idx: i, token, activation, embedding });
  }

  // Sort by activation (descending) or random shuffle
  if (cfg.randomSample) {
    // Fisher-Yates shuffle
    for (let i = tokenData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [tokenData[i], tokenData[j]] = [tokenData[j], tokenData[i]];
    }
  } else {
    tokenData.sort((a, b) => b.activation - a.activation);
  }

  // Take top N bubbles
  const selectedTokens = tokenData.slice(0, cfg.maxBubbles);

  // Convert to bubbles
  return selectedTokens.map((data, idx) => ({
    id: `bubble-${data.idx}`,
    label: data.token,
    embedding: data.embedding,
    activation: data.activation,
    tag: assignTag(data.token),
    ts: Date.now() - idx * 100 // Slight offset for ordering
  }));
}

/**
 * Extract bubbles from recent training context.
 *
 * Uses the most recently seen tokens during training to create
 * contextually relevant bubbles.
 *
 * @param model The neural language model
 * @param recentTokens Array of recently processed token indices
 * @param config Extraction configuration
 * @returns Array of CerebroBubble objects
 */
export function extractBubblesFromContext(
  model: ProNeuralLM,
  recentTokens: number[],
  config: BubbleExtractionConfig = {}
): CerebroBubble[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const weights = model.getWeights();
  const embeddings = weights.embedding;

  const idxToWord = (model as any).idxToWord as Map<number, string>;

  // Deduplicate tokens while preserving order
  const uniqueTokens = [...new Set(recentTokens)];

  // Extract bubbles for unique tokens
  const bubbles: CerebroBubble[] = [];

  for (const idx of uniqueTokens) {
    if (idx < 0 || idx >= embeddings.length) continue;

    const token = idxToWord.get(idx) ?? `token-${idx}`;
    if (token.startsWith('<') && token.endsWith('>')) continue;

    const embedding = embeddings[idx];
    const activation = calculateActivation(embedding);

    if (activation < cfg.minActivation) continue;

    bubbles.push({
      id: `ctx-bubble-${idx}`,
      label: token,
      embedding,
      activation,
      tag: assignTag(token),
      ts: Date.now()
    });

    if (bubbles.length >= cfg.maxBubbles) break;
  }

  return bubbles;
}
