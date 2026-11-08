/**
 * Attributions of a language model's next-token prediction to its context tokens.
 *
 * The explained quantity is f(context) = log p(target | context), where `target` is the model's
 * most likely next token after the input text and `context` is the window the model actually
 * reads (the last contextSize tokens, left-padded with BOS). A context token that is "absent"
 * is replaced by the BOS padding embedding.
 *
 * - 'shap': Shapley values of the context positions — exact (all 2^n coalitions) for windows of
 *   up to 10 tokens, Monte-Carlo permutations otherwise. They sum to f(context) − f(all BOS).
 * - 'integrated-gradients': integrated gradients in embedding space from the all-BOS baseline,
 *   using the model's exact input gradient; per-token scores sum (up to the Riemann error) to
 *   the same difference.
 * - 'attention-rollout': attention rollout over the transformer's real attention maps
 *   (transformer models only); scores are the share of each context position in the final
 *   representation and sum to 1.
 */

import { ProNeuralLM } from '../lib/ProNeuralLM';
import { TransformerLM } from '../lib/TransformerLM';
import { estimateShapValues } from './shap';
import { integratedGradients } from './integratedGradients';
import { attentionRollout } from './attentionRollout';

export type AttributionMethod = 'shap' | 'integrated-gradients' | 'attention-rollout';

export interface NextTokenExplanation {
  method: AttributionMethod;
  /** Context window tokens (oldest first; BOS marks padding). */
  tokens: string[];
  /** Explained next token and its probability. */
  target: string;
  targetProbability: number;
  /** One score per context token. */
  scores: number[];
  /** f(context) − f(baseline) in nats (shap / integrated-gradients). */
  totalEffect?: number;
}

const EXACT_SHAPLEY_MAX_TOKENS = 10;

function factorial(n: number): number {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

/** Exact Shapley values for a set function v over n players given as a bit mask. */
export function exactShapley(n: number, v: (mask: number) => number): number[] {
  const values = new Float64Array(1 << n);
  for (let mask = 0; mask < 1 << n; mask++) values[mask] = v(mask);
  const weights = Array.from(
    { length: n },
    (_, s) => (factorial(s) * factorial(n - s - 1)) / factorial(n)
  );
  const phi = new Array(n).fill(0);
  for (let mask = 0; mask < 1 << n; mask++) {
    let size = 0;
    for (let i = 0; i < n; i++) if (mask & (1 << i)) size++;
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) continue;
      phi[i] += weights[size] * (values[mask | (1 << i)] - values[mask]);
    }
  }
  return phi;
}

export async function explainNextToken(
  model: ProNeuralLM,
  text: string,
  method: AttributionMethod,
  options: { steps?: number; permutations?: number } = {}
): Promise<NextTokenExplanation> {
  const window = model.getContextWindow(text);
  const vocab = model.getVocab();
  const tokens = window.map((i) => vocab[i] ?? '<UNK>');
  const logits = await model.getLogitsForContext(window);
  let target = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[target]) target = i;
  let maxLogit = -Infinity;
  for (const z of logits) if (z > maxLogit) maxLogit = z;
  const partition = logits.reduce((s, z) => s + Math.exp(z - maxLogit), 0);
  const targetProbability = Math.exp(logits[target] - maxLogit) / partition;

  const base = { method, tokens, target: vocab[target] ?? '<UNK>', targetProbability };

  if (method === 'attention-rollout') {
    if (!(model instanceof TransformerLM)) {
      throw new Error('Attention Rollout is only available for Transformer models');
    }
    const rollout = attentionRollout(model.getAttentionMaps(window), { addIdentity: true });
    const causal = model.getTransformerInfo().objective === 'causal';
    // Causal models predict from the last position; pooled models average every position.
    const scores = causal
      ? rollout[rollout.length - 1]
      : rollout[0].map((_, j) => rollout.reduce((s, row) => s + row[j], 0) / rollout.length);
    return { ...base, scores };
  }

  const bos = model.getTokenIndex('<BOS>');
  const input = model.getEmbeddingRows(window);
  const baseline = model.getEmbeddingRows(window.map(() => (bos >= 0 ? bos : 0)));
  const logProb = (rows: number[][]) => model.explainLogProb(rows, target).logProb;
  const totalEffect = logProb(input) - logProb(baseline);

  if (method === 'shap') {
    const n = window.length;
    const rowsFor = (present: (i: number) => boolean) =>
      input.map((row, i) => (present(i) ? row : baseline[i]));
    const scores =
      n <= EXACT_SHAPLEY_MAX_TOKENS
        ? exactShapley(n, (mask) => logProb(rowsFor((i) => (mask & (1 << i)) !== 0)))
        : estimateShapValues(
            (presence) => logProb(rowsFor((i) => presence[i] > 0.5)),
            new Array(n).fill(1),
            { baseline: new Array(n).fill(0), permutations: options.permutations ?? 8 * n }
          );
    return { ...base, scores, totalEffect };
  }

  // Integrated gradients in embedding space
  const H = input[0]?.length ?? 0;
  const flatten = (rows: number[][]) => rows.flat();
  const unflatten = (flat: number[]) =>
    Array.from({ length: window.length }, (_, t) => flat.slice(t * H, (t + 1) * H));
  const result = integratedGradients(
    (flat) => flatten(model.explainLogProb(unflatten(flat), target).gradient),
    flatten(input),
    { baseline: flatten(baseline), steps: options.steps ?? 64 }
  );
  const scores = window.map((_, t) =>
    result.attributions.slice(t * H, (t + 1) * H).reduce((s, v) => s + v, 0)
  );
  return { ...base, scores, totalEffect };
}
