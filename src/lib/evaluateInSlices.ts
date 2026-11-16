import { ProNeuralLM, type EvaluationResult, type EvaluationSums } from './ProNeuralLM';
import { TransformerLM } from './TransformerLM';

export interface SlicedEvaluationOptions {
  /**
   * Windows per slice with exact scoring; with `fast`, chunks per slice (default 16 and 2).
   */
  sliceWindows?: number;
  /** Longest stretch of work between two yields to the event loop (default 50 ms). */
  yieldIntervalMs?: number;
  /**
   * Causal transformers only: score in chunks of half a context window
   * (`TransformerLM.evaluateChunkedSums`) instead of one forward pass per token. About
   * contextSize / 2 times cheaper, with a slightly higher loss; meant for early stopping.
   * Other models are always scored exactly.
   */
  fast?: boolean;
}

/**
 * Held-out scoring in slices, yielding to the event loop between slices so the page stays
 * responsive, and stopping when `signal` aborts (returns null then). Without `fast` the result
 * is identical to `model.evaluate(text)` (the sums are added in order).
 */
export async function evaluateInSlices(
  model: ProNeuralLM,
  text: string,
  signal?: AbortSignal | null,
  options: SlicedEvaluationOptions = {}
): Promise<EvaluationResult | null> {
  const interval = options.yieldIntervalMs ?? 50;
  const chunked =
    options.fast && model instanceof TransformerLM && model.evaluateChunkedSums(text, 0, 0) !== null
      ? model
      : null;
  const slice = Math.max(1, options.sliceWindows ?? (chunked ? 2 : 16));
  const parts: EvaluationSums[] = [];
  let lastYield = Date.now();
  let start = 0;
  let total = Infinity;
  while (start < total) {
    if (signal?.aborted) return null;
    const sums = chunked
      ? chunked.evaluateChunkedSums(text, start, start + slice)!
      : await model.evaluateSums(text, start, start + slice);
    parts.push(sums);
    total = sums.total;
    start += slice;
    if (Date.now() - lastYield >= interval) {
      await new Promise((resolve) => setTimeout(resolve, 0));
      lastYield = Date.now();
    }
  }
  return ProNeuralLM.evaluationResult(parts);
}
