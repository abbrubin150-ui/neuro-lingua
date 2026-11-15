import { ProNeuralLM, type EvaluationResult, type EvaluationSums } from './ProNeuralLM';

/**
 * `model.evaluate(text)` in slices of windows, yielding to the event loop between slices so the
 * page stays responsive, and stopping when `signal` aborts (returns null then). The result is
 * identical to `evaluate` (the sums are added in order).
 */
export async function evaluateInSlices(
  model: ProNeuralLM,
  text: string,
  signal?: AbortSignal | null,
  options: { sliceWindows?: number; yieldIntervalMs?: number } = {}
): Promise<EvaluationResult | null> {
  const slice = Math.max(1, options.sliceWindows ?? 16);
  const interval = options.yieldIntervalMs ?? 50;
  const parts: EvaluationSums[] = [];
  let lastYield = Date.now();
  let start = 0;
  let total = Infinity;
  while (start < total) {
    if (signal?.aborted) return null;
    const sums = await model.evaluateSums(text, start, start + slice);
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
