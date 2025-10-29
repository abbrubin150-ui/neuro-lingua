/**
 * Lightweight SHAP value estimation via permutation sampling.
 */

export type PredictFn = (input: number[]) => number;

export interface ShapOptions {
  /** Baseline input used to represent the model's "missing" features. */
  baseline?: number[];
  /**
   * Number of random permutations to average over. More permutations lead to a
   * better Monte Carlo approximation of the exact Shapley values.
   */
  permutations?: number;
  /** Optional random seed for deterministic experiments. */
  seed?: number;
}

function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) % 0xffffffff;
    return state / 0xffffffff;
  };
}

function randomPermutation(size: number, rng: () => number): number[] {
  const perm = Array.from({ length: size }, (_, i) => i);
  for (let i = size - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  return perm;
}

function clone(input: number[]): number[] {
  return input.slice();
}

/**
 * Estimate SHAP values for a single input using permutation sampling.
 */
export function estimateShapValues(
  model: PredictFn,
  input: number[],
  options: ShapOptions = {}
): number[] {
  if (input.length === 0) return [];
  const baseline = options.baseline ?? new Array(input.length).fill(0);
  if (baseline.length !== input.length) {
    throw new Error('Baseline dimension must match the input dimension.');
  }
  const permutations = options.permutations ?? Math.max(32, input.length * 4);
  const rng = createRng(options.seed ?? 5489);
  const contributions = new Array(input.length).fill(0);

  for (let t = 0; t < permutations; t++) {
    const order = randomPermutation(input.length, rng);
    const current = clone(baseline);
    let prevOutput = model(current);
    for (const index of order) {
      current[index] = input[index];
      const nextOutput = model(current);
      const marginal = nextOutput - prevOutput;
      contributions[index] += marginal;
      prevOutput = nextOutput;
    }
  }

  const inv = 1 / permutations;
  for (let i = 0; i < contributions.length; i++) {
    contributions[i] *= inv;
  }

  return contributions;
}
