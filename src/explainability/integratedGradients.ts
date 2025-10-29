/**
 * Integrated Gradients attribution for black-box gradient evaluators.
 */

export type GradientFn = (input: number[]) => number[];

export interface IntegratedGradientsOptions {
  /** Baseline used to start the integration path. Defaults to the zero vector. */
  baseline?: number[];
  /** Number of Riemann summation steps along the straight-line path. */
  steps?: number;
  /** Whether to return the intermediate points for visualisation. */
  returnPath?: boolean;
}

export interface IntegratedGradientsResult {
  attributions: number[];
  path?: number[][];
}

function interpolate(baseline: number[], input: number[], alpha: number): number[] {
  const result = new Array(input.length);
  for (let i = 0; i < input.length; i++) {
    result[i] = baseline[i] + alpha * (input[i] - baseline[i]);
  }
  return result;
}

function ensureDimensions(input: number[], baseline: number[]): void {
  if (input.length !== baseline.length) {
    throw new Error('Input and baseline must share the same dimensionality.');
  }
}

/**
 * Compute Integrated Gradients for the provided input.
 *
 * The caller supplies a gradient function (e.g. powered by auto-diff) that
 * returns $\nabla_x f(x)$ for the model output of interest. The integration is
 * performed along the straight line between the baseline and the input.
 */
export function integratedGradients(
  gradientFn: GradientFn,
  input: number[],
  options: IntegratedGradientsOptions = {}
): IntegratedGradientsResult {
  if (input.length === 0) return { attributions: [] };
  const baseline = options.baseline ?? new Array(input.length).fill(0);
  ensureDimensions(input, baseline);
  const steps = Math.max(1, options.steps ?? 50);
  const path: number[][] = [];
  const attributions = new Array(input.length).fill(0);
  const stepSize = 1 / steps;

  for (let s = 1; s <= steps; s++) {
    const alpha = s * stepSize;
    const point = interpolate(baseline, input, alpha);
    const grad = gradientFn(point);
    if (grad.length !== input.length) {
      throw new Error('Gradient dimensionality mismatch.');
    }
    for (let i = 0; i < input.length; i++) {
      attributions[i] += grad[i];
    }
    if (options.returnPath) {
      path.push(point);
    }
  }

  for (let i = 0; i < attributions.length; i++) {
    attributions[i] *= (input[i] - baseline[i]) * stepSize;
  }

  return { attributions, path: options.returnPath ? path : undefined };
}
