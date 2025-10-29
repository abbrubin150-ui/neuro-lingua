/**
 * Advanced loss functions used by the research notebooks and experiment
 * harness. Each function operates on native number arrays to avoid tying the
 * implementation to a specific tensor backend.
 */

export function focalLoss(
  logits: number[],
  targets: number[],
  options: { gamma?: number; alpha?: number } = {}
): number {
  const { gamma = 2, alpha = 0.25 } = options;
  if (logits.length !== targets.length) {
    throw new Error('focalLoss: logits and targets must share the same shape.');
  }
  let loss = 0;
  for (let i = 0; i < logits.length; i++) {
    const logit = logits[i];
    const target = targets[i];
    const prob = 1 / (1 + Math.exp(-logit));
    const pt = target ? prob : 1 - prob;
    loss += -alpha * (1 - pt) ** gamma * Math.log(Math.max(pt, Number.EPSILON));
  }
  return loss / logits.length;
}

export function labelSmoothingCrossEntropy(
  logits: number[],
  targetIndex: number,
  classes: number,
  smoothing = 0.1
): number {
  const logProbs = softmax(logits).map((p) => Math.log(Math.max(p, Number.EPSILON)));
  const smoothTarget = smoothing / classes;
  let loss = 0;
  for (let i = 0; i < classes; i++) {
    const target = i === targetIndex ? 1 - smoothing + smoothTarget : smoothTarget;
    loss += -target * logProbs[i];
  }
  return loss;
}

export function symmetricCrossEntropy(
  logits: number[],
  targets: number[],
  alpha = 1,
  beta = 1
): number {
  const probs = softmax(logits);
  const eps = 1e-12;
  let forward = 0;
  let reverse = 0;
  for (let i = 0; i < logits.length; i++) {
    const target = Math.min(Math.max(targets[i], eps), 1 - eps);
    const prob = Math.min(Math.max(probs[i], eps), 1 - eps);
    forward += -target * Math.log(prob);
    reverse += -prob * Math.log(target);
  }
  return alpha * forward + beta * reverse;
}

export function cosineEmbeddingLoss(x: number[], y: number[], label: 1 | -1, margin = 0.0): number {
  if (x.length !== y.length) {
    throw new Error('cosineEmbeddingLoss expects vectors of equal length.');
  }
  const dot = x.reduce((sum, value, idx) => sum + value * y[idx], 0);
  const normX = Math.sqrt(x.reduce((sum, value) => sum + value * value, 0));
  const normY = Math.sqrt(y.reduce((sum, value) => sum + value * value, 0));
  const cosine = dot / Math.max(normX * normY, Number.EPSILON);
  return label === 1 ? 1 - cosine : Math.max(0, cosine - margin);
}

function softmax(vector: number[]): number[] {
  const maxVal = Math.max(...vector);
  const exps = vector.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((acc, val) => acc + val, 0) || Number.EPSILON;
  return exps.map((v) => v / sum);
}
