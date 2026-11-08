import { describe, expect, it } from 'vitest';
import {
  blockBackward,
  blockForward,
  createBlockGrads,
  type BlockConfig,
  type BlockParams
} from '../../src/models/transformer_backprop';
import { MiniTransformerBlock } from '../../src/models/mini_transformer';

function rng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

function randMat(r: number, c: number, next: () => number, scale = 0.6): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => next() * scale));
}

function makeParams(D: number, kvDim: number, F: number, seed: number): BlockParams {
  const next = rng(seed);
  return {
    attention: {
      query: randMat(D, D, next),
      key: randMat(D, kvDim, next),
      value: randMat(D, kvDim, next)
    },
    ff1: randMat(D, 2 * F, next),
    ff2: randMat(F, D, next),
    attnGamma: Array.from({ length: D }, () => 1 + next() * 0.4),
    ffnGamma: Array.from({ length: D }, () => 1 + next() * 0.4),
    attnEpsilon: 1e-6,
    ffnEpsilon: 1e-6
  };
}

/** Scalar loss L = Σ w ⊙ out, with fixed random weights w. */
function lossOf(
  X: number[][],
  params: BlockParams,
  config: BlockConfig,
  positions: number[],
  W: number[][]
): number {
  const { out } = blockForward(X, params, config, positions);
  let s = 0;
  out.forEach((row, t) => row.forEach((v, j) => (s += v * W[t][j])));
  return s;
}

function checkGradients(config: BlockConfig, D: number, seed: number) {
  const hd = D / config.numHeads;
  const kvDim = config.numKVHeads * hd;
  const F = 3;
  const T = 3;
  const params = makeParams(D, kvDim, F, seed);
  const next = rng(seed + 99);
  const X = randMat(T, D, next, 2);
  const W = randMat(T, D, next, 2);
  const positions = [0, 1, 2];

  const { cache } = blockForward(X, params, config, positions);
  const grads = createBlockGrads(params);
  const dX = blockBackward(W, cache, params, config, positions, grads);

  const h = 1e-5;
  const numeric = (get: () => number, set: (v: number) => void) => {
    const orig = get();
    set(orig + h);
    const lp = lossOf(X, params, config, positions, W);
    set(orig - h);
    const lm = lossOf(X, params, config, positions, W);
    set(orig);
    return (lp - lm) / (2 * h);
  };
  const expectClose = (analytic: number, num: number) => {
    expect(Math.abs(analytic - num)).toBeLessThan(1e-5 + 1e-4 * Math.abs(num));
  };

  const matrices: Array<[number[][], number[][]]> = [
    [params.attention.query, grads.query],
    [params.attention.key, grads.key],
    [params.attention.value, grads.value],
    [params.ff1, grads.ff1],
    [params.ff2, grads.ff2],
    [X, dX]
  ];
  for (const [P, G] of matrices) {
    for (let i = 0; i < P.length; i++) {
      for (let j = 0; j < P[i].length; j++) {
        expectClose(
          G[i][j],
          numeric(
            () => P[i][j],
            (v) => (P[i][j] = v)
          )
        );
      }
    }
  }
  const vectors: Array<[number[], number[]]> = [
    [params.attnGamma, grads.attnGamma],
    [params.ffnGamma, grads.ffnGamma]
  ];
  for (const [P, G] of vectors) {
    for (let i = 0; i < P.length; i++) {
      expectClose(
        G[i],
        numeric(
          () => P[i],
          (v) => (P[i] = v)
        )
      );
    }
  }
}

describe('transformer_backprop', () => {
  it('matches finite-difference gradients (standard MHA)', () => {
    checkGradients({ numHeads: 2, numKVHeads: 2, ropeBase: 10000 }, 4, 7);
  });

  it('matches finite-difference gradients (GQA, 4 query heads / 2 KV heads)', () => {
    checkGradients({ numHeads: 4, numKVHeads: 2, ropeBase: 500000 }, 8, 11);
  });

  it('matches finite-difference gradients (MQA, 1 KV head)', () => {
    checkGradients({ numHeads: 2, numKVHeads: 1, ropeBase: 100 }, 4, 23);
  });

  it('produces the same forward output as MiniTransformerBlock (dropout disabled)', async () => {
    const D = 8;
    const config: BlockConfig = { numHeads: 4, numKVHeads: 2, ropeBase: 500000 };
    const params = makeParams(D, 4, 4, 3);
    const X = randMat(4, D, rng(5), 2);
    const positions = [0, 1, 2, 3];
    const block = new MiniTransformerBlock({
      modelDim: D,
      heads: 4,
      numKVHeads: 2,
      ff: { hiddenDim: 4 },
      attentionDropout: 0,
      dropConnectRate: 0,
      attentionRms: { gamma: params.attnGamma, epsilon: 1e-6 },
      ffnRms: { gamma: params.ffnGamma, epsilon: 1e-6 },
      ropeBase: 500000
    });
    const reference = await block.forward(X, params.attention, params.ff1, params.ff2, {
      positions
    });
    const { out } = blockForward(X, params, config, positions);
    out.forEach((row, t) => row.forEach((v, j) => expect(v).toBeCloseTo(reference[t][j], 10)));
  });
});
