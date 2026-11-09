import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

/**
 * The 'newton' optimizer uses the exact Gauss-Newton diagonal G = Jᵀ·A·J (A = diag(p) − ppᵀ).
 * Verify every entry against J computed by finite differences of the logits.
 */
type Internals = {
  wOutput: number[][];
  bOutput: number[];
  wHidden: number[][];
  bHidden: number[];
  forward(
    ctx: number[],
    train?: boolean
  ): Promise<{
    h: number[];
    probs: number[];
    logits: number[];
    avgEmb: number[];
    preAct: number[];
    z?: number[];
    dropMask: number[] | null;
  }>;
  gaussNewtonDiagonal(
    h: number[],
    probs: number[],
    avgEmb: number[],
    z: number[],
    dropMask: number[] | null
  ): { wOutput: number[][]; bOutput: number[]; wHidden: number[][]; bHidden: number[] };
};

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c'];

describe('Gauss-Newton diagonal', () => {
  it('matches diag(Jᵀ·A·J) with J from finite differences', async () => {
    const model = new ProNeuralLM(VOCAB, 5, 0.1, 3, 'newton', 0.9, 0, 9);
    const m = model as unknown as Internals;
    let s = 11;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5) * 1.6;
    m.wOutput = m.wOutput.map((r) => r.map(() => rnd()));
    m.wHidden = m.wHidden.map((r) => r.map(() => rnd()));
    m.bHidden = m.bHidden.map(() => 0.2 + rnd() * 0.1);
    const ctx = [1, 4, 5];

    const cache = await m.forward(ctx, false);
    const G = m.gaussNewtonDiagonal(
      cache.h,
      cache.probs,
      cache.avgEmb,
      cache.z ?? cache.preAct,
      cache.dropMask
    );
    const p = cache.probs;
    const ggnEntry = async (get: () => number, set: (v: number) => void) => {
      const eps = 1e-6;
      const o = get();
      set(o + eps);
      const zp = (await m.forward(ctx, false)).logits;
      set(o - eps);
      const zm = (await m.forward(ctx, false)).logits;
      set(o);
      const J = zp.map((v, k) => (v - zm[k]) / (2 * eps));
      // Jᵀ A J = Σ_k p_k J_k² − (Σ_k p_k J_k)²
      let m1 = 0;
      let m2 = 0;
      J.forEach((jk, k) => {
        m1 += p[k] * jk;
        m2 += p[k] * jk * jk;
      });
      return m2 - m1 * m1;
    };
    const close = (a: number, b: number) =>
      expect(Math.abs(a - b)).toBeLessThan(1e-7 + 1e-4 * Math.abs(b));

    for (const key of ['wOutput', 'wHidden'] as const) {
      for (let i = 0; i < m[key].length; i++) {
        for (let j = 0; j < m[key][i].length; j++) {
          close(
            G[key][i][j],
            await ggnEntry(
              () => m[key][i][j],
              (v) => (m[key][i][j] = v)
            )
          );
        }
      }
    }
    for (const key of ['bOutput', 'bHidden'] as const) {
      for (let i = 0; i < m[key].length; i++) {
        close(
          G[key][i],
          await ggnEntry(
            () => m[key][i],
            (v) => (m[key][i] = v)
          )
        );
      }
    }
  });

  it('takes damped steps: training stays finite and reduces the loss on a simple sequence', async () => {
    const corpus = 'a b c a b c a b c a b c a b c a b c';
    const model = new ProNeuralLM(VOCAB, 16, 0.03, 3, 'newton', 0.9, 0, 4);
    const first = await model.train(corpus, 1);
    const later = await model.train(corpus, 20);
    const last = later.history[later.history.length - 1].loss;
    expect(Number.isFinite(last)).toBe(true);
    expect(last).toBeLessThan(first.loss);
  });
});
