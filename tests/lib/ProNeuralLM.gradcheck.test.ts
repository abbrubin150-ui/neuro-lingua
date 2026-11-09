import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

/**
 * Finite-difference verification of ProNeuralLM.backward.
 *
 * With optimizer 'momentum', momentum = 0 and learning rate 1, a single backward step applies
 * ΔW = −∂L/∂W exactly, so the analytic gradient can be read back from the parameter change and
 * compared with central differences of the cross-entropy loss.
 */

type Internals = {
  embedding: number[][];
  wHidden: number[][];
  wOutput: number[][];
  bHidden: number[];
  bOutput: number[];
  forward(inputs: number[], train?: boolean): Promise<{ probs: number[] }>;
  backward(inputs: number[], target: number, cache: unknown): Promise<void>;
};

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];

function snapshot(m: Internals) {
  return {
    embedding: m.embedding.map((r) => r.slice()),
    wHidden: m.wHidden.map((r) => r.slice()),
    wOutput: m.wOutput.map((r) => r.slice()),
    bHidden: m.bHidden.slice(),
    bOutput: m.bOutput.slice()
  };
}

function restore(m: Internals, s: ReturnType<typeof snapshot>) {
  m.embedding = s.embedding.map((r) => r.slice());
  m.wHidden = s.wHidden.map((r) => r.slice());
  m.wOutput = s.wOutput.map((r) => r.slice());
  m.bHidden = s.bHidden.slice();
  m.bOutput = s.bOutput.slice();
}

async function lossAt(m: Internals, ctx: number[], target: number) {
  const { probs } = await m.forward(ctx, false);
  return -Math.log(probs[target]);
}

describe('ProNeuralLM backward (finite-difference check)', () => {
  it('matches numeric gradients for every parameter, including repeated context tokens', async () => {
    const model = new ProNeuralLM(VOCAB, 6, 1, 4, 'momentum', 0, 0, 31);
    const m = model as unknown as Internals;
    // Larger random weights so that ReLU units are active and gradients are informative.
    let seed = 7;
    const rnd = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648 - 0.5;
    };
    m.embedding = m.embedding.map((r) => r.map(() => rnd()));
    m.wHidden = m.wHidden.map((r) => r.map(() => rnd()));
    m.wOutput = m.wOutput.map((r) => r.map(() => rnd()));
    m.bHidden = m.bHidden.map(() => 0.1 + rnd() * 0.1);

    const ctx = [1, 1, 4, 5]; // BOS appears twice
    const target = 6;
    const before = snapshot(m);

    const cache = await m.forward(ctx, false);
    await m.backward(ctx, target, cache);
    const after = snapshot(m);
    restore(m, before);

    const h = 1e-6;
    const check = async (
      get: () => number,
      set: (v: number) => void,
      analytic: number,
      label: string
    ) => {
      const orig = get();
      set(orig + h);
      const lp = await lossAt(m, ctx, target);
      set(orig - h);
      const lm = await lossAt(m, ctx, target);
      set(orig);
      const numeric = (lp - lm) / (2 * h);
      expect(Math.abs(analytic - numeric), label).toBeLessThan(1e-6 + 1e-4 * Math.abs(numeric));
    };

    for (const key of ['wHidden', 'wOutput', 'embedding'] as const) {
      for (let i = 0; i < before[key].length; i++) {
        for (let j = 0; j < before[key][i].length; j++) {
          const analytic = before[key][i][j] - after[key][i][j];
          await check(
            () => m[key][i][j],
            (v) => (m[key][i][j] = v),
            analytic,
            `${key}[${i}][${j}]`
          );
        }
      }
    }
    for (const key of ['bHidden', 'bOutput'] as const) {
      for (let i = 0; i < before[key].length; i++) {
        const analytic = before[key][i] - after[key][i];
        await check(
          () => m[key][i],
          (v) => (m[key][i] = v),
          analytic,
          `${key}[${i}]`
        );
      }
    }
  });

  it('applies exactly one optimizer step per unique context token (Adam)', async () => {
    const lr = 0.01;
    const model = new ProNeuralLM(VOCAB, 6, lr, 4, 'adam', 0.9, 0, 3);
    const m = model as unknown as Internals;
    const ctx = [1, 1, 1, 4]; // BOS repeated three times
    const before = m.embedding[1].slice();
    const cache = await m.forward(ctx, false);
    await m.backward(ctx, 5, cache);
    // First Adam step moves each coordinate by ≈ lr (m̂/√v̂ = ±1). Applying the update once per
    // occurrence would move it by ≈ 3·lr instead.
    m.embedding[1].forEach((v, i) => {
      const delta = Math.abs(v - before[i]);
      if (delta > 0) expect(delta).toBeLessThan(lr * 1.01);
    });
  });
});
