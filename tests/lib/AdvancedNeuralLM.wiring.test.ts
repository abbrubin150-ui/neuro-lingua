import { beforeEach, describe, expect, it } from 'vitest';
import { AdvancedNeuralLM, type AdvancedConfig } from '../../src/lib/AdvancedNeuralLM';
import { cosineAnnealingLR } from '../../src/lib/MathUtils';

/**
 * These tests verify that every AdvancedNeuralLM option acts on the standard train()/forward()
 * path used by the UI (previously activation and LayerNorm were never applied and the LR
 * schedule / weight decay only ran inside trainAdvanced(), which the UI never called).
 */

type Internals = {
  embedding: number[][];
  wHidden: number[][];
  wOutput: number[][];
  bHidden: number[];
  bOutput: number[];
  layerNormGamma: number[];
  layerNormBeta: number[];
  forward(inputs: number[], train?: boolean): Promise<{ probs: number[]; logits: number[] }>;
  backward(inputs: number[], target: number, cache: unknown): Promise<void>;
};

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];
const CORPUS = 'a b c d a b c d a c b d a b d c';

function randomize(m: Internals, seed: number) {
  let s = seed;
  const rnd = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648 - 0.5;
  };
  m.embedding = m.embedding.map((r) => r.map(() => rnd()));
  m.wHidden = m.wHidden.map((r) => r.map(() => rnd()));
  m.wOutput = m.wOutput.map((r) => r.map(() => rnd()));
  m.bHidden = m.bHidden.map(() => rnd() * 0.2);
  if (m.layerNormGamma.length) {
    m.layerNormGamma = m.layerNormGamma.map(() => 1 + rnd() * 0.5);
    m.layerNormBeta = m.layerNormBeta.map(() => rnd() * 0.2);
  }
}

function makeModel(config: AdvancedConfig, optimizer: 'momentum' | 'adam' = 'momentum') {
  return new AdvancedNeuralLM(VOCAB, 6, 1, 3, optimizer, 0, 0, 11, undefined, {
    initialization: 'default',
    weightDecay: 0,
    gradientClipNorm: 0,
    lrSchedule: 'constant',
    ...config
  });
}

describe('AdvancedNeuralLM wiring', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it.each(['gelu', 'elu', 'leaky_relu'] as const)(
    'backprop through %s + LayerNorm matches finite differences',
    async (activation) => {
      const model = makeModel({ activation, useLayerNorm: true });
      const m = model as unknown as Internals;
      randomize(m, 5);
      const ctx = [1, 4, 4];
      const target = 6;

      const snap = () => ({
        wHidden: m.wHidden.map((r) => r.slice()),
        wOutput: m.wOutput.map((r) => r.slice()),
        embedding: m.embedding.map((r) => r.slice()),
        bHidden: m.bHidden.slice(),
        bOutput: m.bOutput.slice(),
        layerNormGamma: m.layerNormGamma.slice(),
        layerNormBeta: m.layerNormBeta.slice()
      });
      const before = snap();
      await m.backward(ctx, target, await m.forward(ctx, false));
      const after = snap();
      Object.assign(m, {
        wHidden: before.wHidden.map((r) => r.slice()),
        wOutput: before.wOutput.map((r) => r.slice()),
        embedding: before.embedding.map((r) => r.slice()),
        bHidden: before.bHidden.slice(),
        bOutput: before.bOutput.slice(),
        layerNormGamma: before.layerNormGamma.slice(),
        layerNormBeta: before.layerNormBeta.slice()
      });

      const loss = async () => -Math.log((await m.forward(ctx, false)).probs[target]);
      const h = 1e-6;
      const numeric = async (get: () => number, set: (v: number) => void) => {
        const o = get();
        set(o + h);
        const lp = await loss();
        set(o - h);
        const lm = await loss();
        set(o);
        return (lp - lm) / (2 * h);
      };

      for (const key of ['wHidden', 'wOutput', 'embedding'] as const) {
        for (let i = 0; i < before[key].length; i++) {
          for (let j = 0; j < before[key][i].length; j++) {
            const analytic = before[key][i][j] - after[key][i][j];
            const num = await numeric(
              () => m[key][i][j],
              (v) => (m[key][i][j] = v)
            );
            expect(Math.abs(analytic - num), `${key}[${i}][${j}]`).toBeLessThan(
              1e-6 + 1e-4 * Math.abs(num)
            );
          }
        }
      }
      for (const key of [
        'bHidden',
        'bOutput',
        'layerNormGamma',
        'layerNormBeta'
      ] as const) {
        for (let i = 0; i < before[key].length; i++) {
          const analytic = before[key][i] - after[key][i];
          const num = await numeric(
            () => m[key][i],
            (v) => (m[key][i] = v)
          );
          expect(Math.abs(analytic - num), `${key}[${i}]`).toBeLessThan(
            1e-6 + 1e-4 * Math.abs(num)
          );
        }
      }
    }
  );

  it('the configured activation changes the forward pass', async () => {
    const relu = makeModel({ activation: 'relu' });
    const gelu = makeModel({ activation: 'gelu' });
    randomize(relu as unknown as Internals, 9);
    randomize(gelu as unknown as Internals, 9);
    const ctx = [1, 4, 5];
    const a = await relu.getLogitsForContext(ctx);
    const b = await gelu.getLogitsForContext(ctx);
    expect(a).not.toEqual(b);
  });

  it('LayerNorm is applied in forward and its parameters are trained by train()', async () => {
    const model = makeModel({ useLayerNorm: true }, 'adam');
    (model as unknown as { learningRate: number }).learningRate = 0.01;
    const m = model as unknown as Internals;
    const gammaBefore = m.layerNormGamma.slice();
    const plain = makeModel({ useLayerNorm: false });
    randomize(m, 3);
    randomize(plain as unknown as Internals, 3);
    const ctx = [1, 4, 5];
    expect(await model.getLogitsForContext(ctx)).not.toEqual(
      await plain.getLogitsForContext(ctx)
    );
    await model.train(CORPUS, 2);
    expect(m.layerNormGamma).not.toEqual(gammaBefore);
    expect(model.getParametersCount()).toBe(plain.getParametersCount() + 2 * 6);
  });

  it('follows the LR schedule when trained one epoch per call (UI path)', async () => {
    const base = 0.05;
    const model = new AdvancedNeuralLM(VOCAB, 6, base, 3, 'momentum', 0.9, 0, 1, undefined, {
      lrSchedule: 'cosine',
      lrMin: 0.001,
      weightDecay: 0
    });
    const total = 6;
    model.setTrainingHorizon(total);
    const lrs: number[] = [];
    for (let e = 0; e < total; e++) {
      await model.train(CORPUS, 1);
      lrs.push(model.getLearningRate());
    }
    lrs.forEach((lr, e) => expect(lr).toBeCloseTo(cosineAnnealingLR(e, total, base, 0.001), 12));
    expect(lrs[total - 1]).toBeLessThan(lrs[0]);
  });

  it('applies weight decay in the regular train() path', async () => {
    const norm = (w: number[][]) => Math.sqrt(w.flat().reduce((s, v) => s + v * v, 0));
    const run = async (weightDecay: number) => {
      const model = new AdvancedNeuralLM(VOCAB, 6, 0.05, 3, 'momentum', 0.9, 0, 4, undefined, {
        lrSchedule: 'constant',
        weightDecay
      });
      for (let e = 0; e < 5; e++) await model.train(CORPUS, 1);
      return norm((model as unknown as Internals).wHidden);
    };
    expect(await run(0.5)).toBeLessThan(await run(0));
  });

  it('round-trips through toJSON/fromJSON and localStorage with identical predictions', async () => {
    const model = new AdvancedNeuralLM(VOCAB, 6, 0.05, 3, 'adam', 0.9, 0, 21, undefined, {
      activation: 'gelu',
      useLayerNorm: true,
      lrSchedule: 'cosine'
    });
    model.setTrainingHorizon(4);
    for (let e = 0; e < 2; e++) await model.train(CORPUS, 1);

    const json = model.toJSON();
    expect(json.architecture).toBe('advanced');
    expect(json.learningRate).toBe(0.05); // base rate, not the scheduled value
    const restored = AdvancedNeuralLM.fromJSON(JSON.parse(JSON.stringify(json)));
    const ctx = [1, 4, 5];
    expect(await restored.getLogitsForContext(ctx)).toEqual(await model.getLogitsForContext(ctx));
    expect(restored.getLearningRate()).toBeCloseTo(model.getLearningRate(), 15);

    model.saveToLocalStorage('adv-key');
    const loaded = AdvancedNeuralLM.loadFromLocalStorage('adv-key');
    expect(loaded).not.toBeNull();
    expect(await loaded!.getLogitsForContext(ctx)).toEqual(await model.getLogitsForContext(ctx));
  });

  it('evaluate() reports held-out loss without modifying weights', async () => {
    const model = makeModel({}, 'adam');
    (model as unknown as { learningRate: number }).learningRate = 0.01;
    await model.train(CORPUS, 3);
    const before = JSON.stringify(model.getWeights());
    const result = await model.evaluate('a b c d');
    expect(JSON.stringify(model.getWeights())).toBe(before);
    expect(result.tokens).toBe(5); // 4 tokens + EOS
    expect(result.perplexity).toBeCloseTo(Math.exp(result.loss), 10);
    expect(result.oovRate).toBe(0);
    const unknown = await model.evaluate('a zebra');
    expect(unknown.oovRate).toBeCloseTo(1 / 3, 10);
  });
});
