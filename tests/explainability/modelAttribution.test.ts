import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';
import { exactShapley, explainNextToken } from '../../src/explainability/modelAttribution';
import { attentionRollout } from '../../src/explainability/attentionRollout';

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];
const CORPUS = 'a b c d a b c d a c b d a b d c a b c d';

async function trained(): Promise<ProNeuralLM[]> {
  const models: ProNeuralLM[] = [
    new ProNeuralLM(VOCAB, 12, 0.01, 4, 'adam', 0.9, 0, 1),
    new AdvancedNeuralLM(VOCAB, 12, 0.01, 4, 'adam', 0.9, 0, 2, undefined, {
      activation: 'gelu',
      useLayerNorm: true
    }),
    new TransformerLM(VOCAB, 8, 0.01, 4, 'adam', 0.9, 0, 3, undefined, {
      numLayers: 2,
      numHeads: 2,
      ffHiddenDim: 8,
      attentionDropout: 0,
      dropConnectRate: 0
    }),
    new TransformerLM(VOCAB, 8, 0.01, 4, 'adam', 0.9, 0, 4, undefined, {
      numLayers: 1,
      numHeads: 2,
      objective: 'pooled',
      attentionDropout: 0,
      dropConnectRate: 0
    })
  ];
  for (const m of models) await m.train(CORPUS, 3);
  return models;
}

describe('model attributions', () => {
  it('explainLogProb gradients match finite differences for every architecture', async () => {
    for (const model of await trained()) {
      const rows = model.getEmbeddingRows([1, 4, 5, 6]);
      const { logProb, gradient } = model.explainLogProb(rows, 7);
      expect(Number.isFinite(logProb)).toBe(true);
      const h = 1e-6;
      for (let t = 0; t < rows.length; t++) {
        for (let i = 0; i < rows[t].length; i++) {
          const plus = rows.map((r) => r.slice());
          const minus = rows.map((r) => r.slice());
          plus[t][i] += h;
          minus[t][i] -= h;
          const numeric =
            (model.explainLogProb(plus, 7).logProb - model.explainLogProb(minus, 7).logProb) /
            (2 * h);
          expect(Math.abs(gradient[t][i] - numeric)).toBeLessThan(1e-6 + 1e-4 * Math.abs(numeric));
        }
      }
    }
  });

  it('Shapley values sum exactly to the total effect; IG satisfies completeness', async () => {
    for (const model of await trained()) {
      const shap = await explainNextToken(model, 'a b c', 'shap');
      const sum = shap.scores.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(shap.totalEffect!, 10);

      const ig = await explainNextToken(model, 'a b c', 'integrated-gradients', { steps: 400 });
      const igSum = ig.scores.reduce((s, v) => s + v, 0);
      expect(Math.abs(igSum - ig.totalEffect!)).toBeLessThan(0.02 + 0.02 * Math.abs(ig.totalEffect!));
      expect(ig.tokens).toEqual(['<BOS>', 'a', 'b', 'c']);
    }
  });

  it('attention rollout uses the transformer attention and sums to 1', async () => {
    const [, , causal, pooled] = await trained();
    for (const model of [causal, pooled]) {
      const r = await explainNextToken(model, 'a b c', 'attention-rollout');
      expect(r.scores.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 6);
      r.scores.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));
    }
    const [mlp] = await trained();
    await expect(explainNextToken(mlp, 'a b', 'attention-rollout')).rejects.toThrow(/Transformer/);
  });
});

describe('exactShapley', () => {
  it('recovers additive weights and splits a pure interaction equally', () => {
    const w = [0.5, -1, 2];
    const additive = exactShapley(3, (mask) =>
      w.reduce((s, wi, i) => s + (mask & (1 << i) ? wi : 0), 0)
    );
    additive.forEach((v, i) => expect(v).toBeCloseTo(w[i], 12));
    const interaction = exactShapley(3, (mask) => ((mask & 0b011) === 0b011 ? 1 : 0));
    expect(interaction[0]).toBeCloseTo(0.5, 12);
    expect(interaction[1]).toBeCloseTo(0.5, 12);
    expect(interaction[2]).toBeCloseTo(0, 12);
  });
});

describe('attentionRollout', () => {
  it('composes layers as Â_L ⋯ Â_1 with the residual ½A + ½I', () => {
    // Layer 1: position 1 attends fully to position 0. Layer 2: position 0 attends to 1.
    const layer1 = [
      [1, 0],
      [1, 0]
    ];
    const layer2 = [
      [0, 1],
      [0, 1]
    ];
    const r = attentionRollout([[layer1], [layer2]], { addIdentity: true });
    // Â1 = [[1,0],[.5,.5]], Â2 = [[.5,.5],[0,1]]  →  Â2·Â1 = [[.75,.25],[.5,.5]]
    expect(r[0][0]).toBeCloseTo(0.75, 5);
    expect(r[0][1]).toBeCloseTo(0.25, 5);
    expect(r[1][0]).toBeCloseTo(0.5, 5);
    expect(r[1][1]).toBeCloseTo(0.5, 5);
  });
});
