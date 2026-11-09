import { describe, expect, it } from 'vitest';
import { computeEdgeLearningDiagnostics } from '../../src/backend/edgeLearning';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c'];
const TEXT = 'a b c a c b';

type Internals = {
  wOutput: number[][];
  bOutput: number[];
  wHidden: number[][];
  bHidden: number[];
  embedding: number[][];
  forward(ctx: number[], train?: boolean): Promise<unknown>;
  backward(ctx: number[], tgt: number, cache: unknown): Promise<void>;
  createTrainingSequences(text: string): [number[], number][];
};

describe('computeEdgeLearningDiagnostics', () => {
  it('Fisher trace equals the mean squared output-layer gradient from backprop', async () => {
    const model = new ProNeuralLM(VOCAB, 6, 1, 2, 'momentum', 0, 0, 5);
    await model.train(TEXT, 2); // non-trivial weights (momentum 0, lr 1)
    const m = model as unknown as Internals;

    const diag = await computeEdgeLearningDiagnostics(model, TEXT, { maxSamples: 1000 });

    // Brute force: one SGD step with lr = 1 and momentum = 0 moves W by −∂ℓ/∂W.
    let sumSq = 0;
    const seqs = m.createTrainingSequences(TEXT);
    for (const [ctx, tgt] of seqs) {
      const W = m.wOutput.map((r) => r.slice());
      const b = m.bOutput.slice();
      const Wh = m.wHidden.map((r) => r.slice());
      const bh = m.bHidden.slice();
      const E = m.embedding.map((r) => r.slice());
      await m.backward(ctx, tgt, await m.forward(ctx, false));
      m.wOutput.forEach((row, i) => row.forEach((v, j) => (sumSq += (W[i][j] - v) ** 2)));
      m.bOutput.forEach((v, j) => (sumSq += (b[j] - v) ** 2));
      // restore every parameter so that each window is evaluated on the same model
      Object.assign(m, { wOutput: W, bOutput: b, wHidden: Wh, bHidden: bh, embedding: E });
    }
    const P = 6 * VOCAB.length + VOCAB.length;
    expect(diag.samples).toBe(seqs.length);
    expect(diag.fisherInformation).toBeCloseTo(sumSq / seqs.length / P, 9);
    expect(diag.cramerRaoBound).toBeCloseTo(1 / (seqs.length * diag.fisherInformation), 12);
  });

  it('a uniform predictor has entropy ln V and Fisher trace (‖h‖² + 1)(1 − 1/V)', async () => {
    const model = new ProNeuralLM(VOCAB, 6, 0.01, 2, 'adam', 0.9, 0, 5);
    const m = model as unknown as Internals;
    m.wOutput = m.wOutput.map((r) => r.map(() => 0));
    m.bOutput = m.bOutput.map(() => 0);
    const diag = await computeEdgeLearningDiagnostics(model, TEXT);
    expect(diag.entropy).toBeCloseTo(Math.log(VOCAB.length), 12);
    expect(diag.variance).toBeCloseTo(0, 12); // every loss equals ln V
    const { representations } = await model.collectRepresentations(TEXT);
    const V = VOCAB.length;
    const expected =
      representations.reduce((s, h) => s + (h.reduce((a, v) => a + v * v, 0) + 1), 0) /
      representations.length;
    expect(diag.fisherInformation).toBeCloseTo((expected * (1 - 1 / V)) / (6 * V + V), 12);
  });
});
