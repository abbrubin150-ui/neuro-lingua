import { describe, expect, it } from 'vitest';
import { estimateInformationPlane } from '../../src/losses/information_bottleneck';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';

describe('estimateInformationPlane (binning estimator)', () => {
  it('I(Z;Y) = H(Y) = ln K when the representation determines the label', () => {
    const K = 4;
    const reps: number[][] = [];
    const labels: number[] = [];
    for (let i = 0; i < 400; i++) {
      const y = i % K;
      reps.push([y, 3 - y]); // distinct, well separated codes per class
      labels.push(y);
    }
    const est = estimateInformationPlane(reps, labels, 0.5, 8);
    expect(est.predictionMI).toBeCloseTo(Math.log(K), 10);
    expect(est.compressionMI).toBeCloseTo(Math.log(K), 10); // 4 distinct symbols
    expect(est.ibLoss).toBeCloseTo(-Math.log(K) + 0.5 * Math.log(K), 10);
    expect(est.samples).toBe(400);
  });

  it('a constant representation carries no information', () => {
    const reps = Array.from({ length: 50 }, () => [0.3, 0.3]);
    const labels = Array.from({ length: 50 }, (_, i) => i % 5);
    const est = estimateInformationPlane(reps, labels, 1);
    expect(est.compressionMI).toBe(0);
    expect(est.predictionMI).toBeCloseTo(0, 12);
  });

  it('is bounded by ln(N) and by H(Y)', () => {
    let s = 1;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const reps = Array.from({ length: 64 }, () => [rnd(), rnd(), rnd()]);
    const labels = Array.from({ length: 64 }, () => Math.floor(rnd() * 3));
    const est = estimateInformationPlane(reps, labels, 1, 20);
    expect(est.compressionMI).toBeLessThanOrEqual(Math.log(64) + 1e-12);
    expect(est.predictionMI).toBeLessThanOrEqual(Math.log(3) + 1e-12);
  });

  it('collects representations from every architecture', async () => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c'];
    const text = 'a b c a b c a b c';
    const models = [
      new ProNeuralLM(vocab, 8, 0.05, 2, 'adam', 0.9, 0, 1),
      new AdvancedNeuralLM(vocab, 8, 0.05, 2, 'adam', 0.9, 0, 1),
      new TransformerLM(vocab, 8, 0.01, 2, 'adam', 0.9, 0, 1, undefined, {
        numLayers: 1,
        numHeads: 2
      })
    ];
    for (const model of models) {
      const { representations, labels } = await model.collectRepresentations(text, 5);
      expect(representations.length).toBe(labels.length);
      expect(representations.length).toBeLessThanOrEqual(5);
      expect(representations[0].length).toBe(8);
      const est = estimateInformationPlane(representations, labels, 0.1);
      expect(Number.isFinite(est.compressionMI)).toBe(true);
    }
  });
});
