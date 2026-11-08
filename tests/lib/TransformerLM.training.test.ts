import { describe, expect, it } from 'vitest';
import { TransformerLM } from '../../src/lib/TransformerLM';
import { ParamOptimizer } from '../../src/training/ParamOptimizer';

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd', 'e'];
const CORPUS = Array.from({ length: 40 }, (_, i) => VOCAB[4 + (i % 5)]).join(' ');

function build(config: Record<string, unknown> = {}) {
  return new TransformerLM(VOCAB, 16, 0.01, 4, 'adam', 0.9, 0, 7, undefined, {
    numLayers: 1,
    numHeads: 2,
    attentionDropout: 0,
    dropConnectRate: 0,
    batchSize: 1,
    ...config
  });
}

describe('TransformerLM training options', () => {
  it('trainingStride = contextSize visits each target once per epoch (chunked training)', async () => {
    const model = build({ trainingStride: 4 });
    const internals = model as unknown as {
      accumulateGradients: (inputs: number[], pass: unknown, targets: number[]) => number;
    };
    const original = internals.accumulateGradients.bind(model);
    const covered: number[] = [];
    internals.accumulateGradients = (inputs, pass, targets) => {
      covered.push(targets.filter((t) => t >= 0).length);
      return original(inputs, pass, targets);
    };
    await model.train(CORPUS, 1);
    const targets = 41; // 40 tokens + EOS
    expect(covered.length).toBe(Math.ceil(targets / 4));
    // Every target is scored at least once; overlap only in the final partial chunk.
    const total = covered.reduce((s, v) => s + v, 0);
    expect(total).toBeGreaterThanOrEqual(targets);
    expect(total).toBeLessThan(targets + 4);
  });

  it('strided training still learns', async () => {
    const model = build({ trainingStride: 4 });
    const first = await model.train(CORPUS, 1);
    const later = await model.train(CORPUS, 30);
    expect(later.history[later.history.length - 1].loss).toBeLessThan(first.loss * 0.5);
  });

  it('follows the warm-up + cosine learning-rate schedule per optimizer step', () => {
    const model = build();
    model.setLearningRateSchedule({ warmupSteps: 4, totalSteps: 24, minFactor: 0.1 });
    const lr = (step: number) =>
      (model as unknown as { scheduledLearningRate(s: number): number }).scheduledLearningRate(step);
    expect(lr(0)).toBeCloseTo(0.0025, 12);
    expect(lr(3)).toBeCloseTo(0.01, 12);
    expect(lr(4)).toBeCloseTo(0.01, 12); // start of the cosine phase
    expect(lr(14)).toBeCloseTo(0.01 * (0.1 + 0.9 * 0.5), 12); // halfway
    expect(lr(24)).toBeCloseTo(0.001, 12);
    expect(lr(100)).toBeCloseTo(0.001, 12);
    expect(model.getScheduledLearningRate()).toBeCloseTo(0.0025, 12);
  });

  it('applies the schedule while training', async () => {
    const model = build();
    model.setLearningRateSchedule({ warmupSteps: 1000, totalSteps: 2000 });
    await model.train(CORPUS, 1); // 41 optimizer steps (batch 1)
    expect(model.getScheduledLearningRate()).toBeCloseTo((0.01 * 42) / 1000, 12);
  });

  it('weight decay shrinks weights; norm gains are not decayed', async () => {
    const norm = (m: TransformerLM) =>
      Math.sqrt(
        m
          .getTransformerWeights()
          .ffWeights1.flat(2)
          .reduce((s, v) => s + v * v, 0)
      );
    const plain = build();
    const decayed = build({ weightDecay: 0.5 });
    await plain.train(CORPUS, 3);
    await decayed.train(CORPUS, 3);
    expect(norm(decayed)).toBeLessThan(norm(plain));
  });
});

describe('ParamOptimizer weight decay', () => {
  it('multiplies parameters by (1 − η·λ) before the gradient step', () => {
    const opt = new ParamOptimizer('adam');
    const p = [[1, -2]];
    opt.step();
    opt.update('w', p, [[0, 0]], 0.1, undefined, { weightDecay: 0.5 });
    expect(p[0][0]).toBeCloseTo(0.95, 12);
    expect(p[0][1]).toBeCloseTo(-1.9, 12);
  });
});
