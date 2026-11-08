// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { TransformerLM, type TransformerConfig } from '../../src/lib/TransformerLM';
import { inProcessBackend } from '../../src/factory/pipeline';
import {
  DataParallelTrainer,
  dataParallelBackend,
  splitEvenly
} from '../../src/factory/node/parallel';

const TEXT = Array.from(
  { length: 12 },
  (_, i) =>
    `the cat ${i % 3 === 0 ? 'sat' : 'ran'} on the mat while the dog ${i % 2 ? 'slept' : 'barked'} near the log.`
).join(' ');
const VALIDATION = 'the dog sat on the log while the cat slept near the mat.';

function build(config: TransformerConfig = {}, dropout = 0) {
  const vocab = ProNeuralLM.buildVocabulary(TEXT + ' ' + VALIDATION);
  return new TransformerLM(vocab, 16, 0.01, 6, 'adam', 0.9, dropout, 11, undefined, {
    numLayers: 1,
    numHeads: 2,
    attentionDropout: 0,
    dropConnectRate: 0,
    batchSize: 4,
    trainingStride: 2,
    ...config
  });
}

const setup = (model: TransformerLM) => ({
  model,
  trainText: TEXT,
  validationText: VALIDATION,
  seed: 5
});

async function trainWith(model: TransformerLM, workers: number, epochs: number) {
  const backend =
    workers > 1
      ? await DataParallelTrainer.start(setup(model), { workers })
      : inProcessBackend(setup(model));
  try {
    const losses: number[] = [];
    for (let e = 0; e < epochs; e++) losses.push((await backend.trainEpoch(e)).loss);
    const evaluation = await backend.evaluate();
    return { losses, evaluation, params: model.exportParameters() };
  } finally {
    await backend.dispose();
  }
}

const maxDiff = (a: Float64Array, b: Float64Array) => {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

describe('splitEvenly', () => {
  it('makes contiguous chunks whose sizes differ by at most one', () => {
    expect(splitEvenly([1, 2, 3, 4, 5], 2)).toEqual([[1, 2, 3], [4, 5]]);
    expect(splitEvenly([1, 2], 4)).toEqual([[1], [2]]);
    expect(splitEvenly([1, 2, 3, 4, 5, 6, 7], 3).map((c) => c.length)).toEqual([3, 2, 2]);
  });
});

describe('data-parallel transformer training', () => {
  it('without dropout, matches single-process training up to summation order', async () => {
    const single = await trainWith(build(), 1, 2);
    const parallel = await trainWith(build(), 3, 2);
    expect(maxDiff(single.params, parallel.params)).toBeLessThan(1e-9);
    parallel.losses.forEach((l, i) => expect(l).toBeCloseTo(single.losses[i], 9));
    expect(parallel.evaluation.tokens).toBe(single.evaluation.tokens);
    expect(parallel.evaluation.loss).toBeCloseTo(single.evaluation.loss, 9);
    expect(parallel.evaluation.accuracy).toBe(single.evaluation.accuracy);
  }, 60000);

  it('with dropout and DropConnect, is reproducible for a given seed and worker count', async () => {
    const config = { attentionDropout: 0.1, dropConnectRate: 0.1 };
    const a = await trainWith(build(config, 0.1), 2, 2);
    const b = await trainWith(build(config, 0.1), 2, 2);
    expect(maxDiff(a.params, b.params)).toBe(0);
    expect(a.losses).toEqual(b.losses);
    // The model still learns.
    expect(a.losses[1]).toBeLessThan(a.losses[0]);
  }, 60000);

  it('records epochs in the model history and keeps the replicas in sync', async () => {
    const model = build();
    const trainer = await DataParallelTrainer.start(setup(model), { workers: 2 });
    try {
      await trainer.trainEpoch(0);
      expect(model.getTrainingHistory()).toHaveLength(1);
      // Distributed evaluation uses the replicas' parameters: equal to the coordinator's model.
      const distributed = await trainer.evaluate();
      const local = await model.evaluate(VALIDATION);
      expect(distributed.loss).toBeCloseTo(local.loss, 10);
      expect(distributed.tokens).toBe(local.tokens);
    } finally {
      await trainer.dispose();
    }
  }, 60000);

  it('the backend factory falls back to in-process training for one worker or an MLP', async () => {
    const mlp = new ProNeuralLM(ProNeuralLM.buildVocabulary(TEXT), 8, 0.01, 3);
    const backend = await dataParallelBackend({ workers: 4 })({
      model: mlp,
      trainText: TEXT,
      validationText: VALIDATION,
      seed: 1
    });
    expect(backend.description).toBe('in-process');
    const single = await dataParallelBackend({ workers: 1 })(setup(build()));
    expect(single.description).toBe('in-process');
  });
});
