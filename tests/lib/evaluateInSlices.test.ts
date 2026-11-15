import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { evaluateInSlices } from '../../src/lib/evaluateInSlices';

async function trainedModel() {
  const corpus = 'the cat sat on the mat . the dog sat on the log . '.repeat(8);
  const tokenizer = { mode: 'unicode' as const };
  const vocab = ProNeuralLM.buildVocabulary(corpus, tokenizer);
  const model = new ProNeuralLM(vocab, 16, 0.05, 3, 'adam', 0.9, 0, 11, tokenizer);
  await model.train(corpus, 3);
  return model;
}

describe('evaluateInSlices', () => {
  it('gives the same result as evaluate()', async () => {
    const model = await trainedModel();
    const heldOut = 'the cat sat on the log . the dog sat on the mat .';
    const whole = await model.evaluate(heldOut);
    const sliced = await evaluateInSlices(model, heldOut, null, {
      sliceWindows: 4,
      yieldIntervalMs: 0
    });
    expect(sliced).not.toBeNull();
    expect(sliced!.tokens).toBe(whole.tokens);
    expect(sliced!.loss).toBeCloseTo(whole.loss, 12);
    expect(sliced!.accuracy).toBe(whole.accuracy);
  });

  it('returns null when aborted', async () => {
    const model = await trainedModel();
    const controller = new AbortController();
    controller.abort();
    expect(await evaluateInSlices(model, 'the cat sat on the mat .', controller.signal)).toBeNull();
  });
});
