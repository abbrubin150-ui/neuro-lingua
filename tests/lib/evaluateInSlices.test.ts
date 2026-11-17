import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { evaluateInSlices } from '../../src/lib/evaluateInSlices';
import { TransformerLM } from '../../src/lib/TransformerLM';

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

describe('fast (chunked) scoring of a causal transformer', () => {
  const corpus =
    'the cat sat on the mat . the dog sat on the log . a bird sat on the wall . '.repeat(6);
  const heldOut = 'the bird sat on the mat . a cat sat on the log . the dog sat on the wall .';

  async function transformer(objective: 'causal' | 'pooled') {
    const tokenizer = { mode: 'unicode' as const };
    const vocab = ProNeuralLM.buildVocabulary(corpus, tokenizer);
    const model = new TransformerLM(vocab, 16, 0.01, 8, 'adam', 0.9, 0, 5, tokenizer, {
      numLayers: 1,
      numHeads: 2,
      ffHiddenDim: 32,
      attentionDropout: 0,
      dropConnectRate: 0,
      objective
    });
    await model.train(corpus, 2);
    return model;
  }

  it('scores every token once and stays close to the exact loss', async () => {
    const model = await transformer('causal');
    const exact = await model.evaluate(heldOut);
    const fast = await evaluateInSlices(model, heldOut, null, { fast: true });
    expect(fast!.tokens).toBe(exact.tokens);
    expect(Number.isFinite(fast!.loss)).toBe(true);
    // Same model, same targets; only the context of some tokens is shorter (at least half).
    expect(Math.abs(fast!.loss - exact.loss) / exact.loss).toBeLessThan(0.15);
  });

  it('gives the same sums whatever the slice size', async () => {
    const model = await transformer('causal');
    const whole = model.evaluateChunkedSums(heldOut)!;
    const a = await evaluateInSlices(model, heldOut, null, { fast: true, sliceWindows: 1 });
    const b = await evaluateInSlices(model, heldOut, null, { fast: true, sliceWindows: 3 });
    expect(a!.loss).toBeCloseTo(whole.nll / whole.count, 12);
    expect(b!.loss).toBeCloseTo(a!.loss, 12);
    expect(a!.tokens).toBe(whole.count);
  });

  it('does not change the model or its random stream', async () => {
    const model = await transformer('causal');
    const before = JSON.stringify(model.toJSON());
    model.evaluateChunkedSums(heldOut);
    expect(JSON.stringify(model.toJSON())).toBe(before);
  });

  it('falls back to exact scoring for the pooled objective and for MLPs', async () => {
    const pooled = await transformer('pooled');
    expect(pooled.evaluateChunkedSums(heldOut)).toBeNull();
    const exact = await pooled.evaluate(heldOut);
    const fast = await evaluateInSlices(pooled, heldOut, null, { fast: true });
    expect(fast!.loss).toBeCloseTo(exact.loss, 12);

    const mlp = await trainedModel();
    const mlpExact = await mlp.evaluate('the cat sat on the log .');
    const mlpFast = await evaluateInSlices(mlp, 'the cat sat on the log .', null, { fast: true });
    expect(mlpFast!.loss).toBeCloseTo(mlpExact.loss, 12);
  });
});
