import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';
import { createProNeuralLMAdapter } from '../../src/lib/expandable/ProNeuralLMAdapter';
import { createAdvancedNeuralLMAdapter } from '../../src/lib/expandable/AdvancedNeuralLMAdapter';
import { createTransformerLMAdapter } from '../../src/lib/expandable/TransformerLMAdapter';
import { InjectionEngine } from '../../src/lib/expandable/InjectionEngine';
import type { InjectionTarget } from '../../src/types/injection';

/**
 * Cerebro neuron injection must leave every model in a working state: shapes consistent,
 * training and generation still possible, and rollback restoring the exact model.
 */

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];
const CORPUS = 'a b c d a b c d a c b d a b d c';
const CTX = [1, 4, 5];

async function trainsAfterInjection(model: ProNeuralLM) {
  const result = await model.train(CORPUS, 2);
  expect(Number.isFinite(result.loss)).toBe(true);
  const logits = await model.getLogitsForContext(CTX);
  expect(logits.every(Number.isFinite)).toBe(true);
  expect(typeof (await model.generate('a', 3))).toBe('string');
}

describe('neuron injection integrity', () => {
  it('ProNeuralLM: injection preserves predictions, training continues, rollback is exact', async () => {
    const model = new ProNeuralLM(VOCAB, 16, 0.05, 3, 'adam', 0.9, 0, 1);
    await model.train(CORPUS, 3);
    const adapter = createProNeuralLMAdapter(model, 'ff');
    const snapshot = adapter.exportWeights();
    const before = await model.getLogitsForContext(CTX);

    adapter.inject(4, 'random_he');
    expect(model.getHiddenSize()).toBe(20);
    const after = await model.getLogitsForContext(CTX);
    after.forEach((v, i) => expect(v).toBeCloseTo(before[i], 12)); // function-preserving

    await trainsAfterInjection(model);

    adapter.importWeights(snapshot);
    expect(model.getHiddenSize()).toBe(16);
    const restored = await model.getLogitsForContext(CTX);
    restored.forEach((v, i) => expect(v).toBeCloseTo(before[i], 5)); // Float32 snapshot
  });

  it('AdvancedNeuralLM (GELU, LayerNorm): training continues and rollback is exact', async () => {
    const model = new AdvancedNeuralLM(VOCAB, 16, 0.05, 3, 'adam', 0.9, 0, 2, undefined, {
      activation: 'gelu',
      useLayerNorm: true
    });
    await model.train(CORPUS, 3);
    const adapter = createAdvancedNeuralLMAdapter(model, 'adv');
    const snapshot = adapter.exportWeights();
    const before = await model.getLogitsForContext(CTX);

    adapter.inject(3, 'random_he');
    await trainsAfterInjection(model);

    adapter.importWeights(snapshot);
    const restored = await model.getLogitsForContext(CTX);
    restored.forEach((v, i) => expect(v).toBeCloseTo(before[i], 5));
  });

  it.each([
    ['MHA', 4],
    ['GQA', 2],
    ['MQA', 1]
  ])('TransformerLM (%s): injection keeps heads, trains, and rolls back exactly', async (_, kv) => {
    const model = new TransformerLM(VOCAB, 16, 0.01, 3, 'adam', 0.9, 0, 3, undefined, {
      numLayers: 2,
      numHeads: 4,
      numKVHeads: kv,
      ffHiddenDim: 12
    });
    await model.train(CORPUS, 2);
    const adapter = createTransformerLMAdapter(model, 'tf');
    const snapshot = adapter.exportWeights();
    const before = await model.getLogitsForContext(CTX);

    adapter.inject(8, 'random_he');
    const info = model.getTransformerInfo();
    expect(model.getHiddenSize()).toBe(24);
    expect(info.numHeads).toBe(4);
    expect(info.numKVHeads).toBe(kv);
    const w = model.getTransformerWeights();
    expect(w.attentionWeights[0].key[0].length).toBe(kv * 6); // kvDim = numKVHeads · headDim
    await trainsAfterInjection(model);

    adapter.importWeights(snapshot);
    expect(model.getHiddenSize()).toBe(16);
    expect(model.getTransformerInfo().numKVHeads).toBe(kv);
    const restored = await model.getLogitsForContext(CTX);
    restored.forEach((v, i) => expect(v).toBeCloseTo(before[i], 4));
  });

  it('TransformerLM.expandHiddenLayer with k not divisible by the head count stays usable', async () => {
    const model = new TransformerLM(VOCAB, 16, 0.01, 3, 'adam', 0.9, 0, 4, undefined, {
      numLayers: 1,
      numHeads: 4,
      numKVHeads: 2
    });
    model.expandHiddenLayer(2); // 18 is not divisible by 4 heads → layout re-derived
    const info = model.getTransformerInfo();
    expect(18 % info.numHeads).toBe(0);
    expect(info.numHeads % info.numKVHeads).toBe(0);
    await trainsAfterInjection(model);
  });

  it('InjectionEngine rounds transformer proposals up to a multiple of the head count', () => {
    const engine = new InjectionEngine();
    const target: InjectionTarget = {
      modelId: 'm',
      layerId: 'transformer-0',
      type: 'transformer',
      dModel: 16,
      hiddenSize: 16,
      metadata: { numLayers: 2, numHeads: 4, ffHiddenDim: 32, vocabSize: 8, contextSize: 3 }
    };
    const proposal = engine.propose(
      { meanResidual: 0.2, tracePerp: 1, estimatedGain: 0.1, suggestedK: 5 },
      target
    );
    expect(proposal.k).toBe(8);
  });
});
