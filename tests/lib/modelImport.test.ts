import { describe, expect, it } from 'vitest';
import { parseModelImport } from '../../src/lib/modelImport';
import { createTraceExport } from '../../src/lib/traceExport';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c'];
const CORPUS = 'a b c a b c a c b';
const CTX = [1, 4, 5];

async function roundTrip(model: ProNeuralLM, format: 'standard' | 'trace') {
  const json = model.toJSON();
  const file = format === 'trace' ? createTraceExport(json) : json;
  // Simulate saving to disk and reading the file back.
  return parseModelImport(JSON.parse(JSON.stringify(file)));
}

describe('parseModelImport', () => {
  it.each(['standard', 'trace'] as const)(
    'restores all three architectures from the %s format',
    async (format) => {
      const ff = new ProNeuralLM(VOCAB, 8, 0.05, 3, 'adam', 0.9, 0, 1);
      const adv = new AdvancedNeuralLM(VOCAB, 8, 0.05, 3, 'adam', 0.9, 0, 2, undefined, {
        activation: 'gelu',
        useLayerNorm: true
      });
      const tf = new TransformerLM(VOCAB, 8, 0.01, 3, 'adam', 0.9, 0, 3, undefined, {
        numLayers: 1,
        numHeads: 2
      });
      for (const m of [ff, adv, tf]) await m.train(CORPUS, 2);

      const pf = await roundTrip(ff, format);
      expect(pf.format).toBe(format);
      expect(pf.architecture).toBe('feedforward');
      expect(await ProNeuralLM.fromJSON(pf.modelData).getLogitsForContext(CTX)).toEqual(
        await ff.getLogitsForContext(CTX)
      );

      const pa = await roundTrip(adv, format);
      expect(pa.architecture).toBe('advanced');
      expect(await AdvancedNeuralLM.fromJSON(pa.modelData).getLogitsForContext(CTX)).toEqual(
        await adv.getLogitsForContext(CTX)
      );

      const pt = await roundTrip(tf, format);
      expect(pt.architecture).toBe('transformer');
      expect(await TransformerLM.fromJSON(pt.modelData).getLogitsForContext(CTX)).toEqual(
        await tf.getLogitsForContext(CTX)
      );
    }
  );

  it('rejects files that are not model exports with a readable error', () => {
    expect(() => parseModelImport([1, 2, 3])).toThrow(/JSON object/);
    expect(() => parseModelImport({ hello: 'world' })).toThrow(/vocab/);
    expect(() => parseModelImport({ modelWeights: 5, config: {}, exportedAt: 1 })).toThrow(
      /modelWeights/
    );
  });
});
