import { describe, expect, it } from 'vitest';
import {
  base64ToBytes,
  bytesToBase64,
  fromFloat16Bits,
  loadModelFromPackage,
  packModel,
  renderModelCard,
  toFloat16Bits,
  unpackModel,
  type ModelCard
} from '../../src/lib/modelPackage';
import { parseModelImport } from '../../src/lib/modelImport';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];
const CORPUS = 'a b c d a b c d a c b d a b d c';
const CTX = [1, 4, 5, 6];

async function models(): Promise<ProNeuralLM[]> {
  const list: ProNeuralLM[] = [
    new ProNeuralLM(VOCAB, 32, 0.01, 4, 'adam', 0.9, 0, 1),
    new AdvancedNeuralLM(VOCAB, 32, 0.01, 4, 'adam', 0.9, 0, 2, undefined, {
      activation: 'gelu',
      useLayerNorm: true
    }),
    new TransformerLM(VOCAB, 32, 0.01, 4, 'adam', 0.9, 0, 3, undefined, {
      numLayers: 2,
      numHeads: 4,
      numKVHeads: 2
    })
  ];
  for (const m of list) await m.train(CORPUS, 2);
  return list;
}

describe('binary helpers', () => {
  it('base64 matches Node Buffer for every length remainder', () => {
    for (const n of [0, 1, 2, 3, 4, 5, 31, 256]) {
      const bytes = Uint8Array.from({ length: n }, (_, i) => (i * 37 + 11) & 255);
      const encoded = bytesToBase64(bytes);
      expect(encoded).toBe(Buffer.from(bytes).toString('base64'));
      expect(Array.from(base64ToBytes(encoded))).toEqual(Array.from(bytes));
    }
  });

  it('float16 round-trips representable values and bounds the relative error', () => {
    for (const v of [0, 1, -2, 0.5, 65504, -0.25, 2 ** -24, 2 ** -14]) {
      expect(fromFloat16Bits(toFloat16Bits(v))).toBe(v);
    }
    expect(fromFloat16Bits(toFloat16Bits(1e6))).toBe(Infinity);
    expect(Number.isNaN(fromFloat16Bits(toFloat16Bits(NaN)))).toBe(true);
    for (let i = 1; i < 500; i++) {
      const v = Math.sin(i) * 10 ** ((i % 6) - 3);
      const r = fromFloat16Bits(toFloat16Bits(v));
      expect(Math.abs(r - v)).toBeLessThanOrEqual(Math.abs(v) * 2 ** -11 + 2 ** -25);
    }
  });
});

describe('model packages', () => {
  it('restore every architecture (float32 and float16) and shrink the file', async () => {
    for (const model of await models()) {
      const json = model.toJSON() as unknown as Record<string, unknown>;
      const reference = await model.getLogitsForContext(CTX);
      const rawSize = JSON.stringify(json).length;

      const f32 = packModel(json);
      const r32 = loadModelFromPackage(JSON.parse(JSON.stringify(f32)));
      expect(r32.constructor).toBe(model.constructor);
      (await r32.getLogitsForContext(CTX)).forEach((v, i) =>
        expect(Math.abs(v - reference[i])).toBeLessThan(1e-4)
      );
      expect(JSON.stringify(f32).length * 3.5).toBeLessThan(rawSize); // measured: 3.9x (transformer) to 10x (MLP)

      const f16 = packModel(json, { precision: 'float16' });
      const r16 = loadModelFromPackage(JSON.parse(JSON.stringify(f16)));
      (await r16.getLogitsForContext(CTX)).forEach((v, i) =>
        expect(Math.abs(v - reference[i])).toBeLessThan(0.05)
      );
      expect(JSON.stringify(f16).length * 1.7).toBeLessThan(JSON.stringify(f32).length);
    }
  });

  it('keeps optimizer state only when asked', async () => {
    const [mlp] = await models();
    const json = mlp.toJSON() as unknown as Record<string, unknown>;
    expect(unpackModel(packModel(json))).not.toHaveProperty('aWOutput');
    expect(unpackModel(packModel(json, { includeOptimizerState: true }))).toHaveProperty('aWOutput');
  });

  it('detects corrupted tensor data', async () => {
    const [mlp] = await models();
    const pkg = packModel(mlp.toJSON() as unknown as Record<string, unknown>);
    const tampered = JSON.parse(JSON.stringify(pkg));
    const data: string = tampered.model.wOutput.data;
    tampered.model.wOutput.data = (data[0] === 'A' ? 'B' : 'A') + data.slice(1);
    expect(() => unpackModel(tampered)).toThrow(/checksum/);
  });

  it('is accepted by the Import parser and carries the model card', async () => {
    const [, , transformer] = await models();
    const card: ModelCard = {
      name: 'test',
      createdAt: '2026-11-08T00:00:00Z',
      architecture: 'transformer',
      parameters: transformer.getParametersCount(),
      vocabularySize: VOCAB.length,
      contextSize: 4,
      tokenizer: { mode: 'unicode' },
      provenance: { offline: true, externalServices: [], statement: 'Trained locally.' }
    };
    const pkg = packModel(transformer.toJSON() as unknown as Record<string, unknown>, { card });
    const parsed = parseModelImport(JSON.parse(JSON.stringify(pkg)));
    expect(parsed.format).toBe('package');
    expect(parsed.architecture).toBe('transformer');
    expect(parsed.card?.name).toBe('test');
    expect(renderModelCard(card)).toContain('External services**: none');
  });
});
