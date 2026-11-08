import { afterAll, describe, expect, it } from 'vitest';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import type { AddressInfo } from 'node:net';
import { runModelFactory, evaluateText, type FactoryConfig } from '../../src/factory/pipeline';
import { listCorpusFiles, readCorpusDocuments } from '../../src/factory/node/files';
import { createInferenceServer } from '../../src/factory/node/server';
import { loadModelFromPackage, unpackModel } from '../../src/lib/modelPackage';
import { TransformerLM } from '../../src/lib/TransformerLM';

const SENTENCES = [
  'the quick brown fox jumps over the lazy dog',
  'a quick red fox runs past the sleeping dog',
  'the lazy dog sleeps under the old tree',
  'a brown dog chases the quick red fox',
  'the old tree stands near the quiet river',
  'foxes and dogs play near the quiet river'
];
const DOCS = Array.from({ length: 12 }, (_, d) => ({
  id: `doc${d}`,
  text: SENTENCES.map((s, i) => `${s} number ${d * 6 + i}.`).join('\n\n')
}));

function config(overrides: Partial<FactoryConfig['training']> = {}): FactoryConfig {
  return {
    name: 'test-model',
    architecture: 'transformer',
    tokenizer: { mode: 'bpe', vocabSize: 120 },
    model: {
      hiddenSize: 16,
      contextSize: 8,
      transformer: { numLayers: 1, numHeads: 2, ffHiddenDim: 16, trainingStride: 8 }
    },
    training: { epochs: 3, learningRate: 0.01, warmupSteps: 5, seed: 3, ...overrides },
    data: { validationFraction: 0.2 }
  };
}

const temp: string[] = [];
afterAll(async () => {
  for (const dir of temp) await rm(dir, { recursive: true, force: true });
});

describe('model factory pipeline', () => {
  it('produces a validated, packaged model with an offline model card', async () => {
    const logs: string[] = [];
    const result = await runModelFactory(DOCS, config(), { onLog: (m) => logs.push(m) });
    expect(result.history.length).toBe(3);
    expect(result.history[2].validationLoss).toBeLessThan(result.history[0].validationLoss);
    expect(result.card.provenance).toMatchObject({ offline: true, externalServices: [] });
    expect(result.card.tokenizer).toMatchObject({ mode: 'bpe' });
    // The package restores the best checkpoint exactly.
    const restored = loadModelFromPackage(JSON.parse(JSON.stringify(result.package)));
    expect(restored).toBeInstanceOf(TransformerLM);
    const ctx = restored.getContextWindow('the quick brown');
    const a = await restored.getLogitsForContext(ctx);
    const b = await result.model.getLogitsForContext(ctx);
    a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 4));
    const metrics = await evaluateText(result.model, 'the quick brown fox runs past the river');
    expect(Number.isFinite(metrics.bitsPerCharacter)).toBe(true);
    expect(logs.some((l) => l.includes('best'))).toBe(true);
  });

  it('stops early when validation stops improving and keeps the best epoch', async () => {
    const result = await runModelFactory(DOCS, config({ epochs: 6, learningRate: 0, patience: 1 }));
    expect(result.history.length).toBe(2); // no improvement after epoch 1
    expect(result.bestEpoch).toBe(1);
  });

  it('continues training an existing model and keeps its tokenizer', async () => {
    const first = await runModelFactory(DOCS, config({ epochs: 1 }));
    const json = unpackModel(first.package);
    const second = await runModelFactory(
      DOCS.slice(0, 6),
      config({ epochs: 1 }),
      {},
      TransformerLM.fromJSON(json)
    );
    expect(second.card.tokenizer).toMatchObject({ reusedFromInitialModel: true });
    expect(second.model.getVocab()).toEqual(first.model.getVocab());
  });
});

describe('corpus files', () => {
  it('lists supported files recursively and skips hidden / dependency folders', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'nl-corpus-'));
    temp.push(dir);
    await mkdir(path.join(dir, 'sub'));
    await mkdir(path.join(dir, 'node_modules'));
    await mkdir(path.join(dir, '.hidden'));
    await writeFile(path.join(dir, 'a.txt'), 'alpha text');
    await writeFile(path.join(dir, 'sub', 'b.md'), '# beta');
    await writeFile(path.join(dir, 'sub', 'c.jsonl'), '{"text":"gamma"}\n{"text":"delta"}');
    await writeFile(path.join(dir, 'image.png'), 'binary');
    await writeFile(path.join(dir, 'node_modules', 'x.txt'), 'dependency');
    await writeFile(path.join(dir, '.hidden', 'y.txt'), 'secret');
    const files = await listCorpusFiles([dir]);
    expect(files.map((f) => path.relative(dir, f))).toEqual(['a.txt', 'sub/b.md', 'sub/c.jsonl']);
    const { documents } = await readCorpusDocuments([dir]);
    expect(documents.map((d) => d.text)).toEqual(['alpha text', '# beta', 'gamma', 'delta']);
  });
});

describe('local inference server', () => {
  it('serves completions and chat completions from the local model only', async () => {
    const { model } = await runModelFactory(DOCS, config({ epochs: 1 }));
    for (const cors of [false, true]) {
      const server = createInferenceServer(model, { modelName: 'local-test', cors });
      await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
      const { port } = server.address() as AddressInfo;
      const base = `http://127.0.0.1:${port}`;
      try {
        const health = await fetch(`${base}/health`);
        expect(await health.json()).toEqual({ status: 'ok' });
        expect(health.headers.get('access-control-allow-origin')).toBe(cors ? '*' : null);

        const models = await (await fetch(`${base}/v1/models`)).json();
        expect(models.data[0].id).toBe('local-test');

        const completion = await (
          await fetch(`${base}/v1/completions`, {
            method: 'POST',
            body: JSON.stringify({ prompt: 'the quick', max_tokens: 5, temperature: 0.5 })
          })
        ).json();
        expect(completion.object).toBe('text_completion');
        expect(typeof completion.choices[0].text).toBe('string');
        expect(completion.usage.prompt_tokens).toBeGreaterThan(0);

        const chat = await (
          await fetch(`${base}/v1/chat/completions`, {
            method: 'POST',
            body: JSON.stringify({ messages: [{ role: 'user', content: 'the dog' }], max_tokens: 4 })
          })
        ).json();
        expect(chat.choices[0].message.role).toBe('assistant');

        // The same seed gives the same sampled completion.
        const seeded = async () =>
          (
            await (
              await fetch(`${base}/v1/completions`, {
                method: 'POST',
                body: JSON.stringify({ prompt: 'the', max_tokens: 6, temperature: 1.5, seed: 11 })
              })
            ).json()
          ).choices[0].text;
        expect(await seeded()).toBe(await seeded());

        expect((await fetch(`${base}/v1/completions`, { method: 'POST', body: '{}' })).status).toBe(400);
        expect((await fetch(`${base}/nope`)).status).toBe(404);
      } finally {
        await new Promise((resolve) => server.close(resolve));
      }
    }
  });
});
