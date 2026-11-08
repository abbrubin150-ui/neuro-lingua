// @vitest-environment node
import { describe, expect, it } from 'vitest';
import {
  runModelFactory,
  type FactoryCheckpoint,
  type FactoryConfig,
  type TrainingBackendFactory
} from '../../src/factory/pipeline';
import { dataParallelBackend } from '../../src/factory/node/parallel';
import {
  decodeTensors,
  encodeTensors,
  packModel,
  restoreModel,
  unpackModel,
  type ModelPackage
} from '../../src/lib/modelPackage';

const DOCS = [
  {
    id: 'a.txt',
    text: Array.from(
      { length: 30 },
      (_, i) =>
        `Paragraph ${i}: the quick brown fox ${i % 2 ? 'jumps over' : 'runs past'} the lazy dog near the river bank.`
    ).join('\n\n')
  },
  {
    id: 'b.md',
    format: 'markdown' as const,
    text: Array.from(
      { length: 20 },
      (_, i) => `## Note ${i}\n\nA small model learns ${i % 3 ? 'words' : 'letters'} from local text only.`
    ).join('\n\n')
  }
];

const CONFIG: FactoryConfig = {
  name: 'resume-test',
  architecture: 'transformer',
  tokenizer: { mode: 'bpe', vocabSize: 120 },
  model: {
    hiddenSize: 16,
    contextSize: 8,
    dropout: 0.1,
    transformer: {
      numLayers: 1,
      numHeads: 2,
      numKVHeads: 1,
      ffHiddenDim: 32,
      attentionDropout: 0.1,
      dropConnectRate: 0.1,
      batchSize: 4,
      trainingStride: 8,
      weightDecay: 0.01
    }
  },
  training: { epochs: 4, learningRate: 0.01, warmupSteps: 3, patience: 0, seed: 7 },
  data: { validationFraction: 0.2, minParagraphChars: 10 }
};

/** What the CLI writes after an epoch: a lossless checkpoint package, the state, the best model. */
interface Saved {
  model: string;
  checkpoint: string;
  best: string;
}

function persist(model: { toJSON(): unknown }, checkpoint: FactoryCheckpoint, best: unknown): Saved {
  return {
    model: JSON.stringify(
      packModel(model.toJSON() as Record<string, unknown>, {
        precision: 'float64',
        // As the CLI does: a transformer's optimizer state travels in the checkpoint itself.
        includeOptimizerState: false
      })
    ),
    checkpoint: JSON.stringify(encodeTensors(checkpoint)),
    best: JSON.stringify(best)
  };
}

const withoutTime = (json: unknown) =>
  JSON.stringify(json, (key, value) =>
    key === 'timestamp' || key === 'lastUpdatedAt' ? undefined : value
  );
const historyWithoutTime = (history: { seconds: number }[]) =>
  history.map(({ seconds: _seconds, ...rest }) => rest);

async function straight(backend?: TrainingBackendFactory) {
  return runModelFactory(DOCS, CONFIG, {}, undefined, backend ? { backend } : {});
}

async function interruptedAndResumed(after: number, backend?: TrainingBackendFactory) {
  let saved: Saved | null = null;
  let bestJson: unknown = null;
  await expect(
    runModelFactory(
      DOCS,
      CONFIG,
      {
        onEpoch: (report, model, checkpoint) => {
          if (report.improved) bestJson = JSON.parse(JSON.stringify(model.toJSON()));
          if (report.epoch === after) {
            saved = persist(model, checkpoint, bestJson);
            throw new Error('interrupted');
          }
        }
      },
      undefined,
      backend ? { backend } : {}
    )
  ).rejects.toThrow('interrupted');

  // A new process would read these files back.
  const files = saved as unknown as Saved;
  const model = restoreModel(unpackModel(JSON.parse(files.model) as ModelPackage));
  const checkpoint = decodeTensors(JSON.parse(files.checkpoint)) as FactoryCheckpoint;
  return runModelFactory(DOCS, CONFIG, {}, undefined, {
    ...(backend ? { backend } : {}),
    resume: { model, checkpoint, bestModel: JSON.parse(files.best) }
  });
}

describe('resuming an interrupted factory run', () => {
  it('in-process: interrupted after epoch 2 and resumed equals an uninterrupted run, bit for bit', async () => {
    const a = await straight();
    const b = await interruptedAndResumed(2);
    // The compared model was trained after the interruption.
    expect(a.bestEpoch).toBeGreaterThan(2);
    expect(withoutTime(b.model.toJSON())).toBe(withoutTime(a.model.toJSON()));
    expect(historyWithoutTime(b.history)).toEqual(historyWithoutTime(a.history));
    expect(b.bestEpoch).toBe(a.bestEpoch);
    expect(b.package.checksum).toBe(a.package.checksum);
  }, 120000);

  it('data-parallel (2 workers): resumed run equals an uninterrupted one, bit for bit', async () => {
    const backend = dataParallelBackend({ workers: 2 });
    const a = await straight(backend);
    const b = await interruptedAndResumed(3, backend);
    expect(a.bestEpoch).toBeGreaterThan(3);
    expect(withoutTime(b.model.toJSON())).toBe(withoutTime(a.model.toJSON()));
    expect(historyWithoutTime(b.history)).toEqual(historyWithoutTime(a.history));
  }, 120000);

  it('refuses to resume on different data', async () => {
    let saved: Saved | null = null;
    await expect(
      runModelFactory(DOCS, CONFIG, {
        onEpoch: (_report, model, checkpoint) => {
          saved = persist(model, checkpoint, model.toJSON());
          throw new Error('interrupted');
        }
      })
    ).rejects.toThrow('interrupted');
    const files = saved as unknown as Saved;
    const changed = [...DOCS, { id: 'c.txt', text: 'An extra paragraph of new training text.' }];
    await expect(
      runModelFactory(changed, CONFIG, {}, undefined, {
        resume: {
          model: restoreModel(unpackModel(JSON.parse(files.model) as ModelPackage)),
          checkpoint: decodeTensors(JSON.parse(files.checkpoint)) as FactoryCheckpoint,
          bestModel: JSON.parse(files.best)
        }
      })
    ).rejects.toThrow('differs from the interrupted run');
  }, 120000);
});
