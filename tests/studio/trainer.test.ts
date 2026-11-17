import { describe, expect, it } from 'vitest';
import { DEFAULT_STUDIO_SETTINGS, structuredCloneSettings } from '../../src/studio/settings';
import { TrainError, trainModel, type TrainEvent } from '../../src/studio/trainer';
import { generateText, importModelText, exportPackage, summarizeModel } from '../../src/studio/modelIO';
import { TransformerLM } from '../../src/lib/TransformerLM';

const SENTENCES = [
  'the cat sat on the mat.',
  'the dog ran in the park.',
  'a bird sang in the tree.',
  'the cat ran after the bird.',
  'the dog sat by the door.'
];
// Enough sentences for a held-out split (every 10th sentence, at least 200 characters).
const LONG_TEXT = Array.from({ length: 120 }, (_, i) => SENTENCES[i % SENTENCES.length]).join(' ');

function mlpSettings() {
  const s = structuredCloneSettings(DEFAULT_STUDIO_SETTINGS);
  s.architecture = 'feedforward';
  s.tokenizer = 'unicode';
  s.hiddenSize = 16;
  s.contextSize = 3;
  s.epochs = 3;
  return s;
}

describe('Studio trainer', () => {
  it('trains a new model with a held-out split and reports every epoch', async () => {
    const events: TrainEvent[] = [];
    const reports: number[] = [];
    const outcome = await trainModel(
      { text: LONG_TEXT, settings: mlpSettings(), mode: 'new', current: null, origin: undefined },
      {
        signal: new AbortController().signal,
        onEvent: (e) => events.push(e),
        onEpoch: (r) => reports.push(r.epoch)
      }
    );
    expect(events[0].kind).toBe('split');
    expect(events.some((e) => e.kind === 'init')).toBe(true);
    expect(events.some((e) => e.kind === 'kept' || e.kind === 'best')).toBe(true);
    expect(reports).toEqual([1, 2, 3].slice(0, outcome.epochsRun));
    expect(outcome.validationBpc).toBeGreaterThan(0);
    expect(outcome.origin.training?.epochs).toBe(outcome.epochsRun);
    expect(outcome.reason).not.toBe('stopped');
  });

  it('continues the same model and refuses a text with unknown words', async () => {
    const settings = mlpSettings();
    settings.validation = false;
    const first = await trainModel(
      { text: LONG_TEXT, settings, mode: 'new', current: null, origin: undefined },
      { signal: new AbortController().signal }
    );
    const before = first.model.getTrainingHistory().length;
    const second = await trainModel(
      { text: LONG_TEXT, settings, mode: 'continue', current: first.model, origin: first.origin },
      { signal: new AbortController().signal }
    );
    expect(second.model).toBe(first.model);
    expect(second.model.getTrainingHistory().length).toBe(before + settings.epochs);

    await expect(
      trainModel(
        {
          text: LONG_TEXT + ' zebras juggle quietly',
          settings,
          mode: 'continue',
          current: first.model,
          origin: first.origin
        },
        { signal: new AbortController().signal }
      )
    ).rejects.toMatchObject({ code: 'vocab-mismatch' });
  });

  it('stops when aborted', async () => {
    const controller = new AbortController();
    const settings = mlpSettings();
    settings.epochs = 50;
    const outcome = await trainModel(
      { text: LONG_TEXT, settings, mode: 'new', current: null, origin: undefined },
      {
        signal: controller.signal,
        onEpoch: (r) => {
          if (r.epoch === 2) controller.abort();
        }
      }
    );
    expect(outcome.reason).toBe('stopped');
    expect(outcome.epochsRun).toBe(2);
  });

  it('rejects empty text', async () => {
    await expect(
      trainModel(
        { text: '  ', settings: mlpSettings(), mode: 'new', current: null, origin: undefined },
        { signal: new AbortController().signal }
      )
    ).rejects.toBeInstanceOf(TrainError);
  });

  it('trains a small BPE transformer, generates, and round-trips through a package', async () => {
    const s = structuredCloneSettings(DEFAULT_STUDIO_SETTINGS);
    s.hiddenSize = 16;
    s.contextSize = 8;
    s.epochs = 1;
    s.bpeVocabSize = 120;
    s.validation = false;
    s.transformer = { ...s.transformer, numLayers: 1, numHeads: 2, numKVHeads: 1, ffHiddenDim: 32 };
    const outcome = await trainModel(
      { text: LONG_TEXT, settings: s, mode: 'new', current: null, origin: undefined },
      { signal: new AbortController().signal }
    );
    expect(outcome.model).toBeInstanceOf(TransformerLM);
    expect(summarizeModel(outcome.model)).toMatchObject({ layers: 1, heads: 2, tokenizer: 'bpe' });

    const g = { ...s.generation, sampling: 'greedy' as const, maxTokens: 10 };
    const generated = await generateText(outcome.model, 'the cat', g);
    expect(generated.tokens).toBeLessThanOrEqual(10);

    const pkg = exportPackage(outcome.model, outcome.origin);
    const restored = importModelText(pkg.text, pkg.fileName);
    expect(restored.format).toBe('package');
    expect(restored.model.getVocab()).toEqual(outcome.model.getVocab());
  });
});
