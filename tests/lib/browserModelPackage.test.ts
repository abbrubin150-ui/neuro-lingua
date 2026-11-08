import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';
import { trainSubwordModel } from '../../src/tokenizer/subword';
import {
  BROWSER_PROVENANCE_STATEMENT,
  browserProvenance,
  createBrowserModelCard,
  originFromImport,
  packageBrowserModel,
  packageFileName,
  recordTraining,
  textFingerprint
} from '../../src/lib/modelExport';
import { renderModelCard, restoreModel, unpackModel } from '../../src/lib/modelPackage';
import { parseModelImport } from '../../src/lib/modelImport';

const TEXT = 'the cat sat on the mat. the dog sat on the log. the cat saw the dog. a cat sat.';

describe('model provenance in the browser app', () => {
  it('a model initialised and trained in the app is offline', () => {
    const origin = recordTraining(undefined, { continued: false, text: TEXT, epochs: 3 });
    expect(origin.base).toBeUndefined();
    expect(origin.training).toMatchObject({ characters: TEXT.length, epochs: 3 });
    expect(browserProvenance(origin)).toEqual({
      offline: true,
      externalServices: [],
      statement: BROWSER_PROVENANCE_STATEMENT
    });
  });

  it('continued training keeps the provenance of the imported starting model', () => {
    const card = {
      name: 'factory-model',
      createdAt: '2026-01-01T00:00:00.000Z',
      architecture: 'transformer',
      parameters: 1,
      vocabularySize: 1,
      contextSize: 1,
      tokenizer: {},
      provenance: { offline: true, externalServices: [], statement: 'Trained locally.' }
    };
    const imported = originFromImport('factory.nlm.json', card);
    const continued = recordTraining(imported, { continued: true, text: TEXT, epochs: 1 });
    const provenance = browserProvenance(continued);
    expect(provenance.offline).toBe(true);
    expect(provenance.statement).toContain('"factory-model"');
    expect(provenance.statement).toContain('Trained locally.');
    expect(provenance.statement).toMatch(/^Training continued/);
  });

  it('never claims "offline" for a model of unknown origin', () => {
    expect(browserProvenance(originFromImport('weights.json')).offline).toBeNull();
    // Saved by an older version (no origin recorded) or continued from such a model.
    expect(browserProvenance(undefined).offline).toBeNull();
    const continued = recordTraining(undefined, { continued: true, text: TEXT, epochs: 1 });
    expect(browserProvenance(continued).offline).toBeNull();
    // A fresh model trained afterwards is offline again.
    const fresh = recordTraining(continued, { continued: false, text: TEXT, epochs: 1 });
    expect(browserProvenance(fresh).offline).toBe(true);
  });

  it('fingerprints texts without storing them', () => {
    expect(textFingerprint(TEXT)).toMatch(/^[0-9a-f]{16}$/);
    expect(textFingerprint(TEXT)).toBe(textFingerprint(TEXT));
    expect(textFingerprint(TEXT)).not.toBe(textFingerprint(TEXT + ' '));
  });

  it('makes safe package file names', () => {
    expect(packageFileName('Neuro-Lingua transformer 2026/11/08')).toBe(
      'neuro-lingua-transformer-2026-11-08.nlm.json'
    );
    expect(packageFileName('מודל עברי')).toBe('מודל-עברי.nlm.json');
    expect(packageFileName('***')).toBe('model.nlm.json');
  });
});

describe('model packages exported from the app', () => {
  it('packages a trained model with a card that matches the model, and imports it back', async () => {
    const config = { mode: 'bpe' as const, bpe: trainSubwordModel(TEXT, { vocabSize: 60 }) };
    const model = new TransformerLM(
      ProNeuralLM.buildVocabulary(TEXT, config),
      16,
      0.01,
      4,
      'adam',
      0.9,
      0,
      7,
      config,
      { numLayers: 1, numHeads: 2 }
    );
    await model.train(TEXT, 2);
    const origin = recordTraining(undefined, { continued: false, text: TEXT, epochs: 2 });

    const pkg = packageBrowserModel(model, origin, { name: 'demo', appVersion: '4.4.0' });
    const card = pkg.card!;
    expect(card).toMatchObject({
      name: 'demo',
      architecture: 'transformer',
      parameters: model.getParametersCount(),
      vocabularySize: model.getVocabSize(),
      contextSize: model.getContextSize(),
      provenance: { offline: true }
    });
    expect(card.tokenizer).toMatchObject({ mode: 'bpe', merges: config.bpe.merges.length });
    expect(card.training).toMatchObject({ epochsRecorded: 2, lastRunEpochs: 2 });
    expect(card.metrics?.finalEpochTrainingLoss).toBeCloseTo(
      model.getTrainingHistory()[1].loss,
      4
    );
    expect(renderModelCard(card)).toContain('**Offline**: yes');
    expect(pkg.includesOptimizerState).toBe(false);

    // The app's Import path: parse, verify the checksum, restore the class.
    const parsed = parseModelImport(JSON.parse(JSON.stringify(pkg)));
    expect(parsed.format).toBe('package');
    const restored = restoreModel(parsed.modelData, parsed.architecture);
    expect(restored).toBeInstanceOf(TransformerLM);
    const ctx = model.getContextWindow('the cat sat');
    const expected = await model.getLogitsForContext(ctx);
    const actual = await restored.getLogitsForContext(ctx);
    actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 4));
    expect(Object.keys(unpackModel(pkg))).toContain('vocab');
  });

  it('describes an untrained model without inventing metrics', () => {
    const model = new ProNeuralLM(ProNeuralLM.buildVocabulary(TEXT), 8, 0.01, 3);
    const card = createBrowserModelCard(model, undefined, { name: 'untrained' });
    expect(card.metrics).toBeUndefined();
    expect(card.data).toBeUndefined();
    expect(card.training).toEqual({ epochsRecorded: 0 });
    expect(renderModelCard(card)).toContain('**Offline**: unknown');
  });
});
