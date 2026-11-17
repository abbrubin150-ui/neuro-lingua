import { describe, expect, it } from 'vitest';
import {
  DEFAULT_STUDIO_SETTINGS,
  applyPreset,
  mergeSettings,
  structuredCloneSettings,
  validateSettings,
  withArchitecture
} from '../../src/studio/settings';

describe('Studio settings', () => {
  it('defaults are valid', () => {
    expect(validateSettings(DEFAULT_STUDIO_SETTINGS)).toEqual([]);
  });

  it('presets are valid and set the transformer size', () => {
    const tiny = applyPreset(DEFAULT_STUDIO_SETTINGS, 'transformer-tiny');
    expect(tiny.transformer).toMatchObject({ numLayers: 2, numHeads: 4, numKVHeads: 2 });
    expect(validateSettings(tiny)).toEqual([]);
    expect(validateSettings(applyPreset(DEFAULT_STUDIO_SETTINGS, 'transformer-small'))).toEqual(
      []
    );
    const mlp = applyPreset(DEFAULT_STUDIO_SETTINGS, 'mlp');
    expect(mlp.architecture).toBe('feedforward');
    expect(validateSettings(mlp)).toEqual([]);
  });

  it('switching architecture follows the default context unless it was changed', () => {
    const mlp = withArchitecture(DEFAULT_STUDIO_SETTINGS, 'feedforward');
    expect(mlp.contextSize).toBe(4);
    const custom = { ...structuredCloneSettings(DEFAULT_STUDIO_SETTINGS), contextSize: 48 };
    expect(withArchitecture(custom, 'feedforward').contextSize).toBe(48);
  });

  it('rejects head counts that do not divide', () => {
    const s = structuredCloneSettings(DEFAULT_STUDIO_SETTINGS);
    s.hiddenSize = 100;
    s.transformer.numHeads = 8;
    s.transformer.numKVHeads = 3;
    const errors = validateSettings(s);
    expect(errors).toContain('hiddenSize % numHeads ≠ 0');
    expect(errors).toContain('numHeads % numKVHeads ≠ 0');
  });

  it('merges stored settings field by field and drops bad values', () => {
    const merged = mergeSettings({
      architecture: 'quantum',
      epochs: 5,
      learningRate: 'fast',
      transformer: { numLayers: 2 },
      generation: { sampling: 'mirostat', noRepeatNgram: 4 }
    });
    expect(merged.architecture).toBe(DEFAULT_STUDIO_SETTINGS.architecture);
    expect(merged.epochs).toBe(5);
    expect(merged.learningRate).toBe(DEFAULT_STUDIO_SETTINGS.learningRate);
    expect(merged.transformer.numLayers).toBe(2);
    expect(merged.transformer.numHeads).toBe(DEFAULT_STUDIO_SETTINGS.transformer.numHeads);
    expect(merged.generation.sampling).toBe(DEFAULT_STUDIO_SETTINGS.generation.sampling);
    expect(merged.generation.noRepeatNgram).toBe(4);
    expect(mergeSettings('garbage')).toEqual(DEFAULT_STUDIO_SETTINGS);
  });
});
