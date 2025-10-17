import { beforeEach, describe, expect, it } from 'vitest';

import { ProNeuralLM } from '../src/lib/ProNeuralLM';

const storage = new Map<string, string>();
const localStorageMock = {
  getItem(key: string) {
    return storage.has(key) ? storage.get(key)! : null;
  },
  setItem(key: string, value: string) {
    storage.set(key, value);
  },
  removeItem(key: string) {
    storage.delete(key);
  },
  clear() {
    storage.clear();
  },
  key(index: number) {
    return Array.from(storage.keys())[index] ?? null;
  },
  get length() {
    return storage.size;
  }
};

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: false,
  configurable: true
});

type Matrix = number[][];

type Vector = number[];

function matricesClose(a: Matrix, b: Matrix, eps = 1e-9) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(a[i].length).toBe(b[i].length);
    for (let j = 0; j < a[i].length; j++) {
      expect(Math.abs(a[i][j] - b[i][j])).toBeLessThan(eps);
    }
  }
}

function vectorsClose(a: Vector, b: Vector, eps = 1e-9) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThan(eps);
  }
}

function optimizerStateClose(base: ProNeuralLM, other: ProNeuralLM) {
  matricesClose((base as any).mEmbedding, (other as any).mEmbedding);
  matricesClose((base as any).mWHidden, (other as any).mWHidden);
  matricesClose((base as any).mWOutput, (other as any).mWOutput);
  vectorsClose((base as any).mBHidden, (other as any).mBHidden);
  vectorsClose((base as any).mBOutput, (other as any).mBOutput);

  matricesClose((base as any).aEmbedding.m, (other as any).aEmbedding.m);
  matricesClose((base as any).aEmbedding.v, (other as any).aEmbedding.v);
  matricesClose((base as any).aWHidden.m, (other as any).aWHidden.m);
  matricesClose((base as any).aWHidden.v, (other as any).aWHidden.v);
  matricesClose((base as any).aWOutput.m, (other as any).aWOutput.m);
  matricesClose((base as any).aWOutput.v, (other as any).aWOutput.v);
  vectorsClose((base as any).aBHidden.m, (other as any).aBHidden.m);
  vectorsClose((base as any).aBHidden.v, (other as any).aBHidden.v);
  vectorsClose((base as any).aBOutput.m, (other as any).aBOutput.m);
  vectorsClose((base as any).aBOutput.v, (other as any).aBOutput.v);
}

function modelStateClose(base: ProNeuralLM, other: ProNeuralLM) {
  matricesClose((base as any).embedding, (other as any).embedding);
  matricesClose((base as any).wHidden, (other as any).wHidden);
  matricesClose((base as any).wOutput, (other as any).wOutput);
  vectorsClose((base as any).bHidden, (other as any).bHidden);
  vectorsClose((base as any).bOutput, (other as any).bOutput);
}

describe('ProNeuralLM persistence', () => {
  beforeEach(() => {
    storage.clear();
  });

  it('restores optimizer and model state deterministically', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
    const text = 'hello world';
    const key = 'test-pro-neural-lm';

    const original = new ProNeuralLM(vocab, 8, 0.05, 2, 'adam', 0.9, 0, 42);
    original.train(text, 1);
    original.saveToLocalStorage(key);

    const resumed = ProNeuralLM.loadFromLocalStorage(key);
    expect(resumed).toBeTruthy();

    const seqs: [number[], number][] = (original as any).createTrainingSequences(text);
    const step = (model: ProNeuralLM) => {
      for (const [ctx, tgt] of seqs) {
        const cache = (model as any).forward(ctx, true);
        (model as any).backward(ctx, tgt, cache);
      }
    };

    step(original);
    step(resumed!);

    modelStateClose(original, resumed!);
    optimizerStateClose(original, resumed!);
    expect((resumed as any).adamT).toEqual((original as any).adamT);
  });

  it('restores RNG state and dropout masks', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'foo', 'bar'];
    const text = 'foo bar foo';
    const key = 'test-pro-neural-lm-rng';
    const seed = 2024;
    const dropout = 0.25;

    const original = new ProNeuralLM(vocab, 8, 0.05, 2, 'momentum', 0.9, dropout, seed);
    const seqs: [number[], number][] = (original as any).createTrainingSequences(text);
    expect(seqs.length).toBeGreaterThanOrEqual(2);

    (original as any).forward(seqs[0][0], true);
    original.saveToLocalStorage(key);

    const raw = storage.get(key);
    expect(raw).toBeTruthy();
    const saved = JSON.parse(raw!);
    expect(saved.rngSeed).toEqual(seed >>> 0);
    expect(typeof saved.rngState).toBe('number');

    const resumed = ProNeuralLM.loadFromLocalStorage(key);
    expect(resumed).toBeTruthy();
    expect((resumed as any).rngSeed).toEqual((original as any).rngSeed);
    expect((resumed as any).rngState).toEqual((original as any).rngState);

    const originalCache = (original as any).forward(seqs[1][0], true);
    const resumedCache = (resumed as any).forward(seqs[1][0], true);

    expect(originalCache.dropMask).toBeTruthy();
    expect(resumedCache.dropMask).toBeTruthy();
    vectorsClose(originalCache.dropMask!, resumedCache.dropMask!);
  });

  it('persists tokenizer configuration and metadata', () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'alpha', 'βeta'];
    const key = 'test-tokenizer-config';
    const customPattern = '[^\\p{L}\\d]+';

    const model = new ProNeuralLM(vocab, 8, 0.05, 2, 'momentum', 0.8, 0.1, 99, {
      mode: 'custom',
      pattern: customPattern
    });
    model.train('alpha βeta alpha', 1);
    model.saveToLocalStorage(key);

    const resumed = ProNeuralLM.loadFromLocalStorage(key);
    expect(resumed).toBeTruthy();
    expect(resumed!.getTokenizerConfig()).toEqual({ mode: 'custom', pattern: customPattern });
    const updatedAt = resumed!.getLastUpdatedAt();
    expect(typeof updatedAt).toBe('number');
    expect(updatedAt).toBeGreaterThan(0);
  });
});
