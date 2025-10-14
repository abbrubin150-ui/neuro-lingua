import assert from 'node:assert/strict';
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
  assert.equal(a.length, b.length, 'matrix row mismatch');
  for (let i = 0; i < a.length; i++) {
    assert.equal(a[i].length, b[i].length, 'matrix column mismatch');
    for (let j = 0; j < a[i].length; j++) {
      assert.ok(Math.abs(a[i][j] - b[i][j]) < eps, `matrix diff at ${i},${j}`);
    }
  }
}

function vectorsClose(a: Vector, b: Vector, eps = 1e-9) {
  assert.equal(a.length, b.length, 'vector length mismatch');
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps, `vector diff at ${i}`);
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

(function regressionTest() {
  const vocab = ['<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
  const text = 'hello world';
  const key = 'test-pro-neural-lm';

  const original = new ProNeuralLM(vocab, 8, 0.05, 2, 'adam', 0.9, 0, 42);
  original.train(text, 1);
  original.saveToLocalStorage(key);

  const resumed = ProNeuralLM.loadFromLocalStorage(key);
  assert.ok(resumed, 'model should load');

  const seqs: [number[], number][] = (original as any).createTrainingSequences(text);
  const step = (model: ProNeuralLM) => {
    for (const [ctx, tgt] of seqs) {
      const cache = (model as any).forward(ctx, true);
      (model as any).backward(ctx, tgt, cache);
    }
  };

  // Continue training deterministically for one more pass on both instances.
  step(original);
  step(resumed!);

  modelStateClose(original, resumed!);
  optimizerStateClose(original, resumed!);
  assert.equal((original as any).adamT, (resumed as any).adamT, 'adamT should persist');
})();

console.log('ProNeuralLM serialization regression test passed');
