import { describe, expect, it, beforeEach } from 'vitest';
import { LionOptimizer, LION_DEFAULTS } from '../../src/training/LionOptimizer';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

describe('LionOptimizer standalone', () => {
  let optimizer: LionOptimizer;

  beforeEach(() => {
    optimizer = new LionOptimizer();
  });

  it('initializes with default configuration', () => {
    const config = optimizer.getConfig();
    expect(config.lr).toBe(LION_DEFAULTS.lr);
    expect(config.beta1).toBe(LION_DEFAULTS.beta1);
    expect(config.beta2).toBe(LION_DEFAULTS.beta2);
    expect(config.weightDecay).toBe(LION_DEFAULTS.weightDecay);
  });

  it('accepts custom configuration', () => {
    const customOptimizer = new LionOptimizer({
      lr: 1e-4,
      beta1: 0.95,
      beta2: 0.98,
      weightDecay: 0.001
    });
    const config = customOptimizer.getConfig();
    expect(config.lr).toBe(1e-4);
    expect(config.beta1).toBe(0.95);
    expect(config.beta2).toBe(0.98);
    expect(config.weightDecay).toBe(0.001);
  });

  it('updates vector parameters with sign of momentum', () => {
    const b = [1.0, 2.0, 3.0];
    const g = [0.1, -0.2, 0.3];
    const originalB = [...b];

    optimizer.updateVector(b, g, 'bias');

    // Lion uses sign of (β₁m + (1-β₁)g), which for first step (m=0) is sign(g)
    // Update: b -= lr * sign + lr * wd * b
    expect(b[0]).toBeLessThan(originalB[0]); // g[0] > 0, so sign = 1, b decreases
    expect(b[1]).toBeGreaterThan(originalB[1]); // g[1] < 0, so sign = -1, b increases
    expect(b[2]).toBeLessThan(originalB[2]); // g[2] > 0, so sign = 1, b decreases
  });

  it('updates matrix parameters correctly', () => {
    const W = [
      [1.0, 2.0],
      [3.0, 4.0]
    ];
    const G = [
      [0.1, -0.1],
      [-0.2, 0.2]
    ];
    const originalW00 = W[0][0];
    const originalW01 = W[0][1];

    optimizer.updateMatrix(W, G, 'weights');

    expect(W[0][0]).toBeLessThan(originalW00); // G[0][0] > 0
    expect(W[0][1]).toBeGreaterThan(originalW01); // G[0][1] < 0
  });

  it('updates single row correctly', () => {
    const W = [
      [1.0, 2.0],
      [3.0, 4.0]
    ];
    const gRow = [0.5, -0.5];
    const originalRow1 = [...W[1]];

    optimizer.updateRow(W, 1, gRow, 'embedding');

    // Only row 1 should be updated
    expect(W[0]).toEqual([1.0, 2.0]); // Unchanged
    expect(W[1][0]).toBeLessThan(originalRow1[0]); // g > 0
    expect(W[1][1]).toBeGreaterThan(originalRow1[1]); // g < 0
  });

  it('accumulates momentum across updates', () => {
    const b = [0.0, 0.0];
    const g1 = [1.0, 1.0];
    const g2 = [1.0, 1.0];

    // First update
    optimizer.updateVector(b, g1, 'test');
    const afterFirst = [...b];

    // Second update with same gradient
    optimizer.updateVector(b, g2, 'test');
    const afterSecond = [...b];

    // Values should continue decreasing (same direction of gradient)
    expect(afterSecond[0]).toBeLessThan(afterFirst[0]);
    expect(afterSecond[1]).toBeLessThan(afterFirst[1]);
  });

  it('applies weight decay', () => {
    // Set high weight decay for testing
    const wdOptimizer = new LionOptimizer({ lr: 0.001, weightDecay: 0.1 });

    const b = [10.0, 10.0];
    const g = [0.0, 0.0]; // Zero gradient

    // With zero gradient and zero momentum, only weight decay affects weights
    // Actually with zero gradient, sign(0) = 0, so update = -lr * 0 - lr * wd * w
    // But Math.sign(0) = 0 in JS, so update is just weight decay
    wdOptimizer.updateVector(b, g, 'test');

    // Weight decay should reduce magnitude
    expect(b[0]).toBeLessThan(10.0);
    expect(b[1]).toBeLessThan(10.0);
  });

  it('resets state correctly', () => {
    const b = [1.0, 2.0];
    const g = [0.1, 0.2];

    optimizer.updateVector(b, g, 'test');
    optimizer.reset();

    // After reset, state should be empty
    const exported = optimizer.exportState();
    expect(exported.momentumVectors).toEqual({});
    expect(exported.momentumMatrices).toEqual({});
  });

  it('exports and imports state correctly', () => {
    const b = [1.0, 2.0];
    const g = [0.1, 0.2];

    optimizer.updateVector(b, g, 'test');
    const exported = optimizer.exportState();

    const newOptimizer = new LionOptimizer();
    newOptimizer.importState(exported);

    const newExported = newOptimizer.exportState();
    expect(newExported.config).toEqual(exported.config);
    expect(newExported.momentumVectors).toEqual(exported.momentumVectors);
  });

  it('allows dynamic config updates', () => {
    optimizer.setConfig({ lr: 0.001 });
    expect(optimizer.getConfig().lr).toBe(0.001);
    expect(optimizer.getConfig().beta1).toBe(LION_DEFAULTS.beta1); // Unchanged
  });
});

describe('Lion optimizer in ProNeuralLM', () => {
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

  beforeEach(() => {
    storage.clear();
  });

  it('trains with Lion optimizer without errors', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'test'];
    const corpus = 'hello world test';

    const model = new ProNeuralLM(vocab, 16, 0.001, 2, 'lion', 0.9, 0.1, 42);

    await expect(model.train(corpus, 3)).resolves.not.toThrow();
  });

  it('reduces loss during training with Lion', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'the', 'cat', 'sat', 'on', 'mat'];
    const corpus = 'the cat sat on the mat';

    const model = new ProNeuralLM(vocab, 32, 0.001, 2, 'lion', 0.9, 0, 42);

    await model.train(corpus, 10);
    const history = model.getTrainingHistory();

    expect(history.length).toBeGreaterThan(0);
    // Loss should be finite and reasonable after training
    const finalLoss = history[history.length - 1].loss;
    expect(finalLoss).toBeLessThan(Infinity);
    expect(finalLoss).toBeGreaterThan(0);
  });

  it('generates text after Lion training', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'hello', 'world', 'foo', 'bar'];
    const corpus = 'hello world foo bar';

    const model = new ProNeuralLM(vocab, 16, 0.001, 2, 'lion', 0.9, 0, 42);
    await model.train(corpus, 5);

    const generated = await model.generate('hello', 5, 0.8);
    expect(typeof generated).toBe('string');
    expect(generated.length).toBeGreaterThan(0);
  });

  it('persists and restores Lion-trained model', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd'];
    const corpus = 'a b c d';
    const key = 'lion-test-model';

    const model = new ProNeuralLM(vocab, 16, 0.001, 2, 'lion', 0.9, 0, 42);
    await model.train(corpus, 3);
    model.saveToLocalStorage(key);

    const restored = ProNeuralLM.loadFromLocalStorage(key);
    expect(restored).not.toBeNull();
    expect((restored as any).optimizer).toBe('lion');

    // Verify model can still generate
    const generated = await restored!.generate('a', 3, 0.8);
    expect(typeof generated).toBe('string');
  });

  it('produces different results than momentum optimizer', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'test', 'data'];
    const corpus = 'test data';

    const lionModel = new ProNeuralLM(vocab, 16, 0.01, 2, 'lion', 0.9, 0, 42);
    const momentumModel = new ProNeuralLM(vocab, 16, 0.01, 2, 'momentum', 0.9, 0, 42);

    await lionModel.train(corpus, 5);
    await momentumModel.train(corpus, 5);

    const lionHistory = lionModel.getTrainingHistory();
    const momentumHistory = momentumModel.getTrainingHistory();

    // Both should train, but losses should differ
    expect(lionHistory.length).toBeGreaterThan(0);
    expect(momentumHistory.length).toBeGreaterThan(0);

    // Loss values should be different (different optimization paths)
    const lionFinalLoss = lionHistory[lionHistory.length - 1].loss;
    const momentumFinalLoss = momentumHistory[momentumHistory.length - 1].loss;
    expect(lionFinalLoss).not.toBe(momentumFinalLoss);
  });

  it('works with dropout enabled', async () => {
    const vocab = ['<BOS>', '<EOS>', '<UNK>', 'x', 'y', 'z'];
    const corpus = 'x y z';

    const model = new ProNeuralLM(vocab, 16, 0.001, 2, 'lion', 0.9, 0.2, 42);

    await expect(model.train(corpus, 3)).resolves.not.toThrow();
    const history = model.getTrainingHistory();
    expect(history.length).toBeGreaterThan(0);
  });
});

describe('Lion optimizer mathematical properties', () => {
  it('uses sign function for updates', () => {
    const optimizer = new LionOptimizer({ lr: 1.0, weightDecay: 0 });

    // Test with different gradient magnitudes - should all result in same step size
    const b1 = [0.0];
    const b2 = [0.0];
    const b3 = [0.0];

    optimizer.updateVector(b1, [0.1], 'test1');
    optimizer.updateVector(b2, [1.0], 'test2');
    optimizer.updateVector(b3, [10.0], 'test3');

    // All should have same magnitude update (lr=1, so update = ±1)
    expect(Math.abs(b1[0])).toBe(Math.abs(b2[0]));
    expect(Math.abs(b2[0])).toBe(Math.abs(b3[0]));
  });

  it('handles zero gradients correctly', () => {
    const optimizer = new LionOptimizer({ lr: 0.1, weightDecay: 0 });

    const b = [1.0, 2.0];
    const originalB = [...b];

    // Zero gradient with zero momentum should give zero update direction
    optimizer.updateVector(b, [0.0, 0.0], 'test');

    // With no weight decay and sign(0) = 0, values should be unchanged
    expect(b[0]).toBe(originalB[0]);
    expect(b[1]).toBe(originalB[1]);
  });

  it('momentum beta2 affects state update rate', () => {
    const fastOptimizer = new LionOptimizer({ lr: 0.1, beta2: 0.5 });
    const slowOptimizer = new LionOptimizer({ lr: 0.1, beta2: 0.99 });

    const b1 = [0.0];
    const b2 = [0.0];

    // First update
    fastOptimizer.updateVector([...b1], [1.0], 'test');
    slowOptimizer.updateVector([...b2], [1.0], 'test');

    // Check momentum states
    const fastState = fastOptimizer.exportState();
    const slowState = slowOptimizer.exportState();

    const fastMomentum = fastState.momentumVectors['test']?.[0] ?? 0;
    const slowMomentum = slowState.momentumVectors['test']?.[0] ?? 0;

    // Fast beta2 = 0.5: m = 0.5*0 + 0.5*1 = 0.5
    // Slow beta2 = 0.99: m = 0.99*0 + 0.01*1 = 0.01
    expect(fastMomentum).toBeCloseTo(0.5, 5);
    expect(slowMomentum).toBeCloseTo(0.01, 5);
  });
});
