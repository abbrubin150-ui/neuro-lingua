import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

describe('Sophia hyper-parameters reach the optimizer', () => {
  it('setSophiaConfig replaces the optimizer configuration (lr stays the model LR)', () => {
    const model = new ProNeuralLM(['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a'], 8, 0.002, 2, 'sophia');
    model.setSophiaConfig({ beta1: 0.9, beta2: 0.95, rho: 0.5, hessianUpdateFreq: 3 });
    const config = (model as unknown as { sophiaOptimizer: { config: Record<string, number> } })
      .sophiaOptimizer.config;
    expect(config).toMatchObject({ beta1: 0.9, beta2: 0.95, rho: 0.5, hessianUpdateFreq: 3 });
    expect(config.lr).toBe(0.002);
  });

  it('is a no-op for other optimizers', () => {
    const model = new ProNeuralLM(['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a'], 8, 0.002, 2, 'adam');
    model.setSophiaConfig({ rho: 0.5 });
    expect((model as unknown as { sophiaOptimizer: unknown }).sophiaOptimizer).toBeNull();
  });
});
