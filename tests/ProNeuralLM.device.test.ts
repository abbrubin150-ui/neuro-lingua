import { describe, expect, it } from 'vitest';

import { GPUNeuralOps } from '../src/backend/gpu_neural_ops';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';

type TrainingMetrics = {
  loss: number;
  perplexity: number;
};

const corpus = 'hello world hello ai hello world ai models';
const vocab = [
  '<PAD>',
  '<BOS>',
  '<EOS>',
  '<UNK>',
  ...Array.from(new Set(ProNeuralLM.tokenizeText(corpus)))
];

async function trainWithDevice(gpuOps: GPUNeuralOps | null): Promise<TrainingMetrics> {
  const model = new ProNeuralLM(vocab, 6, 0.05, 2, 'momentum', 0.9, 0, 1234);

  if (gpuOps) {
    model.setGPUOps(gpuOps);
  }

  const result = await model.train(corpus, 3);
  return { loss: result.loss, perplexity: Math.exp(result.loss) };
}

describe('ProNeuralLM device parity', () => {
  it('keeps loss and perplexity consistent across CPU and GPU (when available)', async () => {
    const cpuMetrics = await trainWithDevice(null);

    const gpuOps = new GPUNeuralOps();
    const gpuReady = await gpuOps.initialize();

    if (!gpuReady || !gpuOps.isEnabled()) {
      // Graceful skip when WebGPU is unavailable in the environment
      expect(gpuOps.isEnabled()).toBe(false);
      return;
    }

    const gpuMetrics = await trainWithDevice(gpuOps);

    // Metrics should stay within a small tolerance across devices
    expect(gpuMetrics.loss).toBeCloseTo(cpuMetrics.loss, 1);
    expect(gpuMetrics.perplexity).toBeCloseTo(cpuMetrics.perplexity, 1);
  });
});
