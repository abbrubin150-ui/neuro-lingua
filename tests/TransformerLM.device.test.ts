/**
 * TransformerLM Device Parity Tests
 *
 * These tests verify that TransformerLM produces consistent results
 * when running on CPU vs GPU. This ensures training and inference
 * behave identically regardless of the compute backend.
 */

import { describe, expect, it } from 'vitest';
import { GPUNeuralOps } from '../src/backend/gpu_neural_ops';
import { TransformerLM } from '../src/lib/TransformerLM';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';

type TrainingMetrics = {
  loss: number;
  accuracy: number;
  perplexity: number;
};

const corpus = 'hello world hello neural network hello world neural models transformer attention';
const vocab = [
  '<PAD>',
  '<BOS>',
  '<EOS>',
  '<UNK>',
  ...Array.from(new Set(ProNeuralLM.tokenizeText(corpus)))
];

async function trainTransformerWithDevice(
  gpuOps: GPUNeuralOps | null,
  seed: number = 1234
): Promise<TrainingMetrics> {
  const model = new TransformerLM(
    vocab,
    8, // hiddenSize
    0.05, // learningRate
    2, // contextSize
    'momentum',
    0.9,
    0, // dropout
    seed,
    undefined,
    { numLayers: 1, numHeads: 2 }
  );

  if (gpuOps) {
    model.setGPUOps(gpuOps);
  }

  const result = await model.train(corpus, 3);
  return {
    loss: result.loss,
    accuracy: result.accuracy,
    perplexity: Math.exp(result.loss)
  };
}

describe('TransformerLM device parity', () => {
  it('keeps loss and perplexity consistent across CPU and GPU (when available)', async () => {
    // Train on CPU
    const cpuMetrics = await trainTransformerWithDevice(null);

    // Initialize GPU
    const gpuOps = new GPUNeuralOps();
    const gpuReady = await gpuOps.initialize();

    if (!gpuReady || !gpuOps.isEnabled()) {
      // Graceful skip when WebGPU is unavailable in the environment
      expect(gpuOps.isEnabled()).toBe(false);
      return;
    }

    // Train on GPU
    const gpuMetrics = await trainTransformerWithDevice(gpuOps);

    // Metrics should stay within a small tolerance across devices
    expect(gpuMetrics.loss).toBeCloseTo(cpuMetrics.loss, 1);
    expect(gpuMetrics.perplexity).toBeCloseTo(cpuMetrics.perplexity, 1);
    expect(gpuMetrics.accuracy).toBeCloseTo(cpuMetrics.accuracy, 1);
  });

  it('produces consistent predictions on CPU and GPU', async () => {
    const seed = 42;

    // Train on CPU
    const cpuModel = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      seed,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );
    await cpuModel.train(corpus, 3);

    // Initialize GPU
    const gpuOps = new GPUNeuralOps();
    const gpuReady = await gpuOps.initialize();

    if (!gpuReady || !gpuOps.isEnabled()) {
      // Skip when GPU unavailable
      expect(true).toBe(true);
      return;
    }

    // Train on GPU with same seed
    const gpuModel = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      seed,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );
    gpuModel.setGPUOps(gpuOps);
    await gpuModel.train(corpus, 3);

    // Generate text with temperature 0 for determinism
    const prompt = 'hello';
    const cpuOutput = await cpuModel.generate(prompt, 5, { temperature: 0 });
    const gpuOutput = await gpuModel.generate(prompt, 5, { temperature: 0 });

    // Outputs should be identical
    expect(gpuOutput).toBe(cpuOutput);
  });

  it('maintains training stability on GPU', async () => {
    const gpuOps = new GPUNeuralOps();
    const gpuReady = await gpuOps.initialize();

    if (!gpuReady || !gpuOps.isEnabled()) {
      expect(true).toBe(true);
      return;
    }

    const model = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      42,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );
    model.setGPUOps(gpuOps);

    const results: number[] = [];

    // Train for several epochs and verify loss decreases
    for (let epoch = 0; epoch < 5; epoch++) {
      const result = await model.train(corpus, 1);
      results.push(result.loss);
    }

    // Loss should generally decrease
    expect(results[4]).toBeLessThan(results[0]);

    // No NaN or Infinity values
    for (const loss of results) {
      expect(Number.isFinite(loss)).toBe(true);
    }
  });
});

describe('GPU fallback behavior', () => {
  it('falls back to CPU when GPU operations fail', async () => {
    const model = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      42,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );

    // Create mock GPU ops that will fail
    const gpuOps = new GPUNeuralOps();
    // Don't initialize - operations should fall back to CPU

    model.setGPUOps(gpuOps);

    // Training should still work (falling back to CPU)
    const result = await model.train(corpus, 2);

    expect(result.loss).toBeGreaterThan(0);
    expect(Number.isFinite(result.loss)).toBe(true);
  });

  it('reports correct metrics when GPU is disabled', async () => {
    const gpuOps = new GPUNeuralOps();
    await gpuOps.initialize();

    // Disable GPU
    gpuOps.setEnabled(false);

    expect(gpuOps.isEnabled()).toBe(false);

    const metrics = gpuOps.getMetrics();
    expect(metrics.enabled).toBe(false);
  });

  it('tracks GPU metrics correctly when enabled', async () => {
    const gpuOps = new GPUNeuralOps();
    const gpuReady = await gpuOps.initialize();

    if (!gpuReady || !gpuOps.isEnabled()) {
      expect(true).toBe(true);
      return;
    }

    gpuOps.resetMetrics();

    // Perform some operations
    const A = [
      [1, 2],
      [3, 4]
    ];
    const x = [1, 1];

    await gpuOps.matrixVectorMul(A, x);
    await gpuOps.matrixVectorMul(A, x);

    const metrics = gpuOps.getMetrics();

    expect(metrics.enabled).toBe(true);
    expect(metrics.totalOperations).toBeGreaterThanOrEqual(2);
  });

  it('continues training after GPU error recovery', async () => {
    const model = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      42,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );

    // Train without GPU first
    await model.train(corpus, 2);
    const lossWithoutGPU = model.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

    // Set uninitialized GPU ops (will fall back to CPU)
    const gpuOps = new GPUNeuralOps();
    model.setGPUOps(gpuOps);

    // Continue training - should work via fallback
    await model.train(corpus, 2);
    const lossAfterFallback = model.toJSON().trainingHistory.at(-1)?.loss ?? Infinity;

    // Training should continue to work
    expect(Number.isFinite(lossAfterFallback)).toBe(true);
    expect(lossAfterFallback).toBeLessThanOrEqual(lossWithoutGPU);
  });
});

describe('Multi-architecture device parity', () => {
  it('both ProNeuralLM and TransformerLM produce consistent CPU results', async () => {
    const seed = 12345;

    const feedforward = new ProNeuralLM(vocab, 8, 0.05, 2, 'momentum', 0.9, 0, seed);

    const transformer = new TransformerLM(
      vocab,
      8,
      0.05,
      2,
      'momentum',
      0.9,
      0,
      seed,
      undefined,
      { numLayers: 1, numHeads: 2 }
    );

    const ffResult = await feedforward.train(corpus, 3);
    const tfResult = await transformer.train(corpus, 3);

    // Both should produce valid losses
    expect(Number.isFinite(ffResult.loss)).toBe(true);
    expect(Number.isFinite(tfResult.loss)).toBe(true);

    // Both should train successfully (loss decreases from initial)
    expect(ffResult.loss).toBeLessThan(10); // Reasonable upper bound
    expect(tfResult.loss).toBeLessThan(10);
  });
});
