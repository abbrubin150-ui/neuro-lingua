import { describe, expect, it } from 'vitest';
import { GPUNeuralOps } from '../src/backend/gpu_neural_ops';
import { stableSoftmax } from '../src/lib/MathUtils';

type TensorLike = {
  data: Float32Array;
  shape: number[];
  size: number;
  toArray: () => Promise<Float32Array>;
  dispose: () => void;
};

class FakeGPUTensor implements TensorLike {
  readonly data: Float32Array;
  readonly shape: number[];
  readonly size: number;

  constructor(values: Float32Array, shape: number[]) {
    this.data = values;
    this.shape = shape;
    this.size = shape.reduce((a, b) => a * b, 1);
  }

  async toArray(): Promise<Float32Array> {
    return this.data;
  }

  dispose(): void {
    // no-op for fake tensors
  }
}

class FakeGPUBackend {
  async createTensor(data: Float32Array, shape: number[]): Promise<FakeGPUTensor> {
    return new FakeGPUTensor(new Float32Array(data), shape);
  }

  async matMul(a: TensorLike, b: TensorLike): Promise<FakeGPUTensor> {
    const [m, kA] = a.shape;
    const [kB, n] = b.shape;
    if (kA !== kB) {
      throw new Error('Inner dimensions must match for matMul.');
    }

    const out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < kA; k++) {
          sum += a.data[i * kA + k] * b.data[k * n + j];
        }
        out[i * n + j] = sum;
      }
    }

    return new FakeGPUTensor(out, [m, n]);
  }
}

const ABS_TOLERANCE = 1e-5;
const REL_TOLERANCE = 1e-4;

function expectClose(actual: number, expected: number) {
  const diff = Math.abs(actual - expected);
  const rel = Math.abs(expected) > 0 ? diff / Math.abs(expected) : diff;
  expect(diff).toBeLessThanOrEqual(ABS_TOLERANCE + REL_TOLERANCE * Math.abs(expected));
  expect(rel).toBeLessThanOrEqual(REL_TOLERANCE * 10);
}

async function createGpuOps(): Promise<GPUNeuralOps> {
  const ops = new GPUNeuralOps();
  (ops as any).backend = new FakeGPUBackend();
  (ops as any).enabled = true;
  (ops as any).available = true;
  return ops;
}

async function computeAttention(
  ops: GPUNeuralOps,
  queries: number[][],
  keys: number[][],
  values: number[][]
): Promise<{ weights: number[][]; context: number[][] }> {
  const keyT = keys[0] ? keys[0].map((_, j) => keys.map((row) => row[j])) : [];
  const scores = await ops.matrixMultiply(queries, keyT);
  const scale = 1 / Math.sqrt(keys[0]?.length ?? 1);
  const scaled = scores.map((row) => row.map((v) => v * scale));
  const weights = scaled.map((row) => stableSoftmax(row));
  const context = await ops.matrixMultiply(weights, values);
  return { weights, context };
}

describe('Transformer core kernel parity (CPU vs GPU)', () => {
  it('matches matrix multiplication outputs within tolerance', async () => {
    const gpuOps = await createGpuOps();
    const cpuOps = new GPUNeuralOps();

    const A = [
      [1.5, -2.0, 0.75],
      [0.5, 1.25, -1.0]
    ];
    const B = [
      [2.0, -1.0],
      [0.1, 0.5],
      [1.5, 2.0]
    ];

    const gpuResult = await gpuOps.matrixMultiply(A, B);
    const cpuResult = await cpuOps.matrixMultiply(A, B);

    expect(gpuResult.length).toBe(cpuResult.length);
    for (let i = 0; i < gpuResult.length; i++) {
      for (let j = 0; j < gpuResult[i].length; j++) {
        expectClose(gpuResult[i][j], cpuResult[i][j]);
      }
    }
  });

  it('aligns scaled dot-product attention outputs', async () => {
    const gpuOps = await createGpuOps();
    const cpuOps = new GPUNeuralOps();

    const queries = [
      [0.1, 0.3, -0.2],
      [0.05, -0.1, 0.4]
    ];
    const keys = [
      [0.2, 0.25, -0.15],
      [0.15, 0.05, 0.35]
    ];
    const values = [
      [0.4, -0.2],
      [-0.1, 0.3]
    ];

    const gpuAttention = await computeAttention(gpuOps, queries, keys, values);
    const cpuAttention = await computeAttention(cpuOps, queries, keys, values);

    expect(gpuAttention.weights.length).toBe(cpuAttention.weights.length);
    for (let i = 0; i < gpuAttention.weights.length; i++) {
      for (let j = 0; j < gpuAttention.weights[i].length; j++) {
        expectClose(gpuAttention.weights[i][j], cpuAttention.weights[i][j]);
      }
    }

    for (let i = 0; i < gpuAttention.context.length; i++) {
      for (let j = 0; j < gpuAttention.context[i].length; j++) {
        expectClose(gpuAttention.context[i][j], cpuAttention.context[i][j]);
      }
    }
  });

  it('keeps activation outputs consistent after matmul', async () => {
    const gpuOps = await createGpuOps();
    const cpuOps = new GPUNeuralOps();

    const inputs = [
      [0.2, -0.4],
      [1.0, 0.5]
    ];
    const weights = [
      [0.5, -0.3, 0.8],
      [1.2, 0.1, -0.6]
    ];

    const gpuMatmul = await gpuOps.matrixMultiply(inputs, weights);
    const cpuMatmul = await cpuOps.matrixMultiply(inputs, weights);

    const gpuActivated = gpuMatmul.map((row) => row.map((v) => Math.max(0, v)));
    const cpuActivated = cpuMatmul.map((row) => row.map((v) => Math.max(0, v)));

    for (let i = 0; i < gpuActivated.length; i++) {
      for (let j = 0; j < gpuActivated[i].length; j++) {
        expectClose(gpuActivated[i][j], cpuActivated[i][j]);
      }
    }
  });
});
