import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { WebGPUBackend, WebGPUTensor } from '../../src/backend/webgpu';
import { GPUNeuralOps } from '../../src/backend/gpu_neural_ops';

// Mock GPU implementation for testing
function createMockGPUDevice() {
  const buffers = new Map<any, Float32Array>();

  const mockBuffer = (size: number) => {
    const data = new Float32Array(size / 4);
    const buffer = {
      destroy: vi.fn(),
      mapAsync: vi.fn(async () => {}),
      getMappedRange: vi.fn(() => data.buffer),
      unmap: vi.fn(),
      size
    };
    buffers.set(buffer, data);
    return buffer;
  };

  const mockDevice = {
    createBuffer: vi.fn((descriptor: any) => mockBuffer(descriptor.size)),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer: vi.fn(),
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn()
      })),
      finish: vi.fn(() => ({}))
    })),
    createShaderModule: vi.fn(() => ({})),
    createComputePipelineAsync: vi.fn(async () => ({
      getBindGroupLayout: vi.fn(() => ({}))
    })),
    createBindGroup: vi.fn(() => ({})),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn((buffer: any, offset: number, data: ArrayBuffer) => {
        const bufferData = buffers.get(buffer);
        if (bufferData && data instanceof ArrayBuffer) {
          const srcView = new Float32Array(data);
          for (let i = 0; i < srcView.length; i++) {
            bufferData[offset / 4 + i] = srcView[i];
          }
        }
      })
    }
  };

  return { mockDevice, buffers };
}

describe('WebGPUBackend', () => {
  let originalNavigator: any;

  beforeEach(() => {
    originalNavigator = globalThis.navigator;
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      writable: true,
      configurable: true
    });
  });

  describe('Static factory methods', () => {
    it('throws error when WebGPU is not available', async () => {
      Object.defineProperty(globalThis, 'navigator', {
        value: {},
        writable: true,
        configurable: true
      });

      await expect(WebGPUBackend.create()).rejects.toThrow('WebGPU is not available');
    });

    it('throws error when adapter is null', async () => {
      Object.defineProperty(globalThis, 'navigator', {
        value: {
          gpu: {
            requestAdapter: vi.fn(async () => null)
          }
        },
        writable: true,
        configurable: true
      });

      await expect(WebGPUBackend.create()).rejects.toThrow('Failed to acquire GPU adapter');
    });

    it('creates backend from existing device', () => {
      const { mockDevice } = createMockGPUDevice();
      const backend = WebGPUBackend.fromDevice(mockDevice as any);

      expect(backend).toBeDefined();
      expect(backend.device).toBe(mockDevice);
      expect(backend.queue).toBe(mockDevice.queue);
    });
  });

  describe('Tensor operations', () => {
    let backend: WebGPUBackend;
    let mockDevice: any;

    beforeEach(() => {
      const mock = createMockGPUDevice();
      mockDevice = mock.mockDevice;
      backend = WebGPUBackend.fromDevice(mockDevice);
    });

    it('creates tensor with correct shape', async () => {
      const data = [1, 2, 3, 4, 5, 6];
      const tensor = await backend.createTensor(new Float32Array(data), [2, 3]);

      expect(tensor.shape).toEqual([2, 3]);
      expect(tensor.size).toBe(6);
      expect(mockDevice.createBuffer).toHaveBeenCalled();
    });

    it('creates 1D tensor', async () => {
      const data = [1, 2, 3, 4];
      const tensor = await backend.createTensor(new Float32Array(data), [4]);

      expect(tensor.shape).toEqual([4]);
      expect(tensor.size).toBe(4);
    });

    it('creates tensor from regular array', async () => {
      const data = [1.5, 2.5, 3.5];
      const tensor = await backend.createTensor(data as any, [3]);

      expect(tensor.size).toBe(3);
    });

    it('reads tensor data back', async () => {
      const data = [1, 2, 3, 4, 5];
      const tensor = await backend.createTensor(new Float32Array(data), [5]);

      const result = await tensor.toArray();

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(5);
    });

    it('disposes tensor correctly', async () => {
      const tensor = await backend.createTensor(new Float32Array([1, 2, 3]), [3]);
      const destroySpy = vi.spyOn(tensor.buffer, 'destroy');

      tensor.dispose();

      expect(destroySpy).toHaveBeenCalled();
    });
  });

  describe('Matrix operations', () => {
    let backend: WebGPUBackend;

    beforeEach(() => {
      const { mockDevice } = createMockGPUDevice();
      backend = WebGPUBackend.fromDevice(mockDevice as any);
    });

    it('validates matrix dimensions for matMul', async () => {
      const tensorA = await backend.createTensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const tensorB = await backend.createTensor(new Float32Array([1, 2, 3]), [3, 1]);

      // Incompatible dimensions
      await expect(backend.matMul(tensorA, tensorB)).rejects.toThrow('Inner dimensions must match');
    });

    it('validates tensor rank for matMul', async () => {
      const tensorA = await backend.createTensor(new Float32Array([1, 2, 3]), [3]);
      const tensorB = await backend.createTensor(new Float32Array([1, 2]), [2]);

      // Non-2D tensors
      await expect(backend.matMul(tensorA, tensorB)).rejects.toThrow('supports 2D tensors only');
    });

    it('creates output tensor with correct shape for matMul', async () => {
      const tensorA = await backend.createTensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
      const tensorB = await backend.createTensor(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]);

      const result = await backend.matMul(tensorA, tensorB);

      // 2x3 @ 3x2 = 2x2
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('Element-wise operations', () => {
    let backend: WebGPUBackend;

    beforeEach(() => {
      const { mockDevice } = createMockGPUDevice();
      backend = WebGPUBackend.fromDevice(mockDevice as any);
    });

    it('validates tensor sizes for elementwise binary ops', async () => {
      const tensorA = await backend.createTensor(new Float32Array([1, 2, 3]), [3]);
      const tensorB = await backend.createTensor(new Float32Array([1, 2]), [2]);

      await expect(backend.elementwiseBinary('add', tensorA, tensorB)).rejects.toThrow('equal size');
    });

    it('creates output tensor for addition', async () => {
      const tensorA = await backend.createTensor(new Float32Array([1, 2, 3]), [3]);
      const tensorB = await backend.createTensor(new Float32Array([4, 5, 6]), [3]);

      const result = await backend.elementwiseBinary('add', tensorA, tensorB);

      expect(result.shape).toEqual([3]);
      expect(result.size).toBe(3);

      // Clean up
      tensorA.dispose();
      tensorB.dispose();
      result.dispose();
    });
  });
});

describe('GPUNeuralOps', () => {
  describe('Initialization and availability', () => {
    it('is not available without initialization', () => {
      const gpuOps = new GPUNeuralOps();

      expect(gpuOps.isAvailable()).toBe(false);
      expect(gpuOps.isEnabled()).toBe(false);
    });

    it('cannot enable GPU before initialization', () => {
      const gpuOps = new GPUNeuralOps();
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      gpuOps.setEnabled(true);

      expect(gpuOps.isEnabled()).toBe(false);
      expect(consoleSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe('CPU fallback operations', () => {
    it('falls back to CPU for matrix-vector multiply', async () => {
      const gpuOps = new GPUNeuralOps();

      const matrix = [
        [1, 2],
        [3, 4]
      ];
      const vector = [5, 6];

      const result = await gpuOps.matrixVectorMul(matrix, vector);

      expect(result).toHaveLength(2);
      expect(result[0]).toBe(17); // 1*5 + 2*6
      expect(result[1]).toBe(39); // 3*5 + 4*6
    });

    it('performs vector addition on CPU', async () => {
      const gpuOps = new GPUNeuralOps();

      const vec1 = [1, 2, 3];
      const vec2 = [4, 5, 6];

      const result = await gpuOps.vectorAdd(vec1, vec2);

      expect(result).toEqual([5, 7, 9]);
    });

    it('performs element-wise vector multiplication on CPU', async () => {
      const gpuOps = new GPUNeuralOps();

      const vec1 = [2, 3, 4];
      const vec2 = [5, 6, 7];

      const result = await gpuOps.vectorMul(vec1, vec2);

      expect(result).toEqual([10, 18, 28]);
    });
  });

  describe('Performance metrics', () => {
    it('provides metrics even without GPU', () => {
      const gpuOps = new GPUNeuralOps();

      const metrics = gpuOps.getMetrics();

      expect(metrics.enabled).toBe(false);
      expect(metrics.available).toBe(false);
      expect(metrics.totalOperations).toBe(0);
      expect(metrics.averageTimeMs).toBe(0);
    });

    it('does not track operations when GPU is disabled', async () => {
      const gpuOps = new GPUNeuralOps();

      const initialMetrics = gpuOps.getMetrics();
      expect(initialMetrics.totalOperations).toBe(0);

      // Operations fall back to CPU and don't increment totalOperations
      await gpuOps.vectorAdd([1, 2], [3, 4]);
      await gpuOps.vectorMul([1, 2], [3, 4]);

      const updatedMetrics = gpuOps.getMetrics();
      expect(updatedMetrics.totalOperations).toBe(0); // CPU fallback doesn't track
    });

    it('resets metrics correctly', () => {
      const gpuOps = new GPUNeuralOps();

      // Manually set some metrics to test reset
      (gpuOps as any).totalOperations = 10;
      (gpuOps as any).totalTimeMs = 100;

      expect(gpuOps.getMetrics().totalOperations).toBe(10);

      gpuOps.resetMetrics();

      const metrics = gpuOps.getMetrics();
      expect(metrics.totalOperations).toBe(0);
      expect(metrics.totalTimeMs).toBe(0);
    });
  });

  describe('Edge cases', () => {
    it('handles empty vectors', async () => {
      const gpuOps = new GPUNeuralOps();

      const result1 = await gpuOps.vectorAdd([], []);
      expect(result1).toEqual([]);

      const result2 = await gpuOps.vectorMul([], []);
      expect(result2).toEqual([]);
    });

    it('handles empty matrix', async () => {
      const gpuOps = new GPUNeuralOps();

      const result = await gpuOps.matrixVectorMul([], []);
      expect(result).toEqual([]);
    });

    it('handles single element vectors', async () => {
      const gpuOps = new GPUNeuralOps();

      const result = await gpuOps.vectorAdd([5], [3]);
      expect(result).toEqual([8]);
    });

    it('handles identity matrix correctly', async () => {
      const gpuOps = new GPUNeuralOps();

      const identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
      ];
      const vector = [5, 6, 7];

      const result = await gpuOps.matrixVectorMul(identity, vector);
      expect(result).toEqual([5, 6, 7]);
    });

    it('handles mismatched vector dimensions gracefully', async () => {
      const gpuOps = new GPUNeuralOps();

      const vec1 = [1, 2, 3];
      const vec2 = [1, 2]; // Wrong size

      // Implementation returns NaN for mismatched dimensions rather than throwing
      const result1 = await gpuOps.vectorAdd(vec1, vec2);
      expect(result1).toHaveLength(3);
      expect(result1[2]).toBeNaN();

      const result2 = await gpuOps.vectorMul(vec1, vec2);
      expect(result2).toHaveLength(3);
      expect(result2[2]).toBeNaN();
    });

    it('handles mismatched matrix-vector dimensions gracefully', async () => {
      const gpuOps = new GPUNeuralOps();

      const matrix = [[1, 2, 3]];
      const vector = [1, 2]; // Wrong size (should be 3)

      // Implementation returns NaN for mismatched dimensions rather than throwing
      const result = await gpuOps.matrixVectorMul(matrix, vector);
      expect(result).toHaveLength(1);
      expect(result[0]).toBeNaN();
    });
  });
});
