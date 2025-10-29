/**
 * GPU-accelerated neural network operations
 *
 * Provides high-level operations for neural networks using WebGPU backend.
 * Includes automatic fallback to CPU for unsupported environments.
 */

import { WebGPUBackend, WebGPUTensor } from './webgpu';

export interface GPUMetrics {
  enabled: boolean;
  available: boolean;
  totalOperations: number;
  totalTimeMs: number;
  averageTimeMs: number;
  deviceInfo?: string;
}

export class GPUNeuralOps {
  private backend: WebGPUBackend | null = null;
  private enabled: boolean = false;
  private available: boolean = false;
  private totalOperations: number = 0;
  private totalTimeMs: number = 0;

  /**
   * Initialize GPU backend
   */
  async initialize(): Promise<boolean> {
    try {
      this.backend = await WebGPUBackend.create();
      this.available = true;
      this.enabled = true;
      console.log('✅ WebGPU backend initialized successfully');
      return true;
    } catch (error) {
      console.warn('⚠️ WebGPU not available, falling back to CPU:', error);
      this.available = false;
      this.enabled = false;
      return false;
    }
  }

  /**
   * Check if GPU is available
   */
  isAvailable(): boolean {
    return this.available;
  }

  /**
   * Enable/disable GPU acceleration
   */
  setEnabled(enabled: boolean): void {
    if (!this.available && enabled) {
      console.warn('⚠️ Cannot enable GPU: WebGPU is not available');
      return;
    }
    this.enabled = enabled;
  }

  /**
   * Check if GPU is currently enabled
   */
  isEnabled(): boolean {
    return this.enabled && this.available;
  }

  /**
   * Get GPU metrics
   */
  getMetrics(): GPUMetrics {
    return {
      enabled: this.enabled,
      available: this.available,
      totalOperations: this.totalOperations,
      totalTimeMs: this.totalTimeMs,
      averageTimeMs: this.totalOperations > 0 ? this.totalTimeMs / this.totalOperations : 0,
      deviceInfo: this.backend ? 'WebGPU Device' : undefined
    };
  }

  /**
   * Reset metrics
   */
  resetMetrics(): void {
    this.totalOperations = 0;
    this.totalTimeMs = 0;
  }

  /**
   * Matrix-vector multiplication: y = A @ x
   * A: [m, n] matrix
   * x: [n] vector
   * Returns: [m] vector
   */
  async matrixVectorMul(A: number[][], x: number[]): Promise<number[]> {
    if (!this.enabled || !this.backend) {
      return this.cpuMatrixVectorMul(A, x);
    }

    const startTime = performance.now();
    try {
      const m = A.length;
      const n = A[0].length;

      // Flatten matrix A
      const flatA = new Float32Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          flatA[i * n + j] = A[i][j];
        }
      }

      // Convert x to column vector [n, 1]
      const flatX = new Float32Array(x);

      // Create GPU tensors
      const tensorA = await this.backend.createTensor(flatA, [m, n]);
      const tensorX = await this.backend.createTensor(flatX, [n, 1]);

      // Perform matrix multiplication
      const result = await this.backend.matMul(tensorA, tensorX);
      const resultArray = await result.toArray();

      // Clean up
      tensorA.dispose();
      tensorX.dispose();
      result.dispose();

      const endTime = performance.now();
      this.totalOperations++;
      this.totalTimeMs += endTime - startTime;

      return Array.from(resultArray);
    } catch (error) {
      console.warn('⚠️ GPU operation failed, falling back to CPU:', error);
      this.enabled = false;
      return this.cpuMatrixVectorMul(A, x);
    }
  }

  /**
   * Matrix-vector multiplication (transposed): y = A^T @ x
   * A: [m, n] matrix
   * x: [m] vector
   * Returns: [n] vector
   */
  async matrixVectorMulTranspose(A: number[][], x: number[]): Promise<number[]> {
    if (!this.enabled || !this.backend) {
      return this.cpuMatrixVectorMulTranspose(A, x);
    }

    const startTime = performance.now();
    try {
      const m = A.length;
      const n = A[0].length;

      // Create transposed matrix
      const flatAT = new Float32Array(n * m);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          flatAT[j * m + i] = A[i][j];
        }
      }

      const flatX = new Float32Array(x);

      const tensorAT = await this.backend.createTensor(flatAT, [n, m]);
      const tensorX = await this.backend.createTensor(flatX, [m, 1]);

      const result = await this.backend.matMul(tensorAT, tensorX);
      const resultArray = await result.toArray();

      tensorAT.dispose();
      tensorX.dispose();
      result.dispose();

      const endTime = performance.now();
      this.totalOperations++;
      this.totalTimeMs += endTime - startTime;

      return Array.from(resultArray);
    } catch (error) {
      console.warn('⚠️ GPU operation failed, falling back to CPU:', error);
      this.enabled = false;
      return this.cpuMatrixVectorMulTranspose(A, x);
    }
  }

  /**
   * Element-wise vector addition: z = x + y
   */
  async vectorAdd(x: number[], y: number[]): Promise<number[]> {
    if (!this.enabled || !this.backend) {
      return this.cpuVectorAdd(x, y);
    }

    const startTime = performance.now();
    try {
      const tensorX = await this.backend.createTensor(new Float32Array(x), [x.length]);
      const tensorY = await this.backend.createTensor(new Float32Array(y), [y.length]);

      const result = await this.backend.elementwiseBinary('add', tensorX, tensorY);
      const resultArray = await result.toArray();

      tensorX.dispose();
      tensorY.dispose();
      result.dispose();

      const endTime = performance.now();
      this.totalOperations++;
      this.totalTimeMs += endTime - startTime;

      return Array.from(resultArray);
    } catch (error) {
      console.warn('⚠️ GPU operation failed, falling back to CPU:', error);
      this.enabled = false;
      return this.cpuVectorAdd(x, y);
    }
  }

  /**
   * Element-wise vector multiplication: z = x * y
   */
  async vectorMul(x: number[], y: number[]): Promise<number[]> {
    if (!this.enabled || !this.backend) {
      return this.cpuVectorMul(x, y);
    }

    const startTime = performance.now();
    try {
      const tensorX = await this.backend.createTensor(new Float32Array(x), [x.length]);
      const tensorY = await this.backend.createTensor(new Float32Array(y), [y.length]);

      const result = await this.backend.elementwiseBinary('mul', tensorX, tensorY);
      const resultArray = await result.toArray();

      tensorX.dispose();
      tensorY.dispose();
      result.dispose();

      const endTime = performance.now();
      this.totalOperations++;
      this.totalTimeMs += endTime - startTime;

      return Array.from(resultArray);
    } catch (error) {
      console.warn('⚠️ GPU operation failed, falling back to CPU:', error);
      this.enabled = false;
      return this.cpuVectorMul(x, y);
    }
  }

  // CPU fallback implementations

  private cpuMatrixVectorMul(A: number[][], x: number[]): number[] {
    const y = new Array(A.length).fill(0);
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      const row = A[i];
      for (let j = 0; j < row.length; j++) {
        s += row[j] * x[j];
      }
      y[i] = s;
    }
    return y;
  }

  private cpuMatrixVectorMulTranspose(A: number[][], x: number[]): number[] {
    if (A.length === 0) return [];
    const cols = A[0].length;
    const y = new Array(cols).fill(0);
    for (let i = 0; i < A.length; i++) {
      const row = A[i];
      const xi = x[i];
      for (let j = 0; j < cols; j++) {
        y[j] += row[j] * xi;
      }
    }
    return y;
  }

  private cpuVectorAdd(x: number[], y: number[]): number[] {
    return x.map((xi, i) => xi + y[i]);
  }

  private cpuVectorMul(x: number[], y: number[]): number[] {
    return x.map((xi, i) => xi * y[i]);
  }

  /**
   * Dispose of GPU resources
   */
  dispose(): void {
    if (this.backend) {
      // Note: WebGPU backend doesn't have a dispose method in the current implementation
      // but we can add it if needed
      this.backend = null;
    }
    this.enabled = false;
  }
}

/**
 * Singleton instance for global GPU operations
 */
let globalGPUOps: GPUNeuralOps | null = null;

/**
 * Get or create the global GPU operations instance
 */
export async function getGPUOps(forceNew: boolean = false): Promise<GPUNeuralOps> {
  if (!globalGPUOps || forceNew) {
    globalGPUOps = new GPUNeuralOps();
    await globalGPUOps.initialize();
  }
  return globalGPUOps;
}

/**
 * Check if WebGPU is available in the current environment
 */
export function isWebGPUAvailable(): boolean {
  if (typeof navigator === 'undefined') return false;
  return 'gpu' in navigator;
}
