#!/usr/bin/env tsx
/**
 * WebGPU Matrix Multiplication Benchmark
 *
 * Compares CPU vs GPU performance for matrix operations relevant to neural network training.
 * Run with: npx tsx scripts/benchmark_gpu.ts
 */

import { WebGPUBackend } from '../src/backend/webgpu';

interface BenchmarkResult {
  operation: string;
  size: string;
  cpuTimeMs: number;
  gpuTimeMs: number;
  speedup: number;
  iterations: number;
}

/**
 * CPU matrix multiplication (naive implementation)
 */
function cpuMatMul(
  A: Float32Array,
  B: Float32Array,
  m: number,
  k: number,
  n: number
): Float32Array {
  const C = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
  return C;
}

/**
 * Benchmark a single matrix multiplication
 */
async function benchmarkMatMul(
  backend: WebGPUBackend,
  m: number,
  k: number,
  n: number,
  iterations: number = 10
): Promise<BenchmarkResult> {
  console.log(`\n📊 Benchmarking matmul: [${m}×${k}] × [${k}×${n}]`);

  // Generate random matrices
  const A = new Float32Array(m * k);
  const B = new Float32Array(k * n);
  for (let i = 0; i < A.length; i++) A[i] = Math.random();
  for (let i = 0; i < B.length; i++) B[i] = Math.random();

  // CPU benchmark
  console.log(`  Running ${iterations} iterations on CPU...`);
  const cpuStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    cpuMatMul(A, B, m, k, n);
  }
  const cpuEnd = performance.now();
  const cpuTimeMs = (cpuEnd - cpuStart) / iterations;

  // GPU benchmark
  console.log(`  Running ${iterations} iterations on GPU...`);
  const gpuTensorA = await backend.createTensor(A, [m, k]);
  const gpuTensorB = await backend.createTensor(B, [k, n]);

  const gpuStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    const result = await backend.matMul(gpuTensorA, gpuTensorB);
    result.dispose();
  }
  const gpuEnd = performance.now();
  const gpuTimeMs = (gpuEnd - gpuStart) / iterations;

  gpuTensorA.dispose();
  gpuTensorB.dispose();

  const speedup = cpuTimeMs / gpuTimeMs;

  console.log(
    `  ✅ CPU: ${cpuTimeMs.toFixed(2)}ms, GPU: ${gpuTimeMs.toFixed(2)}ms, Speedup: ${speedup.toFixed(2)}x`
  );

  return {
    operation: 'matmul',
    size: `${m}×${k} × ${k}×${n}`,
    cpuTimeMs,
    gpuTimeMs,
    speedup,
    iterations
  };
}

/**
 * Benchmark elementwise operations
 */
async function benchmarkElementwise(
  backend: WebGPUBackend,
  size: number,
  iterations: number = 100
): Promise<BenchmarkResult> {
  console.log(`\n📊 Benchmarking elementwise add: [${size}] elements`);

  // Generate random vectors
  const A = new Float32Array(size);
  const B = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    A[i] = Math.random();
    B[i] = Math.random();
  }

  // CPU benchmark
  console.log(`  Running ${iterations} iterations on CPU...`);
  const cpuStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    const C = new Float32Array(size);
    for (let j = 0; j < size; j++) {
      C[j] = A[j] + B[j];
    }
  }
  const cpuEnd = performance.now();
  const cpuTimeMs = (cpuEnd - cpuStart) / iterations;

  // GPU benchmark
  console.log(`  Running ${iterations} iterations on GPU...`);
  const gpuTensorA = await backend.createTensor(A, [size]);
  const gpuTensorB = await backend.createTensor(B, [size]);

  const gpuStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    const result = await backend.elementwiseBinary('add', gpuTensorA, gpuTensorB);
    result.dispose();
  }
  const gpuEnd = performance.now();
  const gpuTimeMs = (gpuEnd - gpuStart) / iterations;

  gpuTensorA.dispose();
  gpuTensorB.dispose();

  const speedup = cpuTimeMs / gpuTimeMs;

  console.log(
    `  ✅ CPU: ${cpuTimeMs.toFixed(2)}ms, GPU: ${gpuTimeMs.toFixed(2)}ms, Speedup: ${speedup.toFixed(2)}x`
  );

  return {
    operation: 'elementwise_add',
    size: `${size}`,
    cpuTimeMs,
    gpuTimeMs,
    speedup,
    iterations
  };
}

/**
 * Print summary table
 */
function printSummary(results: BenchmarkResult[]) {
  console.log('\n' + '='.repeat(80));
  console.log('📈 BENCHMARK SUMMARY');
  console.log('='.repeat(80));
  console.log();
  console.log(
    'Operation'.padEnd(20) +
      'Size'.padEnd(20) +
      'CPU (ms)'.padEnd(12) +
      'GPU (ms)'.padEnd(12) +
      'Speedup'
  );
  console.log('-'.repeat(80));

  for (const result of results) {
    console.log(
      result.operation.padEnd(20) +
        result.size.padEnd(20) +
        result.cpuTimeMs.toFixed(2).padEnd(12) +
        result.gpuTimeMs.toFixed(2).padEnd(12) +
        `${result.speedup.toFixed(2)}x`
    );
  }

  console.log('-'.repeat(80));

  const avgSpeedup = results.reduce((sum, r) => sum + r.speedup, 0) / results.length;
  console.log(`\n🚀 Average Speedup: ${avgSpeedup.toFixed(2)}x`);
  console.log();
}

/**
 * Main benchmark function
 */
async function main() {
  console.log('🔧 Initializing WebGPU backend...');

  try {
    const backend = await WebGPUBackend.create();
    console.log('✅ WebGPU backend initialized successfully!');
    console.log(`   Device: ${backend.device.constructor.name}`);

    const results: BenchmarkResult[] = [];

    // Test different matrix sizes relevant to neural language models
    // These sizes are typical for embedding and hidden layer operations

    // Small matrices (typical for small vocabulary/hidden size)
    results.push(await benchmarkMatMul(backend, 64, 64, 64, 20));

    // Medium matrices (typical for medium models)
    results.push(await benchmarkMatMul(backend, 128, 128, 256, 10));

    // Larger matrices (typical for larger vocabularies)
    results.push(await benchmarkMatMul(backend, 256, 256, 512, 5));

    // Very large matrices (stress test)
    results.push(await benchmarkMatMul(backend, 512, 512, 1024, 3));

    // Elementwise operations (common in activation functions and updates)
    results.push(await benchmarkElementwise(backend, 10000, 100));
    results.push(await benchmarkElementwise(backend, 100000, 50));

    printSummary(results);

    // Recommendations based on results
    console.log('💡 RECOMMENDATIONS:');
    console.log();

    const matmulResults = results.filter((r) => r.operation === 'matmul');
    const avgMatmulSpeedup =
      matmulResults.reduce((sum, r) => sum + r.speedup, 0) / matmulResults.length;

    if (avgMatmulSpeedup > 2) {
      console.log('✅ GPU acceleration shows significant speedup (>2x) for matrix operations.');
      console.log('   Recommendation: Enable GPU acceleration by default for training.');
    } else if (avgMatmulSpeedup > 1.2) {
      console.log('⚠️  GPU acceleration shows moderate speedup (1.2-2x) for matrix operations.');
      console.log(
        '   Recommendation: Offer GPU as optional feature with clear performance expectations.'
      );
    } else {
      console.log('❌ GPU acceleration shows minimal speedup (<1.2x) for matrix operations.');
      console.log('   Recommendation: WebGPU overhead may exceed benefits for small models.');
      console.log('   Consider GPU only for larger model sizes (hidden_size > 256).');
    }

    console.log();
    console.log('📝 Note: These benchmarks include data transfer overhead between CPU and GPU.');
    console.log(
      '   In practice, keeping data on GPU throughout training will yield better performance.'
    );
    console.log();
  } catch (error) {
    console.error('❌ Failed to initialize WebGPU:', error);
    console.log();
    console.log('💡 WebGPU is not available. Possible reasons:');
    console.log(
      '   - Running in Node.js without @webgpu/dawn (install with: npm install @webgpu/dawn)'
    );
    console.log('   - Browser does not support WebGPU (try Chrome/Edge 113+)');
    console.log('   - GPU drivers need updating');
    console.log();
    process.exit(1);
  }
}

// Run benchmark
main().catch(console.error);
