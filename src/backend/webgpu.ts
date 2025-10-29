import { logSumExp } from '../lib/MathUtils';

type TypedArray = Float32Array | Float64Array | Int32Array | Uint32Array;

export type TensorShape = number[];

defineGPUConstants();

function defineGPUConstants() {
  if (typeof (globalThis as any).GPUBufferUsage === 'undefined') {
    (globalThis as any).GPUBufferUsage = {
      MAP_READ: 0x0001,
      MAP_WRITE: 0x0002,
      COPY_SRC: 0x0004,
      COPY_DST: 0x0008,
      INDEX: 0x0010,
      VERTEX: 0x0020,
      UNIFORM: 0x0040,
      STORAGE: 0x0080,
      INDIRECT: 0x0100,
      QUERY_RESOLVE: 0x0200
    };
  }
  if (typeof (globalThis as any).GPUMapMode === 'undefined') {
    (globalThis as any).GPUMapMode = {
      READ: 0x0001,
      WRITE: 0x0002
    };
  }
}

function defaultRng(): number {
  return Math.random();
}

export class WebGPUTensor {
  readonly size: number;

  constructor(
    readonly backend: WebGPUBackend,
    readonly shape: TensorShape,
    readonly buffer: GPUBuffer,
    readonly dtype: 'float32' | 'int32' = 'float32'
  ) {
    this.size = shape.reduce((acc, dim) => acc * dim, 1);
  }

  async toArray(): Promise<Float32Array> {
    const device = this.backend.device;
    const bytes = this.size * 4;
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.buffer, 0, staging, 0, bytes);
    device.queue.submit([commandEncoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange();
    const result = new Float32Array(copy.slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  dispose(): void {
    this.buffer.destroy();
  }
}

export interface ElementwiseOptions {
  workgroupSize?: number;
}

export class WebGPUBackend {
  private constructor(
    readonly device: GPUDevice,
    readonly queue: GPUQueue
  ) {}

  static async create(): Promise<WebGPUBackend> {
    const nav = (globalThis as any).navigator as (Navigator & { gpu?: GPU }) | undefined;
    if (!nav?.gpu) {
      throw new Error('WebGPU is not available in this environment.');
    }
    const adapter = await nav.gpu.requestAdapter();
    if (!adapter) throw new Error('Failed to acquire GPU adapter.');
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device, device.queue);
  }

  static fromDevice(device: GPUDevice): WebGPUBackend {
    return new WebGPUBackend(device, device.queue);
  }

  async createTensor(data: TypedArray, shape: TensorShape): Promise<WebGPUTensor> {
    const array = data instanceof Float32Array ? data : new Float32Array(data as ArrayLike<number>);
    const buffer = this.device.createBuffer({
      size: array.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.queue.writeBuffer(buffer, 0, array.buffer, array.byteOffset, array.byteLength);
    return new WebGPUTensor(this, shape, buffer);
  }

  private async dispatchCompute(
    shaderCode: string,
    bindings: GPUBindGroupEntry[],
    workgroups: number | [number, number] | [number, number, number]
  ): Promise<void> {
    const module = this.device.createShaderModule({ code: shaderCode });
    const pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    if (Array.isArray(workgroups)) {
      const [x, y = 1, z = 1] = workgroups;
      pass.dispatchWorkgroups(x, y, z);
    } else {
      pass.dispatchWorkgroups(workgroups);
    }
    pass.end();
    this.queue.submit([encoder.finish()]);
  }

  private getWorkgroupCount(size: number, workgroupSize = 64): number {
    return Math.ceil(size / workgroupSize);
  }

  async elementwiseBinary(
    op: 'add' | 'sub' | 'mul' | 'div',
    a: WebGPUTensor,
    b: WebGPUTensor,
    options: ElementwiseOptions = {}
  ): Promise<WebGPUTensor> {
    if (a.size !== b.size) {
      throw new Error('Elementwise operations require tensors of equal size.');
    }

    const workgroupSize = options.workgroupSize ?? 64;
    const output = this.device.createBuffer({
      size: a.size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const shader = `
      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, read_write> C: array<f32>;

      @compute @workgroup_size(${workgroupSize}, 1, 1)
      fn main(@builtin(global_invocation_id) GlobalId : vec3<u32>) {
        let index = GlobalId.x;
        if (index >= arrayLength(&C)) {
          return;
        }
        let aVal = A[index];
        let bVal = B[index];
        var value: f32 = 0.0;
        ${this.binaryOpSnippet(op)}
        C[index] = value;
      }
    `;

    await this.dispatchCompute(
      shader,
      [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: output } }
      ],
      this.getWorkgroupCount(a.size, workgroupSize)
    );

    return new WebGPUTensor(this, [...a.shape], output);
  }

  private binaryOpSnippet(op: 'add' | 'sub' | 'mul' | 'div'): string {
    switch (op) {
      case 'add':
        return 'value = aVal + bVal;';
      case 'sub':
        return 'value = aVal - bVal;';
      case 'mul':
        return 'value = aVal * bVal;';
      case 'div':
        return 'value = aVal / bVal;';
      default:
        return 'value = aVal;';
    }
  }

  async matMul(a: WebGPUTensor, b: WebGPUTensor): Promise<WebGPUTensor> {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('matMul currently supports 2D tensors only.');
    }
    const [m, kA] = a.shape;
    const [kB, n] = b.shape;
    if (kA !== kB) {
      throw new Error('Inner dimensions must match for matMul.');
    }

    const output = this.device.createBuffer({
      size: m * n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const shader = `
      struct MatrixDimensions {
        rowsA: u32,
        colsA: u32,
        colsB: u32,
      };

      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read> B: array<f32>;
      @group(0) @binding(2) var<storage, read_write> C: array<f32>;
      @group(0) @binding(3) var<uniform> dims: MatrixDimensions;

      fn index(row: u32, col: u32, width: u32) -> u32 {
        return row * width + col;
      }

      @compute @workgroup_size(8, 8, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x >= dims.rowsA || gid.y >= dims.colsB) {
          return;
        }
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < dims.colsA; i = i + 1u) {
          let aIndex = index(gid.x, i, dims.colsA);
          let bIndex = index(i, gid.y, dims.colsB);
          sum = sum + A[aIndex] * B[bIndex];
        }
        let outIndex = index(gid.x, gid.y, dims.colsB);
        C[outIndex] = sum;
      }
    `;

    const dimsBuffer = this.device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const dimsArray = new Uint32Array([m, kA, n]);
    this.queue.writeBuffer(dimsBuffer, 0, dimsArray.buffer, dimsArray.byteOffset, dimsArray.byteLength);

    await this.dispatchCompute(
      shader,
      [
        { binding: 0, resource: { buffer: a.buffer } },
        { binding: 1, resource: { buffer: b.buffer } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: dimsBuffer } }
      ],
      [Math.ceil(m / 8), Math.ceil(n / 8), 1]
    );

    dimsBuffer.destroy();
    return new WebGPUTensor(this, [m, n], output);
  }

  async exp(tensor: WebGPUTensor): Promise<WebGPUTensor> {
    const workgroupSize = 64;
    const output = this.device.createBuffer({
      size: tensor.size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const shader = `
      @group(0) @binding(0) var<storage, read> A: array<f32>;
      @group(0) @binding(1) var<storage, read_write> B: array<f32>;

      @compute @workgroup_size(${workgroupSize}, 1, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= arrayLength(&B)) {
          return;
        }
        B[idx] = exp(A[idx]);
      }
    `;

    await this.dispatchCompute(
      shader,
      [
        { binding: 0, resource: { buffer: tensor.buffer } },
        { binding: 1, resource: { buffer: output } }
      ],
      this.getWorkgroupCount(tensor.size, workgroupSize)
    );

    return new WebGPUTensor(this, [...tensor.shape], output);
  }

  async logSumExp(tensor: WebGPUTensor): Promise<number> {
    const data = await tensor.toArray();
    return logSumExp(Array.from(data));
  }

  async softmax(tensor: WebGPUTensor, temperature = 1): Promise<WebGPUTensor> {
    const data = await tensor.toArray();
    const scaled = data.map((value) => value / Math.max(temperature, 1e-6));
    const norm = logSumExp(scaled);
    const probs = new Float32Array(scaled.map((v) => Math.exp(v - norm)));
    return this.createTensor(probs, [...tensor.shape]);
  }

  async normalize(tensor: WebGPUTensor): Promise<WebGPUTensor> {
    const data = await tensor.toArray();
    const sum = data.reduce((acc, v) => acc + v, 0);
    const inv = sum === 0 ? 0 : 1 / sum;
    const normalized = new Float32Array(data.map((v) => v * inv));
    return this.createTensor(normalized, [...tensor.shape]);
  }

  async random(shape: TensorShape, rng: () => number = defaultRng): Promise<WebGPUTensor> {
    const size = shape.reduce((acc, dim) => acc * dim, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = rng();
    }
    return this.createTensor(data, shape);
  }
}

export async function withWebGPU<T>(callback: (backend: WebGPUBackend) => Promise<T>): Promise<T> {
  const backend = await WebGPUBackend.create();
  try {
    return await callback(backend);
  } finally {
    const device = backend.device as GPUDevice & { destroy?: () => void };
    device.destroy?.();
  }
}
