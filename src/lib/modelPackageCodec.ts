/**
 * Compact, self-describing model package ("neuro-lingua-model" format).
 *
 * A package wraps the object produced by `model.toJSON()`:
 * - every numeric array / matrix (≥ 8 values) is stored as a binary tensor, float32 or float16,
 *   base64-encoded — several times smaller and faster to parse than nested JSON numbers;
 * - optimizer state (Adam / momentum buffers) can be left out for an inference package;
 * - an FNV-1a checksum over the tensor payload detects corruption (it is not a signature);
 * - a model card records how the model was produced (data, tokenizer, training, metrics).
 *
 * Packages are plain JSON, so they work in the browser (Import button), in Node (model-factory
 * CLI, local server) and in the desktop app, without any network access.
 *
 * This module has no dependency on the model classes (ProNeuralLM itself uses it to save
 * compactly); `modelPackage.ts` adds the class-aware loader.
 */

export const MODEL_PACKAGE_FORMAT = 'neuro-lingua-model';
export const MODEL_PACKAGE_VERSION = 1;

export type TensorPrecision = 'float32' | 'float16';

export interface ModelCard {
  name: string;
  description?: string;
  createdAt: string;
  architecture: string;
  parameters: number;
  vocabularySize: number;
  contextSize: number;
  tokenizer: Record<string, unknown>;
  training?: Record<string, unknown>;
  data?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  environment?: Record<string, unknown>;
  /** How the model was produced with respect to external services. */
  provenance: {
    /** true: produced without external services; null: unknown (history not recorded). */
    offline: boolean | null;
    externalServices: string[];
    statement: string;
  };
}

export interface ModelPackage {
  format: typeof MODEL_PACKAGE_FORMAT;
  formatVersion: number;
  createdAt: string;
  architecture: 'feedforward' | 'advanced' | 'transformer';
  precision: TensorPrecision;
  includesOptimizerState: boolean;
  checksum: string;
  card?: ModelCard;
  model: unknown;
}

interface EncodedTensor {
  $tensor: number[];
  dtype: 'f32' | 'f16';
  data: string;
}

const OPTIMIZER_STATE_KEYS = new Set([
  'adamT',
  'mEmbedding',
  'mWHidden',
  'mWOutput',
  'mBHidden',
  'mBOutput',
  'aEmbedding',
  'aWHidden',
  'aWOutput',
  'aBHidden',
  'aBOutput'
]);

const MIN_TENSOR_SIZE = 8;

// ---------------------------------------------------------------------------
// Base64 and float16 helpers (no dependency on Buffer / btoa)
// ---------------------------------------------------------------------------

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = new Uint8Array(128);
for (let i = 0; i < B64.length; i++) B64_LOOKUP[B64.charCodeAt(i)] = i;

export function bytesToBase64(bytes: Uint8Array): string {
  let out = '';
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out += B64[n >> 18] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + B64[n & 63];
  }
  const rest = bytes.length - i;
  if (rest === 1) {
    const n = bytes[i] << 16;
    out += B64[n >> 18] + B64[(n >> 12) & 63] + '==';
  } else if (rest === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += B64[n >> 18] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + '=';
  }
  return out;
}

export function base64ToBytes(text: string): Uint8Array {
  const clean = text.replace(/=+$/, '');
  const bytes = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const a = B64_LOOKUP[clean.charCodeAt(i)];
    const b = B64_LOOKUP[clean.charCodeAt(i + 1)];
    const c = i + 2 < clean.length ? B64_LOOKUP[clean.charCodeAt(i + 2)] : 0;
    const d = i + 3 < clean.length ? B64_LOOKUP[clean.charCodeAt(i + 3)] : 0;
    const n = (a << 18) | (b << 12) | (c << 6) | d;
    if (o < bytes.length) bytes[o++] = (n >> 16) & 255;
    if (o < bytes.length) bytes[o++] = (n >> 8) & 255;
    if (o < bytes.length) bytes[o++] = n & 255;
  }
  return bytes;
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** IEEE-754 binary16 encoding with round-to-nearest-even. */
export function toFloat16Bits(value: number): number {
  f32Scratch[0] = value;
  const x = u32Scratch[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // Inf / NaN
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow → Inf
  if (e <= 0) {
    if (e < -10) return sign; // underflow → ±0
    mant = (mant | 0x800000) >>> (1 - e);
    if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) === 0x1000 && mant & 0x2000)) mant += 0x2000;
    return sign | (mant >>> 13);
  }
  if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) === 0x1000 && mant & 0x2000)) {
    mant += 0x2000;
    if (mant & 0x800000) {
      mant = 0;
      e += 1;
      if (e >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (e << 10) | (mant >>> 13);
}

export function fromFloat16Bits(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

// ---------------------------------------------------------------------------
// Tensor encoding
// ---------------------------------------------------------------------------

/** Shape of a (nested) rectangular numeric array, or null if it is not one. */
function numericShape(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  if (value.length === 0) return null;
  if (value.every((v) => typeof v === 'number')) return [value.length];
  const inner = numericShape(value[0]);
  if (!inner) return null;
  for (const v of value) {
    const s = numericShape(v);
    if (!s || s.length !== inner.length || s.some((d, i) => d !== inner[i])) return null;
  }
  return [value.length, ...inner];
}

function flatten(value: unknown, out: number[]): void {
  if (Array.isArray(value)) for (const v of value) flatten(v, out);
  else out.push(value as number);
}

function encodeTensor(
  value: unknown[],
  shape: number[],
  precision: TensorPrecision
): EncodedTensor {
  const flat: number[] = [];
  flatten(value, flat);
  let bytes: Uint8Array;
  if (precision === 'float16') {
    const half = new Uint16Array(flat.length);
    for (let i = 0; i < flat.length; i++) half[i] = toFloat16Bits(flat[i]);
    bytes = new Uint8Array(half.buffer);
  } else {
    bytes = new Uint8Array(Float32Array.from(flat).buffer);
  }
  return {
    $tensor: shape,
    dtype: precision === 'float16' ? 'f16' : 'f32',
    data: bytesToBase64(bytes)
  };
}

function decodeTensor(t: EncodedTensor): unknown {
  const bytes = base64ToBytes(t.data);
  const values: ArrayLike<number> =
    t.dtype === 'f16'
      ? Array.from(
          new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2),
          fromFloat16Bits
        )
      : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const total = t.$tensor.reduce((a, b) => a * b, 1);
  if (values.length !== total) throw new Error('Tensor payload does not match its shape');
  let offset = 0;
  const build = (dim: number): unknown => {
    const size = t.$tensor[dim];
    if (dim === t.$tensor.length - 1) {
      const row = new Array<number>(size);
      for (let i = 0; i < size; i++) row[i] = values[offset++];
      return row;
    }
    return Array.from({ length: size }, () => build(dim + 1));
  };
  return build(0);
}

function isEncodedTensor(value: unknown): value is EncodedTensor {
  return (
    !!value &&
    typeof value === 'object' &&
    Array.isArray((value as EncodedTensor).$tensor) &&
    typeof (value as EncodedTensor).data === 'string'
  );
}

/** 64-bit FNV-1a (two 32-bit lanes) over every tensor payload, in traversal order. */
function checksumOf(node: unknown, state = { a: 0x811c9dc5, b: 0xcbf29ce4 }): typeof state {
  if (isEncodedTensor(node)) {
    for (let i = 0; i < node.data.length; i++) {
      const c = node.data.charCodeAt(i);
      state.a = Math.imul(state.a ^ c, 0x01000193) >>> 0;
      state.b = Math.imul(state.b ^ c, 0x01000193 ^ 0x5bd1e995) >>> 0;
    }
  } else if (Array.isArray(node)) {
    for (const v of node) checksumOf(v, state);
  } else if (node && typeof node === 'object') {
    for (const key of Object.keys(node).sort())
      checksumOf((node as Record<string, unknown>)[key], state);
  }
  return state;
}

function formatChecksum(state: { a: number; b: number }): string {
  return `fnv1a64:${state.a.toString(16).padStart(8, '0')}${state.b.toString(16).padStart(8, '0')}`;
}

function encodeNode(node: unknown, precision: TensorPrecision, keepOptimizer: boolean): unknown {
  if (Array.isArray(node)) {
    const shape = numericShape(node);
    if (shape && shape.reduce((a, b) => a * b, 1) >= MIN_TENSOR_SIZE) {
      return encodeTensor(node, shape, precision);
    }
    return node.map((v) => encodeNode(v, precision, keepOptimizer));
  }
  if (node && typeof node === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      if (!keepOptimizer && OPTIMIZER_STATE_KEYS.has(key)) continue;
      out[key] = encodeNode(value, precision, keepOptimizer);
    }
    return out;
  }
  return node;
}

function decodeNode(node: unknown): unknown {
  if (isEncodedTensor(node)) return decodeTensor(node);
  if (Array.isArray(node)) return node.map(decodeNode);
  if (node && typeof node === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      out[key] = decodeNode(value);
    }
    return out;
  }
  return node;
}

function architectureOf(modelJson: Record<string, unknown>): ModelPackage['architecture'] {
  return modelJson.architecture === 'transformer' || modelJson.architecture === 'advanced'
    ? modelJson.architecture
    : 'feedforward';
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function packModel(
  modelJson: Record<string, unknown>,
  options: { precision?: TensorPrecision; includeOptimizerState?: boolean; card?: ModelCard } = {}
): ModelPackage {
  const precision = options.precision ?? 'float32';
  const includeOptimizerState = options.includeOptimizerState ?? false;
  // Round-trip through JSON first so that typed arrays / undefined values are normalised.
  const plain = JSON.parse(JSON.stringify(modelJson)) as Record<string, unknown>;
  const model = encodeNode(plain, precision, includeOptimizerState);
  return {
    format: MODEL_PACKAGE_FORMAT,
    formatVersion: MODEL_PACKAGE_VERSION,
    createdAt: new Date().toISOString(),
    architecture: architectureOf(plain),
    precision,
    includesOptimizerState: includeOptimizerState,
    checksum: formatChecksum(checksumOf(model)),
    ...(options.card ? { card: options.card } : {}),
    model
  };
}

export function isModelPackage(value: unknown): value is ModelPackage {
  return (
    !!value &&
    typeof value === 'object' &&
    (value as ModelPackage).format === MODEL_PACKAGE_FORMAT &&
    typeof (value as ModelPackage).formatVersion === 'number'
  );
}

/** Verify the checksum and restore the plain `toJSON()` object. */
export function unpackModel(pkg: ModelPackage): Record<string, unknown> {
  if (!isModelPackage(pkg)) throw new Error('Not a neuro-lingua model package');
  if (pkg.formatVersion > MODEL_PACKAGE_VERSION) {
    throw new Error(`Unsupported package version ${pkg.formatVersion}`);
  }
  const actual = formatChecksum(checksumOf(pkg.model));
  if (actual !== pkg.checksum) {
    throw new Error(`Package checksum mismatch (expected ${pkg.checksum}, got ${actual})`);
  }
  return decodeNode(pkg.model) as Record<string, unknown>;
}

/** Render the model card as Markdown. */
export function renderModelCard(card: ModelCard): string {
  const section = (title: string, data?: Record<string, unknown>) =>
    data && Object.keys(data).length
      ? [
          `## ${title}`,
          '',
          ...Object.entries(data).map(
            ([k, v]) => `- **${k}**: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`
          ),
          ''
        ]
      : [];
  return [
    `# ${card.name}`,
    '',
    ...(card.description ? [card.description, ''] : []),
    `- **Architecture**: ${card.architecture}`,
    `- **Parameters**: ${card.parameters.toLocaleString('en-US')}`,
    `- **Vocabulary**: ${card.vocabularySize}`,
    `- **Context window**: ${card.contextSize}`,
    `- **Created**: ${card.createdAt}`,
    '',
    ...section('Tokenizer', card.tokenizer),
    ...section('Data', card.data),
    ...section('Training', card.training),
    ...section('Metrics', card.metrics),
    ...section('Environment', card.environment),
    '## Provenance',
    '',
    card.provenance.statement,
    '',
    `- **Offline**: ${
      card.provenance.offline === null ? 'unknown' : card.provenance.offline ? 'yes' : 'no'
    }`,
    `- **External services**: ${
      card.provenance.externalServices.length ? card.provenance.externalServices.join(', ') : 'none'
    }`,
    ''
  ].join('\n');
}

/**
 * Read a model saved in localStorage-like storage: either the plain `toJSON()` object or a model
 * package (used when the plain JSON does not fit). Returns the plain object, or null.
 */
export function parseStoredModel(raw: string | null): Record<string, unknown> | null {
  if (!raw) return null;
  const data = JSON.parse(raw);
  return isModelPackage(data) ? unpackModel(data) : (data as Record<string, unknown>);
}

/**
 * Save a model's `toJSON()` object with `setItem`, falling back to progressively smaller
 * encodings when the storage quota is exceeded: plain JSON → float32 package → float16 package.
 * Returns the encoding that was stored, or null when even the smallest one did not fit.
 */
export function storeModelCompactly(
  setItem: (value: string) => void,
  modelJson: Record<string, unknown>
): 'json' | 'float32' | 'float16' | null {
  const attempts: Array<['json' | 'float32' | 'float16', () => string]> = [
    ['json', () => JSON.stringify(modelJson)],
    ['float32', () => JSON.stringify(packModel(modelJson, { precision: 'float32' }))],
    ['float16', () => JSON.stringify(packModel(modelJson, { precision: 'float16' }))]
  ];
  for (const [encoding, encode] of attempts) {
    try {
      setItem(encode());
      return encoding;
    } catch {
      // quota exceeded (or serialization failure): try the next, smaller encoding
    }
  }
  return null;
}
