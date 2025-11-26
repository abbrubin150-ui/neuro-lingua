/**
 * Int8 Quantization Module
 *
 * Converts float32 model weights to int8 format for:
 * - 4x size reduction (32 bits → 8 bits)
 * - Faster loading and serialization
 * - Minimal accuracy loss (<2% typically)
 *
 * Quantization formula:
 *   quantized = round((value - min) / (max - min) * 255)
 *   dequantized = quantized / 255 * (max - min) + min
 */

export interface QuantizationParams {
  min: number;
  max: number;
  scale: number;
  zeroPoint: number;
}

export interface QuantizedWeights {
  values: Int8Array;
  params: QuantizationParams;
  originalShape: number[];
}

export interface QuantizedModel {
  embedding: QuantizedWeights;
  wHidden: QuantizedWeights;
  wOutput: QuantizedWeights;
  bHidden: QuantizedWeights;
  bOutput: QuantizedWeights;
  quantizationMethod: 'symmetric' | 'asymmetric';
  compressionRatio: number;
}

/**
 * Quantize a 1D array to int8
 * Uses symmetric quantization: scale = max(abs(min), abs(max)) / 127
 */
export function quantizeArray(arr: number[]): QuantizedWeights {
  if (arr.length === 0) {
    return {
      values: new Int8Array(0),
      params: { min: 0, max: 0, scale: 1, zeroPoint: 0 },
      originalShape: [0]
    };
  }

  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const absMax = Math.max(Math.abs(min), Math.abs(max));

  // Symmetric quantization (zero-centered)
  const scale = absMax / 127;
  const zeroPoint = 0;

  const values = new Int8Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const quantized = Math.round(arr[i] / scale);
    values[i] = Math.max(-128, Math.min(127, quantized));
  }

  return {
    values,
    params: { min, max, scale, zeroPoint },
    originalShape: [arr.length]
  };
}

/**
 * Quantize a 2D matrix to int8
 */
export function quantizeMatrix(matrix: number[][]): QuantizedWeights {
  if (matrix.length === 0 || matrix[0].length === 0) {
    return {
      values: new Int8Array(0),
      params: { min: 0, max: 0, scale: 1, zeroPoint: 0 },
      originalShape: [0, 0]
    };
  }

  const rows = matrix.length;
  const cols = matrix[0].length;

  // Flatten matrix
  const flat: number[] = [];
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      flat.push(matrix[i][j]);
    }
  }

  const quantized = quantizeArray(flat);
  quantized.originalShape = [rows, cols];
  return quantized;
}

/**
 * Dequantize int8 array back to float32
 */
export function dequantizeArray(quantized: QuantizedWeights): number[] {
  const result: number[] = [];
  const { values, params } = quantized;

  for (let i = 0; i < values.length; i++) {
    result.push(values[i] * params.scale);
  }

  return result;
}

/**
 * Dequantize int8 matrix back to float32 2D array
 */
export function dequantizeMatrix(quantized: QuantizedWeights): number[][] {
  const [rows, cols] = quantized.originalShape;
  const flat = dequantizeArray(quantized);

  const matrix: number[][] = [];
  for (let i = 0; i < rows; i++) {
    matrix.push(flat.slice(i * cols, (i + 1) * cols));
  }

  return matrix;
}

/**
 * Calculate quantization error (MSE)
 */
export function calculateQuantizationError(
  original: number[],
  quantized: QuantizedWeights
): number {
  const dequantized = dequantizeArray(quantized);
  let sumSquaredError = 0;

  for (let i = 0; i < original.length; i++) {
    const error = original[i] - dequantized[i];
    sumSquaredError += error * error;
  }

  return sumSquaredError / original.length;
}

/**
 * Calculate compression ratio
 */
export function calculateCompressionRatio(
  originalSize: number,
  compressedSize: number
): number {
  return originalSize / compressedSize;
}

/**
 * Estimate size in bytes for quantized data
 */
export function estimateQuantizedSize(quantized: QuantizedWeights): number {
  // Int8Array size + params (4 floats) + shape (2 ints)
  return quantized.values.length + 4 * 4 + 2 * 4;
}

/**
 * Convert quantized weights to JSON-serializable format
 */
export function serializeQuantizedWeights(quantized: QuantizedWeights): object {
  return {
    values: Array.from(quantized.values),
    params: quantized.params,
    originalShape: quantized.originalShape
  };
}

/**
 * Restore quantized weights from JSON
 */
export function deserializeQuantizedWeights(data: any): QuantizedWeights {
  return {
    values: new Int8Array(data.values),
    params: data.params,
    originalShape: data.originalShape
  };
}
