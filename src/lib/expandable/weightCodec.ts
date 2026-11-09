/**
 * Helpers for the Cerebro adapters' flat Float32Array weight snapshots.
 *
 * Snapshots are taken before an injection and restored on rollback, i.e. when the model has a
 * *different* width than the snapshot. Shapes must therefore be inferred from the snapshot
 * itself (array lengths and the hidden size encoded by the bias vector), never from the live
 * model.
 */

/** Reshape a row-major flat array into `rows` rows (columns = length / rows). */
export function unflatten(flat: ArrayLike<number>, rows: number): number[][] {
  if (rows <= 0) return [];
  const cols = Math.floor(flat.length / rows);
  if (cols * rows !== flat.length) {
    throw new Error(`Cannot reshape ${flat.length} values into ${rows} rows`);
  }
  const out: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Array<number>(cols);
    for (let j = 0; j < cols; j++) row[j] = flat[i * cols + j];
    out[i] = row;
  }
  return out;
}

/**
 * Rebuild the ProNeuralLM base tensors from the first five snapshot arrays
 * [wHidden, bHidden, wOutput, bOutput, embedding].
 *
 * Shapes: wHidden H×E, bHidden H, wOutput H×V, bOutput V, embedding V×E.
 */
export function decodeBaseWeights(
  weights: Float32Array[],
  vocabSize: number
): {
  wHidden: number[][];
  bHidden: number[];
  wOutput: number[][];
  bOutput: number[];
  embedding: number[][];
} {
  if (weights.length < 5) {
    throw new Error('Invalid weights array: expected 5 base arrays');
  }
  const bHidden = Array.from(weights[1]);
  const hiddenSize = bHidden.length;
  return {
    wHidden: unflatten(weights[0], hiddenSize),
    bHidden,
    wOutput: unflatten(weights[2], hiddenSize),
    bOutput: Array.from(weights[3]),
    embedding: unflatten(weights[4], vocabSize)
  };
}
