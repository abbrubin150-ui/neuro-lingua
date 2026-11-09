/**
 * Exact (analytic) forward/backward pass for the pre-norm transformer block used by
 * TransformerLM.
 *
 * Forward (dropout disabled it mirrors MiniTransformerBlock exactly):
 *
 *   N1  = RMSNorm(X; γ_attn)
 *   Q   = RoPE(N1·W̃q)     K = RoPE(N1·W̃k)     V = N1·W̃v        (GQA: K/V have kvDim columns)
 *   P_h = softmax(Q_h·K_gᵀ / √d_h  [+ causal mask])               g = KV-group of query head h
 *   O_h = (P_h ⊙ M_h)·V_g                                           M_h: attention-dropout mask
 *   A   = concat(O_h) / √d_h
 *   R   = X + A
 *   N2  = RMSNorm(R; γ_ffn)
 *   P   = N2·W1 = [a | b]
 *   Y   = R + (a ⊙ swish(b))·W2
 *
 * W̃ = W ⊙ D / (1 − p) when DropConnect is active (training only), otherwise W̃ = W.
 * Every parameter (Wq, Wk, Wv, W1, W2, γ_attn, γ_ffn) and the block input receive their true
 * gradients, including through the dropout masks. The implementation is CPU-only and written for
 * clarity; correctness is verified by finite-difference gradient checks in
 * tests/models/transformer_backprop.test.ts.
 */

import type { AttentionWeights, Matrix } from './attention';

export interface BlockParams {
  attention: AttentionWeights;
  ff1: Matrix;
  ff2: Matrix;
  attnGamma: number[];
  ffnGamma: number[];
  attnEpsilon: number;
  ffnEpsilon: number;
}

export interface BlockConfig {
  numHeads: number;
  numKVHeads: number;
  ropeBase: number;
  /** Causal (autoregressive) attention: position i only attends to positions j ≤ i. */
  causal?: boolean;
  /**
   * Apply RoPE independently inside every head with head-dimension frequencies
   * θ_i = base^(−2i/d_h) (standard RoPE). When false, the legacy layout rotates pairs across the
   * whole projection row with θ_i = base^(−2i/width), which gives query and key pairs different
   * frequencies under GQA; it is kept only to reproduce models saved before per-head RoPE.
   */
  ropePerHead?: boolean;
}

/** Training-time stochastic regularisation for one block. */
export interface BlockDropout {
  /** Probability of zeroing an attention probability (after softmax). */
  attention: number;
  /** Probability of zeroing an individual Q/K/V weight (DropConnect). */
  dropConnect: number;
  /** Uniform [0, 1) random source (use the model's seeded RNG for reproducibility). */
  random: () => number;
}

/** Inverted-dropout masks (entries are 0 or 1/(1−p)); null when the regulariser is off. */
export interface BlockMasks {
  attention: Matrix[] | null;
  query: Matrix | null;
  key: Matrix | null;
  value: Matrix | null;
}

export interface BlockGrads {
  query: Matrix;
  key: Matrix;
  value: Matrix;
  ff1: Matrix;
  ff2: Matrix;
  attnGamma: number[];
  ffnGamma: number[];
}

export interface BlockCache {
  X: Matrix;
  rms1: number[];
  N1: Matrix;
  Qr: Matrix;
  Kr: Matrix;
  V: Matrix;
  /** Attention probabilities per query head [T x T] (before dropout). */
  attn: Matrix[];
  /** Attention weights actually applied to V (after dropout; same arrays when no dropout). */
  attnUsed: Matrix[];
  R: Matrix;
  rms2: number[];
  N2: Matrix;
  P: Matrix;
  G: Matrix;
  /** Effective (DropConnect-masked) projection weights used in this pass. */
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
  masks: BlockMasks;
}

export interface BlockForwardOptions {
  /** Sample fresh dropout masks (training). Ignored when `masks` is given. */
  dropout?: BlockDropout;
  /** Re-use previously sampled masks (e.g. for gradient checking). */
  masks?: BlockMasks;
}

/**
 * Sample DropConnect masks for a block's Q/K/V weights. A caller that keeps the masks fixed for
 * several passes (e.g. a mini-batch) can run the block on W̃ = W ⊙ M with no DropConnect inside
 * and convert the accumulated gradient with ∂L/∂W = M ⊙ ∂L/∂W̃.
 */
export function sampleWeightMasks(
  params: BlockParams,
  rate: number,
  random: () => number
): Pick<BlockMasks, 'query' | 'key' | 'value'> {
  if (rate <= 0) return { query: null, key: null, value: null };
  const shapeMask = (M: Matrix) => sampleMask(M.length, M[0]?.length ?? 0, rate, random);
  return {
    query: shapeMask(params.attention.query),
    key: shapeMask(params.attention.key),
    value: shapeMask(params.attention.value)
  };
}

// ---------------------------------------------------------------------------
// Small dense helpers
// ---------------------------------------------------------------------------

export function zeros(r: number, c: number): Matrix {
  return Array.from({ length: r }, () => new Array(c).fill(0));
}

/** A[T x n] · B[n x m] */
function matmul(A: Matrix, B: Matrix): Matrix {
  const T = A.length;
  const n = B.length;
  const m = n > 0 ? B[0].length : 0;
  const out = zeros(T, m);
  for (let t = 0; t < T; t++) {
    const a = A[t];
    const o = out[t];
    for (let k = 0; k < n; k++) {
      const v = a[k];
      if (v === 0) continue;
      const b = B[k];
      for (let j = 0; j < m; j++) o[j] += v * b[j];
    }
  }
  return out;
}

/** A[T x n] · B[m x n]ᵀ  →  [T x m] */
function matmulBT(A: Matrix, B: Matrix): Matrix {
  const out = zeros(A.length, B.length);
  for (let t = 0; t < A.length; t++) {
    const a = A[t];
    for (let j = 0; j < B.length; j++) {
      const b = B[j];
      let s = 0;
      for (let k = 0; k < a.length; k++) s += a[k] * b[k];
      out[t][j] = s;
    }
  }
  return out;
}

/** accumulate Aᵀ[n x T] · B[T x m] into G[n x m] */
function accumATB(G: Matrix, A: Matrix, B: Matrix): void {
  for (let t = 0; t < A.length; t++) {
    const a = A[t];
    const b = B[t];
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (v === 0) continue;
      const g = G[i];
      for (let j = 0; j < b.length; j++) g[j] += v * b[j];
    }
  }
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function rmsForward(X: Matrix, gamma: number[], eps: number): { N: Matrix; rms: number[] } {
  const rms: number[] = [];
  const N = X.map((row) => {
    let ss = 0;
    for (const v of row) ss += v * v;
    const r = Math.sqrt(ss / row.length + eps);
    rms.push(r);
    return row.map((v, i) => (v / r) * gamma[i]);
  });
  return { N, rms };
}

/**
 * Backward through y = (x / r) ⊙ γ with r = sqrt(mean(x²) + ε).
 * dx_j = γ_j·dy_j / r − x_j·Σ_i(γ_i·dy_i·x_i) / (D·r³)
 */
export function rmsBackward(
  dN: Matrix,
  X: Matrix,
  rms: number[],
  gamma: number[],
  dGamma: number[]
): Matrix {
  return X.map((x, t) => {
    const r = rms[t];
    const D = x.length;
    const dy = dN[t];
    let dot = 0;
    for (let i = 0; i < D; i++) {
      dGamma[i] += (dy[i] * x[i]) / r;
      dot += gamma[i] * dy[i] * x[i];
    }
    const k = dot / (D * r * r * r);
    const dx = new Array(D);
    for (let j = 0; j < D; j++) dx[j] = (gamma[j] * dy[j]) / r - x[j] * k;
    return dx;
  });
}

function ropeThetas(dim: number, base: number): number[] {
  const half = Math.floor(dim / 2);
  const theta: number[] = [];
  for (let i = 0; i < half; i++) theta.push(Math.pow(base, (-2 * i) / dim));
  return theta;
}

/**
 * Rotate (even, odd) pairs by pos·θ. `sign = -1` applies the inverse (transpose) rotation.
 *
 * headDim = null → legacy full-row layout (no rotation at all when the row width is odd).
 * headDim = d    → per-head layout: each consecutive slice of width d is rotated with
 *                  θ_i = base^(−2i/d); with odd d the last component of every head is left as is.
 */
export function rope(
  M: Matrix,
  positions: number[],
  base: number,
  sign: 1 | -1,
  headDim: number | null = null
): Matrix {
  const width = M[0]?.length ?? 0;
  if (width === 0 || positions.length === 0) return M.map((r) => r.slice());
  if (headDim === null && width % 2 !== 0) return M.map((r) => r.slice());
  const span = headDim ?? width;
  const theta = ropeThetas(span, base);
  const numSlices = Math.floor(width / span);
  return M.map((row, t) => {
    const pos = positions[t] ?? t;
    const out = row.slice();
    for (let s = 0; s < numSlices; s++) {
      const offset = s * span;
      for (let i = 0; i < theta.length; i++) {
        const ang = pos * theta[i];
        const c = Math.cos(ang);
        const sn = sign * Math.sin(ang);
        const e = row[offset + 2 * i];
        const o = row[offset + 2 * i + 1];
        out[offset + 2 * i] = e * c - o * sn;
        out[offset + 2 * i + 1] = e * sn + o * c;
      }
    }
    return out;
  });
}

function softmaxRow(row: number[]): number[] {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  const ex = row.map((v) => Math.exp(v - max));
  let sum = 0;
  for (const v of ex) sum += v;
  return ex.map((v) => v / sum);
}

// ---------------------------------------------------------------------------
// Block forward / backward
// ---------------------------------------------------------------------------

const NO_MASKS: BlockMasks = { attention: null, query: null, key: null, value: null };

function sampleMask(rows: number, cols: number, rate: number, random: () => number): Matrix {
  const keep = 1 - rate;
  const scaleKept = 1 / keep;
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (random() < keep ? scaleKept : 0))
  );
}

function sampleMasks(
  T: number,
  params: BlockParams,
  config: BlockConfig,
  dropout?: BlockDropout
): BlockMasks {
  if (!dropout) return NO_MASKS;
  const { attention, dropConnect, random } = dropout;
  return {
    attention:
      attention > 0
        ? Array.from({ length: config.numHeads }, () => sampleMask(T, T, attention, random))
        : null,
    ...sampleWeightMasks(params, dropConnect, random)
  };
}

/** Element-wise product W ⊙ M (returns W itself when M is null). */
export function hadamard(W: Matrix, M: Matrix | null): Matrix {
  if (!M) return W;
  return W.map((row, i) => row.map((v, j) => v * M[i][j]));
}

export function blockForward(
  X: Matrix,
  params: BlockParams,
  config: BlockConfig,
  positions: number[],
  options: BlockForwardOptions = {}
): { out: Matrix; cache: BlockCache } {
  const T = X.length;
  const D = X[0].length;
  const hd = D / config.numHeads;
  const group = config.numHeads / config.numKVHeads;
  const scale = 1 / Math.sqrt(hd);
  const causal = config.causal ?? false;
  const masks = options.masks ?? sampleMasks(T, params, config, options.dropout);

  const wq = hadamard(params.attention.query, masks.query);
  const wk = hadamard(params.attention.key, masks.key);
  const wv = hadamard(params.attention.value, masks.value);

  const { N: N1, rms: rms1 } = rmsForward(X, params.attnGamma, params.attnEpsilon);
  const ropeSpan = config.ropePerHead ? hd : null;
  const Qr = rope(matmul(N1, wq), positions, config.ropeBase, 1, ropeSpan);
  const Kr = rope(matmul(N1, wk), positions, config.ropeBase, 1, ropeSpan);
  const V = matmul(N1, wv);

  const attnOut = zeros(T, D);
  const attn: Matrix[] = [];
  const attnUsed: Matrix[] = [];
  for (let h = 0; h < config.numHeads; h++) {
    const qo = h * hd;
    const ko = Math.floor(h / group) * hd;
    const A: Matrix = [];
    const Aused: Matrix = [];
    const dropMask = masks.attention ? masks.attention[h] : null;
    for (let i = 0; i < T; i++) {
      const scores = new Array(T);
      for (let j = 0; j < T; j++) {
        if (causal && j > i) {
          scores[j] = -Infinity;
          continue;
        }
        let s = 0;
        for (let d = 0; d < hd; d++) s += Qr[i][qo + d] * Kr[j][ko + d];
        scores[j] = s * scale;
      }
      const a = softmaxRow(scores);
      const used = dropMask ? a.map((v, j) => v * dropMask[i][j]) : a;
      A.push(a);
      Aused.push(used);
      for (let d = 0; d < hd; d++) {
        let s = 0;
        for (let j = 0; j < T; j++) s += used[j] * V[j][ko + d];
        attnOut[i][qo + d] = s * scale; // output projection scale 1/√d_h
      }
    }
    attn.push(A);
    attnUsed.push(Aused);
  }

  const R = X.map((row, t) => row.map((v, j) => v + attnOut[t][j]));
  const { N: N2, rms: rms2 } = rmsForward(R, params.ffnGamma, params.ffnEpsilon);
  const P = matmul(N2, params.ff1);
  const F = P[0].length / 2;
  const G = P.map((p) => {
    const g = new Array(F);
    for (let i = 0; i < F; i++) {
      const b = p[F + i];
      g[i] = p[i] * b * sigmoid(b);
    }
    return g;
  });
  const FF = matmul(G, params.ff2);
  const out = R.map((row, t) => row.map((v, j) => v + FF[t][j]));

  return {
    out,
    cache: { X, rms1, N1, Qr, Kr, V, attn, attnUsed, R, rms2, N2, P, G, wq, wk, wv, masks }
  };
}

export function createBlockGrads(params: BlockParams): BlockGrads {
  const shape = (M: Matrix) => zeros(M.length, M[0]?.length ?? 0);
  return {
    query: shape(params.attention.query),
    key: shape(params.attention.key),
    value: shape(params.attention.value),
    ff1: shape(params.ff1),
    ff2: shape(params.ff2),
    attnGamma: new Array(params.attnGamma.length).fill(0),
    ffnGamma: new Array(params.ffnGamma.length).fill(0)
  };
}

/** grads += mask ⊙ (Aᵀ·B)  (mask null → plain accumulation) */
function accumATBMasked(G: Matrix, A: Matrix, B: Matrix, mask: Matrix | null): void {
  if (!mask) {
    accumATB(G, A, B);
    return;
  }
  const tmp = zeros(G.length, G[0]?.length ?? 0);
  accumATB(tmp, A, B);
  for (let i = 0; i < G.length; i++) {
    for (let j = 0; j < G[i].length; j++) G[i][j] += tmp[i][j] * mask[i][j];
  }
}

/**
 * Backpropagate dOut through one block. Parameter gradients are accumulated into `grads`;
 * the gradient w.r.t. the block input is returned.
 */
export function blockBackward(
  dOut: Matrix,
  cache: BlockCache,
  params: BlockParams,
  config: BlockConfig,
  positions: number[],
  grads: BlockGrads
): Matrix {
  const { X, rms1, N1, Qr, Kr, V, attn, attnUsed, R, rms2, N2, P, G, wq, wk, wv, masks } = cache;
  const T = X.length;
  const D = X[0].length;
  const hd = D / config.numHeads;
  const group = config.numHeads / config.numKVHeads;
  const scale = 1 / Math.sqrt(hd);
  const F = G[0].length;

  // ---- FFN sub-layer ----
  accumATB(grads.ff2, G, dOut);
  const dG = matmulBT(dOut, params.ff2); // [T x F]
  const dP = P.map((p, t) => {
    const row = new Array(2 * F);
    for (let i = 0; i < F; i++) {
      const a = p[i];
      const b = p[F + i];
      const sg = sigmoid(b);
      const sw = b * sg;
      const dsw = sg + b * sg * (1 - sg);
      row[i] = dG[t][i] * sw;
      row[F + i] = dG[t][i] * a * dsw;
    }
    return row;
  });
  accumATB(grads.ff1, N2, dP);
  const dN2 = matmulBT(dP, params.ff1);
  const dRfromFfn = rmsBackward(dN2, R, rms2, params.ffnGamma, grads.ffnGamma);
  const dR = dOut.map((row, t) => row.map((v, j) => v + dRfromFfn[t][j]));

  // ---- Attention sub-layer ----
  const kvDim = V[0].length;
  const dQr = zeros(T, D);
  const dKr = zeros(T, kvDim);
  const dV = zeros(T, kvDim);
  for (let h = 0; h < config.numHeads; h++) {
    const qo = h * hd;
    const ko = Math.floor(h / group) * hd;
    const A = attn[h];
    const Aused = attnUsed[h];
    const dropMask = masks.attention ? masks.attention[h] : null;
    for (let i = 0; i < T; i++) {
      // dO_h[i] = scale · dR[i, head slice]
      const dO = new Array(hd);
      for (let d = 0; d < hd; d++) dO[d] = dR[i][qo + d] * scale;
      // dA'[i][j] = dO · V_j ; dV_j += A'[i][j] · dO ; dA = dA' ⊙ mask
      const dA = new Array(T);
      let rowDot = 0;
      for (let j = 0; j < T; j++) {
        let s = 0;
        const aUsed = Aused[i][j];
        for (let d = 0; d < hd; d++) {
          s += dO[d] * V[j][ko + d];
          dV[j][ko + d] += aUsed * dO[d];
        }
        if (dropMask) s *= dropMask[i][j];
        dA[j] = s;
        rowDot += s * A[i][j];
      }
      // softmax backward then score scaling (masked causal entries have A = 0 → dS = 0)
      for (let j = 0; j < T; j++) {
        const dS = A[i][j] * (dA[j] - rowDot) * scale;
        if (dS === 0) continue;
        for (let d = 0; d < hd; d++) {
          dQr[i][qo + d] += dS * Kr[j][ko + d];
          dKr[j][ko + d] += dS * Qr[i][qo + d];
        }
      }
    }
  }

  const ropeSpan = config.ropePerHead ? hd : null;
  const dQ = rope(dQr, positions, config.ropeBase, -1, ropeSpan);
  const dK = rope(dKr, positions, config.ropeBase, -1, ropeSpan);
  accumATBMasked(grads.query, N1, dQ, masks.query);
  accumATBMasked(grads.key, N1, dK, masks.key);
  accumATBMasked(grads.value, N1, dV, masks.value);

  const dN1 = matmulBT(dQ, wq);
  const dN1k = matmulBT(dK, wk);
  const dN1v = matmulBT(dV, wv);
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < D; j++) dN1[t][j] += dN1k[t][j] + dN1v[t][j];
  }
  const dXfromAttn = rmsBackward(dN1, X, rms1, params.attnGamma, grads.attnGamma);

  return dR.map((row, t) => row.map((v, j) => v + dXfromAttn[t][j]));
}
