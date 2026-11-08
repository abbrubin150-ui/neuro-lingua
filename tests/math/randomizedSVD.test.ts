import { describe, expect, it } from 'vitest';
import { randomizedSVD } from '../../src/math/approximation';

/** Orthonormal columns via Gram-Schmidt of a deterministic pseudo-random matrix. */
function orthonormal(rows: number, cols: number, seed: number): number[][] {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5);
  const cols_: number[][] = [];
  for (let c = 0; c < cols; c++) {
    const v = Array.from({ length: rows }, rnd);
    for (const q of cols_) {
      const d = v.reduce((acc, x, i) => acc + x * q[i], 0);
      for (let i = 0; i < rows; i++) v[i] -= d * q[i];
    }
    const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));
    cols_.push(v.map((x) => x / norm));
  }
  return cols_; // list of column vectors
}

describe('randomizedSVD', () => {
  it.each([
    [12, 8],
    [8, 12],
    [10, 10]
  ])('recovers the leading singular triplets of a %ix%i matrix with known spectrum', (m, n) => {
    const r = Math.min(m, n);
    const sigma = Array.from({ length: r }, (_, i) => 10 * Math.pow(0.5, i));
    const U = orthonormal(m, r, 7);
    const V = orthonormal(n, r, 11);
    const A = Array.from({ length: m }, (_, i) =>
      Array.from({ length: n }, (_, j) => sigma.reduce((acc, sk, k) => acc + sk * U[k][i] * V[k][j], 0))
    );

    const k = 3;
    const result = randomizedSVD(A, k);
    result.singularValues.forEach((s, i) => expect(s).toBeCloseTo(sigma[i], 6));
    expect(result.errorBound).toBeCloseTo(sigma[k], 6);

    // Orthonormal singular vectors that reproduce A·v = σ·u
    for (let a = 0; a < k; a++) {
      const u = result.U[a];
      const v = result.V[a];
      expect(u.length).toBe(m);
      expect(v.length).toBe(n);
      expect(u.reduce((acc, x) => acc + x * x, 0)).toBeCloseTo(1, 8);
      const Av = A.map((row) => row.reduce((acc, x, j) => acc + x * v[j], 0));
      Av.forEach((x, i) => expect(x).toBeCloseTo(result.singularValues[a] * u[i], 6));
    }
  });
});
