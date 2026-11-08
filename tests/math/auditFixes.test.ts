import { describe, expect, it } from 'vitest';
import { knnEntropy } from '../../src/math/information_theory';
import { computeLaplacian, analyzeAttentionGraph } from '../../src/math/spectral_graph';
import { estimateConditionNumber } from '../../src/math/numerics';
import { computeEValue } from '../../src/math/dag_operations';

function gaussianSampler(seed: number) {
  let s = seed;
  const u = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296 || 1e-12);
  return () => Math.sqrt(-2 * Math.log(u())) * Math.cos(2 * Math.PI * u());
}

describe('Kozachenko-Leonenko entropy (knnEntropy)', () => {
  it('matches the analytic entropy of N(0, σ²) (no spurious d·ln 2 term)', () => {
    const randn = gaussianSampler(3);
    const X = Array.from({ length: 2000 }, () => [2 * randn()]);
    const analytic = 0.5 * Math.log(2 * Math.PI * Math.E * 4);
    expect(Math.abs(knnEntropy(X, 3) - analytic)).toBeLessThan(0.06);
  });

  it('is ≈ 0 for the uniform distribution on the unit square', () => {
    let s = 9;
    const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
    const X = Array.from({ length: 2000 }, () => [rnd(), rnd()]);
    expect(Math.abs(knnEntropy(X, 3))).toBeLessThan(0.06);
  });
});

describe('graph Laplacian and spectrum', () => {
  it('symmetrises directed attention and ignores self-loops', () => {
    const A = [
      [0.5, 0.4, 0.1],
      [0.1, 0.8, 0.1],
      [0.3, 0.3, 0.4]
    ];
    const L = computeLaplacian(A, false);
    for (let i = 0; i < 3; i++) {
      expect(L[i].reduce((s, v) => s + v, 0)).toBeCloseTo(0, 12); // L·1 = 0
      for (let j = 0; j < 3; j++) expect(L[i][j]).toBeCloseTo(L[j][i], 12);
    }
    expect(analyzeAttentionGraph(A).eigenvalues[0]).toBeCloseTo(0, 10);
  });

  it('recovers the full spectrum of a 15-node path graph (more than 10 nodes)', () => {
    const n = 15;
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (Math.abs(i - j) === 1 ? 1 : 0))
    );
    const { eigenvalues, algebraicConnectivity, numComponents } = analyzeAttentionGraph(A);
    const expected = Array.from({ length: n }, (_, k) => 1 - Math.cos((Math.PI * k) / (n - 1)));
    eigenvalues.forEach((v, k) => expect(v).toBeCloseTo(expected[k], 10));
    expect(algebraicConnectivity).toBeCloseTo(expected[1], 10);
    expect(numComponents).toBe(1);
  });

  it('pairs the Fiedler vector with λ₂: two weakly linked cliques are separated', () => {
    const n = 8;
    const A = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        if (i === j) return 0;
        const sameClique = i < 4 === j < 4;
        return sameClique ? 1 : 0.02;
      })
    );
    const { communityStructure, numComponents } = analyzeAttentionGraph(A);
    expect(numComponents).toBe(1);
    const groups = communityStructure.map((g) => [...g].sort((a, b) => a - b));
    expect(groups).toContainEqual([0, 1, 2, 3]);
    expect(groups).toContainEqual([4, 5, 6, 7]);
  });

  it('counts connected components from zero eigenvalues', () => {
    const A = Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 6 }, (_, j) => (i !== j && i < 3 === j < 3 ? 1 : 0))
    );
    expect(analyzeAttentionGraph(A).numComponents).toBe(2);
  });
});

describe('estimateConditionNumber (κ₁)', () => {
  it('is exact for diagonal and Hilbert matrices', () => {
    expect(
      estimateConditionNumber([
        [1, 0],
        [0, 1e-3]
      ])
    ).toBeCloseTo(1000, 8);
    const H = Array.from({ length: 5 }, (_, i) => Array.from({ length: 5 }, (_, j) => 1 / (i + j + 1)));
    expect(estimateConditionNumber(H) / 943656).toBeCloseTo(1, 6);
  });

  it('is infinite for a singular matrix', () => {
    expect(
      estimateConditionNumber([
        [1, 2],
        [2, 4]
      ])
    ).toBe(Infinity);
  });
});

describe('computeEValue confidence-interval rule', () => {
  it('is 1 when the confidence interval contains the null', () => {
    expect(computeEValue(1.4, 0.8, 2.1).ciLowerBound).toBe(1);
    expect(computeEValue(0.7, 0.5, 1.1).ciLowerBound).toBe(1);
  });

  it('uses the limit closest to the null (upper limit for protective effects)', () => {
    const e = (rr: number) => rr + Math.sqrt(rr * (rr - 1));
    expect(computeEValue(2, 1.5, 2.7).ciLowerBound).toBeCloseTo(e(1.5), 12);
    expect(computeEValue(0.5, 0.3, 0.8).ciLowerBound).toBeCloseTo(e(1 / 0.8), 12);
  });
});
