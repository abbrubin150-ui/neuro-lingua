import { describe, expect, it } from 'vitest';

import { analyzeLyapunov, spectralRadius } from '../../src/math/analysis';
import {
  computeDiagonalHessian,
  dampedNewtonStep,
  createSecondOrderState,
  quasiNewtonStep
} from '../../src/training/optimizer';

function diagonalMatrix(values: number[]): number[][] {
  return values.map((value, row) => values.map((_, col) => (row === col ? value : 0)));
}

describe('spectral analysis', () => {
  it('computes spectral radius of a diagonal matrix', () => {
    const matrix = diagonalMatrix([2, -3, 0.5]);
    const { spectralRadius: radius, converged } = spectralRadius(matrix, {
      tolerance: 1e-10,
      maxIterations: 256
    });
    expect(converged).toBe(true);
    expect(radius).toBeCloseTo(3, 1e-6);
  });

  it('flags Lyapunov stability for contractive systems', () => {
    const matrix = diagonalMatrix([0.6, 0.2]);
    const result = analyzeLyapunov(matrix, { tolerance: 1e-9, maxIterations: 128 });
    expect(result.stable).toBe(true);
    expect(result.stabilityMargin).toBeGreaterThan(0.3);
    expect(result.lyapunovExponent).toBeLessThan(0);
  });

  it('detects instability when spectral radius exceeds one', () => {
    const matrix = diagonalMatrix([1.4, 0.7]);
    const result = analyzeLyapunov(matrix, { tolerance: 1e-9, maxIterations: 128 });
    expect(result.stable).toBe(false);
    expect(result.spectralRadius).toBeGreaterThan(1);
  });
});

describe('second-order optimizer primitives', () => {
  it('builds a damped Newton step opposite to the gradient', () => {
    const grad = Float64Array.from([1, -2]);
    const diag = computeDiagonalHessian(grad, 1e-9);
    const step = dampedNewtonStep(grad, diag, { learningRate: 1, damping: 1e-4 });
    expect(step[0]).toBeLessThan(0);
    expect(step[1]).toBeGreaterThan(0);
    expect(Math.abs(step[0])).toBeLessThanOrEqual(1);
    expect(Math.abs(step[1])).toBeLessThan(1);
  });

  it('quasi-Newton updates converge on a quadratic bowl', () => {
    const state = createSecondOrderState(5);
    let params = Float64Array.from([1, -1]);
    for (let iter = 0; iter < 8; iter++) {
      const grad = Float64Array.from(params);
      const step = quasiNewtonStep(params, grad, state, { learningRate: 0.8, epsilon: 1e-12 });
      params = Float64Array.from(params.map((value, i) => value + step[i]));
    }
    const norm = Math.hypot(...params);
    expect(norm).toBeLessThan(1e-3);
  });
});
