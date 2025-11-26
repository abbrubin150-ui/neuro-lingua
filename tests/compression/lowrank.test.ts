import { describe, it, expect } from 'vitest';
import {
  transpose,
  matmul,
  frobeniusNorm,
  lowRankApproximation,
  reconstructFromLowRank,
  approximationError,
  findOptimalRank
} from '../../src/compression/lowrank';

describe('Low-Rank Approximation', () => {
  describe('Matrix Operations', () => {
    it('should transpose a matrix correctly', () => {
      const matrix = [
        [1, 2, 3],
        [4, 5, 6]
      ];

      const transposed = transpose(matrix);

      expect(transposed).toEqual([
        [1, 4],
        [2, 5],
        [3, 6]
      ]);
    });

    it('should handle empty matrix transpose', () => {
      const matrix: number[][] = [];
      const transposed = transpose(matrix);
      expect(transposed).toEqual([]);
    });

    it('should multiply matrices correctly', () => {
      const A = [
        [1, 2],
        [3, 4]
      ];

      const B = [
        [5, 6],
        [7, 8]
      ];

      const C = matmul(A, B);

      expect(C).toEqual([
        [19, 22],
        [43, 50]
      ]);
    });

    it('should throw on incompatible dimensions', () => {
      const A = [[1, 2, 3]];
      const B = [[1], [2]];

      expect(() => matmul(A, B)).toThrow();
    });

    it('should calculate Frobenius norm correctly', () => {
      const matrix = [
        [1, 0],
        [0, 1]
      ];

      const norm = frobeniusNorm(matrix);

      expect(norm).toBeCloseTo(Math.sqrt(2), 5);
    });
  });

  describe('Low-Rank Compression', () => {
    it('should compress a matrix to lower rank', () => {
      const matrix = [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [3, 6, 9, 12]
      ];

      const rank = 1;
      const compressed = lowRankApproximation(matrix, rank);

      expect(compressed.rank).toBe(rank);
      expect(compressed.U.length).toBe(3); // m rows
      expect(compressed.U[0].length).toBe(rank); // k columns
      expect(compressed.SV.length).toBe(rank); // k rows
      expect(compressed.SV[0].length).toBe(4); // n columns
    });

    it('should reconstruct matrix from low-rank form', () => {
      const matrix = [
        [1, 2],
        [2, 4]
      ];

      const rank = 1;
      const compressed = lowRankApproximation(matrix, rank);
      const reconstructed = reconstructFromLowRank(compressed);

      expect(reconstructed.length).toBe(2);
      expect(reconstructed[0].length).toBe(2);

      // For rank-deficient matrix, should be nearly perfect
      for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
          expect(Math.abs(matrix[i][j] - reconstructed[i][j])).toBeLessThan(0.5);
        }
      }
    });

    it('should calculate approximation error', () => {
      const original = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ];

      const rank = 2;
      const compressed = lowRankApproximation(original, rank);
      const reconstructed = reconstructFromLowRank(compressed);

      const error = approximationError(original, reconstructed);

      expect(error).toBeGreaterThanOrEqual(0);
      // Should be relatively small for rank 2 approximation
      expect(error).toBeLessThan(10);
    });

    it('should achieve compression ratio > 1', () => {
      const m = 100;
      const n = 50;
      const matrix = Array(m)
        .fill(0)
        .map(() =>
          Array(n)
            .fill(0)
            .map(() => Math.random())
        );

      const rank = 10;
      const compressed = lowRankApproximation(matrix, rank);

      // Original: m × n = 5000 params
      // Compressed: k(m+n) = 10(150) = 1500 params
      // Ratio: 5000/1500 = 3.33x

      expect(compressed.compressionRatio).toBeGreaterThan(1);
      expect(compressed.compressionRatio).toBeCloseTo(3.33, 1);
    });
  });

  describe('Optimal Rank Finding', () => {
    it('should find rank for target compression ratio', () => {
      const m = 100;
      const n = 50;
      const targetRatio = 2.0;

      const rank = findOptimalRank(m, n, targetRatio);

      // k = mn / (targetRatio * (m+n))
      // k = 5000 / (2 * 150) = 16.67 → 16

      expect(rank).toBeGreaterThan(0);
      expect(rank).toBeLessThan(Math.min(m, n));

      // Verify it gives approximately the target ratio
      const actualRatio = (m * n) / (rank * (m + n));
      expect(actualRatio).toBeCloseTo(targetRatio, 0);
    });

    it('should handle edge cases', () => {
      // Square matrix
      const rank1 = findOptimalRank(100, 100, 2.0);
      expect(rank1).toBeGreaterThan(0);

      // Rectangular matrix
      const rank2 = findOptimalRank(200, 50, 3.0);
      expect(rank2).toBeGreaterThan(0);

      // High compression
      const rank3 = findOptimalRank(100, 100, 5.0);
      expect(rank3).toBeGreaterThan(0);
    });

    it('should clamp rank to valid range', () => {
      const m = 10;
      const n = 10;
      const targetRatio = 100.0; // Unrealistic

      const rank = findOptimalRank(m, n, targetRatio);

      // Should be at least 1
      expect(rank).toBeGreaterThanOrEqual(1);
      // Should not exceed min dimension
      expect(rank).toBeLessThanOrEqual(Math.min(m, n));
    });
  });

  describe('Shape Preservation', () => {
    it('should preserve original shape in metadata', () => {
      const matrix = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
      ];

      const compressed = lowRankApproximation(matrix, 1);

      expect(compressed.originalShape).toEqual([2, 5]);
    });

    it('should reconstruct with correct dimensions', () => {
      const m = 7;
      const n = 5;
      const matrix = Array(m)
        .fill(0)
        .map(() =>
          Array(n)
            .fill(0)
            .map(() => Math.random())
        );

      const compressed = lowRankApproximation(matrix, 3);
      const reconstructed = reconstructFromLowRank(compressed);

      expect(reconstructed.length).toBe(m);
      expect(reconstructed[0].length).toBe(n);
    });
  });
});
