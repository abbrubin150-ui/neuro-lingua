/**
 * Tests for Sparse Attention Mechanisms
 *
 * Verifies:
 * 1. Local (sliding window) attention mask
 * 2. Strided attention mask
 * 3. Dilated attention mask
 * 4. BigBird attention mask
 * 5. Longformer attention mask
 * 6. Block sparse attention mask
 * 7. Sparse attention computation
 * 8. Memory/FLOPS savings calculations
 * 9. Mask combination operations
 */

import { describe, it, expect } from 'vitest';
import {
  createLocalMask,
  createStridedMask,
  createDilatedMask,
  createBigBirdMask,
  createLongformerMask,
  createBlockSparseMask,
  createAxialMask,
  createSparseMask,
  sparseScaledDotProductAttention,
  analyzeSparsity,
  getRecommendedPattern,
  combineMasks,
  intersectMasks,
  SPARSE_ATTENTION_PRESETS
} from '../../src/models/sparse_attention';

describe('Local Attention Mask', () => {
  it('should create sliding window mask', () => {
    const mask = createLocalMask(5, 1);

    // Window size 1 means each token attends to itself ± 1
    expect(mask[0]).toEqual([true, true, false, false, false]); // 0 attends to 0,1
    expect(mask[1]).toEqual([true, true, true, false, false]); // 1 attends to 0,1,2
    expect(mask[2]).toEqual([false, true, true, true, false]); // 2 attends to 1,2,3
    expect(mask[3]).toEqual([false, false, true, true, true]); // 3 attends to 2,3,4
    expect(mask[4]).toEqual([false, false, false, true, true]); // 4 attends to 3,4
  });

  it('should apply causal masking', () => {
    const mask = createLocalMask(5, 2, true);

    // Causal: can only attend to past (including self)
    expect(mask[0]).toEqual([true, false, false, false, false]);
    expect(mask[2]).toEqual([true, true, true, false, false]); // 2 attends to 0,1,2
    expect(mask[4]).toEqual([false, false, true, true, true]); // 4 attends to 2,3,4
  });

  it('should handle full window size', () => {
    const mask = createLocalMask(3, 5); // Window larger than sequence

    // Should be all true (full attention)
    expect(mask.every((row) => row.every((v) => v))).toBe(true);
  });
});

describe('Strided Attention Mask', () => {
  it('should create strided pattern', () => {
    const mask = createStridedMask(6, 2);

    // Stride 2: position i attends to j where j%2 == i%2 or j%2 == 0
    // Position 0 (even): attends to 0, 2, 4 (all even)
    expect(mask[0][0]).toBe(true);
    expect(mask[0][2]).toBe(true);
    expect(mask[0][4]).toBe(true);
  });

  it('should apply causal masking', () => {
    const mask = createStridedMask(6, 2, true);

    // Position 2 can only attend to 0, 2 (past positions with matching stride)
    expect(mask[2][0]).toBe(true);
    expect(mask[2][2]).toBe(true);
    expect(mask[2][4]).toBe(false); // Future
  });
});

describe('Dilated Attention Mask', () => {
  it('should create dilated pattern', () => {
    const mask = createDilatedMask(8, [1, 2, 4]);

    // Dilation rates [1, 2, 4] mean:
    // - Rate 1: attend to adjacent (diff % 1 == 0 → all)
    // - Rate 2: attend to every 2nd
    // - Rate 4: attend to every 4th
    // Combined: position can reach many positions

    // Position 4 can reach 0 (diff=4, 4%4==0), 2 (diff=2, 2%2==0), etc.
    expect(mask[4][0]).toBe(true);
    expect(mask[4][2]).toBe(true);
    expect(mask[4][4]).toBe(true);
  });
});

describe('BigBird Attention Mask', () => {
  it('should include global tokens', () => {
    const mask = createBigBirdMask(10, 2, 2, 0);

    // First and last token are global (numGlobalTokens=2)
    // Position 0 should attend to all
    expect(mask[0].every((v) => v)).toBe(true);

    // All positions should attend to position 0 (global)
    expect(mask.every((row) => row[0])).toBe(true);
  });

  it('should include local window', () => {
    const mask = createBigBirdMask(10, 2, 0, 0);

    // Position 5 with window 2 should attend to 3,4,5,6,7
    expect(mask[5][3]).toBe(true);
    expect(mask[5][4]).toBe(true);
    expect(mask[5][5]).toBe(true);
    expect(mask[5][6]).toBe(true);
    expect(mask[5][7]).toBe(true);
  });

  it('should include random tokens', () => {
    const mask = createBigBirdMask(20, 1, 0, 5);

    // Check that non-local positions have some attention (random)
    // Position 10 with window 1: local = 9,10,11
    // Should have at least some random connections (may overlap with local)
    const nonLocalCount = mask[10].filter((v, i) => v && Math.abs(i - 10) > 1).length;
    expect(nonLocalCount).toBeGreaterThanOrEqual(3);
  });

  it('should apply causal masking', () => {
    const mask = createBigBirdMask(10, 2, 2, 0, true);

    // Position 5 should not attend to position 9 (future)
    expect(mask[5][9]).toBe(false);
  });
});

describe('Longformer Attention Mask', () => {
  it('should make specified positions global', () => {
    const globalPositions = [0, 5, 9];
    const mask = createLongformerMask(10, 2, globalPositions);

    // Position 5 is global - should attend to all
    expect(mask[5].every((v) => v)).toBe(true);

    // All positions should attend to global position 5
    expect(mask.every((row) => row[5])).toBe(true);
  });

  it('should use local window for non-global', () => {
    const mask = createLongformerMask(10, 2, [0]);

    // Position 7 (non-global, window 2): attends to 0 (global), 5,6,7,8,9
    expect(mask[7][0]).toBe(true); // Global
    expect(mask[7][5]).toBe(true); // Window
    expect(mask[7][6]).toBe(true);
    expect(mask[7][3]).toBe(false); // Outside window and not global
  });
});

describe('Block Sparse Attention Mask', () => {
  it('should create block diagonal pattern', () => {
    const mask = createBlockSparseMask(8, 4);

    // Block 0: positions 0-3
    // Block 1: positions 4-7
    // Position 0 (block 0) attends to block 0 and block 1 (adjacent)
    expect(mask[0][0]).toBe(true);
    expect(mask[0][3]).toBe(true);
    expect(mask[0][4]).toBe(true); // Adjacent block
    expect(mask[0][7]).toBe(true);
  });

  it('should not cross block boundaries by more than 1', () => {
    const mask = createBlockSparseMask(12, 4);

    // Block 0: 0-3, Block 1: 4-7, Block 2: 8-11
    // Position 0 should NOT attend to block 2 (distance > 1)
    expect(mask[0][8]).toBe(false);
    expect(mask[0][11]).toBe(false);
  });
});

describe('Axial Attention Mask', () => {
  it('should create row-wise attention', () => {
    const mask = createAxialMask(3, 3, 'row');

    // 3x3 grid, row attention
    // Position 0 (row 0, col 0) attends to row 0: positions 0,1,2
    expect(mask[0][0]).toBe(true);
    expect(mask[0][1]).toBe(true);
    expect(mask[0][2]).toBe(true);
    expect(mask[0][3]).toBe(false); // Different row
  });

  it('should create column-wise attention', () => {
    const mask = createAxialMask(3, 3, 'col');

    // Position 0 (row 0, col 0) attends to col 0: positions 0,3,6
    expect(mask[0][0]).toBe(true);
    expect(mask[0][3]).toBe(true);
    expect(mask[0][6]).toBe(true);
    expect(mask[0][1]).toBe(false); // Different column
  });
});

describe('createSparseMask factory', () => {
  it('should create local mask', () => {
    const mask = createSparseMask({ pattern: 'local', seqLen: 10, windowSize: 2 });
    expect(mask.pattern).toBe('local');
    expect(mask.mask.length).toBe(10);
  });

  it('should create bigbird mask', () => {
    const mask = createSparseMask({
      pattern: 'bigbird',
      seqLen: 20,
      windowSize: 3,
      numGlobalTokens: 2,
      numRandomTokens: 2
    });
    expect(mask.pattern).toBe('bigbird');
  });

  it('should calculate sparsity correctly', () => {
    const fullMask = createSparseMask({ pattern: 'local', seqLen: 4, windowSize: 4 });
    expect(fullMask.sparsity).toBeCloseTo(0, 2); // Full attention = 0 sparsity

    const sparseMask = createSparseMask({ pattern: 'local', seqLen: 10, windowSize: 1 });
    expect(sparseMask.sparsity).toBeGreaterThan(0.5); // Significant sparsity
  });

  it('should throw for custom without mask', () => {
    expect(() =>
      createSparseMask({ pattern: 'custom', seqLen: 10 })
    ).toThrow('Custom pattern requires customMask');
  });
});

describe('Sparse Attention Computation', () => {
  it('should compute attention with sparse mask', () => {
    const queries = [
      [1, 0],
      [0, 1]
    ];
    const keys = [
      [1, 0],
      [0, 1]
    ];
    const values = [
      [1, 0],
      [0, 1]
    ];

    const sparseMask = createSparseMask({
      pattern: 'local',
      seqLen: 2,
      windowSize: 1
    });

    const { output } = sparseScaledDotProductAttention(queries, keys, values, sparseMask);

    expect(output.length).toBe(2);
    expect(output[0].length).toBe(2);
  });

  it('should return attention weights when requested', () => {
    const queries = [[1, 0]];
    const keys = [[1, 0]];
    const values = [[1, 0]];

    const sparseMask = createSparseMask({ pattern: 'local', seqLen: 1, windowSize: 1 });

    const { attention } = sparseScaledDotProductAttention(
      queries,
      keys,
      values,
      sparseMask,
      { returnAttention: true }
    );

    expect(attention).toBeDefined();
    expect(attention!.length).toBe(1);
  });

  it('should produce valid output with full local mask', () => {
    const queries = [
      [1, 0],
      [0, 1]
    ];
    const keys = [
      [1, 0],
      [0, 1]
    ];
    const values = [
      [1, 0],
      [0, 1]
    ];

    // Use full attention (window covers all) to avoid -inf issues
    const fullMask = createSparseMask({
      pattern: 'local',
      seqLen: 2,
      windowSize: 2 // Larger than seq, so full attention
    });

    const { output, attention } = sparseScaledDotProductAttention(
      queries,
      keys,
      values,
      fullMask,
      { returnAttention: true }
    );

    // Should produce valid output
    expect(output.length).toBe(2);
    expect(Number.isFinite(output[0][0])).toBe(true);
    expect(Number.isFinite(output[1][0])).toBe(true);

    // Attention should sum to approximately 1 for each row
    const attnSum0 = attention![0].reduce((a, b) => a + b, 0);
    expect(attnSum0).toBeCloseTo(1, 2);
  });
});

describe('Sparsity Analysis', () => {
  it('should calculate memory savings', () => {
    const analysis = analyzeSparsity({
      pattern: 'local',
      seqLen: 100,
      windowSize: 10
    });

    expect(analysis.fullAttentionMemory).toBe(100 * 100 * 4);
    expect(analysis.sparseAttentionMemory).toBeLessThan(analysis.fullAttentionMemory);
    expect(analysis.memorySavingsPercent).toBeGreaterThan(0);
  });

  it('should calculate FLOPS savings', () => {
    const analysis = analyzeSparsity({
      pattern: 'bigbird',
      seqLen: 1000,
      windowSize: 64,
      numGlobalTokens: 2,
      numRandomTokens: 3
    });

    expect(analysis.flopsSavingsPercent).toBeGreaterThan(50);
  });
});

describe('getRecommendedPattern', () => {
  it('should recommend full attention for short sequences', () => {
    const config = getRecommendedPattern(100, 1000);
    expect(config.windowSize).toBe(100); // Full attention
  });

  it('should recommend BigBird for medium sequences with tight memory', () => {
    // 512^2 * 4 = ~1MB, need memory < 1MB to trigger BigBird
    const config = getRecommendedPattern(512, 0.5);
    expect(config.pattern).toBe('bigbird');
  });

  it('should recommend local pattern when memory is sufficient', () => {
    // 512^2 * 4 = ~1MB, 10MB is enough
    const config = getRecommendedPattern(512, 10);
    expect(config.pattern).toBe('local');
  });

  it('should recommend block sparse for very long sequences', () => {
    const config = getRecommendedPattern(8000, 10);
    expect(config.pattern).toBe('blockSparse');
  });
});

describe('Mask Operations', () => {
  describe('combineMasks (OR)', () => {
    it('should combine masks with OR', () => {
      const mask1 = createSparseMask({ pattern: 'local', seqLen: 5, windowSize: 0 });
      const mask2 = createSparseMask({ pattern: 'strided', seqLen: 5, stride: 5 });

      const combined = combineMasks([mask1, mask2]);

      // Should have at least diagonal (from local) plus strided positions
      expect(combined.mask[0][0]).toBe(true); // Self attention
    });

    it('should throw for empty array', () => {
      expect(() => combineMasks([])).toThrow('Need at least one mask');
    });
  });

  describe('intersectMasks (AND)', () => {
    it('should intersect masks with AND', () => {
      const mask1 = createSparseMask({ pattern: 'local', seqLen: 5, windowSize: 2 });
      const mask2 = createSparseMask({ pattern: 'local', seqLen: 5, windowSize: 1 });

      const intersected = intersectMasks([mask1, mask2]);

      // Intersection of window 2 and window 1 is window 1
      expect(intersected.mask[0][2]).toBe(false); // Outside window 1
    });
  });
});

describe('Sparse Attention Presets', () => {
  it('should have valid fast preset', () => {
    expect(SPARSE_ATTENTION_PRESETS.fast.pattern).toBe('local');
    expect(SPARSE_ATTENTION_PRESETS.fast.windowSize).toBe(32);
  });

  it('should have valid balanced preset', () => {
    expect(SPARSE_ATTENTION_PRESETS.balanced.pattern).toBe('bigbird');
    expect(SPARSE_ATTENTION_PRESETS.balanced.numGlobalTokens).toBe(2);
  });

  it('should have valid document preset', () => {
    expect(SPARSE_ATTENTION_PRESETS.document.pattern).toBe('longformer');
  });

  it('should have valid efficient preset', () => {
    expect(SPARSE_ATTENTION_PRESETS.efficient.pattern).toBe('blockSparse');
  });
});
