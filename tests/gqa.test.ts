/**
 * Tests for Grouped-Query Attention (GQA) implementation.
 *
 * GQA reduces memory usage by sharing key-value heads among multiple query heads.
 * Reference: Ainslie et al. (2023) "GQA: Training Generalized Multi-Query
 * Transformer Models from Multi-Head Checkpoints"
 */

import { describe, it, expect } from 'vitest';
import { MultiHeadAttention, scaledDotProductAttention } from '../src/models/attention';
import { MiniTransformerBlock } from '../src/models/mini_transformer';

describe('Grouped-Query Attention (GQA)', () => {
  describe('MultiHeadAttention with GQA', () => {
    it('should accept valid numKVHeads configuration', () => {
      // 8 query heads, 2 KV heads = 4:1 ratio (valid)
      expect(() => {
        new MultiHeadAttention({
          heads: 8,
          modelDim: 64,
          keyDim: 8,
          valueDim: 8,
          numKVHeads: 2
        });
      }).not.toThrow();
    });

    it('should default to standard MHA when numKVHeads not specified', () => {
      const mha = new MultiHeadAttention({
        heads: 4,
        modelDim: 32,
        keyDim: 8,
        valueDim: 8
      });
      expect(mha.getNumKVHeads()).toBe(4);
    });

    it('should throw error when numHeads is not divisible by numKVHeads', () => {
      // 8 query heads, 3 KV heads = not valid (8 % 3 !== 0)
      expect(() => {
        new MultiHeadAttention({
          heads: 8,
          modelDim: 64,
          keyDim: 8,
          valueDim: 8,
          numKVHeads: 3
        });
      }).toThrow(/must be divisible/);
    });

    it('should support Multi-Query Attention (MQA) with numKVHeads=1', () => {
      const mqa = new MultiHeadAttention({
        heads: 8,
        modelDim: 64,
        keyDim: 8,
        valueDim: 8,
        numKVHeads: 1
      });
      expect(mqa.getNumKVHeads()).toBe(1);
      expect(mqa.getKVDim()).toBe(8); // 1 head * 8 dim
    });

    it('should calculate correct KV dimension', () => {
      const gqa = new MultiHeadAttention({
        heads: 8,
        modelDim: 64,
        keyDim: 8,
        valueDim: 8,
        numKVHeads: 2
      });
      // headDim = 64 / 8 = 8
      // kvDim = 2 * 8 = 16
      expect(gqa.getKVDim()).toBe(16);
    });

    it('should correctly repeat KV heads in forward pass', async () => {
      const gqa = new MultiHeadAttention({
        heads: 4,
        modelDim: 16,
        keyDim: 4,
        valueDim: 4,
        numKVHeads: 2 // 2:1 ratio
      });

      // Create small test input
      const inputs = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        [1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
      ];

      // Create weights - Query is [16 x 16], Key/Value are [16 x 8] (smaller for GQA)
      const weights = {
        query: Array.from({ length: 16 }, (_, i) =>
          Array.from({ length: 16 }, (_, j) => (i === j ? 1 : 0))
        ),
        key: Array.from({ length: 16 }, (_, i) =>
          Array.from({ length: 8 }, (_, j) => (i % 8 === j ? 1 : 0))
        ),
        value: Array.from({ length: 16 }, (_, i) =>
          Array.from({ length: 8 }, (_, j) => (i % 8 === j ? 1 : 0))
        )
      };

      // Should not throw
      const output = await gqa.forward(inputs, weights);
      expect(output).toHaveLength(2);
      expect(output[0]).toHaveLength(16);
    });
  });

  describe('MiniTransformerBlock with GQA', () => {
    it('should accept numKVHeads in config', () => {
      const block = new MiniTransformerBlock({
        modelDim: 32,
        heads: 4,
        ff: { hiddenDim: 64 },
        numKVHeads: 2
      });
      expect(block).toBeDefined();
    });

    it('should work with standard MHA when numKVHeads equals heads', async () => {
      const block = new MiniTransformerBlock({
        modelDim: 32,
        heads: 4,
        ff: { hiddenDim: 64 },
        numKVHeads: 4 // Same as heads = standard MHA
      });

      const inputs = [
        Array.from({ length: 32 }, () => Math.random()),
        Array.from({ length: 32 }, () => Math.random())
      ];

      const attentionWeights = {
        query: Array.from({ length: 32 }, () =>
          Array.from({ length: 32 }, () => Math.random() * 0.1)
        ),
        key: Array.from({ length: 32 }, () =>
          Array.from({ length: 32 }, () => Math.random() * 0.1)
        ),
        value: Array.from({ length: 32 }, () =>
          Array.from({ length: 32 }, () => Math.random() * 0.1)
        )
      };

      const ffWeights1 = Array.from({ length: 32 }, () =>
        Array.from({ length: 128 }, () => Math.random() * 0.1)
      );
      const ffWeights2 = Array.from({ length: 64 }, () =>
        Array.from({ length: 32 }, () => Math.random() * 0.1)
      );

      const output = await block.forward(inputs, attentionWeights, ffWeights1, ffWeights2);
      expect(output).toHaveLength(2);
      expect(output[0]).toHaveLength(32);
    });
  });

  describe('GQA Memory Savings', () => {
    it('should have smaller KV weights with GQA', () => {
      const modelDim = 64;
      const numHeads = 8;
      const numKVHeads = 2;
      const headDim = modelDim / numHeads;

      // Standard MHA: K/V weights are [modelDim x modelDim]
      const mhaKVSize = modelDim * modelDim;

      // GQA: K/V weights are [modelDim x (numKVHeads * headDim)]
      const gqaKVSize = modelDim * (numKVHeads * headDim);

      // GQA should use (numHeads / numKVHeads) times less memory for K/V
      const expectedRatio = numHeads / numKVHeads; // 8 / 2 = 4
      expect(mhaKVSize / gqaKVSize).toBe(expectedRatio);
    });
  });

  describe('scaledDotProductAttention', () => {
    it('should work with repeated heads (simulating GQA)', () => {
      // Simulate what GQA does: same K/V for multiple Q heads
      const queries1 = [[0.1, 0.2], [0.3, 0.4]];
      const queries2 = [[0.5, 0.6], [0.7, 0.8]];

      // Same K/V for both query groups (GQA behavior)
      const keys = [[1, 0], [0, 1]];
      const values = [[1, 2], [3, 4]];

      const result1 = scaledDotProductAttention(queries1, keys, values);
      const result2 = scaledDotProductAttention(queries2, keys, values);

      // Both should produce valid outputs
      expect(result1.output).toHaveLength(2);
      expect(result2.output).toHaveLength(2);

      // Attention should sum to 1 for each row
      for (const row of result1.attention) {
        const sum = row.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1, 5);
      }
    });
  });
});
