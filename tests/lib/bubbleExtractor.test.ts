import { describe, it, expect, beforeEach } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import {
  extractBubblesFromModel,
  extractBubblesFromContext
} from '../../src/lib/expandable/bubbleExtractor';

describe('bubbleExtractor', () => {
  let model: ProNeuralLM;

  beforeEach(() => {
    const vocab = [
      '<PAD>',
      '<BOS>',
      '<EOS>',
      '<UNK>',
      'hello',
      'world',
      'good',
      'bad',
      'risk',
      'value',
      'want',
      'run',
      'hand'
    ];
    model = new ProNeuralLM(vocab, 16, 0.08, 3, 'momentum', 0.9, 0, 42);
  });

  describe('extractBubblesFromModel', () => {
    it('should extract bubbles from model embeddings', () => {
      const bubbles = extractBubblesFromModel(model);

      expect(bubbles.length).toBeGreaterThan(0);
      expect(bubbles.length).toBeLessThanOrEqual(24); // Default max
    });

    it('should respect maxBubbles config', () => {
      const bubbles = extractBubblesFromModel(model, { maxBubbles: 5 });

      expect(bubbles.length).toBeLessThanOrEqual(5);
    });

    it('should filter out special tokens', () => {
      const bubbles = extractBubblesFromModel(model);

      const hasSpecialTokens = bubbles.some(
        (b) => b.label.startsWith('<') && b.label.endsWith('>')
      );
      expect(hasSpecialTokens).toBe(false);
    });

    it('should include required bubble properties', () => {
      const bubbles = extractBubblesFromModel(model, { maxBubbles: 3 });

      bubbles.forEach((bubble) => {
        expect(bubble.id).toBeDefined();
        expect(bubble.label).toBeDefined();
        expect(bubble.embedding).toBeDefined();
        expect(bubble.embedding.length).toBeGreaterThan(0);
        expect(bubble.activation).toBeGreaterThanOrEqual(0);
        expect(bubble.activation).toBeLessThanOrEqual(1);
        expect(bubble.tag).toBeDefined();
        expect(bubble.ts).toBeDefined();
      });
    });

    it('should assign semantic tags based on token', () => {
      const bubbles = extractBubblesFromModel(model);

      // Find specific tokens and check their tags
      const riskBubble = bubbles.find((b) => b.label === 'risk');
      const valueBubble = bubbles.find((b) => b.label === 'value');
      const wantBubble = bubbles.find((b) => b.label === 'want');
      const runBubble = bubbles.find((b) => b.label === 'run');
      const handBubble = bubbles.find((b) => b.label === 'hand');

      if (riskBubble) expect(riskBubble.tag).toBe('risk');
      if (valueBubble) expect(valueBubble.tag).toBe('value');
      if (wantBubble) expect(wantBubble.tag).toBe('desire');
      if (runBubble) expect(runBubble.tag).toBe('action');
      if (handBubble) expect(handBubble.tag).toBe('body');
    });

    it('should sort by activation by default', () => {
      const bubbles = extractBubblesFromModel(model, { maxBubbles: 10 });

      for (let i = 1; i < bubbles.length; i++) {
        expect(bubbles[i - 1].activation).toBeGreaterThanOrEqual(bubbles[i].activation);
      }
    });

    it('should support random sampling', () => {
      // Run multiple times to check randomness
      const bubbles1 = extractBubblesFromModel(model, { randomSample: true, maxBubbles: 5 });
      const bubbles2 = extractBubblesFromModel(model, { randomSample: true, maxBubbles: 5 });

      // Order should likely be different (not guaranteed but highly probable)
      // Just check that both have valid bubbles
      expect(bubbles1.length).toBeGreaterThan(0);
      expect(bubbles2.length).toBeGreaterThan(0);
    });
  });

  describe('extractBubblesFromContext', () => {
    it('should extract bubbles from recent token indices', () => {
      // Indices for: hello, world, good
      const recentTokens = [4, 5, 6];
      const bubbles = extractBubblesFromContext(model, recentTokens);

      expect(bubbles.length).toBeLessThanOrEqual(recentTokens.length);
    });

    it('should deduplicate tokens', () => {
      const recentTokens = [4, 5, 4, 5, 4]; // hello, world repeated
      const bubbles = extractBubblesFromContext(model, recentTokens);

      const uniqueLabels = new Set(bubbles.map((b) => b.label));
      expect(uniqueLabels.size).toBe(bubbles.length);
    });

    it('should skip special tokens', () => {
      // Indices 0-3 are special tokens
      const recentTokens = [0, 1, 2, 3, 4, 5];
      const bubbles = extractBubblesFromContext(model, recentTokens);

      const hasSpecialTokens = bubbles.some(
        (b) => b.label.startsWith('<') && b.label.endsWith('>')
      );
      expect(hasSpecialTokens).toBe(false);
    });

    it('should handle out-of-range indices', () => {
      const recentTokens = [-1, 1000, 4, 5];
      const bubbles = extractBubblesFromContext(model, recentTokens);

      // Should only have valid tokens
      expect(bubbles.length).toBeLessThanOrEqual(2);
    });

    it('should use context bubble IDs', () => {
      const recentTokens = [4, 5];
      const bubbles = extractBubblesFromContext(model, recentTokens);

      bubbles.forEach((bubble) => {
        expect(bubble.id).toContain('ctx-bubble-');
      });
    });

    it('should respect maxBubbles config', () => {
      const recentTokens = [4, 5, 6, 7, 8, 9, 10, 11, 12];
      const bubbles = extractBubblesFromContext(model, recentTokens, { maxBubbles: 3 });

      expect(bubbles.length).toBeLessThanOrEqual(3);
    });
  });
});
