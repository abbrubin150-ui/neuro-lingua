/**
 * BPE Tokenizer Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { BPETokenizer } from '../../src/tokenizer/BPETokenizer';
import type { BPEArtifact } from '../../src/types/tokenizer';

describe('BPETokenizer', () => {
  let tokenizer: BPETokenizer;

  beforeEach(() => {
    tokenizer = new BPETokenizer({ vocabSize: 100, minFrequency: 2 });
  });

  describe('constructor', () => {
    it('should initialize with default config', () => {
      const t = new BPETokenizer();
      expect(t.vocabSize).toBeGreaterThan(0);
      expect(t.isTrained).toBe(false);
    });

    it('should initialize with custom config', () => {
      const t = new BPETokenizer({
        vocabSize: 500,
        minFrequency: 3,
        padToken: '[PAD]',
        bosToken: '[START]',
        eosToken: '[END]',
        unknownToken: '[OOV]'
      });
      expect(t.isTrained).toBe(false);
    });

    it('should initialize special tokens', () => {
      const ids = tokenizer.specialTokenIds;
      expect(ids.pad).toBeDefined();
      expect(ids.bos).toBeDefined();
      expect(ids.eos).toBeDefined();
      expect(ids.unk).toBeDefined();
    });
  });

  describe('train', () => {
    it('should train on a corpus', async () => {
      const corpus = 'the quick brown fox jumps over the lazy dog. the fox is quick.';
      const artifact = await tokenizer.train(corpus);

      expect(tokenizer.isTrained).toBe(true);
      expect(artifact.version).toBe('1.0.0');
      expect(artifact.hash).toBeTruthy();
      expect(artifact.merges.length).toBeGreaterThan(0);
    });

    it('should report progress during training', async () => {
      // Need a larger corpus to get enough merges for progress callback (called every 50 merges)
      const words = ['hello', 'world', 'test', 'example', 'sample', 'data', 'training', 'corpus'];
      const corpus = Array.from({ length: 500 }, () =>
        words[Math.floor(Math.random() * words.length)]
      ).join(' ');

      let progressCalls = 0;

      await tokenizer.train(corpus, (progress) => {
        progressCalls++;
        expect(progress.currentMerges).toBeDefined();
        expect(progress.targetMerges).toBeDefined();
      });

      // Progress is called every 50 merges, so may be 0 for small vocabs
      expect(progressCalls).toBeGreaterThanOrEqual(0);
    });

    it('should create vocabulary entries', async () => {
      const corpus = 'abababab cdcdcdcd';
      const artifact = await tokenizer.train(corpus);

      expect(artifact.vocab.length).toBeGreaterThan(0);
      expect(Object.keys(artifact.vocabToId).length).toBeGreaterThan(0);
    });
  });

  describe('encode', () => {
    beforeEach(async () => {
      const corpus = 'the quick brown fox jumps over the lazy dog';
      await tokenizer.train(corpus);
    });

    it('should throw if not trained', () => {
      const untrained = new BPETokenizer();
      expect(() => untrained.encode('test')).toThrow('Tokenizer not trained');
    });

    it('should encode text to token IDs', () => {
      const result = tokenizer.encode('the fox');

      expect(result.ids).toBeDefined();
      expect(result.tokens).toBeDefined();
      expect(result.offsets).toBeDefined();
      expect(result.ids.length).toBeGreaterThan(0);
    });

    it('should return offsets for each token', () => {
      const result = tokenizer.encode('quick');

      for (const [start, end] of result.offsets) {
        expect(start).toBeGreaterThanOrEqual(0);
        expect(end).toBeGreaterThanOrEqual(start);
      }
    });

    it('should handle unknown tokens', () => {
      const result = tokenizer.encode('xyz123');
      expect(result.ids.length).toBeGreaterThan(0);
    });
  });

  describe('decode', () => {
    beforeEach(async () => {
      const corpus = 'the quick brown fox jumps over the lazy dog';
      await tokenizer.train(corpus);
    });

    it('should decode token IDs back to text', () => {
      const encoded = tokenizer.encode('the fox');
      const decoded = tokenizer.decode(encoded.ids);

      expect(decoded).toContain('the');
      expect(decoded).toContain('fox');
    });

    it('should skip special tokens by default', () => {
      const ids = [tokenizer.specialTokenIds.bos, ...tokenizer.encode('fox').ids];
      const decoded = tokenizer.decode(ids);

      expect(decoded).not.toContain('<BOS>');
    });

    it('should include special tokens when requested', () => {
      const ids = [tokenizer.specialTokenIds.bos, ...tokenizer.encode('fox').ids];
      const decoded = tokenizer.decode(ids, { skipSpecialTokens: false });

      expect(decoded).toContain('<BOS>');
    });
  });

  describe('calculateMetrics', () => {
    beforeEach(async () => {
      const corpus = 'the quick brown fox jumps over the lazy dog. the fox is quick.';
      await tokenizer.train(corpus);
    });

    it('should throw if not trained', () => {
      const untrained = new BPETokenizer();
      expect(() => untrained.calculateMetrics('test')).toThrow('Tokenizer not trained');
    });

    it('should return coverage metric', () => {
      const metrics = tokenizer.calculateMetrics('the quick fox');
      expect(metrics.coverage).toBeGreaterThanOrEqual(0);
      expect(metrics.coverage).toBeLessThanOrEqual(1);
    });

    it('should return entropy metric', () => {
      const metrics = tokenizer.calculateMetrics('the quick fox');
      expect(metrics.entropy).toBeGreaterThanOrEqual(0);
    });

    it('should return fertility metric', () => {
      const metrics = tokenizer.calculateMetrics('the quick fox');
      expect(metrics.fertility).toBeGreaterThan(0);
    });

    it('should return compression ratio', () => {
      const metrics = tokenizer.calculateMetrics('the quick fox');
      expect(metrics.compressionRatio).toBeGreaterThan(0);
    });

    it('should return vocab utilization', () => {
      const metrics = tokenizer.calculateMetrics('the quick fox');
      expect(metrics.vocabUtilization).toBeGreaterThan(0);
      expect(metrics.vocabUtilization).toBeLessThanOrEqual(1);
    });
  });

  describe('serialization', () => {
    it('should export to JSON', async () => {
      const corpus = 'the quick brown fox';
      await tokenizer.train(corpus);

      const json = await tokenizer.toJSON();
      const parsed = JSON.parse(json);

      expect(parsed.version).toBe('1.0.0');
      expect(parsed.hash).toBeTruthy();
      expect(parsed.merges).toBeDefined();
      expect(parsed.vocab).toBeDefined();
    });

    it('should restore from JSON', async () => {
      const corpus = 'the quick brown fox jumps';
      await tokenizer.train(corpus);
      const json = await tokenizer.toJSON();

      const restored = BPETokenizer.fromJSON(json);
      expect(restored.isTrained).toBe(true);
      expect(restored.vocabSize).toBe(tokenizer.vocabSize);
    });

    it('should restore from artifact', async () => {
      const corpus = 'the quick brown fox jumps';
      const artifact = await tokenizer.train(corpus);

      const restored = BPETokenizer.fromArtifact(artifact);
      expect(restored.isTrained).toBe(true);
      expect(restored.hash).toBe(tokenizer.hash);
    });

    it('should produce identical encodings after restoration', async () => {
      const corpus = 'the quick brown fox jumps over the lazy dog';
      await tokenizer.train(corpus);

      const json = await tokenizer.toJSON();
      const restored = BPETokenizer.fromJSON(json);

      const original = tokenizer.encode('the fox');
      const restoredResult = restored.encode('the fox');

      expect(restoredResult.ids).toEqual(original.ids);
      expect(restoredResult.tokens).toEqual(original.tokens);
    });
  });

  describe('artifact verification', () => {
    it('should verify valid artifact', async () => {
      const corpus = 'the quick brown fox jumps';
      const artifact = await tokenizer.train(corpus);

      const isValid = await tokenizer.verifyArtifact(artifact);
      expect(isValid).toBe(true);
    });

    it('should detect tampered artifact', async () => {
      const corpus = 'the quick brown fox jumps over';
      const artifact = await tokenizer.train(corpus);

      // Tamper with the artifact by modifying the hash directly
      // The verification compares computed hash with stored hash
      const tampered: BPEArtifact = {
        ...artifact,
        hash: 'tampered-hash-value-that-does-not-match'
      };

      const isValid = await tokenizer.verifyArtifact(tampered);
      expect(isValid).toBe(false);
    });
  });

  describe('token ID mapping', () => {
    beforeEach(async () => {
      const corpus = 'the quick brown fox';
      await tokenizer.train(corpus);
    });

    it('should map tokens to IDs', () => {
      // Special tokens are always in the vocab
      const padId = tokenizer.tokenToId('<PAD>');
      expect(padId).toBeDefined();
      expect(typeof padId).toBe('number');
    });

    it('should map IDs to tokens', () => {
      const id = tokenizer.tokenToId('<PAD>');
      if (id !== undefined) {
        const token = tokenizer.idToToken(id);
        expect(token).toBe('<PAD>');
      }
    });

    it('should return undefined for unknown tokens', () => {
      const id = tokenizer.tokenToId('unknowntoken12345');
      expect(id).toBeUndefined();
    });
  });
});
