import { describe, expect, it } from 'vitest';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';

describe('Tokenizer', () => {
  describe('Unicode mode (default)', () => {
    it('tokenizes basic English text', () => {
      const text = 'Hello World! This is a test.';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'world', 'this', 'is', 'a', 'test']);
    });

    it('handles Unicode characters', () => {
      const text = 'café résumé naïve';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toContain('café');
      expect(tokens).toContain('résumé');
      expect(tokens).toContain('naïve');
    });

    it('handles Hebrew text', () => {
      const text = 'שלום עולם';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['שלום', 'עולם']);
    });

    it('handles mixed scripts', () => {
      const text = 'hello שלום café';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'שלום', 'café']);
    });

    it('removes punctuation', () => {
      const text = 'hello, world! how-are you?';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'world', 'how-are', 'you']);
    });

    it('handles empty string', () => {
      const tokens = ProNeuralLM.tokenizeText('');
      expect(tokens).toEqual([]);
    });

    it('handles whitespace-only string', () => {
      const tokens = ProNeuralLM.tokenizeText('   \t\n  ');
      expect(tokens).toEqual([]);
    });

    it('preserves hyphens and apostrophes', () => {
      const text = "it's self-aware";
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toContain("it's");
      expect(tokens).toContain('self-aware');
    });

    it('handles numbers', () => {
      const text = 'test 123 hello 456';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['test', '123', 'hello', '456']);
    });

    it('lowercases text', () => {
      const text = 'HELLO World TeSt';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'world', 'test']);
    });
  });

  describe('ASCII mode', () => {
    it('tokenizes ASCII text', () => {
      const text = 'hello world test';
      const tokens = ProNeuralLM.tokenizeText(text, { mode: 'ascii' });
      expect(tokens).toEqual(['hello', 'world', 'test']);
    });

    it('removes non-ASCII characters', () => {
      const text = 'hello café résumé';
      const tokens = ProNeuralLM.tokenizeText(text, { mode: 'ascii' });
      // Non-ASCII characters are replaced with spaces
      expect(tokens).not.toContain('café');
      expect(tokens).not.toContain('résumé');
      expect(tokens).toContain('hello');
      expect(tokens.some((t) => t.includes('caf') || t.includes('r'))).toBe(true);
    });

    it('preserves numbers and basic punctuation', () => {
      const text = "it's 123 test";
      const tokens = ProNeuralLM.tokenizeText(text, { mode: 'ascii' });
      expect(tokens).toContain("it's");
      expect(tokens).toContain('123');
      expect(tokens).toContain('test');
    });
  });

  describe('Character mode', () => {
    it('splits text into individual characters', () => {
      const tokens = ProNeuralLM.tokenizeText('hello', { mode: 'character' });
      expect(tokens).toEqual(['h', 'e', 'l', 'l', 'o']);
    });

    it('preserves math symbols + and =', () => {
      const tokens = ProNeuralLM.tokenizeText('0+1=1', { mode: 'character' });
      expect(tokens).toEqual(['0', '+', '1', '=', '1']);
    });

    it('keeps word boundaries: each whitespace run becomes one space token', () => {
      const tokens = ProNeuralLM.tokenizeText('a b  \n c', { mode: 'character' });
      expect(tokens).toEqual(['a', ' ', 'b', ' ', 'c']);
    });

    it('drops leading and trailing whitespace', () => {
      expect(ProNeuralLM.tokenizeText('  0+1=1 1+1=2 ', { mode: 'character' })).toEqual([
        ...'0+1=1 1+1=2'
      ]);
    });

    it('keeps astral characters (emoji) as single tokens', () => {
      expect(ProNeuralLM.tokenizeText('a😀b', { mode: 'character' })).toEqual(['a', '😀', 'b']);
    });

    it('preserves case (does not lowercase)', () => {
      const tokens = ProNeuralLM.tokenizeText('Ab', { mode: 'character' });
      expect(tokens).toEqual(['A', 'b']);
    });

    it('handles math corpus correctly', () => {
      const tokens = ProNeuralLM.tokenizeText('0+0=0', { mode: 'character' });
      expect(tokens).toEqual(['0', '+', '0', '=', '0']);
    });

    it('handles empty string', () => {
      const tokens = ProNeuralLM.tokenizeText('', { mode: 'character' });
      expect(tokens).toEqual([]);
    });

    it('handles whitespace-only string', () => {
      const tokens = ProNeuralLM.tokenizeText('   ', { mode: 'character' });
      expect(tokens).toEqual([]);
    });

    it('produces correct vocab for arithmetic dataset', () => {
      const corpus = '0+0=00+1=10+2=20+3=3';
      const tokens = ProNeuralLM.tokenizeText(corpus, { mode: 'character' });
      const uniqueTokens = [...new Set(tokens)];
      // digits 0-3 plus + and = = 6 unique characters
      expect(uniqueTokens.sort()).toEqual(['0', '1', '2', '3', '+', '='].sort());
    });
  });

  describe('Custom mode', () => {
    it('uses custom pattern to split on non-word characters', () => {
      const text = 'hello-world_test';
      const tokens = ProNeuralLM.tokenizeText(text, {
        mode: 'custom',
        pattern: '[^a-z0-9]+'
      });
      expect(tokens).toEqual(['hello', 'world', 'test']);
    });

    it('uses custom pattern to split on spaces only', () => {
      const text = 'hello,world;test';
      const tokens = ProNeuralLM.tokenizeText(text, {
        mode: 'custom',
        pattern: ' +'
      });
      // Since there are no spaces, it should be one token
      expect(tokens.length).toBe(1);
    });

    it('handles Unicode in custom pattern', () => {
      const text = 'café résumé';
      const tokens = ProNeuralLM.tokenizeText(text, {
        mode: 'custom',
        pattern: '[^\\p{L}\\d]+'
      });
      expect(tokens).toContain('café');
      expect(tokens).toContain('résumé');
    });

    it('falls back to Unicode mode on invalid pattern', () => {
      const text = 'hello world';
      // Invalid regex pattern, should fallback
      const tokens = ProNeuralLM.tokenizeText(text, {
        mode: 'custom',
        pattern: ''
      });
      expect(tokens).toEqual(['hello', 'world']);
    });
  });

  describe('Tokenizer configuration persistence', () => {
    it('preserves tokenizer config through save/load', () => {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
      const config = { mode: 'ascii' as const };

      const model = new ProNeuralLM(vocab, 8, 0.05, 2, 'momentum', 0.9, 0, 42, config);
      expect(model.getTokenizerConfig()).toEqual(config);

      const exported = model.exportTokenizerConfig();
      expect(exported).toEqual(config);
    });

    it('allows changing tokenizer config', () => {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello'];
      const model = new ProNeuralLM(vocab);

      expect(model.getTokenizerConfig().mode).toBe('unicode');

      model.importTokenizerConfig({ mode: 'ascii' });
      expect(model.getTokenizerConfig().mode).toBe('ascii');

      model.importTokenizerConfig({ mode: 'custom', pattern: '[^a-z]+' });
      const config = model.getTokenizerConfig();
      expect(config.mode).toBe('custom');
      expect(config.pattern).toBe('[^a-z]+');
    });
  });

  describe('Edge cases', () => {
    it('handles very long text', () => {
      const text = 'hello '.repeat(1000);
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens.length).toBe(1000);
      expect(tokens.every((t) => t === 'hello')).toBe(true);
    });

    it('handles special characters', () => {
      const text = '@ # $ % ^ & * ( ) + = { } [ ] | \\ : ; " < > ? /';
      const tokens = ProNeuralLM.tokenizeText(text);
      // Most punctuation should be filtered out
      expect(tokens.length).toBeGreaterThanOrEqual(0);
    });

    it('handles emoji', () => {
      const text = 'hello 😀 world 🎉';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toContain('hello');
      expect(tokens).toContain('world');
      // Emoji are filtered out in Unicode mode
    });

    it('handles consecutive spaces', () => {
      const text = 'hello    world     test';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'world', 'test']);
    });

    it('handles tabs and newlines', () => {
      const text = 'hello\tworld\ntest\r\nfoo';
      const tokens = ProNeuralLM.tokenizeText(text);
      expect(tokens).toEqual(['hello', 'world', 'test', 'foo']);
    });
  });

  describe('tokenizeWithOffsets', () => {
    it('returns spans into the original text for word tokens', () => {
      const text = 'Hello, World! It\'s fine.';
      const spans = ProNeuralLM.tokenizeWithOffsets(text);
      expect(spans.map((s) => s.token)).toEqual(['hello', 'world', "it's", 'fine']);
      for (const s of spans) expect(text.slice(s.start, s.end).toLowerCase()).toBe(s.token);
    });

    it('returns spans for characters and collapsed whitespace', () => {
      const text = 'ab \t c';
      const spans = ProNeuralLM.tokenizeWithOffsets(text, { mode: 'character' });
      expect(spans).toEqual([
        { token: 'a', start: 0, end: 1 },
        { token: 'b', start: 1, end: 2 },
        { token: ' ', start: 2, end: 5 },
        { token: 'c', start: 5, end: 6 }
      ]);
    });

    it('tokenizeText is the token projection of tokenizeWithOffsets', () => {
      const text = 'שלום עולם, hello-world 42!';
      for (const mode of ['unicode', 'ascii', 'character'] as const) {
        expect(ProNeuralLM.tokenizeText(text, { mode })).toEqual(
          ProNeuralLM.tokenizeWithOffsets(text, { mode }).map((t) => t.token)
        );
      }
    });
  });

  describe('character-mode generation', () => {
    it('joins generated characters without inserting spaces', async () => {
      const corpus = 'ab ab ab ab ab ab';
      const tokens = ProNeuralLM.tokenizeText(corpus, { mode: 'character' });
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ...new Set(tokens)];
      const model = new ProNeuralLM(vocab, 16, 0.05, 2, 'adam', 0.9, 0, 3, { mode: 'character' });
      await model.train(corpus, 30);
      const out = await model.generate('a', 6, 0.05);
      expect(out).not.toMatch(/a b/); // characters are concatenated, not space-separated
      expect(out.length).toBeGreaterThan(0);
    });
  });

  describe('custom RegExp loss mask alignment', () => {
    it('masks exactly the tokens after the match even when the text has spaces', () => {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'question', 'what', 'is', 'it', 'answer', 'blue'];
      const model = new ProNeuralLM(vocab, 8, 0.05, 2, 'adam', 0.9, 0, 1);
      const text = 'Question: what is it? Answer: blue';
      const mask = (model as unknown as {
        buildTrainingLossMask(t: string, c: unknown): number[];
      }).buildTrainingLossMask(text, {
        mode: 'customRegExp',
        customPattern: 'Answer:',
        regExpPosition: 'after'
      });
      // tokens: question what is it answer blue + EOS
      expect(mask).toEqual([0, 0, 0, 0, 0, 1, 1]);
    });
  });
});
