import { describe, expect, it } from 'vitest';
import {
  normalizeText,
  parseCsv,
  parseDocuments,
  prepareCorpus,
  splitParagraphs,
  stripMarkdown
} from '../../src/data/corpusPrep';

describe('corpus preparation', () => {
  it('normalises Unicode, line endings, control characters and spacing', () => {
    const raw = 'café\r\nline  two\u0007\t end\n\n\n\nnext﻿';
    expect(normalizeText(raw)).toBe('café\nline two end\n\nnext');
  });

  it('strips Markdown syntax but keeps the prose', () => {
    const md = [
      '# Title',
      '',
      'Some **bold** and *italic* text with a [link](http://x.y) and `code`.',
      '',
      '```ts',
      'const secret = 1;',
      '```',
      '| a | b |',
      '|---|---|',
      '- item one',
      '> quoted'
    ].join('\n');
    const out = stripMarkdown(md);
    expect(out).toContain('Title');
    expect(out).toContain('Some bold and italic text with a link and code.');
    expect(out).not.toContain('secret');
    expect(out).not.toContain('|');
    expect(out).toContain('item one');
    expect(out).toContain('quoted');
  });

  it('parses jsonl, json, csv and plain text files', () => {
    expect(parseDocuments('a.jsonl', '{"text":"one"}\nbroken\n"two"\n{"content":"three"}').map((d) => d.text)).toEqual(['one', 'two', 'three']);
    expect(parseDocuments('a.json', '[{"body":"x"},"y"]').map((d) => d.text)).toEqual(['x', 'y']);
    expect(parseDocuments('a.csv', 'id,text\n1,"hello, ""world"""\n2,bye').map((d) => d.text)).toEqual(['hello, "world"', 'bye']);
    expect(parseDocuments('notes.md', '# x')[0].format).toBe('markdown');
    expect(parseCsv('a,b\r\n"c\nd",e')).toEqual([['a', 'b'], ['c\nd', 'e']]);
  });

  it('splits paragraphs on blank lines', () => {
    expect(splitParagraphs('a\nb\n\n  \nc')).toEqual(['a b', 'c']);
  });

  it('dedupes, drops short paragraphs and splits deterministically by content', () => {
    const paragraphs = Array.from({ length: 200 }, (_, i) => `Paragraph number ${i} has enough text.`);
    const docs = [
      { id: 'a', text: paragraphs.slice(0, 120).join('\n\n') },
      { id: 'b', text: [...paragraphs.slice(100), 'tiny'].join('\n\n') } // 20 duplicates + 1 short
    ];
    const prepared = prepareCorpus(docs, { validationFraction: 0.2 });
    expect(prepared.stats.duplicateParagraphs).toBe(20);
    expect(prepared.stats.shortParagraphs).toBe(1);
    expect(prepared.stats.trainParagraphs + prepared.stats.validationParagraphs).toBe(200);
    expect(prepared.stats.validationParagraphs).toBeGreaterThan(20);
    expect(prepared.stats.validationParagraphs).toBeLessThan(60);

    // Same split when documents are reordered (assignment depends only on content)
    const reordered = prepareCorpus([...docs].reverse(), { validationFraction: 0.2 });
    const sortParas = (t: string) => t.split('\n\n').sort();
    expect(sortParas(reordered.validation)).toEqual(sortParas(prepared.validation));

    // No paragraph appears in both sets
    const val = new Set(prepared.validation.split('\n\n'));
    for (const p of prepared.train.split('\n\n')) expect(val.has(p)).toBe(false);
  });
});
