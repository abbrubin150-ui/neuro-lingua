import { describe, expect, it } from 'vitest';
import {
  answerSpans,
  formatQaPair,
  parseQaPairs,
  qaDocument,
  qaLossMask,
  qaPrompt
} from '../../src/data/qa';
import { prepareCorpus } from '../../src/data/corpusPrep';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { trainSubwordModel } from '../../src/tokenizer/subword';

describe('Q&A pairs', () => {
  it('reads jsonl with several field names and chat messages', () => {
    const jsonl = [
      JSON.stringify({ question: 'What is BPE?', answer: 'A subword tokenizer.' }),
      JSON.stringify({ prompt: 'Is it offline?', completion: 'Yes, entirely.' }),
      JSON.stringify({ instruction: 'Name a preset', output: 'tiny' }),
      JSON.stringify({
        messages: [
          { role: 'system', content: 'ignored' },
          { role: 'user', content: 'Hi?' },
          { role: 'assistant', content: 'Hello.' },
          { role: 'user', content: 'Bye?' },
          { role: 'assistant', content: 'Goodbye.' }
        ]
      }),
      'not json',
      JSON.stringify({ question: 'No answer' })
    ].join('\n');
    expect(parseQaPairs('faq.jsonl', jsonl)).toEqual([
      { question: 'What is BPE?', answer: 'A subword tokenizer.' },
      { question: 'Is it offline?', answer: 'Yes, entirely.' },
      { question: 'Name a preset', answer: 'tiny' },
      { question: 'Hi?', answer: 'Hello.' },
      { question: 'Bye?', answer: 'Goodbye.' }
    ]);
  });

  it('reads csv with a header (any column order) and Q:/A: text blocks in English and Hebrew', () => {
    expect(parseQaPairs('x.csv', 'answer,question\n"yes, local",offline?\nno,cloud?')).toEqual([
      { question: 'offline?', answer: 'yes, local' },
      { question: 'cloud?', answer: 'no' }
    ]);
    const text = 'Q: What is it?\nA: A language model\nthat runs locally.\n\nש: מה זה?\nת: מודל שפה.';
    expect(parseQaPairs('faq.md', text)).toEqual([
      { question: 'What is it?', answer: 'A language model\nthat runs locally.' },
      { question: 'מה זה?', answer: 'מודל שפה.' }
    ]);
  });

  it('formats one paragraph per pair and finds the answers in the prepared text', () => {
    expect(formatQaPair({ question: 'a\n\nb', answer: 'c' })).toBe('Question: a\nb\nAnswer: c');
    const doc = qaDocument('f.jsonl', '{"q":"Why offline?","a":"Privacy and control."}\n{"q":"How fast?","a":"About 900 tokens per second."}');
    expect(doc.pairs).toBe(2);
    const prepared = prepareCorpus([doc], { validationFraction: 0, minParagraphChars: 1 }).train;
    const answers = answerSpans(prepared).map((s) => prepared.slice(s.start, s.end).trim());
    expect(answers.sort()).toEqual(['About 900 tokens per second.', 'Privacy and control.']);
    expect(qaPrompt('Why offline?')).toBe('Question: Why offline? Answer:');
  });

  it('masks exactly the answer tokens plus the token that ends each answer', () => {
    const text = 'Question: why here? Answer: because it is local.\n\nQuestion: when? Answer: now.';
    const bpe = trainSubwordModel(text, { vocabSize: 60 });
    const spans = ProNeuralLM.tokenizeWithOffsets(text, { mode: 'bpe', bpe });
    const { mask, answerCharacters } = qaLossMask(text, spans);
    expect(mask).toHaveLength(spans.length + 1);
    // Text covered by the learned tokens: the answers (BPE pieces, without the spaces the word
    // markers stand for), the first piece of the next question, and the end of the text.
    const learned = spans
      .filter((_, t) => mask[t] === 1)
      .map((s) => text.slice(s.start, s.end))
      .join('');
    const first = 'becauseitislocal.';
    expect(learned.startsWith(first)).toBe(true);
    expect(learned.endsWith('now.')).toBe(true);
    const stop = learned.slice(first.length, learned.length - 'now.'.length);
    expect(stop.length).toBeGreaterThan(0);
    expect('Question'.startsWith(stop)).toBe(true);
    // Nothing of the questions or labels is learned besides that stop token.
    expect(learned).not.toContain('why');
    expect(learned).not.toContain('Answer');
    expect(mask[spans.length]).toBe(1);
    expect(answerCharacters).toBe(' because it is local.'.length + ' now.'.length);
  });

  it('labels can be configured (e.g. Hebrew)', () => {
    const template = { questionLabel: 'שאלה:', answerLabel: 'תשובה:' };
    const text = formatQaPair({ question: 'מה?', answer: 'כך.' }, template).replace('\n', ' ');
    const spans = answerSpans(text, template);
    expect(text.slice(spans[0].start, spans[0].end)).toBe(' כך.');
    expect(qaPrompt('מה?', template)).toBe('שאלה: מה? תשובה:');
  });
});

describe('answer scoring', () => {
  it('compares answers exactly (ignoring case, spacing, final punctuation) and by word overlap', async () => {
    const { sameAnswer, answerF1 } = await import('../../src/data/qa');
    expect(sameAnswer('Grass is  green.', 'grass is green')).toBe(true);
    expect(sameAnswer('grass is blue', 'grass is green')).toBe(false);
    expect(answerF1('grass is green', 'grass is green')).toBe(1);
    expect(answerF1('grass is blue', 'grass is green')).toBeCloseTo(2 / 3, 10);
    expect(answerF1('snow', 'grass is green')).toBe(0);
    expect(answerF1('מודל שפה', 'מודל שפה קטן')).toBeCloseTo(0.8, 10);
  });
});
