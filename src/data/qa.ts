/**
 * Question–answer data, for models that answer rather than only continue text.
 *
 * Pairs are read from .jsonl / .json (question/answer, prompt/completion, instruction/output,
 * or chat `messages`), .csv (question/answer or prompt/completion columns, else the first two)
 * and text / Markdown files with "Q:" / "A:" blocks (also "Question:" / "Answer:" and the Hebrew
 * "ש:" / "ת:", "שאלה:" / "תשובה:"). Every pair becomes one paragraph,
 * "<questionLabel> question <answerLabel> answer", so the corpus preparation cleans,
 * de-duplicates and splits pairs like any other paragraph.
 *
 * `qaLossMask` then marks, for every token of a prepared text, whether the model should learn to
 * predict it: the tokens of the answers and the token that follows each answer (the start of the
 * next question, or the end of the text), so that the model also learns where an answer stops.
 * Questions and labels are context only.
 */

import { parseCsv, type CorpusDocument } from './corpusPrep';

export interface QaPair {
  question: string;
  answer: string;
}

export interface QaTemplate {
  questionLabel: string;
  answerLabel: string;
}

export const DEFAULT_QA_TEMPLATE: QaTemplate = {
  questionLabel: 'Question:',
  answerLabel: 'Answer:'
};

const QUESTION_KEYS = ['question', 'prompt', 'instruction', 'input', 'q'];
const ANSWER_KEYS = ['answer', 'completion', 'response', 'output', 'a'];

function pick(record: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) return value;
  }
  return null;
}

/** Pairs from one JSON value: a pair object or a chat (`messages`: user → assistant turns). */
function pairsFromRecord(value: unknown): QaPair[] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return [];
  const record = value as Record<string, unknown>;
  if (Array.isArray(record.messages)) {
    const pairs: QaPair[] = [];
    let question: string | null = null;
    for (const m of record.messages as Array<Record<string, unknown>>) {
      if (!m || typeof m.content !== 'string') continue;
      if (m.role === 'user') question = m.content;
      else if (m.role === 'assistant' && question !== null) {
        pairs.push({ question, answer: m.content });
        question = null;
      }
    }
    return pairs;
  }
  const question = pick(record, QUESTION_KEYS);
  const answer = pick(record, ANSWER_KEYS);
  return question && answer ? [{ question, answer }] : [];
}

const Q_LINE = /^\s*(?:Q|Question|ש|שאלה)\s*[:：]\s?(.*)$/iu;
const A_LINE = /^\s*(?:A|Answer|ת|תשובה)\s*[:：]\s?(.*)$/iu;

/** "Q: …" / "A: …" blocks; continuation lines belong to the current question or answer. */
function pairsFromText(content: string): QaPair[] {
  const pairs: QaPair[] = [];
  let question: string[] | null = null;
  let answer: string[] | null = null;
  const flush = () => {
    if (question && answer) {
      pairs.push({ question: question.join('\n'), answer: answer.join('\n') });
    }
    question = null;
    answer = null;
  };
  for (const line of content.replace(/\r\n?/g, '\n').split('\n')) {
    const q = Q_LINE.exec(line);
    const a = q ? null : A_LINE.exec(line);
    if (q) {
      flush();
      question = [q[1]];
    } else if (a && question) {
      answer = [a[1]];
    } else if (answer) {
      answer.push(line);
    } else if (question) {
      question.push(line);
    }
  }
  flush();
  return pairs;
}

/** Question–answer pairs of a file, chosen by its extension (see the module comment). */
export function parseQaPairs(name: string, content: string): QaPair[] {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  let pairs: QaPair[];
  if (ext === 'jsonl' || ext === 'ndjson') {
    pairs = content
      .split('\n')
      .filter((line) => line.trim())
      .flatMap((line) => {
        try {
          return pairsFromRecord(JSON.parse(line));
        } catch {
          return [];
        }
      });
  } else if (ext === 'json') {
    try {
      const data = JSON.parse(content);
      pairs = (Array.isArray(data) ? data : [data]).flatMap(pairsFromRecord);
    } catch {
      pairs = [];
    }
  } else if (ext === 'csv') {
    const rows = parseCsv(content);
    const header = (rows[0] ?? []).map((h) => h.trim().toLowerCase());
    const qi = header.findIndex((h) => QUESTION_KEYS.includes(h));
    const ai = header.findIndex((h) => ANSWER_KEYS.includes(h));
    const hasHeader = qi >= 0 && ai >= 0;
    const [q, a] = hasHeader ? [qi, ai] : [0, 1];
    pairs = rows
      .slice(hasHeader ? 1 : 0)
      .map((r) => ({ question: r[q] ?? '', answer: r[a] ?? '' }));
  } else {
    pairs = pairsFromText(content);
  }
  return pairs
    .map((p) => ({ question: p.question.trim(), answer: p.answer.trim() }))
    .filter((p) => p.question.length > 0 && p.answer.length > 0);
}

/** One pair as a single paragraph (blank lines inside a pair would split it). */
export function formatQaPair(pair: QaPair, template: QaTemplate = DEFAULT_QA_TEMPLATE): string {
  const oneParagraph = (s: string) => s.replace(/\n\s*\n+/g, '\n');
  return `${template.questionLabel} ${oneParagraph(pair.question)}\n${template.answerLabel} ${oneParagraph(pair.answer)}`;
}

/** The pairs of a file as one corpus document (one paragraph per pair). */
export function qaDocument(
  name: string,
  content: string,
  template: QaTemplate = DEFAULT_QA_TEMPLATE
): CorpusDocument & { pairs: number } {
  const pairs = parseQaPairs(name, content);
  return {
    id: name,
    text: pairs.map((p) => formatQaPair(p, template)).join('\n\n'),
    format: 'text',
    pairs: pairs.length
  };
}

/** The prompt that asks a Q&A model for an answer. */
export function qaPrompt(question: string, template: QaTemplate = DEFAULT_QA_TEMPLATE): string {
  return `${template.questionLabel} ${question.trim()} ${template.answerLabel}`;
}

/**
 * Character spans of the answers in a prepared text: in every paragraph that starts with the
 * question label, the text after the first answer label.
 */
export function answerSpans(
  text: string,
  template: QaTemplate = DEFAULT_QA_TEMPLATE
): Array<{ start: number; end: number }> {
  const spans: Array<{ start: number; end: number }> = [];
  let offset = 0;
  for (const paragraph of text.split('\n\n')) {
    const lead = paragraph.length - paragraph.trimStart().length;
    if (paragraph.startsWith(template.questionLabel, lead)) {
      const label = paragraph.indexOf(template.answerLabel, lead + template.questionLabel.length);
      if (label >= 0) {
        const start = label + template.answerLabel.length;
        const end = paragraph.trimEnd().length;
        if (end > start) spans.push({ start: offset + start, end: offset + end });
      }
    }
    offset += paragraph.length + 2;
  }
  return spans;
}

/**
 * Per-token loss mask (length tokens + 1, the last entry being the end of the text) from the
 * tokens' character spans: 1 for tokens that start inside an answer and for the first token
 * after each answer (or the end of the text), 0 elsewhere. Also returns the number of answer
 * characters (for bits per answer character).
 */
export function qaLossMask(
  text: string,
  tokens: Array<{ start: number; end: number }>,
  template: QaTemplate = DEFAULT_QA_TEMPLATE
): { mask: number[]; answerCharacters: number } {
  const spans = answerSpans(text, template);
  const mask = new Array<number>(tokens.length + 1).fill(0);
  let t = 0;
  for (const span of spans) {
    while (t < tokens.length && tokens[t].start < span.start) t++;
    while (t < tokens.length && tokens[t].start < span.end) mask[t++] = 1;
    // The token that ends the answer: the next question's first token, or the end of the text.
    mask[t < tokens.length ? t : tokens.length] = 1;
  }
  const answerCharacters = spans.reduce((n, sp) => n + sp.end - sp.start, 0);
  return { mask, answerCharacters };
}

/**
 * The token a Q&A model emits when an answer is complete: the first token of the next question
 * label, as it appears in the training text (after a paragraph break). Generation stops there.
 */
export function qaStopToken(
  tokenize: (text: string) => Array<{ token: string; start: number }>,
  template: QaTemplate = DEFAULT_QA_TEMPLATE
): string {
  const text = `x.\n\n${template.questionLabel}`;
  const labelStart = text.length - template.questionLabel.length;
  return tokenize(text).find((t) => t.start >= labelStart)?.token ?? template.questionLabel;
}

/** The pairs of a prepared Q&A text (one "<question label> … <answer label> …" per paragraph). */
export function qaPairsFromText(
  text: string,
  template: QaTemplate = DEFAULT_QA_TEMPLATE
): QaPair[] {
  return text.split('\n\n').flatMap((paragraph) => {
    const p = paragraph.trim();
    if (!p.startsWith(template.questionLabel)) return [];
    const label = p.indexOf(template.answerLabel, template.questionLabel.length);
    if (label < 0) return [];
    const question = p.slice(template.questionLabel.length, label).trim();
    const answer = p.slice(label + template.answerLabel.length).trim();
    return question && answer ? [{ question, answer }] : [];
  });
}

/** Answer comparison that ignores case, spacing and trailing punctuation. */
export function sameAnswer(a: string, b: string): boolean {
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/[.!?。]+$/u, '');
  return norm(a) === norm(b);
}

/** Word-overlap F1 between a generated and a reference answer (as in SQuAD), in [0, 1]. */
export function answerF1(answer: string, reference: string): number {
  const words = (s: string) => s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  const a = words(answer);
  const b = words(reference);
  if (a.length === 0 || b.length === 0) return a.length === b.length ? 1 : 0;
  const counts = new Map<string, number>();
  for (const w of b) counts.set(w, (counts.get(w) ?? 0) + 1);
  let common = 0;
  for (const w of a) {
    const n = counts.get(w) ?? 0;
    if (n > 0) {
      common++;
      counts.set(w, n - 1);
    }
  }
  if (common === 0) return 0;
  const precision = common / a.length;
  const recall = common / b.length;
  return (2 * precision * recall) / (precision + recall);
}
