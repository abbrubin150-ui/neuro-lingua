/**
 * Subword (byte-pair-encoding) tokenizer used by the models' `'bpe'` tokenizer mode.
 *
 * - Pre-tokenisation splits text into pieces: runs of letters (with combining marks, so Hebrew
 *   niqqud stays attached), runs of digits, and runs of other symbols. A piece that follows
 *   whitespace — or starts the text — begins with the word marker "▁" (as in SentencePiece), so
 *   spaces are reconstructed exactly by `decodeSubword`.
 * - Training is classic BPE over base symbols (a character with its combining marks): repeatedly
 *   merge the most frequent adjacent pair.
 *   Pair counts are maintained incrementally and the best pair is taken from a lazy max-heap, so a
 *   merge costs time proportional to the words it touches rather than to the whole corpus.
 *   Ties are broken by the pair's string order, which makes training fully deterministic.
 * - Encoding applies the merges by rank within each piece (results are cached per piece).
 *
 * Everything runs locally: no network access and no external model or service is involved.
 */

/** Word-start marker (U+2581 LOWER ONE EIGHTH BLOCK). */
export const WORD_MARKER = '▁';

export interface SubwordModel {
  /** Base symbols (characters with their combining marks, plus the word marker), sorted. */
  alphabet: string[];
  /** Merge rules in rank order: [left, right] → left + right. */
  merges: [string, string][];
  /** Lower-case text before tokenisation. */
  lowercase: boolean;
}

export interface SubwordTrainingOptions {
  /** Target number of subword symbols (alphabet + merges). */
  vocabSize: number;
  /** Stop when the most frequent pair occurs fewer times than this (default 2). */
  minFrequency?: number;
  /** Lower-case the corpus (default false). */
  lowercase?: boolean;
}

export interface SubwordSpan {
  token: string;
  /** [start, end) offsets of the characters covered by this token in the original text. */
  start: number;
  end: number;
}

const PIECE_REGEX = /([\p{L}\p{M}]+|\p{N}+|[^\s\p{L}\p{M}\p{N}]+)/gu;
const SYMBOL_REGEX = /\p{M}+|\P{M}\p{M}*/gu;

/**
 * Split a piece into base symbols: a character together with its combining marks (so Hebrew
 * niqqud or Latin accents are never separate tokens); the word marker is its own symbol.
 */
export function splitSymbols(piece: string): string[] {
  if (piece.startsWith(WORD_MARKER)) {
    return [WORD_MARKER, ...(piece.slice(WORD_MARKER.length).match(SYMBOL_REGEX) ?? [])];
  }
  return piece.match(SYMBOL_REGEX) ?? [];
}

interface Piece {
  /** Piece text including the leading marker when it starts a word. */
  text: string;
  /** Offsets of the piece's characters (without the marker) in the original text. */
  start: number;
  end: number;
  marked: boolean;
}

function normalise(text: string, lowercase: boolean): string {
  // Lower-casing can change the length of a few special characters (e.g. "İ"); offsets are
  // computed on the normalised string, which is the same length for practically all text.
  return lowercase ? text.toLowerCase() : text;
}

function preTokenize(text: string, lowercase: boolean): Piece[] {
  const source = normalise(text, lowercase);
  const pieces: Piece[] = [];
  let previousEnd = 0;
  PIECE_REGEX.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = PIECE_REGEX.exec(source)) !== null) {
    const start = match.index;
    const core = match[0].split(PAIR_SEPARATOR).join('');
    if (core.length === 0) continue;
    const marked = pieces.length === 0 || /\s/u.test(source.slice(previousEnd, start));
    pieces.push({
      text: marked ? WORD_MARKER + core : core,
      start,
      end: start + match[0].length,
      marked
    });
    previousEnd = start + match[0].length;
  }
  return pieces;
}

const PAIR_SEPARATOR = String.fromCharCode(0);
const pairKey = (a: string, b: string) => a + PAIR_SEPARATOR + b;

/** Max-heap of [count, key] with deterministic tie-breaking on the key. */
class PairHeap {
  private items: [number, string][] = [];

  private better(a: [number, string], b: [number, string]): boolean {
    return a[0] > b[0] || (a[0] === b[0] && a[1] < b[1]);
  }

  push(item: [number, string]): void {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.better(items[i], items[parent])) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }

  pop(): [number, string] | undefined {
    const items = this.items;
    if (items.length === 0) return undefined;
    const top = items[0];
    const last = items.pop()!;
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (l < items.length && this.better(items[l], items[best])) best = l;
        if (r < items.length && this.better(items[r], items[best])) best = r;
        if (best === i) break;
        [items[i], items[best]] = [items[best], items[i]];
        i = best;
      }
    }
    return top;
  }
}

/**
 * Learn BPE merges from a corpus. The result is a plain JSON-serialisable object that can be
 * stored in a model's tokenizer configuration.
 */
export function trainSubwordModel(corpus: string, options: SubwordTrainingOptions): SubwordModel {
  const lowercase = options.lowercase ?? false;
  const minFrequency = Math.max(1, options.minFrequency ?? 2);

  const frequencies = new Map<string, number>();
  for (const piece of preTokenize(corpus, lowercase)) {
    frequencies.set(piece.text, (frequencies.get(piece.text) ?? 0) + 1);
  }

  const words = [...frequencies.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([text, freq]) => ({ symbols: splitSymbols(text), freq }));

  const alphabetSet = new Set<string>([WORD_MARKER]);
  for (const word of words) for (const s of word.symbols) alphabetSet.add(s);
  const alphabet = [...alphabetSet].sort();

  const counts = new Map<string, number>();
  const occurrences = new Map<string, Set<number>>();
  const addPairs = (index: number, sign: 1 | -1) => {
    const { symbols, freq } = words[index];
    for (let i = 0; i < symbols.length - 1; i++) {
      const key = pairKey(symbols[i], symbols[i + 1]);
      counts.set(key, (counts.get(key) ?? 0) + sign * freq);
      if (sign > 0) {
        let set = occurrences.get(key);
        if (!set) {
          set = new Set();
          occurrences.set(key, set);
        }
        set.add(index);
      }
    }
  };
  words.forEach((_, i) => addPairs(i, 1));

  const heap = new PairHeap();
  for (const [key, count] of counts) heap.push([count, key]);

  const merges: [string, string][] = [];
  const targetMerges = Math.max(0, Math.floor(options.vocabSize) - alphabet.length);
  while (merges.length < targetMerges) {
    const top = heap.pop();
    if (!top) break;
    const [count, key] = top;
    const current = counts.get(key) ?? 0;
    if (current !== count) {
      if (current > 0) heap.push([current, key]); // stale entry: re-insert with the live count
      continue;
    }
    if (count < minFrequency) break;

    const [left, right] = key.split(PAIR_SEPARATOR);
    const merged = left + right;
    merges.push([left, right]);

    const touched = new Set<string>();
    for (const index of occurrences.get(key) ?? []) {
      const word = words[index];
      if (!word.symbols.some((s, i) => s === left && word.symbols[i + 1] === right)) continue;
      // Remove this word's old pair contributions, merge, then add the new ones.
      for (let i = 0; i < word.symbols.length - 1; i++) {
        const k = pairKey(word.symbols[i], word.symbols[i + 1]);
        counts.set(k, (counts.get(k) ?? 0) - word.freq);
        touched.add(k);
      }
      const next: string[] = [];
      for (let i = 0; i < word.symbols.length; i++) {
        if (
          i < word.symbols.length - 1 &&
          word.symbols[i] === left &&
          word.symbols[i + 1] === right
        ) {
          next.push(merged);
          i++;
        } else {
          next.push(word.symbols[i]);
        }
      }
      word.symbols = next;
      addPairs(index, 1);
      for (let i = 0; i < next.length - 1; i++) touched.add(pairKey(next[i], next[i + 1]));
    }
    counts.delete(key);
    occurrences.delete(key);
    for (const k of touched) {
      const c = counts.get(k) ?? 0;
      if (c > 0) heap.push([c, k]);
    }
  }

  return { alphabet, merges, lowercase };
}

/** Subword vocabulary of a model: alphabet followed by every merged symbol (no duplicates). */
export function subwordVocabulary(model: SubwordModel): string[] {
  const seen = new Set<string>();
  const vocab: string[] = [];
  for (const s of model.alphabet) {
    if (!seen.has(s)) {
      seen.add(s);
      vocab.push(s);
    }
  }
  for (const [a, b] of model.merges) {
    const m = a + b;
    if (!seen.has(m)) {
      seen.add(m);
      vocab.push(m);
    }
  }
  return vocab;
}

interface EncoderState {
  ranks: Map<string, number>;
  cache: Map<string, string[]>;
}

const encoderStates = new WeakMap<SubwordModel, EncoderState>();
const MAX_CACHE_ENTRIES = 50000;

function encoderState(model: SubwordModel): EncoderState {
  let state = encoderStates.get(model);
  if (!state) {
    const ranks = new Map<string, number>();
    model.merges.forEach(([a, b], rank) => {
      const key = pairKey(a, b);
      if (!ranks.has(key)) ranks.set(key, rank);
    });
    state = { ranks, cache: new Map() };
    encoderStates.set(model, state);
  }
  return state;
}

function encodePiece(piece: string, state: EncoderState): string[] {
  const cached = state.cache.get(piece);
  if (cached) return cached;
  let symbols = splitSymbols(piece);
  while (symbols.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < symbols.length - 1; i++) {
      const rank = state.ranks.get(pairKey(symbols[i], symbols[i + 1]));
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex < 0) break;
    symbols = [
      ...symbols.slice(0, bestIndex),
      symbols[bestIndex] + symbols[bestIndex + 1],
      ...symbols.slice(bestIndex + 2)
    ];
  }
  if (state.cache.size >= MAX_CACHE_ENTRIES) state.cache.clear();
  state.cache.set(piece, symbols);
  return symbols;
}

/** Tokenise text into subword tokens with their character spans in `text`. */
export function encodeSubword(text: string, model: SubwordModel): SubwordSpan[] {
  const state = encoderState(model);
  const spans: SubwordSpan[] = [];
  for (const piece of preTokenize(text, model.lowercase)) {
    let offset = piece.start;
    for (const token of encodePiece(piece.text, state)) {
      // The word marker is not a character of the source text.
      const visible = token.startsWith(WORD_MARKER)
        ? token.length - WORD_MARKER.length
        : token.length;
      const end = Math.min(piece.end, offset + visible);
      spans.push({ token, start: offset, end });
      offset = end;
    }
  }
  return spans;
}

/**
 * Reassemble text from subword tokens (markers become spaces). For a continuation of existing
 * text (generation) a leading marker means "space before this word" and is kept as a space.
 */
export function decodeSubword(tokens: string[], options: { continuation?: boolean } = {}): string {
  const text = tokens.join('').split(WORD_MARKER).join(' ');
  return options.continuation ? text : text.replace(/^ /, '');
}

/** Light structural validation for models read from storage or files. */
export function isSubwordModel(value: unknown): value is SubwordModel {
  if (!value || typeof value !== 'object') return false;
  const v = value as Partial<SubwordModel>;
  return (
    Array.isArray(v.alphabet) &&
    v.alphabet.every((s) => typeof s === 'string') &&
    Array.isArray(v.merges) &&
    v.merges.every(
      (m) =>
        Array.isArray(m) && m.length === 2 && typeof m[0] === 'string' && typeof m[1] === 'string'
    ) &&
    typeof v.lowercase === 'boolean'
  );
}
