/**
 * Offline corpus preparation: parse local documents, clean them, remove duplicate paragraphs
 * and split them deterministically into training and validation text.
 *
 * Pure functions with no I/O, usable from the browser and from Node (the model-factory CLI reads
 * the files and passes their contents here).
 */

export interface CorpusDocument {
  /** Identifier (usually the file path or "file#line"). */
  id: string;
  text: string;
  /** Source format, used to decide whether Markdown should be stripped. */
  format?: 'text' | 'markdown';
}

export interface CorpusPrepOptions {
  /** Remove Markdown syntax (code blocks, tables, links, emphasis) from Markdown documents. */
  stripMarkdown?: boolean;
  /** Drop paragraphs shorter than this many characters (default 20). */
  minParagraphChars?: number;
  /** Remove exact duplicate paragraphs after normalisation (default true). */
  dedupe?: boolean;
  /** Fraction of paragraphs held out for validation (default 0.1). */
  validationFraction?: number;
  /** Salt for the hash-based split; changing it gives a different but stable split. */
  splitSeed?: number;
}

export interface CorpusStats {
  documents: number;
  paragraphs: number;
  duplicateParagraphs: number;
  shortParagraphs: number;
  trainParagraphs: number;
  validationParagraphs: number;
  trainCharacters: number;
  validationCharacters: number;
  distinctCharacters: number;
}

export interface PreparedCorpus {
  train: string;
  validation: string;
  stats: CorpusStats;
}

const PARAGRAPH_SEPARATOR = '\n\n';

/**
 * Unicode NFC normalisation, Unix line endings, no control characters (except newline and tab),
 * single spaces inside lines and at most one blank line between paragraphs.
 */
export function normalizeText(text: string): string {
  return removeControlCharacters(text.normalize('NFC').replace(/\r\n?/g, '\n'))
    .replace(/\t/g, ' ')
    .split(NO_BREAK_SPACE)
    .join(' ')
    .replace(/ {2,}/g, ' ')
    .split('\n')
    .map((line) => line.trim())
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

const NO_BREAK_SPACE = String.fromCharCode(0xa0);

/** Drop C0 control characters (except newline and tab), DEL and the byte-order mark. */
function removeControlCharacters(text: string): string {
  let out = '';
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if ((c < 0x20 && c !== 0x0a && c !== 0x09) || c === 0x7f || c === 0xfeff) continue;
    out += ch;
  }
  return out;
}

/** Remove Markdown syntax while keeping the prose. */
export function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '\n') // fenced code blocks
    .replace(/~~~[\s\S]*?~~~/g, '\n')
    .replace(/<!--[\s\S]*?-->/g, '') // HTML comments
    .replace(/^\s*\|.*\|\s*$/gm, '') // table rows
    .replace(/^\s*([-*_]\s*){3,}$/gm, '') // horizontal rules
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1') // images → alt text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') // links → text
    .replace(/<[^>\n]+>/g, '') // inline HTML tags
    .replace(/^\s{0,3}#{1,6}\s+/gm, '') // headings
    .replace(/^\s{0,3}>\s?/gm, '') // block quotes
    .replace(/^\s*[-*+]\s+/gm, '') // bullet markers
    .replace(/^\s*\d+[.)]\s+/gm, '') // numbered list markers
    .replace(/`([^`\n]*)`/g, '$1') // inline code
    .replace(/(\*\*|__)(.*?)\1/g, '$2') // bold
    .replace(/(^|[^\w*])\*(?!\s)([^*\n]+?)\*(?!\w)/g, '$1$2'); // italics
}

/** Split a document into paragraphs (blank-line separated). */
export function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s*\n\s*/g, ' ').trim())
    .filter((p) => p.length > 0);
}

/** Parse a CSV text into rows (RFC 4180 quoting). */
export function parseCsv(content: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < content.length; i++) {
    const ch = content[i];
    if (inQuotes) {
      if (ch === '"') {
        if (content[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && content[i + 1] === '\n') i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim().length > 0));
}

const TEXT_FIELDS = ['text', 'content', 'body', 'document'];

function textOf(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object') {
    for (const key of TEXT_FIELDS) {
      const v = (value as Record<string, unknown>)[key];
      if (typeof v === 'string') return v;
    }
  }
  return null;
}

/**
 * Turn a file's content into documents according to its extension:
 * .txt / .text (one document), .md / .markdown (one Markdown document), .jsonl (one document per
 * line, string or {text|content|body|document}), .json (array of the same), .csv (column named
 * text/content/body/document, or the first column).
 */
export function parseDocuments(name: string, content: string): CorpusDocument[] {
  const ext = name.toLowerCase().split('.').pop() ?? '';
  if (ext === 'md' || ext === 'markdown') return [{ id: name, text: content, format: 'markdown' }];
  if (ext === 'jsonl' || ext === 'ndjson') {
    const docs: CorpusDocument[] = [];
    content.split('\n').forEach((line, i) => {
      if (!line.trim()) return;
      try {
        const text = textOf(JSON.parse(line));
        if (text) docs.push({ id: `${name}#${i + 1}`, text, format: 'text' });
      } catch {
        // skip malformed lines
      }
    });
    return docs;
  }
  if (ext === 'json') {
    try {
      const data = JSON.parse(content);
      const items: unknown[] = Array.isArray(data) ? data : [data];
      return items
        .map((item, i) => ({
          id: `${name}#${i}`,
          text: textOf(item) ?? '',
          format: 'text' as const
        }))
        .filter((d) => d.text.length > 0);
    } catch {
      return [];
    }
  }
  if (ext === 'csv') {
    const rows = parseCsv(content);
    if (rows.length === 0) return [];
    const header = rows[0].map((h) => h.trim().toLowerCase());
    const column = Math.max(
      0,
      header.findIndex((h) => TEXT_FIELDS.includes(h))
    );
    const hasHeader = TEXT_FIELDS.includes(header[column]);
    return rows
      .slice(hasHeader ? 1 : 0)
      .map((r, i) => ({ id: `${name}#${i + 1}`, text: r[column] ?? '', format: 'text' as const }))
      .filter((d) => d.text.trim().length > 0);
  }
  return [{ id: name, text: content, format: 'text' }];
}

/** 32-bit FNV-1a hash. */
export function fnv1a(text: string, seed = 0): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Clean, de-duplicate and split documents. The train/validation assignment depends only on each
 * paragraph's content (and the seed), so it does not change when files are added or reordered.
 */
export function prepareCorpus(
  documents: CorpusDocument[],
  options: CorpusPrepOptions = {}
): PreparedCorpus {
  const minChars = options.minParagraphChars ?? 20;
  const dedupe = options.dedupe ?? true;
  const fraction = Math.min(0.9, Math.max(0, options.validationFraction ?? 0.1));
  const seed = options.splitSeed ?? 0;

  const seen = new Set<string>();
  const train: string[] = [];
  const validation: string[] = [];
  const chars = new Set<string>();
  let paragraphs = 0;
  let duplicates = 0;
  let short = 0;

  for (const doc of documents) {
    const raw =
      doc.format === 'markdown' && (options.stripMarkdown ?? true)
        ? stripMarkdown(doc.text)
        : doc.text;
    for (const paragraph of splitParagraphs(normalizeText(raw))) {
      paragraphs++;
      if (paragraph.length < minChars) {
        short++;
        continue;
      }
      if (dedupe) {
        if (seen.has(paragraph)) {
          duplicates++;
          continue;
        }
        seen.add(paragraph);
      }
      for (const ch of paragraph) chars.add(ch);
      const bucket = fnv1a(paragraph, seed) / 0x100000000;
      (bucket < fraction ? validation : train).push(paragraph);
    }
  }

  // Guarantee a non-empty validation set when a fraction was requested.
  if (fraction > 0 && validation.length === 0 && train.length > 1) {
    validation.push(train.pop()!);
  }

  const trainText = train.join(PARAGRAPH_SEPARATOR);
  const validationText = validation.join(PARAGRAPH_SEPARATOR);
  return {
    train: trainText,
    validation: validationText,
    stats: {
      documents: documents.length,
      paragraphs,
      duplicateParagraphs: duplicates,
      shortParagraphs: short,
      trainParagraphs: train.length,
      validationParagraphs: validation.length,
      trainCharacters: trainText.length,
      validationCharacters: validationText.length,
      distinctCharacters: chars.size
    }
  };
}
