import { ProNeuralLM, type TokenizerConfig } from './ProNeuralLM';
import { SPECIAL_TOKENS } from '../config/constants';

/**
 * Parse raw data into a valid TokenizerConfig
 * @param raw - Unknown input to parse
 * @returns Valid TokenizerConfig
 */
export function parseTokenizerConfig(raw: unknown): TokenizerConfig {
  if (!raw || typeof raw !== 'object') return { mode: 'unicode' };
  const mode = (raw as { mode?: unknown }).mode;
  if (mode === 'ascii') return { mode: 'ascii' };
  if (mode === 'custom') {
    const pattern = (raw as { pattern?: unknown }).pattern;
    if (typeof pattern === 'string' && pattern.length > 0) {
      return { mode: 'custom', pattern };
    }
    return { mode: 'custom', pattern: '' };
  }
  return { mode: 'unicode' };
}

/**
 * Format a timestamp for display
 * @param ts - Unix timestamp in milliseconds
 * @returns Formatted date string
 */
export function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short'
  });
}

/**
 * Check if a string is a valid regex pattern
 * @param pattern - Regex pattern to validate
 * @returns true if valid, false otherwise
 */
export function isValidRegex(pattern: string): boolean {
  try {
    new RegExp(pattern, 'gu');
    return true;
  } catch {
    try {
      new RegExp(pattern, 'g');
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Build vocabulary from text using tokenizer config
 * @param text - Input text to tokenize
 * @param tokenizerConfig - Tokenizer configuration
 * @returns Array of unique vocabulary tokens with special tokens
 */
export function buildVocab(text: string, tokenizerConfig: TokenizerConfig): string[] {
  const tokens = ProNeuralLM.tokenizeText(text, tokenizerConfig);
  const uniq = Array.from(new Set(tokens));
  return Array.from(new Set([...SPECIAL_TOKENS, ...uniq]));
}

/**
 * Download a blob as a file
 * @param blob - Blob to download
 * @param filename - Filename for the download
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Create a CSV blob from training history
 * @param history - Array of training history entries
 * @returns Blob containing CSV data
 */
export function createTrainingHistoryCsv(
  history: { loss: number; accuracy: number; timestamp: number }[]
): Blob {
  const header = 'epoch,loss,accuracy,timestamp';
  const rows = history.map(
    (h, i) =>
      `${i + 1},${h.loss.toFixed(6)},${h.accuracy.toFixed(6)},${new Date(h.timestamp).toISOString()}`
  );
  return new Blob([`${header}\n${rows.join('\n')}`], { type: 'text/csv' });
}
