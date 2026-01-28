/**
 * Loss Mask Utilities for Answer-Only Training
 *
 * Implements loss masking to train models only on answer tokens,
 * preventing memorization of prompts/questions.
 *
 * Supports:
 * - Simple delimiter-based masking (afterEquals, afterAnswerTag)
 * - Custom RegExp pattern matching with multiple position modes
 *
 * @module losses/lossMask
 * @version 4.6.0
 */

import type { LossMaskMode, RegExpMaskPosition } from '../types/project';

/**
 * Simple tokenizer interface for building loss masks
 */
export interface TokenizerLike {
  encode: (text: string) => number[];
  decode?: (ids: number[]) => string;
}

/**
 * Options for RegExp-based loss mask building
 */
export interface RegExpMaskOptions {
  /** The RegExp pattern string */
  pattern: string;
  /** Where to apply loss relative to the match */
  position: RegExpMaskPosition;
  /** Use global flag to find all matches (default: true) */
  global?: boolean;
  /** Case insensitive matching (default: false) */
  caseInsensitive?: boolean;
}

/**
 * Result of a RegExp match with position information
 */
export interface RegExpMatchResult {
  /** Start index in the text */
  start: number;
  /** End index in the text (exclusive) */
  end: number;
  /** The matched text */
  match: string;
}

/**
 * Validate a RegExp pattern string.
 *
 * @param pattern - The pattern to validate
 * @returns Object with valid flag and optional error message
 */
export function validateRegExpPattern(pattern: string): { valid: boolean; error?: string } {
  if (!pattern || pattern.trim() === '') {
    return { valid: false, error: 'Pattern cannot be empty' };
  }

  try {
    new RegExp(pattern, 'gu');
    return { valid: true };
  } catch (e) {
    return { valid: false, error: e instanceof Error ? e.message : 'Invalid RegExp' };
  }
}

/**
 * Find all matches of a RegExp pattern in text.
 *
 * @param text - The text to search
 * @param pattern - The RegExp pattern string
 * @param options - Optional flags for the RegExp
 * @returns Array of match results with start/end positions
 */
export function findRegExpMatches(
  text: string,
  pattern: string,
  options: { global?: boolean; caseInsensitive?: boolean } = {}
): RegExpMatchResult[] {
  const { global = true, caseInsensitive = false } = options;

  let flags = 'u'; // Always use unicode
  if (global) flags += 'g';
  if (caseInsensitive) flags += 'i';

  try {
    const regex = new RegExp(pattern, flags);
    const matches: RegExpMatchResult[] = [];

    if (global) {
      let match: RegExpExecArray | null;
      while ((match = regex.exec(text)) !== null) {
        matches.push({
          start: match.index,
          end: match.index + match[0].length,
          match: match[0]
        });
        // Prevent infinite loop on zero-length matches
        if (match[0].length === 0) {
          regex.lastIndex++;
        }
      }
    } else {
      const match = regex.exec(text);
      if (match) {
        matches.push({
          start: match.index,
          end: match.index + match[0].length,
          match: match[0]
        });
      }
    }

    return matches;
  } catch {
    return [];
  }
}

/**
 * Build a character-level mask from RegExp matches.
 *
 * @param text - The input text
 * @param matches - Array of RegExp match results
 * @param position - Where to apply the mask relative to matches
 * @returns Array of 0s and 1s for each character position
 */
export function buildCharMaskFromMatches(
  text: string,
  matches: RegExpMatchResult[],
  position: RegExpMaskPosition
): number[] {
  const mask = new Array(text.length).fill(0);

  if (matches.length === 0) {
    // No matches found: return all zeros to prevent training on malformed examples
    // Exception: 'exclude' mode with no matches means include everything
    if (position === 'exclude') {
      return mask.map(() => 1);
    }
    return mask;
  }

  switch (position) {
    case 'after': {
      // Compute loss only AFTER the first match
      const firstMatch = matches[0];
      for (let i = firstMatch.end; i < text.length; i++) {
        mask[i] = 1;
      }
      break;
    }
    case 'before': {
      // Compute loss only BEFORE the first match
      const firstMatch = matches[0];
      for (let i = 0; i < firstMatch.start; i++) {
        mask[i] = 1;
      }
      break;
    }
    case 'match': {
      // Compute loss ONLY on matched portions (all matches)
      for (const m of matches) {
        for (let i = m.start; i < m.end; i++) {
          mask[i] = 1;
        }
      }
      break;
    }
    case 'exclude': {
      // Compute loss on everything EXCEPT matched portions
      // Start with all 1s
      mask.fill(1);
      // Zero out matched regions
      for (const m of matches) {
        for (let i = m.start; i < m.end; i++) {
          mask[i] = 0;
        }
      }
      break;
    }
  }

  return mask;
}

/**
 * Map character-level mask to token-level mask.
 *
 * Uses a simple heuristic: a token is included if the majority of its
 * character positions are included in the character mask.
 *
 * @param charMask - Character-level mask array
 * @param text - The original text
 * @param tokenizer - Tokenizer for encoding text
 * @returns Token-level mask array
 */
export function mapCharMaskToTokenMask(
  charMask: number[],
  text: string,
  tokenizer: TokenizerLike
): number[] {
  const tokens = tokenizer.encode(text);
  const T = tokens.length;
  const tokenMask = new Array(T).fill(0);

  if (T === 0) return tokenMask;

  // For character-level tokenizer (like our simple char tokenizer),
  // the mapping is 1:1
  if (text.length === T) {
    return charMask.slice(0, T);
  }

  // For subword tokenizers, we need to estimate token boundaries
  // This is an approximation - ideally the tokenizer would provide offsets
  const avgCharsPerToken = text.length / T;

  for (let t = 0; t < T; t++) {
    // Estimate character range for this token
    const charStart = Math.floor(t * avgCharsPerToken);
    const charEnd = Math.min(Math.floor((t + 1) * avgCharsPerToken), text.length);

    // Token is included if majority of its characters are included
    let included = 0;
    let total = 0;
    for (let c = charStart; c < charEnd; c++) {
      if (c < charMask.length) {
        included += charMask[c];
        total++;
      }
    }

    tokenMask[t] = total > 0 && included > total / 2 ? 1 : 0;
  }

  return tokenMask;
}

/**
 * Build a loss mask using custom RegExp pattern matching.
 *
 * @param text - The raw input text
 * @param tokenizer - Tokenizer with encode method
 * @param options - RegExp mask options
 * @returns Array of 0s and 1s, same length as encoded tokens
 *
 * @example
 * // Mask everything before the answer pattern
 * const mask = buildRegExpLossMask(text, tokenizer, {
 *   pattern: 'Answer:\\s*',
 *   position: 'after'
 * });
 */
export function buildRegExpLossMask(
  text: string,
  tokenizer: TokenizerLike,
  options: RegExpMaskOptions
): number[] {
  const { pattern, position, global = true, caseInsensitive = false } = options;

  // Validate pattern
  const validation = validateRegExpPattern(pattern);
  if (!validation.valid) {
    // Invalid pattern: return all zeros (no loss computed)
    const tokens = tokenizer.encode(text);
    return new Array(tokens.length).fill(0);
  }

  // Find all matches
  const matches = findRegExpMatches(text, pattern, { global, caseInsensitive });

  // Build character-level mask
  const charMask = buildCharMaskFromMatches(text, matches, position);

  // Map to token-level mask
  return mapCharMaskToTokenMask(charMask, text, tokenizer);
}

/**
 * Build a loss mask array based on delimiter position in input tokens.
 *
 * The mask is 0 for tokens before (and including) the delimiter,
 * and 1 for tokens after the delimiter (the answer portion).
 *
 * @param inputIds - Array of token IDs for the full sequence
 * @param tokenizer - Tokenizer with encode method
 * @param mode - Loss mask mode: 'none' | 'afterEquals' | 'afterAnswerTag' | 'customRegExp'
 * @param answerTag - Custom answer tag (default: 'A:')
 * @param regExpOptions - Options for customRegExp mode
 * @returns Array of 0s and 1s, same length as inputIds
 *
 * @example
 * // For input "Q:20+7=A:27" with mode='afterAnswerTag'
 * // Returns [0,0,0,0,0,0,0,1,1] (1s only on "27" tokens)
 */
export function buildAnswerLossMask(
  inputIds: number[],
  tokenizer: TokenizerLike,
  mode: LossMaskMode,
  answerTag = 'A:',
  regExpOptions?: { pattern?: string; position?: RegExpMaskPosition }
): number[] {
  const T = inputIds.length;
  const mask = new Array(T).fill(0);

  // If no masking, compute loss on all tokens
  if (mode === 'none') {
    return mask.map(() => 1);
  }

  // Handle customRegExp mode
  if (mode === 'customRegExp') {
    if (!regExpOptions?.pattern) {
      // No pattern provided: return all zeros
      return mask;
    }

    // For customRegExp mode, we need the raw text
    // If tokenizer has decode, use it; otherwise, fall back to char mapping
    let text = '';
    if (tokenizer.decode) {
      text = tokenizer.decode(inputIds);
    } else {
      // Assume character-level tokenization
      text = String.fromCharCode(...inputIds);
    }

    return buildRegExpLossMask(text, tokenizer, {
      pattern: regExpOptions.pattern,
      position: regExpOptions.position || 'after'
    });
  }

  // Determine delimiter tokens based on mode
  const delimiterText = mode === 'afterEquals' ? '=' : answerTag;
  const delimiterTokens = tokenizer.encode(delimiterText);

  // Find first occurrence of delimiter tokens in inputIds
  let start = -1;
  outer: for (let i = 0; i <= T - delimiterTokens.length; i++) {
    for (let k = 0; k < delimiterTokens.length; k++) {
      if (inputIds[i + k] !== delimiterTokens[k]) continue outer;
    }
    // Found! Start computing loss AFTER the delimiter
    start = i + delimiterTokens.length;
    break;
  }

  // If no delimiter found, return all zeros (no loss computed)
  // This prevents training on malformed examples
  if (start < 0) {
    return mask;
  }

  // Set mask to 1 for all tokens after delimiter
  for (let t = start; t < T; t++) {
    mask[t] = 1;
  }

  return mask;
}

/**
 * Compute masked cross-entropy loss.
 *
 * Only computes loss for positions where mask[t] === 1.
 * Optionally ignores padding tokens.
 *
 * @param logits - 2D array of logits [T][vocab_size]
 * @param targets - Array of target token IDs [T]
 * @param lossMask - Array of 0s and 1s indicating which positions to include
 * @param padId - Optional padding token ID to ignore
 * @returns Average cross-entropy loss over masked positions
 *
 * @example
 * const loss = crossEntropyMasked(logits, targets, mask);
 */
export function crossEntropyMasked(
  logits: number[][],
  targets: number[],
  lossMask: number[],
  padId?: number
): number {
  let loss = 0;
  let count = 0;

  for (let t = 0; t < targets.length; t++) {
    const y = targets[t];

    // Skip padding tokens
    if (padId !== undefined && y === padId) continue;

    // Skip masked positions
    if (!lossMask[t]) continue;

    // Compute cross-entropy: -log(softmax(logits)[y])
    const row = logits[t];
    const max = Math.max(...row);

    // Numerically stable softmax
    let sumExp = 0;
    for (let i = 0; i < row.length; i++) {
      sumExp += Math.exp(row[i] - max);
    }
    const logProb = row[y] - max - Math.log(sumExp);

    loss += -logProb;
    count++;
  }

  // Return average loss (avoid division by zero)
  return count > 0 ? loss / count : 0;
}

/**
 * Compute single-position cross-entropy with mask check.
 *
 * Utility for models that compute loss token-by-token.
 *
 * @param probs - Probability distribution over vocabulary
 * @param target - Target token ID
 * @param maskValue - 0 or 1 indicating whether to include this position
 * @returns Cross-entropy loss or 0 if masked out
 */
export function crossEntropySingleMasked(
  probs: number[],
  target: number,
  maskValue: number
): number {
  if (maskValue === 0) {
    return 0;
  }
  return -Math.log(probs[target] + 1e-10);
}

/**
 * Apply loss mask to gradients (zero out masked positions).
 *
 * Used in backward pass to prevent gradient flow for masked tokens.
 *
 * @param dLogits - Gradient of loss w.r.t. logits
 * @param maskValue - 0 or 1 indicating whether to include this position
 * @returns Masked gradients (zeros if maskValue is 0)
 */
export function applyGradientMask(dLogits: number[], maskValue: number): number[] {
  if (maskValue === 0) {
    return dLogits.map(() => 0);
  }
  return dLogits;
}

/**
 * Build loss masks for a batch of sequences.
 *
 * @param batchInputIds - Array of token ID sequences
 * @param tokenizer - Tokenizer with encode method
 * @param mode - Loss mask mode
 * @param answerTag - Custom answer tag
 * @param regExpOptions - Options for customRegExp mode
 * @returns Array of mask arrays, one per sequence
 */
export function buildBatchLossMasks(
  batchInputIds: number[][],
  tokenizer: TokenizerLike,
  mode: LossMaskMode,
  answerTag = 'A:',
  regExpOptions?: { pattern?: string; position?: RegExpMaskPosition }
): number[][] {
  return batchInputIds.map((inputIds) =>
    buildAnswerLossMask(inputIds, tokenizer, mode, answerTag, regExpOptions)
  );
}

/**
 * Compute the effective number of loss-contributing tokens.
 *
 * Useful for logging and debugging.
 *
 * @param lossMask - Loss mask array
 * @returns Number of positions with mask value 1
 */
export function countActiveMaskPositions(lossMask: number[]): number {
  return lossMask.reduce((sum, v) => sum + v, 0);
}

/**
 * Simple character-level tokenizer for testing.
 *
 * Maps each character to its character code.
 * For production, use the actual tokenizer from the model.
 */
export function createSimpleCharTokenizer(): TokenizerLike {
  return {
    encode: (text: string) => Array.from(text).map((c) => c.charCodeAt(0))
  };
}
