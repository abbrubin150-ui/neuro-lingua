/**
 * Loss Mask Utilities for Answer-Only Training
 *
 * Implements loss masking to train models only on answer tokens,
 * preventing memorization of prompts/questions.
 *
 * @module losses/lossMask
 * @version 4.5.0
 */

import type { LossMaskMode } from '../types/project';

/**
 * Simple tokenizer interface for building loss masks
 */
export interface TokenizerLike {
  encode: (text: string) => number[];
}

/**
 * Build a loss mask array based on delimiter position in input tokens.
 *
 * The mask is 0 for tokens before (and including) the delimiter,
 * and 1 for tokens after the delimiter (the answer portion).
 *
 * @param inputIds - Array of token IDs for the full sequence
 * @param tokenizer - Tokenizer with encode method
 * @param mode - Loss mask mode: 'none' | 'afterEquals' | 'afterAnswerTag'
 * @param answerTag - Custom answer tag (default: 'A:')
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
  answerTag = 'A:'
): number[] {
  const T = inputIds.length;
  const mask = new Array(T).fill(0);

  // If no masking, compute loss on all tokens
  if (mode === 'none') {
    return mask.map(() => 1);
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
 * @returns Array of mask arrays, one per sequence
 */
export function buildBatchLossMasks(
  batchInputIds: number[][],
  tokenizer: TokenizerLike,
  mode: LossMaskMode,
  answerTag = 'A:'
): number[][] {
  return batchInputIds.map((inputIds) => buildAnswerLossMask(inputIds, tokenizer, mode, answerTag));
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
