/**
 * Triadic Operator Implementation
 *
 * Implements the triadic operator 𝕋(A,B,C) which computes a 4-bit vector
 * ⟨W,S,T,N⟩ using only NAND gates.
 *
 * Mathematical Definition:
 * Every operator 𝕋(A,B,C) returns a vector of 4 bits ⟨W,S,T,N⟩ computed
 * exclusively using NAND gates:
 *
 * 1. W (Weak): At least one weak link through B
 *    W = NAND(NAND(A,B), NAND(B,C))
 *
 * 2. S (Strong): Both links are strong (complete triad)
 *    L = NAND(NAND(A,B), NAND(A,B))  # A AND B
 *    R = NAND(NAND(B,C), NAND(B,C))  # B AND C
 *    S = NAND(NAND(L,R), NAND(L,R))  # L AND R
 *
 * 3. T (Tension): Exactly one strong link (XOR)
 *    t0 = NAND(L,R)
 *    T = NAND(NAND(L, t0), NAND(R, t0))
 *
 * 4. N (Null): No weak link at all
 *    N = NAND(W, W)
 */

import type { TriadicVector } from '../types/triadic';

/**
 * NAND gate - the fundamental building block
 * NAND(a, b) = NOT(a AND b)
 *
 * Truth table:
 * a | b | NAND
 * 0 | 0 | 1
 * 0 | 1 | 1
 * 1 | 0 | 1
 * 1 | 1 | 0
 */
function nand(a: boolean, b: boolean): boolean {
  return !(a && b);
}

/**
 * Compute W (Weak): At least one element is present (A OR B OR C)
 * W = (A OR B) OR C
 * Using NAND: A OR B = NAND(NAND(A,A), NAND(B,B))
 *             (A OR B) OR C = NAND(NAND(temp, temp), NAND(C,C))
 */
function computeWeak(a: boolean, b: boolean, c: boolean): boolean {
  // A OR B
  const notA = nand(a, a);
  const notB = nand(b, b);
  const temp = nand(notA, notB); // A OR B

  // (A OR B) OR C
  const notTemp = nand(temp, temp);
  const notC = nand(c, c);
  return nand(notTemp, notC); // (A OR B) OR C
}

/**
 * Compute S (Strong): Both links are strong (complete triad)
 * L = NAND(NAND(A,B), NAND(A,B))  # A AND B
 * R = NAND(NAND(B,C), NAND(B,C))  # B AND C
 * S = NAND(NAND(L,R), NAND(L,R))  # L AND R
 */
function computeStrong(a: boolean, b: boolean, c: boolean): boolean {
  // L = A AND B (using NAND)
  const ab = nand(a, b);
  const L = nand(ab, ab); // NOT(NAND(A,B)) = A AND B

  // R = B AND C (using NAND)
  const bc = nand(b, c);
  const R = nand(bc, bc); // NOT(NAND(B,C)) = B AND C

  // S = L AND R (using NAND)
  const lr = nand(L, R);
  return nand(lr, lr); // NOT(NAND(L,R)) = L AND R
}

/**
 * Compute T (Tension): Exactly one strong link (XOR)
 * L = A AND B
 * R = B AND C
 * t0 = NAND(L,R)
 * T = NAND(NAND(L, t0), NAND(R, t0))
 */
function computeTension(a: boolean, b: boolean, c: boolean): boolean {
  // L = A AND B (using NAND)
  const ab = nand(a, b);
  const L = nand(ab, ab);

  // R = B AND C (using NAND)
  const bc = nand(b, c);
  const R = nand(bc, bc);

  // XOR(L, R) using NAND
  const t0 = nand(L, R);
  const lt0 = nand(L, t0);
  const rt0 = nand(R, t0);
  return nand(lt0, rt0);
}

/**
 * Compute N (Null): No weak link at all
 * N = NAND(W, W) = NOT(W)
 */
function computeNull(w: boolean): boolean {
  return nand(w, w); // NOT(W)
}

/**
 * The triadic operator 𝕋(A,B,C)
 *
 * Computes the complete state vector ⟨W,S,T,N⟩ for three boolean inputs
 * using only NAND gates.
 *
 * @param a - First input (presence/activation of concept A)
 * @param b - Second input (presence/activation of concept B)
 * @param c - Third input (presence/activation of concept C)
 * @returns The complete triadic vector ⟨W,S,T,N⟩
 *
 * @example
 * ```typescript
 * // Complete triad: all concepts present
 * const result = triadicOperator(true, true, true);
 * // result: { weak: true, strong: true, tension: false, null: false }
 *
 * // Partial connection through B
 * const partial = triadicOperator(true, true, false);
 * // partial: { weak: true, strong: false, tension: true, null: false }
 *
 * // No connections
 * const none = triadicOperator(false, false, false);
 * // none: { weak: false, strong: false, tension: false, null: true }
 * ```
 */
export function triadicOperator(a: boolean, b: boolean, c: boolean): TriadicVector {
  const weak = computeWeak(a, b, c);
  const strong = computeStrong(a, b, c);
  const tension = computeTension(a, b, c);
  const nullVal = computeNull(weak);

  return {
    weak,
    strong,
    tension,
    null: nullVal
  };
}

/**
 * Verify the internal consistency of a triadic vector
 *
 * The following invariants should hold:
 * - If strong is true, weak must be true
 * - If null is true, weak must be false
 * - If strong is true, null must be false
 * - strong and tension cannot both be true
 *
 * @param vector - The triadic vector to verify
 * @returns true if the vector is consistent, false otherwise
 */
export function verifyTriadicVector(vector: TriadicVector): boolean {
  // Strong implies weak
  if (vector.strong && !vector.weak) {
    return false;
  }

  // Null implies not weak
  if (vector.null && vector.weak) {
    return false;
  }

  // Strong implies not null
  if (vector.strong && vector.null) {
    return false;
  }

  // Strong and tension are mutually exclusive (they can both be false)
  if (vector.strong && vector.tension) {
    return false;
  }

  return true;
}

/**
 * Convert a triadic vector to a human-readable string
 */
export function triadicVectorToString(vector: TriadicVector): string {
  const parts: string[] = [];
  if (vector.weak) parts.push('W');
  if (vector.strong) parts.push('S');
  if (vector.tension) parts.push('T');
  if (vector.null) parts.push('N');

  return `⟨${parts.join(',')}⟩`;
}

/**
 * Generate a truth table for the triadic operator
 * Useful for debugging and verification
 */
export function generateTriadicTruthTable(): Array<{
  a: boolean;
  b: boolean;
  c: boolean;
  result: TriadicVector;
  valid: boolean;
}> {
  const table: Array<{
    a: boolean;
    b: boolean;
    c: boolean;
    result: TriadicVector;
    valid: boolean;
  }> = [];

  for (let a = 0; a <= 1; a++) {
    for (let b = 0; b <= 1; b++) {
      for (let c = 0; c <= 1; c++) {
        const aVal = Boolean(a);
        const bVal = Boolean(b);
        const cVal = Boolean(c);
        const result = triadicOperator(aVal, bVal, cVal);
        const valid = verifyTriadicVector(result);
        table.push({ a: aVal, b: bVal, c: cVal, result, valid });
      }
    }
  }

  return table;
}
