/**
 * Triadic Operator System Types
 *
 * Defines types for the triadic operator 𝕋 which computes a 4-bit vector
 * ⟨W,S,T,N⟩ using only NAND gates.
 */

/**
 * The result of a triadic operator computation
 * Represents the state vector ⟨W,S,T,N⟩
 */
export interface TriadicVector {
  /** W (Weak): At least one weak link through B */
  weak: boolean;

  /** S (Strong): Both links are strong (complete triad) */
  strong: boolean;

  /** T (Tension): Exactly one strong link (XOR) */
  tension: boolean;

  /** N (Null): No weak link at all */
  null: boolean;
}

/**
 * A cell in the triadic operator table
 * Contains three concepts (A, B, C) and their emoji representations
 */
export interface TriadicCell {
  /** First concept (Noise/Fluctuation/etc.) */
  a: string;

  /** Second concept (Regulation/Order/etc.) */
  b: string;

  /** Third concept (Control/Slaving/etc.) */
  c: string;

  /** The emoji representation of concept A */
  emojiA: string;

  /** The emoji representation of concept B */
  emojiB: string;

  /** The emoji representation of concept C */
  emojiC: string;
}

/**
 * A row in the triadic operator table representing a domain
 */
export interface TriadicDomain {
  /** Domain identifier (e.g., "digital", "logic", "questions") */
  id: string;

  /** Domain name in English */
  nameEn: string;

  /** Domain name in Hebrew */
  nameHe: string;

  /** Domain emoji representation */
  emoji: string;

  /** Cells for this domain across all columns */
  cells: TriadicCell[];
}

/**
 * A column group in the triadic operator table
 */
export interface TriadicColumnGroup {
  /** Column group identifier */
  id: string;

  /** Column group name in English */
  nameEn: string;

  /** Column group name in Hebrew */
  nameHe: string;

  /** Index of this column in the full table */
  index: number;
}

/**
 * The complete triadic operator table
 */
export interface TriadicTable {
  /** All column groups */
  columns: TriadicColumnGroup[];

  /** All domain rows */
  domains: TriadicDomain[];

  /** Metadata about the table */
  metadata: {
    version: string;
    description: string;
    author?: string;
    created?: string;
  };
}
