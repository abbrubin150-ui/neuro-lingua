/**
 * 2D Convolution Matrix Type Definitions
 *
 * This module defines types for the 2D convolution between 14 triads (T1-T14)
 * and 14 digital layers (N1-N14), producing a 196-cell transformation matrix.
 *
 * Formula: Conv(T_i, N_j) = T_i ⊗ S_j → C_ij
 *
 * Based on the Neuro-Lingua Framework Specification
 */

// ============================================================================
// Triad Definitions (Y-axis: T1-T14)
// ============================================================================

export type TriadId =
  | 'T1'
  | 'T2'
  | 'T3'
  | 'T4'
  | 'T5'
  | 'T6'
  | 'T7'
  | 'T8'
  | 'T9'
  | 'T10'
  | 'T11'
  | 'T12'
  | 'T13'
  | 'T14';

/**
 * Triad definition with three components in transformation order
 */
export interface Triad {
  readonly id: TriadId;
  /** Hebrew name */
  readonly nameHe: string;
  /** English name */
  readonly nameEn: string;
  /** Three components: [A → B → C] */
  readonly components: [string, string, string];
  /** Row index (0-13) */
  readonly rowIndex: number;
}

// ============================================================================
// Digital Layer Definitions (X-axis: N1-N14)
// ============================================================================

export type LayerId =
  | 'N1'
  | 'N2'
  | 'N3'
  | 'N4'
  | 'N5'
  | 'N6'
  | 'N7'
  | 'N8'
  | 'N9'
  | 'N10'
  | 'N11'
  | 'N12'
  | 'N13'
  | 'N14';

/**
 * Digital layer definition with keywords
 */
export interface DigitalLayer {
  readonly id: LayerId;
  /** Hebrew name */
  readonly nameHe: string;
  /** English name */
  readonly nameEn: string;
  /** Keywords associated with this layer */
  readonly keywords: string[];
  /** Column index (0-13) */
  readonly colIndex: number;
}

// ============================================================================
// Convolution Cell (196 total cells)
// ============================================================================

/**
 * Convolution depth score K ∈ {0, 1, 2}
 * - 0: Empty/no connection
 * - 1: Relevant/partial connection
 * - 2: Core/full connection
 */
export type ConvDepth = 0 | 1 | 2;

/**
 * Single cell in the convolution matrix
 */
export interface Conv2DCell {
  readonly type: 'CONV_2D_CELL';
  /** Cell ID (e.g., "C0101" for T1×N1) */
  readonly cellId: string;
  /** Triad ID (row) */
  readonly triadId: TriadId;
  /** Layer ID (column) */
  readonly layerId: LayerId;
  /** Row index (0-13) */
  readonly rowIndex: number;
  /** Column index (0-13) */
  readonly colIndex: number;
  /** Transformation content: the convolution result */
  readonly content: string;
  /** Hebrew content (if available) */
  readonly contentHe?: string;
  /** Convolution depth K ∈ {0, 1, 2} */
  readonly depth: ConvDepth;
}

// ============================================================================
// 2D Convolution Matrix
// ============================================================================

/**
 * Full 14×14 convolution matrix
 */
export interface Conv2DMatrix {
  readonly type: 'CONV_2D_MATRIX';
  /** All triads (rows) */
  readonly triads: Record<TriadId, Triad>;
  /** All layers (columns) */
  readonly layers: Record<LayerId, DigitalLayer>;
  /** All 196 cells */
  readonly cells: Conv2DCell[];
  /** Depth matrix K(T,N) for quick lookup */
  readonly depthMatrix: ConvDepth[][];
  /** Matrix statistics */
  readonly statistics: Conv2DStatistics;
  /** Version */
  readonly version: string;
}

/**
 * Matrix statistics
 */
export interface Conv2DStatistics {
  /** Total cells */
  readonly totalCells: 196;
  /** Core cells (K=2) */
  readonly coreCells: number;
  /** Relevant cells (K=1) */
  readonly relevantCells: number;
  /** Empty cells (K=0) */
  readonly emptyCells: number;
  /** Core percentage */
  readonly corePercentage: number;
  /** Relevant percentage */
  readonly relevantPercentage: number;
  /** Empty percentage */
  readonly emptyPercentage: number;
}

// ============================================================================
// Diagonal Analysis
// ============================================================================

/**
 * Main diagonal entry (T=N, reflexivity)
 */
export interface MainDiagonalEntry {
  readonly index: number;
  readonly triadId: TriadId;
  readonly layerId: LayerId;
  readonly cell: Conv2DCell;
  /** Hebrew interpretation */
  readonly meaningHe: string;
  /** English interpretation */
  readonly meaningEn: string;
}

/**
 * Anti-diagonal entry (T+N=15, critical transitions)
 */
export interface AntiDiagonalEntry {
  readonly index: number;
  readonly triadId: TriadId;
  readonly layerId: LayerId;
  readonly cell: Conv2DCell;
  /** Hebrew interpretation */
  readonly meaningHe: string;
  /** English interpretation */
  readonly meaningEn: string;
}

/**
 * Diagonal analysis results
 */
export interface DiagonalAnalysis {
  readonly type: 'DIAGONAL_ANALYSIS';
  /** Main diagonal (T=N): self-reflexivity */
  readonly mainDiagonal: MainDiagonalEntry[];
  /** Anti-diagonal (T+N=15): critical transitions */
  readonly antiDiagonal: AntiDiagonalEntry[];
}

// ============================================================================
// Convolution Formula
// ============================================================================

/**
 * The general convolution formula:
 * Conv_2D(T, N) = Σᵢ₌₁¹⁴ Σⱼ₌₁¹⁴ K(Tᵢ, Nⱼ) · (Tᵢ ⊗ Sⱼ)
 */
export interface ConvolutionFormula {
  readonly type: 'CONVOLUTION_FORMULA';
  /** Formula in LaTeX */
  readonly latex: string;
  /** Properties */
  readonly properties: ConvolutionProperties;
}

/**
 * Matrix properties
 */
export interface ConvolutionProperties {
  /** Partial symmetry: T7 is universal */
  readonly partialSymmetry: boolean;
  /** Universal triad (works with all N) */
  readonly universalTriad: TriadId;
  /** Density: percentage of core cells */
  readonly density: number;
  /** Main diagonal: self-reflexivity in each layer */
  readonly mainDiagonalReflexivity: boolean;
  /** Anti-diagonal: critical transitions between extremes */
  readonly antiDiagonalTransitions: boolean;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Cell lookup by coordinates
 */
export interface CellCoordinates {
  readonly row: number;
  readonly col: number;
}

/**
 * Generate cell ID from coordinates
 */
export function generateConv2DCellId(row: number, col: number): string {
  const rowStr = String(row + 1).padStart(2, '0');
  const colStr = String(col + 1).padStart(2, '0');
  return `C${rowStr}${colStr}`;
}

/**
 * Parse cell ID to coordinates
 */
export function parseConv2DCellId(
  cellId: string
): CellCoordinates | null {
  const match = cellId.match(/^C(\d{2})(\d{2})$/);
  if (!match) return null;
  return {
    row: parseInt(match[1], 10) - 1,
    col: parseInt(match[2], 10) - 1,
  };
}

/**
 * Get triad ID from row index
 */
export function getTriadIdFromRow(row: number): TriadId {
  return `T${row + 1}` as TriadId;
}

/**
 * Get layer ID from column index
 */
export function getLayerIdFromCol(col: number): LayerId {
  return `N${col + 1}` as LayerId;
}

/**
 * Atomic paragraph summary
 */
export interface AtomicParagraph {
  readonly type: 'ATOMIC_PARAGRAPH';
  /** Hebrew content */
  readonly contentHe: string;
  /** English content */
  readonly contentEn: string;
}

/**
 * WHY/HOW assessment
 */
export interface WhyHowAssessment {
  readonly type: 'WHY_HOW_ASSESSMENT';
  /** WHY_RHYTHM score (0-3) */
  readonly whyRhythm: number;
  /** WHY_RHYTHM justification */
  readonly whyJustification: string;
  /** HOW score (0-3) */
  readonly how: number;
  /** HOW steps */
  readonly howSteps: string[];
}

/**
 * Auditor summary
 */
export interface AuditorSummary {
  readonly type: 'AUDITOR_SUMMARY';
  /** Intent */
  readonly intent: string;
  /** Execution */
  readonly execution: string;
  /** Outcome */
  readonly outcome: string;
  /** Delta (potential improvements) */
  readonly delta: string;
}
