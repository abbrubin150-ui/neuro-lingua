/**
 * 2D Convolution Matrix Implementation
 *
 * Implements the full 14×14 convolution between triads (T1-T14) and
 * digital layers (N1-N14), producing 196 transformation cells.
 *
 * Formula: Conv_2D(T, N) = Σᵢ₌₁¹⁴ Σⱼ₌₁¹⁴ K(Tᵢ, Nⱼ) · (Tᵢ ⊗ Sⱼ)
 *
 * Properties:
 * - Partial symmetry: T7 (Drive→Constraint→Mediation) is universal
 * - Density: 90.8% core cells, no empty cells
 * - Main diagonal: self-reflexivity in each layer
 * - Anti-diagonal: critical transitions between extremes
 */

import type {
  TriadId,
  LayerId,
  Triad,
  DigitalLayer,
  Conv2DCell,
  Conv2DMatrix,
  Conv2DStatistics,
  ConvDepth,
  MainDiagonalEntry,
  AntiDiagonalEntry,
  DiagonalAnalysis,
  ConvolutionFormula,
  ConvolutionProperties,
  AtomicParagraph,
  WhyHowAssessment,
  AuditorSummary,
  generateConv2DCellId,
  getTriadIdFromRow,
  getLayerIdFromCol,
} from '../types/conv2d';

import {
  TRIADS,
  LAYERS,
  CELL_CONTENTS,
  DEPTH_MATRIX,
  MAIN_DIAGONAL_MEANINGS,
  ANTI_DIAGONAL_MEANINGS,
  buildAllCells,
} from '../data/conv2dData';

// ============================================================================
// Matrix Creation
// ============================================================================

/**
 * Create the full 2D convolution matrix
 */
export function createConv2DMatrix(): Conv2DMatrix {
  const cells = buildAllCells();
  const statistics = computeStatistics(cells);

  return {
    type: 'CONV_2D_MATRIX',
    triads: TRIADS,
    layers: LAYERS,
    cells,
    depthMatrix: DEPTH_MATRIX,
    statistics,
    version: '1.0.0',
  };
}

/**
 * Compute matrix statistics
 */
export function computeStatistics(cells: Conv2DCell[]): Conv2DStatistics {
  let coreCells = 0;
  let relevantCells = 0;
  let emptyCells = 0;

  for (const cell of cells) {
    switch (cell.depth) {
      case 2:
        coreCells++;
        break;
      case 1:
        relevantCells++;
        break;
      case 0:
        emptyCells++;
        break;
    }
  }

  const totalCells = 196;

  return {
    totalCells,
    coreCells,
    relevantCells,
    emptyCells,
    corePercentage: (coreCells / totalCells) * 100,
    relevantPercentage: (relevantCells / totalCells) * 100,
    emptyPercentage: (emptyCells / totalCells) * 100,
  };
}

// ============================================================================
// Cell Lookup
// ============================================================================

/**
 * Get a cell by triad and layer IDs
 */
export function getCell(
  matrix: Conv2DMatrix,
  triadId: TriadId,
  layerId: LayerId
): Conv2DCell | null {
  const triad = matrix.triads[triadId];
  const layer = matrix.layers[layerId];
  if (!triad || !layer) return null;

  return (
    matrix.cells.find(
      (c) => c.rowIndex === triad.rowIndex && c.colIndex === layer.colIndex
    ) || null
  );
}

/**
 * Get a cell by coordinates
 */
export function getCellByCoordinates(
  matrix: Conv2DMatrix,
  row: number,
  col: number
): Conv2DCell | null {
  if (row < 0 || row >= 14 || col < 0 || col >= 14) return null;
  return matrix.cells.find((c) => c.rowIndex === row && c.colIndex === col) || null;
}

/**
 * Get a cell by cell ID
 */
export function getCellById(matrix: Conv2DMatrix, cellId: string): Conv2DCell | null {
  return matrix.cells.find((c) => c.cellId === cellId) || null;
}

/**
 * Get all cells for a specific triad (row)
 */
export function getCellsByTriad(matrix: Conv2DMatrix, triadId: TriadId): Conv2DCell[] {
  const triad = matrix.triads[triadId];
  if (!triad) return [];
  return matrix.cells.filter((c) => c.rowIndex === triad.rowIndex);
}

/**
 * Get all cells for a specific layer (column)
 */
export function getCellsByLayer(matrix: Conv2DMatrix, layerId: LayerId): Conv2DCell[] {
  const layer = matrix.layers[layerId];
  if (!layer) return [];
  return matrix.cells.filter((c) => c.colIndex === layer.colIndex);
}

/**
 * Get depth value at coordinates
 */
export function getDepth(matrix: Conv2DMatrix, row: number, col: number): ConvDepth {
  if (row < 0 || row >= 14 || col < 0 || col >= 14) return 0;
  return matrix.depthMatrix[row][col];
}

// ============================================================================
// Diagonal Analysis
// ============================================================================

/**
 * Get the main diagonal (T=N, self-reflexivity)
 */
export function getMainDiagonal(matrix: Conv2DMatrix): MainDiagonalEntry[] {
  const entries: MainDiagonalEntry[] = [];

  for (let i = 0; i < 14; i++) {
    const cell = getCellByCoordinates(matrix, i, i);
    if (!cell) continue;

    const meaning = MAIN_DIAGONAL_MEANINGS[i];

    entries.push({
      index: i,
      triadId: cell.triadId,
      layerId: cell.layerId,
      cell,
      meaningHe: meaning.meaningHe,
      meaningEn: meaning.meaningEn,
    });
  }

  return entries;
}

/**
 * Get the anti-diagonal (T+N=15, critical transitions)
 * Note: For 0-indexed, this is row + col = 13
 */
export function getAntiDiagonal(matrix: Conv2DMatrix): AntiDiagonalEntry[] {
  const entries: AntiDiagonalEntry[] = [];

  for (const meaning of ANTI_DIAGONAL_MEANINGS) {
    const cell = getCellByCoordinates(matrix, meaning.triadIndex, meaning.layerIndex);
    if (!cell) continue;

    entries.push({
      index: meaning.triadIndex,
      triadId: cell.triadId,
      layerId: cell.layerId,
      cell,
      meaningHe: meaning.meaningHe,
      meaningEn: meaning.meaningEn,
    });
  }

  return entries;
}

/**
 * Get full diagonal analysis
 */
export function analyzeDiagonals(matrix: Conv2DMatrix): DiagonalAnalysis {
  return {
    type: 'DIAGONAL_ANALYSIS',
    mainDiagonal: getMainDiagonal(matrix),
    antiDiagonal: getAntiDiagonal(matrix),
  };
}

// ============================================================================
// Convolution Properties
// ============================================================================

/**
 * Get convolution formula definition
 */
export function getConvolutionFormula(): ConvolutionFormula {
  return {
    type: 'CONVOLUTION_FORMULA',
    latex: '\\text{Conv}_{2D}(T, N) = \\sum_{i=1}^{14} \\sum_{j=1}^{14} K(T_i, N_j) \\cdot (T_i \\otimes S_j)',
    properties: getConvolutionProperties(),
  };
}

/**
 * Get matrix properties
 */
export function getConvolutionProperties(): ConvolutionProperties {
  return {
    partialSymmetry: true,
    universalTriad: 'T7', // Drive → Constraint → Mediation
    density: 90.8,
    mainDiagonalReflexivity: true,
    antiDiagonalTransitions: true,
  };
}

/**
 * Check if a triad is universal (all depth=2)
 */
export function isUniversalTriad(matrix: Conv2DMatrix, triadId: TriadId): boolean {
  const cells = getCellsByTriad(matrix, triadId);
  return cells.every((c) => c.depth === 2);
}

/**
 * Get all universal triads
 */
export function getUniversalTriads(matrix: Conv2DMatrix): TriadId[] {
  const triadIds = Object.keys(matrix.triads) as TriadId[];
  return triadIds.filter((id) => isUniversalTriad(matrix, id));
}

// ============================================================================
// Filtering and Querying
// ============================================================================

/**
 * Get cells by depth
 */
export function getCellsByDepth(matrix: Conv2DMatrix, depth: ConvDepth): Conv2DCell[] {
  return matrix.cells.filter((c) => c.depth === depth);
}

/**
 * Get core cells (depth = 2)
 */
export function getCoreCells(matrix: Conv2DMatrix): Conv2DCell[] {
  return getCellsByDepth(matrix, 2);
}

/**
 * Get relevant cells (depth = 1)
 */
export function getRelevantCells(matrix: Conv2DMatrix): Conv2DCell[] {
  return getCellsByDepth(matrix, 1);
}

/**
 * Get empty cells (depth = 0)
 */
export function getEmptyCells(matrix: Conv2DMatrix): Conv2DCell[] {
  return getCellsByDepth(matrix, 0);
}

/**
 * Search cells by content
 */
export function searchCells(matrix: Conv2DMatrix, query: string): Conv2DCell[] {
  const lowerQuery = query.toLowerCase();
  return matrix.cells.filter((c) => c.content.toLowerCase().includes(lowerQuery));
}

/**
 * Get cells matching layer keywords
 */
export function getCellsByKeyword(matrix: Conv2DMatrix, keyword: string): Conv2DCell[] {
  const lowerKeyword = keyword.toLowerCase();
  const matchingLayers = Object.values(matrix.layers).filter((layer) =>
    layer.keywords.some((k) => k.toLowerCase().includes(lowerKeyword))
  );

  return matrix.cells.filter((c) =>
    matchingLayers.some((layer) => layer.colIndex === c.colIndex)
  );
}

// ============================================================================
// Matrix Slicing
// ============================================================================

/**
 * Get a submatrix by row and column ranges
 */
export function getSubmatrix(
  matrix: Conv2DMatrix,
  rowStart: number,
  rowEnd: number,
  colStart: number,
  colEnd: number
): Conv2DCell[] {
  return matrix.cells.filter(
    (c) =>
      c.rowIndex >= rowStart &&
      c.rowIndex < rowEnd &&
      c.colIndex >= colStart &&
      c.colIndex < colEnd
  );
}

/**
 * Get quadrant (2×2 split)
 */
export type Quadrant = 'TL' | 'TR' | 'BL' | 'BR';

export function getQuadrant(matrix: Conv2DMatrix, quadrant: Quadrant): Conv2DCell[] {
  const mid = 7;
  switch (quadrant) {
    case 'TL':
      return getSubmatrix(matrix, 0, mid, 0, mid);
    case 'TR':
      return getSubmatrix(matrix, 0, mid, mid, 14);
    case 'BL':
      return getSubmatrix(matrix, mid, 14, 0, mid);
    case 'BR':
      return getSubmatrix(matrix, mid, 14, mid, 14);
  }
}

// ============================================================================
// Convolution Operations
// ============================================================================

/**
 * Apply convolution with depth weighting
 */
export function applyConvolution(matrix: Conv2DMatrix): number {
  let total = 0;
  for (const cell of matrix.cells) {
    total += cell.depth;
  }
  return total;
}

/**
 * Compute row sums (per triad)
 */
export function computeRowSums(matrix: Conv2DMatrix): Record<TriadId, number> {
  const sums: Record<string, number> = {};
  for (const triadId of Object.keys(matrix.triads) as TriadId[]) {
    const cells = getCellsByTriad(matrix, triadId);
    sums[triadId] = cells.reduce((sum, c) => sum + c.depth, 0);
  }
  return sums as Record<TriadId, number>;
}

/**
 * Compute column sums (per layer)
 */
export function computeColSums(matrix: Conv2DMatrix): Record<LayerId, number> {
  const sums: Record<string, number> = {};
  for (const layerId of Object.keys(matrix.layers) as LayerId[]) {
    const cells = getCellsByLayer(matrix, layerId);
    sums[layerId] = cells.reduce((sum, c) => sum + c.depth, 0);
  }
  return sums as Record<LayerId, number>;
}

/**
 * Find the triad with highest connectivity
 */
export function findMostConnectedTriad(matrix: Conv2DMatrix): TriadId {
  const sums = computeRowSums(matrix);
  let maxTriad: TriadId = 'T1';
  let maxSum = 0;
  for (const [triadId, sum] of Object.entries(sums)) {
    if (sum > maxSum) {
      maxSum = sum;
      maxTriad = triadId as TriadId;
    }
  }
  return maxTriad;
}

/**
 * Find the layer with highest connectivity
 */
export function findMostConnectedLayer(matrix: Conv2DMatrix): LayerId {
  const sums = computeColSums(matrix);
  let maxLayer: LayerId = 'N1';
  let maxSum = 0;
  for (const [layerId, sum] of Object.entries(sums)) {
    if (sum > maxSum) {
      maxSum = sum;
      maxLayer = layerId as LayerId;
    }
  }
  return maxLayer;
}

// ============================================================================
// Summary Generation
// ============================================================================

/**
 * Generate atomic paragraph summary
 */
export function generateAtomicParagraph(): AtomicParagraph {
  return {
    type: 'ATOMIC_PARAGRAPH',
    contentHe:
      'הקונבולוציה הדו-מימדית מייצרת מטריצה 14×14 = 196 תאים, כאשר כל תא מכיל טרנספורמציה ייחודית של טריאדה (Noise→Regulation→Control וכו\') על שכבה דיגיטלית (Foundation עד Evolution). האלכסון הראשי מייצג רפלקסיביות עצמית (מערכת שולטת בעצמה), האלכסון המשני מייצג מעברים קריטיים בין קצוות. 90.8% מהתאים הם ליבה (K=2), ללא תאים ריקים. הטריאדה ה-14 (Genesis→Adaptation→Transcendence) נוספה לסגירת המטריצה הריבועית ומייצגת את הציר האבולוציוני של כל מערכת.',
    contentEn:
      'The 2D convolution generates a 14×14 = 196 cell matrix, where each cell contains a unique transformation of a triad (Noise→Regulation→Control etc.) on a digital layer (Foundation to Evolution). The main diagonal represents self-reflexivity (system controls itself), the anti-diagonal represents critical transitions between extremes. 90.8% of cells are core (K=2), with no empty cells. Triad 14 (Genesis→Adaptation→Transcendence) was added to close the square matrix and represents the evolutionary axis of every system.',
  };
}

/**
 * Generate WHY/HOW assessment
 */
export function generateWhyHowAssessment(): WhyHowAssessment {
  return {
    type: 'WHY_HOW_ASSESSMENT',
    whyRhythm: 3,
    whyJustification:
      '2D convolution unifies two orthogonal systems into a shared space',
    how: 3,
    howSteps: [
      'Define 14 triads × 14 layers',
      'Fill 196 cells with unique transformations',
      'Analyze diagonals and patterns',
    ],
  };
}

/**
 * Generate auditor summary
 */
export function generateAuditorSummary(): AuditorSummary {
  return {
    type: 'AUDITOR_SUMMARY',
    intent: '2D convolution between triads and digital layers',
    execution: 'Full 14×14 matrix with depth scores',
    outcome: '196 unique transformations + diagonal analysis',
    delta: 'Could generate network graph or heatmap',
  };
}

// ============================================================================
// Export Utilities
// ============================================================================

/**
 * Export matrix to JSON
 */
export function exportMatrixToJSON(matrix: Conv2DMatrix): string {
  return JSON.stringify(matrix, null, 2);
}

/**
 * Export depth matrix as CSV
 */
export function exportDepthMatrixAsCSV(matrix: Conv2DMatrix): string {
  const header = ['T\\N', ...Object.keys(matrix.layers)].join(',');
  const rows: string[] = [header];

  for (let row = 0; row < 14; row++) {
    const triadId = `T${row + 1}`;
    const values = matrix.depthMatrix[row].join(',');
    rows.push(`${triadId},${values}`);
  }

  return rows.join('\n');
}

/**
 * Export content matrix as Markdown table
 */
export function exportContentAsMarkdown(matrix: Conv2DMatrix, triadId: TriadId): string {
  const cells = getCellsByTriad(matrix, triadId);
  if (cells.length === 0) return '';

  const triad = matrix.triads[triadId];
  const lines: string[] = [];

  lines.push(`### ${triadId}: ${triad.nameEn}`);
  lines.push('');
  lines.push('| N1 | N2 | N3 | N4 | N5 | N6 | N7 |');
  lines.push('|----|----|----|----|----|----|----| ');
  lines.push(`| ${cells.slice(0, 7).map((c) => c.content).join(' | ')} |`);
  lines.push('');
  lines.push('| N8 | N9 | N10 | N11 | N12 | N13 | N14 |');
  lines.push('|----|----|----|-----|-----|-----|-----|');
  lines.push(`| ${cells.slice(7).map((c) => c.content).join(' | ')} |`);

  return lines.join('\n');
}

/**
 * Generate full matrix report
 */
export function generateMatrixReport(matrix: Conv2DMatrix): string {
  const lines: string[] = [];
  const stats = matrix.statistics;

  lines.push('# 2D Convolution Matrix Report');
  lines.push('');
  lines.push('## Statistics');
  lines.push(`- Total cells: ${stats.totalCells}`);
  lines.push(`- Core cells (K=2): ${stats.coreCells} (${stats.corePercentage.toFixed(1)}%)`);
  lines.push(
    `- Relevant cells (K=1): ${stats.relevantCells} (${stats.relevantPercentage.toFixed(1)}%)`
  );
  lines.push(`- Empty cells (K=0): ${stats.emptyCells} (${stats.emptyPercentage.toFixed(1)}%)`);
  lines.push('');

  lines.push('## Universal Triads');
  const universals = getUniversalTriads(matrix);
  for (const triadId of universals) {
    const triad = matrix.triads[triadId];
    lines.push(`- ${triadId}: ${triad.nameEn}`);
  }
  lines.push('');

  lines.push('## Main Diagonal (Self-Reflexivity)');
  const mainDiag = getMainDiagonal(matrix);
  for (const entry of mainDiag) {
    lines.push(`- (${entry.triadId}, ${entry.layerId}): ${entry.meaningEn}`);
  }
  lines.push('');

  lines.push('## Anti-Diagonal (Critical Transitions)');
  const antiDiag = getAntiDiagonal(matrix);
  for (const entry of antiDiag) {
    lines.push(`- (${entry.triadId}, ${entry.layerId}): ${entry.meaningEn}`);
  }

  return lines.join('\n');
}

// ============================================================================
// Visualization Helpers
// ============================================================================

/**
 * Generate heatmap data for visualization
 */
export function generateHeatmapData(
  matrix: Conv2DMatrix
): { row: number; col: number; value: ConvDepth }[] {
  return matrix.cells.map((c) => ({
    row: c.rowIndex,
    col: c.colIndex,
    value: c.depth,
  }));
}

/**
 * Generate network graph edges
 */
export function generateNetworkEdges(
  matrix: Conv2DMatrix
): { source: string; target: string; weight: ConvDepth }[] {
  const edges: { source: string; target: string; weight: ConvDepth }[] = [];

  for (const cell of matrix.cells) {
    if (cell.depth > 0) {
      edges.push({
        source: cell.triadId,
        target: cell.layerId,
        weight: cell.depth,
      });
    }
  }

  return edges;
}

/**
 * Get color for depth value (for visualization)
 */
export function getDepthColor(depth: ConvDepth): string {
  switch (depth) {
    case 2:
      return '#2563eb'; // Blue - core
    case 1:
      return '#93c5fd'; // Light blue - relevant
    case 0:
      return '#e5e7eb'; // Gray - empty
  }
}
