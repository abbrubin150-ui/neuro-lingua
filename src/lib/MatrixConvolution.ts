/**
 * Matrix Structure and Convolution System (E/F modules)
 *
 * E01-E12: Mapping, convolution, and reporting
 * L01-L14: Knowledge levels (rows)
 * T01-T14: Canonical triads (columns)
 * F0101-F1414: 14×14 matrix cells
 */

import type {
  TargetTriadRecord,
  RoleAlignment,
  KDepth,
  KDepthScore,
  ConvolutionOperator,
  NearMissDetector,
  NearMissDetection,
  CanonicalTriadCatalog,
  CatalogEntry,
  SourceCitation,
  MappingReport,
  MatrixGenerator,
  CellValidator,
  CellValidationResult,
  VersionLedger,
  LedgerEntry,
  ExportPack,
  MatrixCell,
  StateSpaceRHS,
  GStatus,
  EvidenceTier,
  KnowledgeLevelId,
  CanonicalTriadId,
} from '../types/kernel';
import {
  KNOWLEDGE_LEVELS,
  CANONICAL_TRIADS,
  generateCellId,
} from '../types/kernel';
import { createStateSpaceRHS, executeCycle, createCycleOperator } from './KernelPrimitives';
import { runFullGenerativityTest, quickGenerativityCheck } from './GenerativityTests';
import { assignEvidenceTier, createEvidenceRecord } from './EvidenceAndMetrics';

// ============================================================================
// E01 - TargetTriadRecord Factory
// ============================================================================

/**
 * Creates a target triad record
 */
export function createTargetTriadRecord(
  id: string,
  x: number[],
  y: number[],
  z: number[],
  context: string,
  sourceRef?: string
): TargetTriadRecord {
  return {
    type: 'TARGET_TRIAD_RECORD',
    id,
    x,
    y,
    z,
    context,
    sourceRef,
  };
}

/**
 * Creates a triad from canonical mapping
 */
export function createTriadFromCanonical(
  levelId: KnowledgeLevelId,
  triadId: CanonicalTriadId,
  dimension: number = 8
): TargetTriadRecord {
  const level = KNOWLEDGE_LEVELS[levelId];
  const triad = CANONICAL_TRIADS[triadId];

  // Generate representative vectors based on level and triad characteristics
  const seed = level.rowIndex * 14 + triad.colIndex;
  const x = generateDeterministicVector(dimension, seed, 0);
  const y = generateDeterministicVector(dimension, seed, 1);
  const z = generateDeterministicVector(dimension, seed, 2);

  return {
    type: 'TARGET_TRIAD_RECORD',
    id: `${levelId}_${triadId}`,
    x,
    y,
    z,
    context: `${level.name} under ${triad.name}`,
    sourceRef: `Cell ${generateCellId(level.rowIndex, triad.colIndex)}`,
  };
}

/**
 * Generates a deterministic vector for reproducibility
 */
function generateDeterministicVector(dimension: number, seed: number, offset: number): number[] {
  const vector: number[] = [];
  let state = seed + offset * 1000;

  for (let i = 0; i < dimension; i++) {
    // Simple PRNG (LCG)
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const value = (state / 0x7fffffff) * 2 - 1; // Range [-1, 1]
    vector.push(value);
  }

  return vector;
}

// ============================================================================
// E02 - RoleAlignment_phi
// ============================================================================

/**
 * Maps (X,Y,Z) → (R,H,S) roles
 */
export function computeRoleAlignment(triad: TargetTriadRecord): RoleAlignment {
  // Analyze characteristics of each component
  const xVariance = computeVariance(triad.x);
  const yVariance = computeVariance(triad.y);
  const zVariance = computeVariance(triad.z);

  const xEntropy = computeEntropy(triad.x);
  const yEntropy = computeEntropy(triad.y);
  const zEntropy = computeEntropy(triad.z);

  // R (Noise) should have highest variance/entropy
  // H (Coherence) should have structure (moderate variance, lower entropy)
  // S (Soleket) should be the emergent mediation

  const components: Array<{
    name: 'X' | 'Y' | 'Z';
    variance: number;
    entropy: number;
    noiseScore: number;
    coherenceScore: number;
    mediationScore: number;
  }> = [
    {
      name: 'X',
      variance: xVariance,
      entropy: xEntropy,
      noiseScore: xVariance * xEntropy,
      coherenceScore: (1 - xEntropy) * xVariance,
      mediationScore: Math.abs(xVariance - (yVariance + zVariance) / 2),
    },
    {
      name: 'Y',
      variance: yVariance,
      entropy: yEntropy,
      noiseScore: yVariance * yEntropy,
      coherenceScore: (1 - yEntropy) * yVariance,
      mediationScore: Math.abs(yVariance - (xVariance + zVariance) / 2),
    },
    {
      name: 'Z',
      variance: zVariance,
      entropy: zEntropy,
      noiseScore: zVariance * zEntropy,
      coherenceScore: (1 - zEntropy) * zVariance,
      mediationScore: Math.abs(zVariance - (xVariance + yVariance) / 2),
    },
  ];

  // Sort by each score to determine mappings
  const byNoise = [...components].sort((a, b) => b.noiseScore - a.noiseScore);
  const byCoherence = [...components].sort((a, b) => b.coherenceScore - a.coherenceScore);

  // Assign roles
  const rMapping = byNoise[0].name;

  // H is the most coherent among remaining
  const hCandidates = byCoherence.filter((c) => c.name !== rMapping);
  const hMapping = hCandidates[0].name;

  // S is the remaining one
  const sMapping = components.find((c) => c.name !== rMapping && c.name !== hMapping)!.name;

  // Compute confidence based on score separation
  const noiseSeparation = byNoise[0].noiseScore - byNoise[1].noiseScore;
  const coherenceSeparation = hCandidates[0].coherenceScore - hCandidates[1]?.coherenceScore || 0;
  const confidence = Math.min(1, (noiseSeparation + coherenceSeparation) / 2);

  return {
    type: 'ROLE_ALIGNMENT',
    rMapping,
    hMapping,
    sMapping,
    confidence,
  };
}

/**
 * Computes variance of a vector
 */
function computeVariance(v: number[]): number {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  return v.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / v.length;
}

/**
 * Computes normalized entropy
 */
function computeEntropy(v: number[]): number {
  // Normalize to probability distribution
  const minVal = Math.min(...v);
  const shifted = v.map((x) => x - minVal + 1e-10);
  const sum = shifted.reduce((a, b) => a + b, 0);
  const prob = shifted.map((x) => x / sum);

  let entropy = 0;
  for (const p of prob) {
    if (p > 0) {
      entropy -= p * Math.log(p);
    }
  }

  // Normalize by maximum entropy
  const maxEntropy = Math.log(v.length);
  return entropy / maxEntropy;
}

// ============================================================================
// E03 - K_DepthScore
// ============================================================================

/**
 * Computes K-depth score for alignment
 */
export function computeKDepthScore(
  triad: TargetTriadRecord,
  alignment: RoleAlignment,
  gStatus: GStatus
): KDepthScore {
  let depth: KDepth = 0;
  let justification = '';

  // K=0: Surface mapping only
  // K=1: Functional correspondence (G1 passes)
  // K=2: Full operational homomorphism (G1+G2+G3 passes)

  if (gStatus.isGenerative) {
    depth = 2;
    justification = 'Full operational homomorphism: G1∧G2∧G3 satisfied';
  } else if (gStatus.g1.passes) {
    depth = 1;
    justification = 'Functional correspondence: G1 (irreducibility) satisfied';
  } else {
    depth = 0;
    justification = 'Surface mapping only: structural similarity without functional equivalence';
  }

  return {
    type: 'K_DEPTH_SCORE',
    depth,
    justification,
  };
}

// ============================================================================
// E04 - ConvolutionOperator
// ============================================================================

/**
 * Creates a convolution operator for sliding RHS kernel over matrix
 */
export function createConvolutionOperator(
  initialState: StateSpaceRHS
): ConvolutionOperator {
  return {
    type: 'CONVOLUTION_OPERATOR',
    kernel: initialState,
    position: { row: 0, col: 0 },
    populatedCount: 0,
  };
}

/**
 * Advances convolution operator to next position
 */
export function advanceConvolution(
  operator: ConvolutionOperator
): ConvolutionOperator {
  let { row, col } = operator.position;

  col++;
  if (col >= 14) {
    col = 0;
    row++;
  }

  return {
    ...operator,
    position: { row, col },
    populatedCount: operator.populatedCount + 1,
  };
}

/**
 * Applies convolution at current position
 */
export function applyConvolution(
  operator: ConvolutionOperator,
  levelId: KnowledgeLevelId,
  triadId: CanonicalTriadId
): { cell: MatrixCell; operator: ConvolutionOperator } {
  // Create triad for this cell
  const triad = createTriadFromCanonical(levelId, triadId);

  // Map triad to RHS
  const alignment = computeRoleAlignment(triad);

  // Create state space from mapped components
  const mappedR = alignment.rMapping === 'X' ? triad.x : alignment.rMapping === 'Y' ? triad.y : triad.z;
  const mappedH = alignment.hMapping === 'X' ? triad.x : alignment.hMapping === 'Y' ? triad.y : triad.z;
  const mappedS = alignment.sMapping === 'X' ? triad.x : alignment.sMapping === 'Y' ? triad.y : triad.z;

  // Create and evolve state space
  const state = createStateSpaceRHS(mappedR.length);
  const cycleOp = createCycleOperator();
  const { state: evolvedState } = executeCycle(state, cycleOp);

  // Run generativity tests
  const gStatus = runFullGenerativityTest(mappedR, mappedH, mappedS, evolvedState);

  // Assign evidence tier
  const evidenceRecord = [
    createEvidenceRecord('convolution_mapping', 'theory', alignment.confidence),
  ];
  const eAssignment = assignEvidenceTier(evidenceRecord, 'convolution');

  // Compute K-depth
  const kDepthScore = computeKDepthScore(triad, alignment, gStatus);

  // Create cell
  const cell: MatrixCell = {
    type: 'MATRIX_CELL',
    cellId: generateCellId(operator.position.row, operator.position.col),
    rowIndex: operator.position.row,
    colIndex: operator.position.col,
    knowledgeLevel: levelId,
    canonicalTriad: triadId,
    rhs: evolvedState,
    gStatus,
    eTier: eAssignment.tier,
    kDepth: kDepthScore.depth,
    narrative: generateCellNarrative(levelId, triadId, gStatus, eAssignment.tier, kDepthScore),
  };

  return { cell, operator: advanceConvolution(operator) };
}

/**
 * Generates narrative for a cell
 */
function generateCellNarrative(
  levelId: KnowledgeLevelId,
  triadId: CanonicalTriadId,
  gStatus: GStatus,
  eTier: EvidenceTier,
  kDepth: KDepthScore
): string {
  const level = KNOWLEDGE_LEVELS[levelId];
  const triad = CANONICAL_TRIADS[triadId];

  const statusStr = gStatus.isGenerative ? 'GENERATIVE' : 'NON-GENERATIVE';
  const testsStr = `G1=${gStatus.g1.passes}, G2=${gStatus.g2.passes}, G3=${gStatus.g3.passes}`;

  return (
    `${level.name} × ${triad.name} (${triad.components.join('/')}): ` +
    `${statusStr} [${testsStr}], Evidence=${eTier}, Depth=K${kDepth.depth}. ` +
    kDepth.justification
  );
}

// ============================================================================
// E05 - NearMissDetector
// ============================================================================

/**
 * Creates a near-miss detector
 */
export function createNearMissDetector(): NearMissDetector {
  return {
    type: 'NEAR_MISS_DETECTOR',
    detections: [],
  };
}

/**
 * Checks for near-miss in a cell
 */
export function detectNearMiss(
  detector: NearMissDetector,
  cell: MatrixCell
): NearMissDetector {
  if (!cell.gStatus) return detector;

  const failedTests: ('G1' | 'G2' | 'G3')[] = [];
  if (!cell.gStatus.g1.passes) failedTests.push('G1');
  if (!cell.gStatus.g2.passes) failedTests.push('G2');
  if (!cell.gStatus.g3.passes) failedTests.push('G3');

  // Near-miss: passes some but not all
  if (failedTests.length > 0 && failedTests.length < 3) {
    const detection: NearMissDetection = {
      triadId: cell.cellId,
      failedTests,
      classification: failedTests.length === 1 ? 'near_generative' : 'taxonomy',
    };

    return {
      ...detector,
      detections: [...detector.detections, detection],
    };
  }

  return detector;
}

// ============================================================================
// E06 - CanonicalTriadCatalog
// ============================================================================

/**
 * Creates an empty canonical triad catalog
 */
export function createCanonicalTriadCatalog(version: string = '1.0.0'): CanonicalTriadCatalog {
  return {
    type: 'CANONICAL_TRIAD_CATALOG',
    triads: [],
    version,
  };
}

/**
 * Registers a triad in the catalog
 */
export function registerTriad(
  catalog: CanonicalTriadCatalog,
  name: string,
  triad: TargetTriadRecord,
  source: SourceCitation
): CanonicalTriadCatalog {
  const entry: CatalogEntry = {
    id: triad.id,
    name,
    triad,
    source,
    acceptedDate: Date.now(),
  };

  return {
    ...catalog,
    triads: [...catalog.triads, entry],
  };
}

/**
 * E07 - Creates a source citation
 */
export function createSourceCitation(
  author: string,
  title: string,
  year: number,
  doi?: string,
  url?: string
): SourceCitation {
  return { author, title, year, doi, url };
}

// ============================================================================
// E08 - MappingReport
// ============================================================================

/**
 * Generates a mapping report for a cell
 */
export function generateMappingReport(cell: MatrixCell): MappingReport {
  return {
    type: 'MAPPING_REPORT',
    cellId: cell.cellId,
    narrative: cell.narrative,
    gStatus: cell.gStatus!,
    eTier: cell.eTier!,
    kDepth: cell.kDepth!,
    generatedAt: Date.now(),
  };
}

/**
 * Generates batch mapping reports
 */
export function generateBatchReports(cells: MatrixCell[]): MappingReport[] {
  return cells
    .filter((cell) => cell.gStatus && cell.eTier !== null && cell.kDepth !== null)
    .map(generateMappingReport);
}

// ============================================================================
// E09 - MatrixGenerator
// ============================================================================

/**
 * Creates a matrix generator
 */
export function createMatrixGenerator(seed: number = 42): MatrixGenerator {
  return {
    type: 'MATRIX_GENERATOR',
    rows: 14,
    cols: 14,
    seed,
  };
}

/**
 * Generates the full 14×14 matrix
 */
export function generateMatrix(generator: MatrixGenerator): MatrixCell[] {
  const cells: MatrixCell[] = [];
  let operator = createConvolutionOperator(createStateSpaceRHS(8));

  const levelIds = Object.keys(KNOWLEDGE_LEVELS) as KnowledgeLevelId[];
  const triadIds = Object.keys(CANONICAL_TRIADS) as CanonicalTriadId[];

  for (const levelId of levelIds) {
    for (const triadId of triadIds) {
      const result = applyConvolution(operator, levelId, triadId);
      cells.push(result.cell);
      operator = result.operator;
    }
  }

  return cells;
}

/**
 * Generates matrix with progress callback
 */
export function generateMatrixAsync(
  generator: MatrixGenerator,
  onProgress?: (current: number, total: number) => void
): Promise<MatrixCell[]> {
  return new Promise((resolve) => {
    const cells: MatrixCell[] = [];
    let operator = createConvolutionOperator(createStateSpaceRHS(8));

    const levelIds = Object.keys(KNOWLEDGE_LEVELS) as KnowledgeLevelId[];
    const triadIds = Object.keys(CANONICAL_TRIADS) as CanonicalTriadId[];
    const total = levelIds.length * triadIds.length;

    let current = 0;
    for (const levelId of levelIds) {
      for (const triadId of triadIds) {
        const result = applyConvolution(operator, levelId, triadId);
        cells.push(result.cell);
        operator = result.operator;
        current++;
        onProgress?.(current, total);
      }
    }

    resolve(cells);
  });
}

// ============================================================================
// E10 - CellValidator
// ============================================================================

/**
 * Creates a cell validator
 */
export function createCellValidator(): CellValidator {
  return {
    type: 'CELL_VALIDATOR',
    results: [],
    isValid: true,
  };
}

/**
 * Validates a single cell
 */
export function validateCell(cell: MatrixCell): CellValidationResult {
  const errors: string[] = [];

  const hasRHS = cell.rhs !== null;
  const hasGStatus = cell.gStatus !== null;
  const hasETier = cell.eTier !== null;
  const hasKDepth = cell.kDepth !== null;

  if (!hasRHS) errors.push('Missing RHS state');
  if (!hasGStatus) errors.push('Missing G-status');
  if (!hasETier) errors.push('Missing E-tier');
  if (!hasKDepth) errors.push('Missing K-depth');

  return {
    cellId: cell.cellId,
    hasRHS,
    hasGStatus,
    hasETier,
    hasKDepth,
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * Validates all cells in matrix
 */
export function validateMatrix(cells: MatrixCell[]): CellValidator {
  const results = cells.map(validateCell);
  const isValid = results.every((r) => r.isValid);

  return {
    type: 'CELL_VALIDATOR',
    results,
    isValid,
  };
}

// ============================================================================
// E11 - VersionLedger
// ============================================================================

/**
 * Creates a version ledger
 */
export function createVersionLedger(initialVersion: string = '0.0.1'): VersionLedger {
  return {
    type: 'VERSION_LEDGER',
    currentVersion: initialVersion,
    history: [
      {
        version: initialVersion,
        timestamp: Date.now(),
        changes: ['Initial creation'],
        author: 'system',
      },
    ],
  };
}

/**
 * Records a new version (merge-only policy)
 */
export function recordVersion(
  ledger: VersionLedger,
  newVersion: string,
  changes: string[],
  author: string
): VersionLedger {
  const entry: LedgerEntry = {
    version: newVersion,
    timestamp: Date.now(),
    changes,
    author,
  };

  return {
    ...ledger,
    currentVersion: newVersion,
    history: [...ledger.history, entry],
  };
}

/**
 * Validates version increment (no downgrades allowed)
 */
export function validateVersionIncrement(current: string, proposed: string): boolean {
  const currentParts = current.split('.').map(Number);
  const proposedParts = proposed.split('.').map(Number);

  for (let i = 0; i < Math.max(currentParts.length, proposedParts.length); i++) {
    const curr = currentParts[i] || 0;
    const prop = proposedParts[i] || 0;

    if (prop > curr) return true;
    if (prop < curr) return false;
  }

  return false; // Equal versions not allowed
}

// ============================================================================
// E12 - ExportPack
// ============================================================================

/**
 * Creates an export pack
 */
export function createExportPack(
  cells: MatrixCell[],
  catalog: CanonicalTriadCatalog,
  ledger: VersionLedger
): ExportPack {
  return {
    type: 'EXPORT_PACK',
    matrix: cells,
    catalog,
    ledger,
    exportedAt: Date.now(),
    formatVersion: '1.0.0',
  };
}

/**
 * Serializes export pack to JSON
 */
export function serializeExportPack(pack: ExportPack): string {
  return JSON.stringify(pack, null, 2);
}

/**
 * Deserializes export pack from JSON
 */
export function deserializeExportPack(json: string): ExportPack {
  return JSON.parse(json) as ExportPack;
}

// ============================================================================
// Matrix Statistics
// ============================================================================

/**
 * Computes statistics for the matrix
 */
export function computeMatrixStatistics(cells: MatrixCell[]): {
  totalCells: number;
  generativeCells: number;
  nonGenerativeCells: number;
  nearMissCells: number;
  byETier: Record<EvidenceTier, number>;
  byKDepth: Record<KDepth, number>;
  generativeRate: number;
} {
  const stats = {
    totalCells: cells.length,
    generativeCells: 0,
    nonGenerativeCells: 0,
    nearMissCells: 0,
    byETier: { E0: 0, E1: 0, E2: 0, E3: 0 } as Record<EvidenceTier, number>,
    byKDepth: { 0: 0, 1: 0, 2: 0 } as Record<KDepth, number>,
    generativeRate: 0,
  };

  for (const cell of cells) {
    if (cell.gStatus?.isGenerative) {
      stats.generativeCells++;
    } else {
      stats.nonGenerativeCells++;

      // Check for near-miss
      if (cell.gStatus) {
        const passCount = [
          cell.gStatus.g1.passes,
          cell.gStatus.g2.passes,
          cell.gStatus.g3.passes,
        ].filter(Boolean).length;
        if (passCount > 0 && passCount < 3) {
          stats.nearMissCells++;
        }
      }
    }

    if (cell.eTier) {
      stats.byETier[cell.eTier]++;
    }

    if (cell.kDepth !== null) {
      stats.byKDepth[cell.kDepth]++;
    }
  }

  stats.generativeRate = stats.totalCells > 0 ? stats.generativeCells / stats.totalCells : 0;

  return stats;
}

/**
 * Gets cell by row and column
 */
export function getCellByPosition(
  cells: MatrixCell[],
  row: number,
  col: number
): MatrixCell | undefined {
  return cells.find((c) => c.rowIndex === row && c.colIndex === col);
}

/**
 * Gets cells by knowledge level
 */
export function getCellsByLevel(cells: MatrixCell[], levelId: KnowledgeLevelId): MatrixCell[] {
  return cells.filter((c) => c.knowledgeLevel === levelId);
}

/**
 * Gets cells by canonical triad
 */
export function getCellsByTriad(cells: MatrixCell[], triadId: CanonicalTriadId): MatrixCell[] {
  return cells.filter((c) => c.canonicalTriad === triadId);
}

// ============================================================================
// Utility Exports
// ============================================================================

export {
  generateDeterministicVector,
  computeVariance,
  computeEntropy,
  generateCellNarrative,
};
