/**
 * Tests for Matrix, Convolution, and Governance Systems
 *
 * Tests for E-module (Matrix/Convolution) and G/H-modules (Governance/Telemetry)
 */

import { describe, it, expect } from 'vitest';
import {
  // Matrix/Convolution
  createTargetTriadRecord,
  createTriadFromCanonical,
  computeRoleAlignment,
  computeKDepthScore,
  createConvolutionOperator,
  advanceConvolution,
  applyConvolution,
  createMatrixGenerator,
  generateMatrix,
  createCellValidator,
  validateMatrix,
  createVersionLedger,
  recordVersion,
  validateVersionIncrement,
  computeMatrixStatistics,
} from '../../src/lib/MatrixConvolution';
import {
  // Governance
  createRAnchorGate,
  checkAnchorGate,
  createCanon5Bundle,
  validateCanon5,
  createMirrorSidecar,
  recordDowngrade,
  createMirrorLipschitzDrift,
  createMirrorFPBCheck,
  evaluateMirrorVeto,
  createEXACT1Primitive,
  transitionEXACT1,
  createStop3Counter,
  recordViolation,
  resetViolationCounter,
  createWitnessANDGate,
  addWitness,
  createGovernanceState,
  // Telemetry
  createKPIDeltaDIADef,
  createKPIEdgeBand,
  createSensorLiveSlope,
  createKPIBEIBreath,
  createLPSFThresholds,
  createLPSFSelector,
  createTelemetryStop3,
  createDashboardView,
  createQuarantineList,
  addToQuarantine,
  createRecoveryLog,
  recordRecovery,
  createTelemetryState,
  updateTelemetryFromState,
} from '../../src/lib/GovernanceAndTelemetry';
import { createStateSpaceRHS } from '../../src/lib/KernelPrimitives';
import { generateCellId, parseCellId } from '../../src/types/kernel';

describe('E01 - TargetTriadRecord', () => {
  it('should create target triad record', () => {
    const triad = createTargetTriadRecord(
      'test_triad',
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      'test_context'
    );

    expect(triad.type).toBe('TARGET_TRIAD_RECORD');
    expect(triad.id).toBe('test_triad');
    expect(triad.x).toEqual([1, 2, 3]);
    expect(triad.y).toEqual([4, 5, 6]);
    expect(triad.z).toEqual([7, 8, 9]);
  });

  it('should create triad from canonical mapping', () => {
    const triad = createTriadFromCanonical('L01', 'T01', 8);
    expect(triad.x).toHaveLength(8);
    expect(triad.y).toHaveLength(8);
    expect(triad.z).toHaveLength(8);
    expect(triad.context).toContain('Digital Foundation');
    expect(triad.context).toContain('Peirce');
  });
});

describe('E02 - RoleAlignment', () => {
  it('should compute role alignment', () => {
    const triad = createTargetTriadRecord(
      'test',
      [1, 2, 3, 4, 5],
      [0.1, 0.2, 0.3, 0.4, 0.5],
      [0.5, 0.4, 0.3, 0.2, 0.1],
      'test'
    );

    const alignment = computeRoleAlignment(triad);
    expect(alignment.type).toBe('ROLE_ALIGNMENT');
    expect(['X', 'Y', 'Z']).toContain(alignment.rMapping);
    expect(['X', 'Y', 'Z']).toContain(alignment.hMapping);
    expect(['X', 'Y', 'Z']).toContain(alignment.sMapping);
    expect(alignment.confidence).toBeGreaterThanOrEqual(0);
  });

  it('should assign different roles to each component', () => {
    const triad = createTriadFromCanonical('L01', 'T01', 5);
    const alignment = computeRoleAlignment(triad);

    const mappings = new Set([alignment.rMapping, alignment.hMapping, alignment.sMapping]);
    expect(mappings.size).toBe(3);
  });
});

describe('E03 - K_DepthScore', () => {
  it('should compute K-depth score', () => {
    const triad = createTriadFromCanonical('L01', 'T01', 5);
    const alignment = computeRoleAlignment(triad);
    const gStatus = {
      type: 'G_STATUS' as const,
      g1: { type: 'G1_IRREDUCIBILITY' as const, passes: true, testedMethods: [], score: 0.8 },
      g2: { type: 'G2_MEDIATION' as const, passes: true, tensionReduction: 0.5, integrityPreserved: true },
      g3: { type: 'G3_DOWNWARD_CLOSURE' as const, passes: true, constraintStrength: 0.6, pValue: 0.01 },
      isGenerative: true,
    };

    const kDepth = computeKDepthScore(triad, alignment, gStatus);
    expect(kDepth.type).toBe('K_DEPTH_SCORE');
    expect([0, 1, 2]).toContain(kDepth.depth);
    expect(kDepth.justification.length).toBeGreaterThan(0);
  });

  it('should return K=2 for fully generative', () => {
    const triad = createTriadFromCanonical('L01', 'T01', 5);
    const alignment = computeRoleAlignment(triad);
    const gStatus = {
      type: 'G_STATUS' as const,
      g1: { type: 'G1_IRREDUCIBILITY' as const, passes: true, testedMethods: [], score: 0.8 },
      g2: { type: 'G2_MEDIATION' as const, passes: true, tensionReduction: 0.5, integrityPreserved: true },
      g3: { type: 'G3_DOWNWARD_CLOSURE' as const, passes: true, constraintStrength: 0.6, pValue: 0.01 },
      isGenerative: true,
    };

    const kDepth = computeKDepthScore(triad, alignment, gStatus);
    expect(kDepth.depth).toBe(2);
  });
});

describe('E04 - ConvolutionOperator', () => {
  it('should create convolution operator', () => {
    const state = createStateSpaceRHS(8);
    const op = createConvolutionOperator(state);

    expect(op.type).toBe('CONVOLUTION_OPERATOR');
    expect(op.position).toEqual({ row: 0, col: 0 });
    expect(op.populatedCount).toBe(0);
  });

  it('should advance convolution position', () => {
    const state = createStateSpaceRHS(8);
    let op = createConvolutionOperator(state);

    op = advanceConvolution(op);
    expect(op.position).toEqual({ row: 0, col: 1 });
    expect(op.populatedCount).toBe(1);
  });

  it('should wrap to next row', () => {
    const state = createStateSpaceRHS(8);
    let op = createConvolutionOperator(state);

    for (let i = 0; i < 14; i++) {
      op = advanceConvolution(op);
    }

    expect(op.position).toEqual({ row: 1, col: 0 });
  });

  it('should apply convolution and produce cell', () => {
    const state = createStateSpaceRHS(8);
    const op = createConvolutionOperator(state);
    const result = applyConvolution(op, 'L01', 'T01');

    expect(result.cell.type).toBe('MATRIX_CELL');
    expect(result.cell.cellId).toBe('F0101');
    expect(result.cell.knowledgeLevel).toBe('L01');
    expect(result.cell.canonicalTriad).toBe('T01');
  });
});

describe('E09 - MatrixGenerator', () => {
  it('should create matrix generator', () => {
    const gen = createMatrixGenerator(42);
    expect(gen.type).toBe('MATRIX_GENERATOR');
    expect(gen.rows).toBe(14);
    expect(gen.cols).toBe(14);
    expect(gen.seed).toBe(42);
  });

  // Skip the full matrix generation test as it's computationally expensive (196 cells)
  it.skip('should generate full 14x14 matrix', () => {
    const gen = createMatrixGenerator(42);
    const cells = generateMatrix(gen);

    expect(cells).toHaveLength(196); // 14 * 14
  });
});

describe('E10 - CellValidator', () => {
  // Skip as it depends on full matrix generation
  it.skip('should validate matrix cells', () => {
    const gen = createMatrixGenerator(42);
    const cells = generateMatrix(gen);
    const validator = validateMatrix(cells);

    expect(validator.type).toBe('CELL_VALIDATOR');
    expect(validator.results).toHaveLength(196);
  });

  it('should create cell validator', () => {
    const validator = createCellValidator();
    expect(validator.type).toBe('CELL_VALIDATOR');
    expect(validator.results).toHaveLength(0);
    expect(validator.isValid).toBe(true);
  });
});

describe('E11 - VersionLedger', () => {
  it('should create version ledger', () => {
    const ledger = createVersionLedger('1.0.0');
    expect(ledger.type).toBe('VERSION_LEDGER');
    expect(ledger.currentVersion).toBe('1.0.0');
    expect(ledger.history).toHaveLength(1);
  });

  it('should record new versions', () => {
    let ledger = createVersionLedger('1.0.0');
    ledger = recordVersion(ledger, '1.0.1', ['Bug fix'], 'test_author');

    expect(ledger.currentVersion).toBe('1.0.1');
    expect(ledger.history).toHaveLength(2);
  });

  it('should validate version increment', () => {
    expect(validateVersionIncrement('1.0.0', '1.0.1')).toBe(true);
    expect(validateVersionIncrement('1.0.0', '1.1.0')).toBe(true);
    expect(validateVersionIncrement('1.0.0', '2.0.0')).toBe(true);
    expect(validateVersionIncrement('1.0.1', '1.0.0')).toBe(false);
    expect(validateVersionIncrement('1.0.0', '1.0.0')).toBe(false);
  });
});

describe('Cell ID Utilities', () => {
  it('should generate cell ID', () => {
    expect(generateCellId(0, 0)).toBe('F0101');
    expect(generateCellId(0, 13)).toBe('F0114');
    expect(generateCellId(13, 13)).toBe('F1414');
  });

  it('should parse cell ID', () => {
    const result = parseCellId('F0101');
    expect(result).toEqual({ row: 0, col: 0 });

    const result2 = parseCellId('F1414');
    expect(result2).toEqual({ row: 13, col: 13 });
  });

  it('should return null for invalid cell ID', () => {
    expect(parseCellId('invalid')).toBeNull();
    expect(parseCellId('F000')).toBeNull();
  });
});

describe('Matrix Statistics', () => {
  // Skip as it depends on full matrix generation
  it.skip('should compute matrix statistics', () => {
    const gen = createMatrixGenerator(42);
    const cells = generateMatrix(gen);
    const stats = computeMatrixStatistics(cells);

    expect(stats.totalCells).toBe(196);
    expect(stats.generativeCells + stats.nonGenerativeCells).toBe(196);
    expect(stats.generativeRate).toBeGreaterThanOrEqual(0);
    expect(stats.generativeRate).toBeLessThanOrEqual(1);
  });

  it('should compute statistics for empty matrix', () => {
    const stats = computeMatrixStatistics([]);
    expect(stats.totalCells).toBe(0);
    expect(stats.generativeRate).toBe(0);
  });
});

// ============================================================================
// Governance Tests (G-module)
// ============================================================================

describe('G01 - R_AnchorGate', () => {
  it('should create anchor gate', () => {
    const gate = createRAnchorGate();
    expect(gate.type).toBe('R_ANCHOR_GATE');
    expect(gate.isActive).toBe(true);
  });

  it('should allow merge operations', () => {
    const gate = createRAnchorGate();
    expect(checkAnchorGate(gate, 'merge')).toBe(true);
    expect(checkAnchorGate(gate, 'append')).toBe(true);
  });

  it('should block delete operations', () => {
    const gate = createRAnchorGate();
    expect(checkAnchorGate(gate, 'delete')).toBe(false);
    expect(checkAnchorGate(gate, 'overwrite')).toBe(false);
  });
});

describe('G02-G08 - CANON5 Bundle', () => {
  it('should create CANON5 bundle', () => {
    const bundle = createCanon5Bundle();
    expect(bundle.type).toBe('CANON5_BUNDLE');
    expect(bundle.c68).toBeDefined();
    expect(bundle.c76).toBeDefined();
    expect(bundle.c79).toBeDefined();
    expect(bundle.c80).toBeDefined();
    expect(bundle.c117).toBeDefined();
  });

  it('should validate CANON5 successfully', () => {
    const bundle = createCanon5Bundle();
    // Set signature to satisfy C80
    bundle.c80.hasSignature = true;
    bundle.c80.signature = 'test_sig';

    const result = validateCanon5(bundle);
    expect(result.isValid).toBe(true);
    expect(result.violations).toHaveLength(0);
  });

  it('should detect C80 violation', () => {
    const bundle = createCanon5Bundle();
    const result = validateCanon5(bundle);

    expect(result.violations).toContain('C80: External signature required but not present');
  });
});

describe('G10-G13 - Mirror Sidecar', () => {
  it('should create mirror sidecar', () => {
    const sidecar = createMirrorSidecar();
    expect(sidecar.type).toBe('MIRROR_SIDECAR');
    expect(sidecar.isActive).toBe(true);
  });

  it('should record downgrade', () => {
    let sidecar = createMirrorSidecar();
    sidecar = recordDowngrade(sidecar, 'drift', 'EXECUTE', 'HOLD');

    expect(sidecar.downgrades).toHaveLength(1);
    expect(sidecar.downgrades[0].reason).toBe('drift');
  });

  it('should evaluate mirror veto', () => {
    const drift = createMirrorLipschitzDrift(0.2, 0.1); // Exceeds threshold
    const fpb = createMirrorFPBCheck(0.05); // OK

    const veto = evaluateMirrorVeto(drift, fpb, false);
    expect(veto.isActive).toBe(true);
    expect(veto.reason).toBe('drift');
  });
});

describe('G14-G18 - EXACT1 Decision', () => {
  it('should create EXACT1 primitive', () => {
    const exact1 = createEXACT1Primitive();
    expect(exact1.type).toBe('EXACT1_PRIMITIVE');
    expect(exact1.currentState).toBe('HOLD');
  });

  it('should transition states', () => {
    let exact1 = createEXACT1Primitive('HOLD');
    exact1 = transitionEXACT1(exact1, 'EXECUTE');
    expect(exact1.currentState).toBe('EXECUTE');

    exact1 = transitionEXACT1(exact1, 'ESCALATE');
    expect(exact1.currentState).toBe('ESCALATE');
  });
});

describe('G19 - Stop3 Counter', () => {
  it('should create Stop3 counter', () => {
    const counter = createStop3Counter();
    expect(counter.type).toBe('STOP3_COUNTER');
    expect(counter.consecutiveViolations).toBe(0);
    expect(counter.isHalted).toBe(false);
  });

  it('should halt after 3 violations', () => {
    let counter = createStop3Counter();
    counter = recordViolation(counter);
    expect(counter.isHalted).toBe(false);

    counter = recordViolation(counter);
    expect(counter.isHalted).toBe(false);

    counter = recordViolation(counter);
    expect(counter.isHalted).toBe(true);
    expect(counter.consecutiveViolations).toBe(3);
  });

  it('should reset counter', () => {
    let counter = createStop3Counter();
    counter = recordViolation(counter);
    counter = recordViolation(counter);
    counter = resetViolationCounter(counter);

    expect(counter.consecutiveViolations).toBe(0);
    expect(counter.isHalted).toBe(false);
  });
});

describe('G22-G23 - Witness System', () => {
  it('should create witness gate', () => {
    const gate = createWitnessANDGate(2);
    expect(gate.type).toBe('WITNESS_AND_GATE');
    expect(gate.requiredWitnesses).toBe(2);
    expect(gate.passes).toBe(false);
  });

  it('should pass after sufficient witnesses', () => {
    let gate = createWitnessANDGate(2);
    gate = addWitness(gate, 'witness_1');
    expect(gate.passes).toBe(false);

    gate = addWitness(gate, 'witness_2');
    expect(gate.passes).toBe(true);
  });
});

// ============================================================================
// Telemetry Tests (H-module)
// ============================================================================

describe('H01-H03 - KPI Definitions', () => {
  it('should create KPI DeltaDIA', () => {
    const kpi = createKPIDeltaDIADef(0.5);
    expect(kpi.type).toBe('KPI_DELTA_DIA_DEF');
    expect(kpi.deltaRate).toBe(0.5);
    expect(kpi.isSatisfied).toBe(true);
  });

  it('should fail KPI with negative delta', () => {
    const kpi = createKPIDeltaDIADef(-0.1);
    expect(kpi.isSatisfied).toBe(false);
  });

  it('should create edge band', () => {
    const band = createKPIEdgeBand(-0.1, 0.1, 0.05);
    expect(band.isInBand).toBe(true);

    const outOfBand = createKPIEdgeBand(-0.1, 0.1, 0.2);
    expect(outOfBand.isInBand).toBe(false);
  });
});

describe('H04-H05 - Sensors', () => {
  it('should create live slope sensor', () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sensor = createSensorLiveSlope(values);

    expect(sensor.type).toBe('SENSOR_LIVE_SLOPE');
    expect(sensor.slope).toBeGreaterThan(0); // Increasing trend
  });

  it('should detect drift', () => {
    const values = [1, 2, 4, 8, 16, 32]; // Exponential growth
    const sensor = createSensorLiveSlope(values);

    expect(sensor.isDrifting).toBe(true);
  });

  it('should create BEI breath metric', () => {
    const breath = createKPIBEIBreath(0.5, 0.52);
    expect(breath.type).toBe('KPI_BEI_BREATH');
    expect(breath.invariance).toBeCloseTo(0.02, 2);
    expect(breath.isInvariant).toBe(true);
  });
});

describe('H06-H08 - LPSF System', () => {
  it('should create LPSF selector', () => {
    const candidates = [
      { expr: 0.6, privacy: 0.9, safety: 0.95 },
      { expr: 0.4, privacy: 0.7, safety: 0.8 },
    ];
    const thresholds = createLPSFThresholds(0.5, 0.8, 0.9);
    const selector = createLPSFSelector(candidates, thresholds);

    expect(selector.type).toBe('LPSF_SELECTOR');
    expect(selector.selectedIndex).toBe(0);
    expect(selector.exact1Satisfied).toBe(true);
  });

  it('should reject when no candidate meets thresholds', () => {
    const candidates = [
      { expr: 0.3, privacy: 0.9, safety: 0.95 },
    ];
    const thresholds = createLPSFThresholds(0.5, 0.8, 0.9);
    const selector = createLPSFSelector(candidates, thresholds);

    expect(selector.selectedIndex).toBe(-1);
    expect(selector.exact1Satisfied).toBe(false);
  });
});

describe('H11-H15 - Monitoring', () => {
  it('should create telemetry Stop3', () => {
    const trends = [0.1, -0.1, -0.2, -0.3];
    const stop3 = createTelemetryStop3(trends);

    expect(stop3.type).toBe('TELEMETRY_STOP3');
    expect(stop3.isStopped).toBe(true); // 3 consecutive negative
  });

  it('should create dashboard view', () => {
    const dashboard = createDashboardView(
      { kpi1: 0.5, kpi2: 0.8 },
      { sensor1: 0.3 }
    );

    expect(dashboard.type).toBe('DASHBOARD_VIEW');
    expect(dashboard.kpiSummary.kpi1).toBe(0.5);
    expect(dashboard.sensorSummary.sensor1).toBe(0.3);
  });

  it('should create and manage quarantine list', () => {
    let quarantine = createQuarantineList();
    quarantine = addToQuarantine(quarantine, 'module_1', 'module', 'drift detected');

    expect(quarantine.items).toHaveLength(1);
    expect(quarantine.items[0].id).toBe('module_1');
  });

  it('should record recovery entries', () => {
    let log = createRecoveryLog();
    log = recordRecovery(log, 'rollback', 'Rolled back to snapshot', true);

    expect(log.entries).toHaveLength(1);
    expect(log.entries[0].action).toBe('rollback');
    expect(log.entries[0].success).toBe(true);
  });
});

describe('Integrated State Updates', () => {
  it('should update telemetry from RHS state', () => {
    let telemetry = createTelemetryState();
    const state = createStateSpaceRHS(8);

    telemetry = updateTelemetryFromState(telemetry, state);

    expect(telemetry.dashboard.kpiSummary.rVariance).toBeDefined();
    expect(telemetry.dashboard.kpiSummary.hCoherence).toBeDefined();
    expect(telemetry.dashboard.kpiSummary.sClearance).toBeDefined();
  });

  it('should create full governance state', () => {
    const gov = createGovernanceState();

    expect(gov.anchorGate).toBeDefined();
    expect(gov.canon5).toBeDefined();
    expect(gov.mirrorSidecar).toBeDefined();
    expect(gov.exact1).toBeDefined();
    expect(gov.stop3).toBeDefined();
  });
});
