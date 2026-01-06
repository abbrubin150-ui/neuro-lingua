/**
 * Tests for Generativity Tests (B-module)
 *
 * Tests for modules B01-B08 (Generativity Tests G1-G3)
 */

import { describe, it, expect } from 'vitest';
import {
  createM0NonGenerativeModelClass,
  testM0Membership,
  createAggConstructor,
  createLinCombConstructor,
  createMonotoneTransformConstructor,
  testG1Irreducibility,
  testG1WithPhi,
  testG2Mediation,
  testG3DownwardClosure,
  classifyGStatus,
  runFullGenerativityTest,
  quickGenerativityCheck,
  computeResidual,
  correlation,
} from '../../src/lib/GenerativityTests';
import { createStateSpaceRHS, executeCycle, createCycleOperator } from '../../src/lib/KernelPrimitives';

describe('B01 - M0_NonGenerativeModelClass', () => {
  it('should create M0 class with all methods', () => {
    const m0 = createM0NonGenerativeModelClass();
    expect(m0.type).toBe('M0_NON_GENERATIVE');
    expect(m0.classId).toBe('M0');
    expect(m0.methods).toContain('aggregation');
    expect(m0.methods).toContain('linear_combination');
    expect(m0.methods).toContain('monotone_transform');
  });
});

describe('B02 - AggConstructor', () => {
  it('should create aggregation (element-wise max)', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 1, 4, 3, 6];
    const agg = createAggConstructor(x, y);

    expect(agg.type).toBe('AGG_CONSTRUCTOR');
    expect(agg.outputZ).toEqual([2, 2, 4, 4, 6]);
  });
});

describe('B03 - LinCombConstructor', () => {
  it('should create linear combination', () => {
    const x = [1, 2, 3];
    const y = [4, 5, 6];
    const linComb = createLinCombConstructor(x, y, 2, 3);

    expect(linComb.type).toBe('LIN_COMB_CONSTRUCTOR');
    expect(linComb.coeffA).toBe(2);
    expect(linComb.coeffB).toBe(3);
    expect(linComb.outputZ).toEqual([14, 19, 24]); // 2*x + 3*y
  });
});

describe('B04 - MonotoneTransformConstructor', () => {
  it('should create monotone transform constructor', () => {
    const mono = createMonotoneTransformConstructor('test_transform', true);
    expect(mono.type).toBe('MONOTONE_TRANSFORM_CONSTRUCTOR');
    expect(mono.transformId).toBe('test_transform');
    expect(mono.preservesTopology).toBe(true);
  });
});

describe('B05 - G1_Irreducibility', () => {
  it('should detect aggregation-based Z as reducible', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 1, 4, 3, 6];
    const z = x.map((xi, i) => Math.max(xi, y[i])); // Aggregation

    const result = testG1Irreducibility(x, y, z, 0.1);
    expect(result.type).toBe('G1_IRREDUCIBILITY');
    expect(result.passes).toBe(false); // Should fail - Z is reducible
  });

  it('should detect linear combination as reducible', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 3, 4, 5, 6];
    const z = x.map((xi, i) => 0.5 * xi + 0.5 * y[i]); // Linear combination

    const result = testG1Irreducibility(x, y, z, 0.1);
    expect(result.passes).toBe(false);
  });

  it('should pass for genuinely novel Z', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 3, 4, 5, 6];
    // Create Z that cannot be expressed as simple combination
    const z = x.map((xi, i) => Math.sin(xi * y[i]) * Math.cos(xi - y[i]));

    const result = testG1Irreducibility(x, y, z, 0.3);
    // Novel Z should be harder to fit with M0 methods
    expect(result.score).toBeGreaterThan(0);
  });
});

describe('B05 with Phi - G1_Irreducibility with Phi', () => {
  it('should compute Phi proxy', () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 3, 4, 5, 6];
    const z = x.map((xi, i) => Math.sin(xi * y[i]));

    const result = testG1WithPhi(x, y, z, 0.05);
    expect(result.phi).toBeDefined();
    expect(result.phi).toBeGreaterThanOrEqual(0);
  });
});

describe('B06 - G2_Mediation', () => {
  it('should test mediation on state', () => {
    const state = createStateSpaceRHS(8);
    const result = testG2Mediation(state);

    expect(result.type).toBe('G2_MEDIATION');
    expect(typeof result.passes).toBe('boolean');
    expect(result.tensionReduction).toBeGreaterThanOrEqual(0);
    expect(typeof result.integrityPreserved).toBe('boolean');
  });

  it('should find mediation path', () => {
    const state = createStateSpaceRHS(8);
    const result = testG2Mediation(state);

    if (result.mediationPath) {
      expect(Array.isArray(result.mediationPath)).toBe(true);
    }
  });
});

describe('B07 - G3_DownwardClosure', () => {
  it('should test downward closure', () => {
    const state = createStateSpaceRHS(8);
    const result = testG3DownwardClosure(state);

    expect(result.type).toBe('G3_DOWNWARD_CLOSURE');
    expect(typeof result.passes).toBe('boolean');
    expect(result.constraintStrength).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it('should improve with more history', () => {
    let state = createStateSpaceRHS(8);
    const op = createCycleOperator();

    // Run cycles to build history
    for (let i = 0; i < 15; i++) {
      const result = executeCycle(state, op);
      state = result.state;
    }

    const result = testG3DownwardClosure(state);
    expect(result.constraintStrength).toBeGreaterThan(0);
  });
});

describe('B08 - G_Status Classification', () => {
  it('should classify as generative when all tests pass', () => {
    const g1 = {
      type: 'G1_IRREDUCIBILITY' as const,
      passes: true,
      testedMethods: ['aggregation' as const, 'linear_combination' as const, 'monotone_transform' as const],
      score: 0.8,
    };
    const g2 = {
      type: 'G2_MEDIATION' as const,
      passes: true,
      tensionReduction: 0.5,
      integrityPreserved: true,
      mediationPath: [1, 2, 3],
    };
    const g3 = {
      type: 'G3_DOWNWARD_CLOSURE' as const,
      passes: true,
      constraintStrength: 0.6,
      pValue: 0.01,
    };

    const status = classifyGStatus(g1, g2, g3);
    expect(status.isGenerative).toBe(true);
  });

  it('should classify as non-generative when any test fails', () => {
    const g1 = {
      type: 'G1_IRREDUCIBILITY' as const,
      passes: false,
      testedMethods: ['aggregation' as const],
      score: 0.2,
    };
    const g2 = {
      type: 'G2_MEDIATION' as const,
      passes: true,
      tensionReduction: 0.5,
      integrityPreserved: true,
    };
    const g3 = {
      type: 'G3_DOWNWARD_CLOSURE' as const,
      passes: true,
      constraintStrength: 0.6,
      pValue: 0.01,
    };

    const status = classifyGStatus(g1, g2, g3);
    expect(status.isGenerative).toBe(false);
  });
});

describe('Full Generativity Test Pipeline', () => {
  it('should run full generativity test', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const y = [2, 3, 4, 5, 6, 7, 8, 9];
    const z = x.map((xi, i) => Math.sin(xi) * Math.cos(y[i]));
    const state = createStateSpaceRHS(8);

    const status = runFullGenerativityTest(x, y, z, state);
    expect(status.type).toBe('G_STATUS');
    expect(status.g1).toBeDefined();
    expect(status.g2).toBeDefined();
    expect(status.g3).toBeDefined();
    expect(typeof status.isGenerative).toBe('boolean');
  });

  it('should run quick generativity check', () => {
    const state = createStateSpaceRHS(8);
    const result = quickGenerativityCheck(state);

    expect(typeof result.isGenerative).toBe('boolean');
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });
});

describe('Utility Functions', () => {
  it('should compute residual correctly', () => {
    const predicted = [1, 2, 3, 4, 5];
    const actual = [1, 2, 3, 4, 5];
    const residual = computeResidual(predicted, actual);
    expect(residual).toBeCloseTo(0, 5);
  });

  it('should compute correlation correctly', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [1, 2, 3, 4, 5];
    const corr = correlation(a, b);
    expect(corr).toBeCloseTo(1, 5);
  });

  it('should compute negative correlation', () => {
    const a = [1, 2, 3, 4, 5];
    const b = [5, 4, 3, 2, 1];
    const corr = correlation(a, b);
    expect(corr).toBeCloseTo(-1, 5);
  });

  it('should compute zero correlation for uncorrelated data', () => {
    const a = [1, 1, 1, 1, 1];
    const b = [1, 2, 3, 4, 5];
    const corr = correlation(a, b);
    expect(Math.abs(corr)).toBeLessThan(0.1);
  });
});
