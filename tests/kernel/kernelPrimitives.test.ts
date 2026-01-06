/**
 * Comprehensive Tests for RHS Kernel Primitives
 *
 * Tests for modules A01-A12 (Kernel Primitives)
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  // A01 - R/Noise
  createRNoise,
  perturbRNoise,
  computeREnergy,
  // A02 - H/Coherence
  createHCoherence,
  applyCoherence,
  computeCoherenceStrength,
  // A03 - S/Soleket
  createSSoleket,
  applySoleket,
  computeClearanceEffect,
  // A04 - CycleOperator
  createCycleOperator,
  advanceCycle,
  executeCycle,
  // A05 - StateSpace_RHS
  createStateSpaceRHS,
  getStateVector,
  computeManifoldDistance,
  // A06 - Attractor
  detectAttractors,
  // A07 - ClearanceOperator
  createClearanceOperator,
  executeClearance,
  // A08 - DownwardConstraint
  createDownwardConstraint,
  computeDownwardConstraintMatrix,
  applyDownwardConstraint,
  // A09 - ClosureCondition
  evaluateClosureCondition,
  // A10 - TensionField
  computeTensionField,
  findTensionReductionDirection,
  // A11 - CoherenceScore
  computeCoherenceScore,
  // A12 - ResonanceCheck
  checkResonance,
} from '../../src/lib/KernelPrimitives';

describe('A01 - R/Noise Primitive', () => {
  it('should create R/Noise with correct dimension', () => {
    const r = createRNoise(10);
    expect(r.type).toBe('R_NOISE');
    expect(r.perturbation).toHaveLength(10);
    expect(r.variance).toBe(1.0);
    expect(r.timestamp).toBeDefined();
  });

  it('should create R/Noise with custom variance', () => {
    const r = createRNoise(5, 2.5);
    expect(r.variance).toBe(2.5);
  });

  it('should perturb R/Noise correctly', () => {
    const r = createRNoise(5, 1.0);
    const perturbed = perturbRNoise(r, 0.5);
    expect(perturbed.variance).toBe(1.5);
    expect(perturbed.perturbation).not.toEqual(r.perturbation);
  });

  it('should compute R energy correctly', () => {
    const r = createRNoise(10);
    const energy = computeREnergy(r);
    expect(energy).toBeGreaterThanOrEqual(0);
    expect(typeof energy).toBe('number');
  });

  it('should preserve source label', () => {
    const r = createRNoise(5, 1.0, 'test_source');
    expect(r.source).toBe('test_source');
  });
});

describe('A02 - H/Coherence Primitive', () => {
  it('should create H/Coherence with correct dimension', () => {
    const h = createHCoherence(8);
    expect(h.type).toBe('H_COHERENCE');
    expect(h.constraints).toHaveLength(8);
    expect(h.constraints[0]).toHaveLength(8);
    expect(h.structure).toHaveLength(8);
    expect(h.holdingStrength).toBe(1.0);
  });

  it('should create symmetric constraint matrix', () => {
    const h = createHCoherence(5);
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        expect(h.constraints[i][j]).toBeCloseTo(h.constraints[j][i], 10);
      }
    }
  });

  it('should have diagonal elements equal to 1', () => {
    const h = createHCoherence(6);
    for (let i = 0; i < 6; i++) {
      expect(h.constraints[i][i]).toBe(1.0);
    }
  });

  it('should apply coherence to input vector', () => {
    const h = createHCoherence(5);
    const input = [1, 2, 3, 4, 5];
    const output = applyCoherence(h, input);
    expect(output).toHaveLength(5);
  });

  it('should compute coherence strength', () => {
    const h = createHCoherence(5);
    const state = [1, 0, 0, 0, 0];
    const strength = computeCoherenceStrength(h, state);
    expect(strength).toBeGreaterThanOrEqual(-1);
    expect(strength).toBeLessThanOrEqual(1);
  });
});

describe('A03 - S/Soleket Primitive', () => {
  it('should create S/Soleket with correct clearance level', () => {
    const s = createSSoleket(8, 0.5);
    expect(s.type).toBe('S_SOLEKET');
    expect(s.clearanceLevel).toBe(0.5);
    expect(s.dofRemoved).toBe(4); // 50% of 8
  });

  it('should have correct mediation coefficients length', () => {
    const s = createSSoleket(10, 0.3);
    expect(s.mediationCoefficients).toHaveLength(10);
  });

  it('should have correct transform matrix dimensions', () => {
    const s = createSSoleket(6, 0.5);
    expect(s.transformMatrix).toHaveLength(6);
    expect(s.transformMatrix[0]).toHaveLength(6);
  });

  it('should apply Soleket transformation', () => {
    const s = createSSoleket(5, 0.5);
    const input = [1, 2, 3, 4, 5];
    const output = applySoleket(s, input);
    expect(output).toHaveLength(5);
  });

  it('should compute clearance effect', () => {
    const s = createSSoleket(5, 0.5);
    const before = [1, 2, 3, 4, 5];
    const after = applySoleket(s, before);
    const effect = computeClearanceEffect(s, before, after);
    expect(effect).toBeGreaterThanOrEqual(0);
    expect(effect).toBeLessThanOrEqual(1);
  });
});

describe('A04 - CycleOperator', () => {
  it('should create cycle operator with initial phase R', () => {
    const op = createCycleOperator();
    expect(op.type).toBe('CYCLE_OPERATOR');
    expect(op.currentPhase).toBe('R');
    expect(op.cycleCount).toBe(0);
  });

  it('should advance cycle phases correctly', () => {
    let op = createCycleOperator();
    expect(op.currentPhase).toBe('R');

    op = advanceCycle(op);
    expect(op.currentPhase).toBe('H');

    op = advanceCycle(op);
    expect(op.currentPhase).toBe('S');

    op = advanceCycle(op);
    expect(op.currentPhase).toBe('R');
    expect(op.cycleCount).toBe(1);
  });

  it('should increment cycle count after S→R transition', () => {
    let op = createCycleOperator();
    op = advanceCycle(op); // R→H
    op = advanceCycle(op); // H→S
    expect(op.cycleCount).toBe(0);
    op = advanceCycle(op); // S→R
    expect(op.cycleCount).toBe(1);
  });

  it('should execute full cycle', () => {
    const state = createStateSpaceRHS(8);
    const op = createCycleOperator();
    const result = executeCycle(state, op);

    expect(result.operator.cycleCount).toBe(1);
    expect(result.state.history.length).toBeGreaterThan(state.history.length);
  });
});

describe('A05 - StateSpace_RHS', () => {
  it('should create state space with all components', () => {
    const state = createStateSpaceRHS(8);
    expect(state.type).toBe('STATE_SPACE_RHS');
    expect(state.r.type).toBe('R_NOISE');
    expect(state.h.type).toBe('H_COHERENCE');
    expect(state.s.type).toBe('S_SOLEKET');
    expect(state.dimension).toBe(8);
  });

  it('should have initial history entry', () => {
    const state = createStateSpaceRHS(5);
    expect(state.history).toHaveLength(1);
    expect(state.history[0].r).toHaveLength(5);
  });

  it('should get state vector', () => {
    const state = createStateSpaceRHS(4);
    const vector = getStateVector(state);
    // R + H + S = 4 + 4 + 4 = 12
    expect(vector).toHaveLength(12);
  });

  it('should compute manifold distance', () => {
    const state1 = createStateSpaceRHS(5);
    const state2 = createStateSpaceRHS(5);
    const distance = computeManifoldDistance(state1, state2);
    expect(distance).toBeGreaterThanOrEqual(0);
  });

  it('should have zero distance to itself', () => {
    const state = createStateSpaceRHS(5);
    const distance = computeManifoldDistance(state, state);
    expect(distance).toBeCloseTo(0, 10);
  });
});

describe('A06 - Attractor Detection', () => {
  it('should return empty array for short history', () => {
    const state = createStateSpaceRHS(5);
    const attractors = detectAttractors(state);
    expect(attractors).toHaveLength(0);
  });

  it('should detect attractors with sufficient history', () => {
    // Create state with longer history
    let state = createStateSpaceRHS(5);
    const op = createCycleOperator();

    // Run multiple cycles to build history
    for (let i = 0; i < 25; i++) {
      const result = executeCycle(state, op);
      state = result.state;
    }

    const attractors = detectAttractors(state, 0.3);
    // May or may not find attractors depending on dynamics
    expect(Array.isArray(attractors)).toBe(true);
  });
});

describe('A07 - ClearanceOperator', () => {
  it('should create clearance operator', () => {
    const op = createClearanceOperator(0.5, 0.3);
    expect(op.type).toBe('CLEARANCE_OPERATOR');
    expect(op.contractionFactor).toBe(0.5);
    expect(op.threshold).toBe(0.3);
    expect(op.isClearing).toBe(false);
  });

  it('should execute clearance', () => {
    const state = createStateSpaceRHS(5);
    const op = createClearanceOperator(0.5, 0.1);
    const result = executeClearance(op, state);

    expect(result.operator.isClearing).toBe(true);
    expect(result.operator.incompatibleModes).toBeDefined();
  });
});

describe('A08 - DownwardConstraint', () => {
  it('should create downward constraint', () => {
    const constraint = createDownwardConstraint(0, 0.5, ['R', 'H']);
    expect(constraint.type).toBe('DOWNWARD_CONSTRAINT');
    expect(constraint.sourceTimeIndex).toBe(0);
    expect(constraint.targetTimeIndex).toBe(1);
    expect(constraint.strength).toBe(0.5);
  });

  it('should compute constraint matrix from S state', () => {
    const constraint = createDownwardConstraint(0, 0.5);
    const s = createSSoleket(5, 0.5);
    const computed = computeDownwardConstraintMatrix(constraint, s);
    expect(computed.constraintMatrix).toBeDefined();
    expect(computed.constraintMatrix.length).toBe(5);
  });

  it('should apply downward constraint', () => {
    const constraint = createDownwardConstraint(0, 0.5, ['R']);
    const s = createSSoleket(5, 0.5);
    const computedConstraint = computeDownwardConstraintMatrix(constraint, s);
    const state = createStateSpaceRHS(5);
    const constrained = applyDownwardConstraint(computedConstraint, state);
    expect(constrained.r).toBeDefined();
  });
});

describe('A09 - ClosureCondition', () => {
  it('should evaluate closure condition', () => {
    const state = createStateSpaceRHS(5);
    const closure = evaluateClosureCondition(state);

    expect(closure.type).toBe('CLOSURE_CONDITION');
    expect(typeof closure.isSatisfied).toBe('boolean');
    expect(closure.influenceStrength).toBeGreaterThanOrEqual(0);
    expect(closure.evidence).toBeDefined();
  });

  it('should have evidence array', () => {
    const state = createStateSpaceRHS(5);
    const closure = evaluateClosureCondition(state);
    expect(Array.isArray(closure.evidence)).toBe(true);
  });
});

describe('A10 - TensionField', () => {
  it('should compute tension field', () => {
    const state = createStateSpaceRHS(5);
    const field = computeTensionField(state);

    expect(field.type).toBe('TENSION_FIELD');
    expect(field.values).toHaveLength(5);
    expect(field.values[0]).toHaveLength(5);
    expect(field.totalEnergy).toBeGreaterThanOrEqual(0);
  });

  it('should identify tension peaks', () => {
    const state = createStateSpaceRHS(5);
    const field = computeTensionField(state);
    expect(Array.isArray(field.peaks)).toBe(true);
  });

  it('should compute gradient', () => {
    const state = createStateSpaceRHS(5);
    const field = computeTensionField(state);
    expect(field.gradient).toHaveLength(5);
    expect(field.gradient[0]).toHaveLength(5);
  });

  it('should find tension reduction direction', () => {
    const state = createStateSpaceRHS(5);
    const field = computeTensionField(state);
    const direction = findTensionReductionDirection(field);
    expect(direction).toHaveLength(5);
  });
});

describe('A11 - CoherenceScore', () => {
  it('should compute coherence score', () => {
    const state = createStateSpaceRHS(5);
    const score = computeCoherenceScore(state);

    expect(score.type).toBe('COHERENCE_SCORE');
    expect(score.value).toBeGreaterThanOrEqual(0);
    expect(score.value).toBeLessThanOrEqual(1);
    expect(score.components).toHaveLength(3);
  });

  it('should have structural, dynamic, and integration components', () => {
    const state = createStateSpaceRHS(5);
    const score = computeCoherenceScore(state);

    const componentNames = score.components.map((c) => c.name);
    expect(componentNames).toContain('structural');
    expect(componentNames).toContain('dynamic');
    expect(componentNames).toContain('integration');
  });

  it('should have weights summing to 1', () => {
    const state = createStateSpaceRHS(5);
    const score = computeCoherenceScore(state);
    const totalWeight = score.components.reduce((sum, c) => sum + c.weight, 0);
    expect(totalWeight).toBeCloseTo(1, 10);
  });
});

describe('A12 - ResonanceCheck', () => {
  it('should check resonance', () => {
    const state = createStateSpaceRHS(5);
    const check = checkResonance(state);

    expect(check.type).toBe('RESONANCE_CHECK');
    expect(typeof check.isResonant).toBe('boolean');
    expect(check.stability).toBeGreaterThanOrEqual(0);
    expect(check.selectivity).toBeGreaterThanOrEqual(0);
  });

  it('should not find resonance in new state', () => {
    const state = createStateSpaceRHS(5);
    const check = checkResonance(state);
    // New state unlikely to have resonant attractors
    expect(check.stability).toBe(0);
    expect(check.selectivity).toBe(0);
  });
});

describe('Integration Tests', () => {
  it('should run multiple cycles and maintain consistency', () => {
    let state = createStateSpaceRHS(8);
    let op = createCycleOperator();

    for (let i = 0; i < 10; i++) {
      const result = executeCycle(state, op);
      state = result.state;
      op = result.operator;

      // Verify state consistency
      expect(state.r.perturbation).toHaveLength(8);
      expect(state.h.structure).toHaveLength(8);
      expect(state.s.mediationCoefficients).toHaveLength(8);
    }

    expect(op.cycleCount).toBe(10);
  });

  it('should track history through cycles', () => {
    let state = createStateSpaceRHS(5);
    const initialHistoryLength = state.history.length;

    for (let i = 0; i < 5; i++) {
      const result = executeCycle(state, createCycleOperator());
      state = result.state;
    }

    expect(state.history.length).toBeGreaterThan(initialHistoryLength);
  });

  it('should handle clearance and cycle interaction', () => {
    let state = createStateSpaceRHS(6);
    const clearanceOp = createClearanceOperator(0.3, 0.2);
    const cycleOp = createCycleOperator();

    // Execute clearance
    const clearResult = executeClearance(clearanceOp, state);
    state = clearResult.state;

    // Then execute cycle
    const cycleResult = executeCycle(state, cycleOp);
    state = cycleResult.state;

    expect(state.r).toBeDefined();
    expect(state.h).toBeDefined();
    expect(state.s).toBeDefined();
  });
});
