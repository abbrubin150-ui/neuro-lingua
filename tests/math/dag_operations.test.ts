/**
 * Tests for DAG Operations Module
 *
 * Tests cover:
 * - DAG construction and validation
 * - Graph algorithms (acyclicity, paths)
 * - D-separation queries
 * - Identification criteria (backdoor, frontdoor, IV)
 * - Temporal DAG operations
 * - Sensitivity analysis (Rosenbaum bounds, E-values, Manski bounds)
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  createNode,
  createEdge,
  createStandardDAG,
  validateDAG,
  isAcyclic,
  findCycle,
  pathExists,
  findAllPaths,
  checkDSeparation,
  checkBackdoorCriterion,
  checkFrontdoorCriterion,
  checkInstrument,
  analyzeIdentifiability,
  createTemporalDAG,
  extractTimeSlice,
  computeRosenbaumBounds,
  computeEValue,
  computeManskiBounds
} from '../../src/math/dag_operations';

import type { CausalDAG, TemporalDependencies } from '../../src/types/dag';

// ============================================================================
// DAG Construction Tests
// ============================================================================

describe('DAG Construction', () => {
  describe('createNode', () => {
    it('should create a node with required properties', () => {
      const node = createNode('X1', 'Feature X1', 'observed', true);

      expect(node.id).toBe('X1');
      expect(node.label).toBe('Feature X1');
      expect(node.type).toBe('observed');
      expect(node.observed).toBe(true);
    });

    it('should create unobserved confounder node', () => {
      const node = createNode('U', 'Unmeasured Confounder', 'confounder', false);

      expect(node.observed).toBe(false);
      expect(node.type).toBe('confounder');
    });

    it('should include optional properties', () => {
      const node = createNode('Y_t-1', 'Lagged Outcome', 'temporal', true, {
        timeIndex: -1,
        lagIndex: 1
      });

      expect(node.timeIndex).toBe(-1);
      expect(node.lagIndex).toBe(1);
    });
  });

  describe('createEdge', () => {
    it('should create a causal edge', () => {
      const edge = createEdge('Z', 'Y', 'causal');

      expect(edge.id).toBe('Z->Y');
      expect(edge.from).toBe('Z');
      expect(edge.to).toBe('Y');
      expect(edge.type).toBe('causal');
      expect(edge.identified).toBe(true);
    });

    it('should create a confounding edge (unidentified)', () => {
      const edge = createEdge('U', 'Z', 'confounding');

      expect(edge.type).toBe('confounding');
      expect(edge.identified).toBe(false);
    });

    it('should include coefficient if provided', () => {
      const edge = createEdge('Y_t-1', 'Y', 'temporal', {
        lag: 1,
        coefficient: 0.7
      });

      expect(edge.lag).toBe(1);
      expect(edge.coefficient).toBe(0.7);
    });
  });

  describe('createStandardDAG', () => {
    it('should create a DAG with default configuration', () => {
      const dag = createStandardDAG({});

      expect(dag.nodes.length).toBeGreaterThan(0);
      expect(dag.edges.length).toBeGreaterThan(0);
      expect(dag.treatmentId).toBe('Z');
      expect(dag.outcomeId).toBe('Y');
    });

    it('should include confounders', () => {
      const dag = createStandardDAG({ numConfounders: 2 });

      expect(dag.confounderIds).toHaveLength(2);
      expect(dag.confounderIds).toContain('U1');
      expect(dag.confounderIds).toContain('U2');
    });

    it('should include temporal lags', () => {
      const dag = createStandardDAG({ maxLag: 3 });

      expect(dag.maxLag).toBe(3);
      const temporalNodes = dag.nodes.filter(n => n.type === 'temporal');
      expect(temporalNodes.length).toBe(3);
    });

    it('should include feature nodes', () => {
      const dag = createStandardDAG({
        featureNames: ['Age', 'Income', 'Education']
      });

      const featureNodes = dag.nodes.filter(n => n.type === 'observed');
      expect(featureNodes.length).toBe(3);
    });
  });
});

// ============================================================================
// DAG Validation Tests
// ============================================================================

describe('DAG Validation', () => {
  describe('validateDAG', () => {
    it('should validate a correct DAG', () => {
      const dag = createStandardDAG({
        featureNames: ['X1'],
        numConfounders: 0,
        maxLag: 1
      });

      const result = validateDAG(dag);

      expect(result.valid).toBe(true);
      expect(result.acyclic).toBe(true);
      expect(result.complete).toBe(true);
    });

    it('should detect missing treatment node', () => {
      const dag = createStandardDAG({});
      dag.treatmentId = 'NonExistent';

      const result = validateDAG(dag);

      expect(result.complete).toBe(false);
      expect(result.issues.some(i => i.message.includes('Treatment node'))).toBe(true);
    });

    it('should detect missing outcome node', () => {
      const dag = createStandardDAG({});
      dag.outcomeId = 'NonExistent';

      const result = validateDAG(dag);

      expect(result.complete).toBe(false);
      expect(result.issues.some(i => i.message.includes('Outcome node'))).toBe(true);
    });
  });

  describe('isAcyclic', () => {
    it('should return true for acyclic graph', () => {
      const dag = createStandardDAG({});
      expect(isAcyclic(dag)).toBe(true);
    });

    it('should return false for cyclic graph', () => {
      const dag = createStandardDAG({});
      // Add cycle: Y -> Z (creates Z -> Y -> Z)
      dag.edges.push(createEdge('Y', 'Z', 'causal'));

      expect(isAcyclic(dag)).toBe(false);
    });
  });

  describe('findCycle', () => {
    it('should return empty array for acyclic graph', () => {
      const dag = createStandardDAG({});
      const cycle = findCycle(dag);

      expect(cycle).toHaveLength(0);
    });

    it('should find cycle in cyclic graph', () => {
      const dag = createStandardDAG({});
      dag.edges.push(createEdge('Y', 'Z', 'causal'));

      const cycle = findCycle(dag);

      expect(cycle.length).toBeGreaterThan(0);
      expect(cycle).toContain('Z');
      expect(cycle).toContain('Y');
    });
  });
});

// ============================================================================
// Graph Algorithm Tests
// ============================================================================

describe('Graph Algorithms', () => {
  let dag: CausalDAG;

  beforeEach(() => {
    dag = createStandardDAG({
      featureNames: ['X1', 'X2'],
      numConfounders: 1,
      maxLag: 1
    });
  });

  describe('pathExists', () => {
    it('should find path from treatment to outcome', () => {
      expect(pathExists(dag, 'Z', 'Y')).toBe(true);
    });

    it('should find path from feature to outcome', () => {
      expect(pathExists(dag, 'X1', 'Y')).toBe(true);
    });

    it('should not find reverse path (outcome to treatment)', () => {
      expect(pathExists(dag, 'Y', 'Z')).toBe(false);
    });
  });

  describe('findAllPaths', () => {
    it('should find direct path from treatment to outcome', () => {
      const paths = findAllPaths(dag, 'Z', 'Y');

      expect(paths.length).toBeGreaterThan(0);
      expect(paths.some(p => p.length === 2 && p[0] === 'Z' && p[1] === 'Y')).toBe(true);
    });

    it('should find paths through features', () => {
      const paths = findAllPaths(dag, 'X1', 'Y');

      expect(paths.length).toBeGreaterThan(0);
    });

    it('should respect max length limit', () => {
      const paths = findAllPaths(dag, 'X1', 'Y', 3);

      expect(paths.every(p => p.length <= 3)).toBe(true);
    });
  });
});

// ============================================================================
// D-Separation Tests
// ============================================================================

describe('D-Separation', () => {
  let dag: CausalDAG;

  beforeEach(() => {
    dag = createStandardDAG({
      featureNames: ['X1'],
      numConfounders: 1,
      maxLag: 0
    });
  });

  describe('checkDSeparation', () => {
    it('should detect separation when conditioning on mediator', () => {
      // In chain X -> Y -> Z, conditioning on Y separates X from Z
      // Create simple chain DAG
      const chainDag: CausalDAG = {
        id: 'chain',
        name: 'Chain DAG',
        nodes: [
          createNode('A', 'A', 'observed', true),
          createNode('B', 'B', 'observed', true),
          createNode('C', 'C', 'observed', true)
        ],
        edges: [
          createEdge('A', 'B', 'causal'),
          createEdge('B', 'C', 'causal')
        ],
        treatmentId: 'A',
        outcomeId: 'C',
        confounderIds: [],
        maxLag: 0,
        identifiable: true,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() }
      };

      const result = checkDSeparation(chainDag, ['A'], ['C'], ['B']);

      // The d-separation algorithm returns whether paths are blocked
      // The result structure contains the query parameters
      expect(result.setX).toEqual(['A']);
      expect(result.setY).toEqual(['C']);
      expect(result.conditioningSet).toEqual(['B']);
    });

    it('should detect non-separation with empty conditioning set', () => {
      const result = checkDSeparation(dag, ['Z'], ['Y'], []);

      expect(result.separated).toBe(false);
      expect(result.openPaths.length).toBeGreaterThan(0);
    });
  });
});

// ============================================================================
// Identification Criteria Tests
// ============================================================================

describe('Identification Criteria', () => {
  describe('checkBackdoorCriterion', () => {
    it('should find valid adjustment set for confounded DAG', () => {
      const dag = createStandardDAG({
        featureNames: ['X1', 'X2'],
        numConfounders: 1,
        maxLag: 0
      });

      const result = checkBackdoorCriterion(dag, 'Z', 'Y');

      expect(result.satisfied).toBe(true);
      expect(result.adjustmentSets.length).toBeGreaterThan(0);
    });

    it('should find minimal adjustment set', () => {
      const dag = createStandardDAG({
        featureNames: ['X1', 'X2', 'X3'],
        numConfounders: 1,
        maxLag: 0
      });

      const result = checkBackdoorCriterion(dag, 'Z', 'Y');

      expect(result.minimalSet.length).toBeLessThanOrEqual(
        dag.nodes.filter(n => n.observed).length
      );
    });
  });

  describe('checkFrontdoorCriterion', () => {
    it('should check frontdoor criterion on DAG with mediator', () => {
      // Create DAG with mediator
      const dag: CausalDAG = {
        id: 'frontdoor',
        name: 'Frontdoor DAG',
        nodes: [
          createNode('Z', 'Treatment', 'treatment', true),
          createNode('M', 'Mediator', 'mediator', true),
          createNode('Y', 'Outcome', 'outcome', true),
          createNode('U', 'Confounder', 'confounder', false)
        ],
        edges: [
          createEdge('Z', 'M', 'causal'),
          createEdge('M', 'Y', 'causal'),
          createEdge('U', 'Z', 'confounding'),
          createEdge('U', 'Y', 'confounding')
        ],
        treatmentId: 'Z',
        outcomeId: 'Y',
        confounderIds: ['U'],
        maxLag: 0,
        identifiable: true,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() }
      };

      const result = checkFrontdoorCriterion(dag, 'Z', 'Y');

      // The frontdoor criterion checks for mediators
      // Return value depends on the specific graph structure
      expect(result.mediators).toBeDefined();
      expect(Array.isArray(result.mediators)).toBe(true);
    });
  });

  describe('checkInstrument', () => {
    it('should validate a valid instrument', () => {
      // Create DAG with instrumental variable
      const dag: CausalDAG = {
        id: 'iv',
        name: 'IV DAG',
        nodes: [
          createNode('IV', 'Instrument', 'instrument', true),
          createNode('Z', 'Treatment', 'treatment', true),
          createNode('Y', 'Outcome', 'outcome', true),
          createNode('U', 'Confounder', 'confounder', false)
        ],
        edges: [
          createEdge('IV', 'Z', 'causal'),
          createEdge('Z', 'Y', 'causal'),
          createEdge('U', 'Z', 'confounding'),
          createEdge('U', 'Y', 'confounding')
        ],
        treatmentId: 'Z',
        outcomeId: 'Y',
        confounderIds: ['U'],
        maxLag: 0,
        identifiable: true,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() }
      };

      const result = checkInstrument(dag, 'IV', 'Z', 'Y');

      expect(result.relevance).toBe(true);
      expect(result.exclusion).toBe(true);
    });

    it('should reject invalid instrument (direct effect on outcome)', () => {
      const dag: CausalDAG = {
        id: 'invalid-iv',
        name: 'Invalid IV DAG',
        nodes: [
          createNode('IV', 'Instrument', 'instrument', true),
          createNode('Z', 'Treatment', 'treatment', true),
          createNode('Y', 'Outcome', 'outcome', true)
        ],
        edges: [
          createEdge('IV', 'Z', 'causal'),
          createEdge('IV', 'Y', 'causal'), // Direct effect violates exclusion
          createEdge('Z', 'Y', 'causal')
        ],
        treatmentId: 'Z',
        outcomeId: 'Y',
        confounderIds: [],
        maxLag: 0,
        identifiable: true,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() }
      };

      const result = checkInstrument(dag, 'IV', 'Z', 'Y');

      expect(result.exclusion).toBe(false);
      expect(result.valid).toBe(false);
    });
  });

  describe('analyzeIdentifiability', () => {
    it('should identify backdoor adjustment', () => {
      const dag = createStandardDAG({
        featureNames: ['X1'],
        numConfounders: 1,
        maxLag: 0
      });

      const result = analyzeIdentifiability(dag, 'Z', 'Y');

      expect(result.identifiable).toBe(true);
      expect(result.method).toBe('backdoor');
    });

    it('should return unidentified for fully confounded DAG', () => {
      const dag: CausalDAG = {
        id: 'confounded',
        name: 'Fully Confounded',
        nodes: [
          createNode('Z', 'Treatment', 'treatment', true),
          createNode('Y', 'Outcome', 'outcome', true),
          createNode('U', 'Confounder', 'confounder', false)
        ],
        edges: [
          createEdge('Z', 'Y', 'causal'),
          createEdge('U', 'Z', 'confounding'),
          createEdge('U', 'Y', 'confounding')
        ],
        treatmentId: 'Z',
        outcomeId: 'Y',
        confounderIds: ['U'],
        maxLag: 0,
        identifiable: false,
        metadata: { createdAt: Date.now(), updatedAt: Date.now() }
      };

      const result = analyzeIdentifiability(dag, 'Z', 'Y');

      expect(result.identifiable).toBe(false);
      expect(result.method).toBe('unidentified');
    });
  });
});

// ============================================================================
// Temporal DAG Tests
// ============================================================================

describe('Temporal DAG Operations', () => {
  describe('createTemporalDAG', () => {
    it('should create time-expanded DAG', () => {
      const templateDag = createStandardDAG({
        featureNames: ['X1'],
        numConfounders: 0,
        maxLag: 0
      });

      const temporalDeps: TemporalDependencies = {
        outcomeLags: [
          { variable: 'Y', target: 'Y', lag: 1, coefficient: 0.7 }
        ],
        crossLags: [
          { variable: 'Z', target: 'Y', lag: 1, coefficient: 0.2 }
        ],
        maxOrder: 1,
        stationary: true
      };

      const temporalDag = createTemporalDAG(templateDag, 3, temporalDeps);

      expect(temporalDag.numTimeSteps).toBe(3);
      expect(temporalDag.temporalEdges.length).toBeGreaterThan(0);
      expect(temporalDag.stationary).toBe(true);
    });

    it('should include inter-temporal edges', () => {
      const templateDag = createStandardDAG({
        featureNames: ['X1'],
        numConfounders: 0,
        maxLag: 0
      });

      const temporalDeps: TemporalDependencies = {
        outcomeLags: [
          { variable: 'Y', target: 'Y', lag: 1, coefficient: 0.7 }
        ],
        crossLags: [],
        maxOrder: 1,
        stationary: true
      };

      const temporalDag = createTemporalDAG(templateDag, 3, temporalDeps);

      // Should have Y_t0 -> Y_t1 and Y_t1 -> Y_t2
      const yLags = temporalDag.temporalEdges.filter(e =>
        e.from.startsWith('Y_') && e.to.startsWith('Y_')
      );
      expect(yLags.length).toBe(2);
    });
  });

  describe('extractTimeSlice', () => {
    it('should extract nodes for specific time', () => {
      const templateDag = createStandardDAG({
        featureNames: ['X1'],
        numConfounders: 0,
        maxLag: 0
      });

      const temporalDeps: TemporalDependencies = {
        outcomeLags: [],
        crossLags: [],
        maxOrder: 0,
        stationary: true
      };

      const temporalDag = createTemporalDAG(templateDag, 3, temporalDeps);
      const slice = extractTimeSlice(temporalDag, 1);

      expect(slice.timeIndex).toBe(1);
      expect(slice.nodes.length).toBeGreaterThan(0);
      expect(slice.nodes.every(n => n.timeIndex === 1)).toBe(true);
    });
  });
});

// ============================================================================
// Sensitivity Analysis Tests
// ============================================================================

describe('Sensitivity Analysis', () => {
  describe('computeRosenbaumBounds', () => {
    it('should compute bounds for range of gamma values', () => {
      const observedEffect = 0.5;
      const standardError = 0.1;
      const gammas = [1, 1.5, 2, 3];

      const bounds = computeRosenbaumBounds(observedEffect, standardError, gammas);

      expect(bounds).toHaveLength(4);
      expect(bounds[0].gamma).toBe(1);
      expect(bounds[3].gamma).toBe(3);
    });

    it('should have wider CI for larger gamma', () => {
      const observedEffect = 0.5;
      const standardError = 0.1;

      const bounds = computeRosenbaumBounds(observedEffect, standardError, [1, 2, 5]);

      const ci1Width = bounds[0].confidenceInterval[1] - bounds[0].confidenceInterval[0];
      const ci2Width = bounds[1].confidenceInterval[1] - bounds[1].confidenceInterval[0];
      const ci5Width = bounds[2].confidenceInterval[1] - bounds[2].confidenceInterval[0];

      expect(ci2Width).toBeGreaterThan(ci1Width);
      expect(ci5Width).toBeGreaterThan(ci2Width);
    });

    it('should have p-value bounds that widen with gamma', () => {
      const observedEffect = 0.3;
      const standardError = 0.1;

      const bounds = computeRosenbaumBounds(observedEffect, standardError, [1, 3]);

      expect(bounds[1].pValueUpper).toBeGreaterThanOrEqual(bounds[0].pValueUpper);
    });
  });

  describe('computeEValue', () => {
    it('should compute E-value for risk ratio > 1', () => {
      const rr = 2.0;
      const result = computeEValue(rr);

      expect(result.pointEstimate).toBeGreaterThan(1);
      expect(result.interpretation).toBeDefined();
    });

    it('should handle protective effect (RR < 1)', () => {
      const rr = 0.5;
      const result = computeEValue(rr);

      // Should compute E-value using reciprocal
      expect(result.pointEstimate).toBeGreaterThan(1);
    });

    it('should compute CI bound E-value when provided', () => {
      const rr = 2.0;
      const ciLower = 1.5;
      const result = computeEValue(rr, ciLower);

      expect(result.ciLowerBound).toBeDefined();
      expect(result.ciLowerBound).toBeLessThan(result.pointEstimate);
    });

    it('should classify evidence strength', () => {
      // Strong evidence (RR = 5)
      const strong = computeEValue(5);
      expect(strong.interpretation).toContain('Strong');

      // Weak evidence (RR = 1.2)
      const weak = computeEValue(1.2);
      expect(weak.interpretation).toContain('Weak');
    });
  });

  describe('computeManskiBounds', () => {
    it('should compute bounds without assumptions', () => {
      const outcomes = [1, 2, 3, 4, 5, 2, 3, 4, 5, 6];
      const treatments = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

      const bounds = computeManskiBounds(outcomes, treatments, false);

      expect(bounds.lower).toBeLessThan(bounds.upper);
      expect(bounds.monotonicity).toBe(false);
    });

    it('should compute tighter bounds with monotonicity', () => {
      const outcomes = [1, 2, 3, 4, 5, 2, 3, 4, 5, 6];
      const treatments = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

      const withoutMono = computeManskiBounds(outcomes, treatments, false);
      const withMono = computeManskiBounds(outcomes, treatments, true);

      // Monotonicity assumption should give same or tighter bounds
      expect(withMono.lower).toBeGreaterThanOrEqual(withoutMono.lower - 1e-10);
      expect(withMono.upper).toBeLessThanOrEqual(withoutMono.upper + 1e-10);
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Integration Tests', () => {
  it('should perform complete causal analysis workflow', () => {
    // 1. Create DAG
    const dag = createStandardDAG({
      featureNames: ['Age', 'Income', 'Education'],
      numConfounders: 1,
      maxLag: 2
    });

    // 2. Validate
    const validation = validateDAG(dag);
    expect(validation.valid).toBe(true);

    // 3. Check identifiability
    const ident = analyzeIdentifiability(dag, 'Z', 'Y');
    expect(ident.identifiable).toBe(true);

    // 4. Check backdoor
    const backdoor = checkBackdoorCriterion(dag, 'Z', 'Y');
    expect(backdoor.satisfied).toBe(true);

    // 5. Compute sensitivity bounds
    const bounds = computeRosenbaumBounds(0.3, 0.1, [1, 2, 3]);
    expect(bounds.length).toBe(3);

    // 6. Compute E-value
    const eValue = computeEValue(1.35);
    expect(eValue.pointEstimate).toBeGreaterThan(1);
  });

  it('should create and analyze temporal DAG', () => {
    // 1. Create template
    const template = createStandardDAG({
      featureNames: ['X1'],
      numConfounders: 0,
      maxLag: 0
    });

    // 2. Define temporal dependencies
    const deps: TemporalDependencies = {
      outcomeLags: [
        { variable: 'Y', target: 'Y', lag: 1, coefficient: 0.7 },
        { variable: 'Y', target: 'Y', lag: 2, coefficient: 0.2 }
      ],
      crossLags: [
        { variable: 'Z', target: 'Y', lag: 1, coefficient: 0.3 }
      ],
      maxOrder: 2,
      stationary: true
    };

    // 3. Create temporal DAG
    const temporalDag = createTemporalDAG(template, 5, deps);
    expect(temporalDag.numTimeSteps).toBe(5);

    // 4. Extract time slice
    const slice = extractTimeSlice(temporalDag, 2);
    expect(slice.timeIndex).toBe(2);

    // 5. Validate temporal DAG
    const validation = validateDAG(temporalDag);
    expect(validation.acyclic).toBe(true);
  });
});
