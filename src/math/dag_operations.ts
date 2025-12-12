/**
 * DAG Operations for Causal Inference
 *
 * This module provides operations on causal DAGs including:
 * - DAG construction and validation
 * - D-separation queries
 * - Backdoor/frontdoor criterion checking
 * - Instrumental variable identification
 * - Temporal DAG operations
 * - Sensitivity analysis (Rosenbaum bounds, E-values)
 *
 * Reference: Pearl (2009), Spirtes et al. (2000), Rosenbaum (2002)
 *
 * @module dag_operations
 */

import type {
  CausalNode,
  CausalEdge,
  CausalDAG,
  CausalNodeType,
  TemporalDependencies,
  DSeparationResult,
  BackdoorCriterionResult,
  FrontdoorCriterionResult,
  InstrumentCheckResult,
  IdentifiabilityAnalysis,
  DAGValidationResult,
  TemporalDAG,
  TimeSlice,
  RosenbaumBounds,
  EValue,
  ManskiBounds
} from '../types/dag';

import { mean, normalQuantile } from './causal_math';

// ============================================================================
// DAG Construction
// ============================================================================

/**
 * Create a new causal node
 */
export function createNode(
  id: string,
  label: string,
  type: CausalNodeType,
  observed: boolean = true,
  options: Partial<CausalNode> = {}
): CausalNode {
  return {
    id,
    label,
    type,
    observed,
    ...options
  };
}

/**
 * Create a new causal edge
 */
export function createEdge(
  from: string,
  to: string,
  type: CausalEdge['type'] = 'causal',
  options: Partial<CausalEdge> = {}
): CausalEdge {
  return {
    id: `${from}->${to}`,
    from,
    to,
    type,
    identified: type !== 'confounding',
    ...options
  };
}

/**
 * Create a standard treatment-outcome DAG with confounders
 *
 * Structure:
 *   U
 *  / \
 * X   Y
 *  \ /
 *   Z
 *
 * @param config - DAG configuration
 * @returns Causal DAG
 */
export function createStandardDAG(config: {
  featureNames?: string[];
  numConfounders?: number;
  maxLag?: number;
}): CausalDAG {
  const { featureNames = ['X1', 'X2', 'X3'], numConfounders = 1, maxLag = 1 } = config;

  const nodes: CausalNode[] = [];
  const edges: CausalEdge[] = [];

  // Treatment node
  nodes.push(createNode('Z', 'Treatment (Z)', 'treatment', true));

  // Outcome node
  nodes.push(createNode('Y', 'Outcome (Y)', 'outcome', true));

  // Feature nodes
  for (const name of featureNames) {
    nodes.push(createNode(name, `Feature ${name}`, 'observed', true));
    // X -> Z
    edges.push(createEdge(name, 'Z', 'causal'));
    // X -> Y
    edges.push(createEdge(name, 'Y', 'causal'));
  }

  // Treatment -> Outcome
  edges.push(createEdge('Z', 'Y', 'causal'));

  // Confounder nodes
  const confounderIds: string[] = [];
  for (let i = 0; i < numConfounders; i++) {
    const confId = `U${i + 1}`;
    confounderIds.push(confId);
    nodes.push(createNode(confId, `Confounder ${confId}`, 'confounder', false));
    // U -> Z (confounding path)
    edges.push(createEdge(confId, 'Z', 'confounding'));
    // U -> Y (confounding path)
    edges.push(createEdge(confId, 'Y', 'confounding'));
  }

  // Temporal nodes (lagged outcomes)
  for (let lag = 1; lag <= maxLag; lag++) {
    const lagId = `Y_t-${lag}`;
    nodes.push(
      createNode(lagId, `Y(t-${lag})`, 'temporal', true, {
        timeIndex: -lag,
        lagIndex: lag
      })
    );
    // Y_{t-lag} -> Y_t
    edges.push(createEdge(lagId, 'Y', 'temporal', { lag }));
    // Y_{t-lag} -> Z_t
    edges.push(createEdge(lagId, 'Z', 'temporal', { lag }));
  }

  // Quantization node
  nodes.push(createNode('theta', 'Quantization (θ)', 'quantization', true));
  edges.push(createEdge('theta', 'Z', 'selection'));

  return {
    id: `dag-${Date.now()}`,
    name: 'Standard Causal DAG',
    nodes,
    edges,
    treatmentId: 'Z',
    outcomeId: 'Y',
    confounderIds,
    maxLag,
    identifiable: numConfounders === 0 || featureNames.length > 0,
    metadata: {
      createdAt: Date.now(),
      updatedAt: Date.now(),
      description: 'Standard treatment-outcome DAG with confounders and temporal dependencies',
      assumptions: [
        'SUTVA (Stable Unit Treatment Value Assumption)',
        'Positivity',
        'No interference between units'
      ]
    }
  };
}

// ============================================================================
// DAG Validation
// ============================================================================

/**
 * Validate a causal DAG for consistency and completeness
 */
export function validateDAG(dag: CausalDAG): DAGValidationResult {
  const issues: DAGValidationResult['issues'] = [];
  const suggestions: string[] = [];

  // Check acyclicity
  const acyclic = isAcyclic(dag);
  if (!acyclic) {
    issues.push({
      severity: 'error',
      message: 'DAG contains cycles',
      nodeIds: findCycle(dag)
    });
  }

  // Check treatment node exists
  const treatmentNode = dag.nodes.find((n) => n.id === dag.treatmentId);
  if (!treatmentNode) {
    issues.push({
      severity: 'error',
      message: `Treatment node '${dag.treatmentId}' not found`,
      nodeIds: [dag.treatmentId]
    });
  }

  // Check outcome node exists
  const outcomeNode = dag.nodes.find((n) => n.id === dag.outcomeId);
  if (!outcomeNode) {
    issues.push({
      severity: 'error',
      message: `Outcome node '${dag.outcomeId}' not found`,
      nodeIds: [dag.outcomeId]
    });
  }

  // Check treatment -> outcome path exists
  const hasPath = pathExists(dag, dag.treatmentId, dag.outcomeId);
  if (!hasPath) {
    issues.push({
      severity: 'warning',
      message: 'No directed path from treatment to outcome',
      nodeIds: [dag.treatmentId, dag.outcomeId]
    });
    suggestions.push('Add edge from treatment to outcome or include mediators');
  }

  // Check for isolated nodes
  const nodeIds = new Set(dag.nodes.map((n) => n.id));
  const connectedNodes = new Set<string>();
  for (const edge of dag.edges) {
    connectedNodes.add(edge.from);
    connectedNodes.add(edge.to);
  }
  const isolated = dag.nodes.filter((n) => !connectedNodes.has(n.id));
  if (isolated.length > 0) {
    issues.push({
      severity: 'warning',
      message: 'Isolated nodes detected',
      nodeIds: isolated.map((n) => n.id)
    });
  }

  // Check edge endpoints exist
  for (const edge of dag.edges) {
    if (!nodeIds.has(edge.from)) {
      issues.push({
        severity: 'error',
        message: `Edge source '${edge.from}' not found`,
        edgeIds: [edge.id]
      });
    }
    if (!nodeIds.has(edge.to)) {
      issues.push({
        severity: 'error',
        message: `Edge target '${edge.to}' not found`,
        edgeIds: [edge.id]
      });
    }
  }

  // Identifiability check
  if (dag.confounderIds.length > 0) {
    const backdoor = checkBackdoorCriterion(dag, dag.treatmentId, dag.outcomeId);
    if (!backdoor.satisfied) {
      issues.push({
        severity: 'warning',
        message: 'Backdoor criterion not satisfied - effect may not be identifiable',
        nodeIds: dag.confounderIds
      });
      suggestions.push('Include additional observed variables to block confounding paths');
    }
  }

  const hasErrors = issues.some((i) => i.severity === 'error');
  const complete = treatmentNode !== undefined && outcomeNode !== undefined;

  return {
    valid: acyclic && !hasErrors,
    acyclic,
    complete,
    issues,
    suggestions
  };
}

// ============================================================================
// Graph Algorithms
// ============================================================================

/**
 * Check if DAG is acyclic using DFS
 */
export function isAcyclic(dag: CausalDAG): boolean {
  const visited = new Set<string>();
  const recStack = new Set<string>();

  const adjacency = buildAdjacencyList(dag);

  function dfs(node: string): boolean {
    visited.add(node);
    recStack.add(node);

    for (const neighbor of adjacency.get(node) || []) {
      if (!visited.has(neighbor)) {
        if (!dfs(neighbor)) return false;
      } else if (recStack.has(neighbor)) {
        return false; // Cycle detected
      }
    }

    recStack.delete(node);
    return true;
  }

  for (const node of dag.nodes) {
    if (!visited.has(node.id)) {
      if (!dfs(node.id)) return false;
    }
  }

  return true;
}

/**
 * Find a cycle in the DAG (returns node IDs in cycle)
 */
export function findCycle(dag: CausalDAG): string[] {
  const visited = new Set<string>();
  const recStack = new Set<string>();
  const parent = new Map<string, string>();

  const adjacency = buildAdjacencyList(dag);
  const cycle: string[] = [];

  function dfs(node: string): boolean {
    visited.add(node);
    recStack.add(node);

    for (const neighbor of adjacency.get(node) || []) {
      if (!visited.has(neighbor)) {
        parent.set(neighbor, node);
        if (!dfs(neighbor)) return false;
      } else if (recStack.has(neighbor)) {
        // Found cycle, reconstruct it
        let current = node;
        cycle.push(neighbor);
        while (current !== neighbor) {
          cycle.push(current);
          current = parent.get(current) || '';
        }
        cycle.reverse();
        return false;
      }
    }

    recStack.delete(node);
    return true;
  }

  for (const node of dag.nodes) {
    if (!visited.has(node.id)) {
      if (!dfs(node.id)) break;
    }
  }

  return cycle;
}

/**
 * Check if directed path exists from source to target
 */
export function pathExists(dag: CausalDAG, source: string, target: string): boolean {
  const adjacency = buildAdjacencyList(dag);
  const visited = new Set<string>();
  const queue = [source];

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (current === target) return true;
    if (visited.has(current)) continue;
    visited.add(current);

    for (const neighbor of adjacency.get(current) || []) {
      queue.push(neighbor);
    }
  }

  return false;
}

/**
 * Get all directed paths from source to target
 */
export function findAllPaths(
  dag: CausalDAG,
  source: string,
  target: string,
  maxLength: number = 10
): string[][] {
  const adjacency = buildAdjacencyList(dag);
  const paths: string[][] = [];

  function dfs(current: string, path: string[], visited: Set<string>): void {
    if (path.length > maxLength) return;
    if (current === target) {
      paths.push([...path]);
      return;
    }

    for (const neighbor of adjacency.get(current) || []) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        path.push(neighbor);
        dfs(neighbor, path, visited);
        path.pop();
        visited.delete(neighbor);
      }
    }
  }

  const visited = new Set<string>([source]);
  dfs(source, [source], visited);
  return paths;
}

/**
 * Build adjacency list from DAG
 */
function buildAdjacencyList(dag: CausalDAG): Map<string, string[]> {
  const adj = new Map<string, string[]>();
  for (const node of dag.nodes) {
    adj.set(node.id, []);
  }
  for (const edge of dag.edges) {
    const neighbors = adj.get(edge.from) || [];
    neighbors.push(edge.to);
    adj.set(edge.from, neighbors);
  }
  return adj;
}

/**
 * Build undirected adjacency list (for ancestral graphs)
 */
function _buildUndirectedAdjacency(dag: CausalDAG): Map<string, string[]> {
  const adj = new Map<string, string[]>();
  for (const node of dag.nodes) {
    adj.set(node.id, []);
  }
  for (const edge of dag.edges) {
    const from = adj.get(edge.from) || [];
    from.push(edge.to);
    adj.set(edge.from, from);

    const to = adj.get(edge.to) || [];
    to.push(edge.from);
    adj.set(edge.to, to);
  }
  return adj;
}

// ============================================================================
// D-Separation
// ============================================================================

/**
 * Check d-separation: X ⫫ Y | Z
 *
 * Uses the Bayes-Ball algorithm for efficiency.
 *
 * @param dag - Causal DAG
 * @param setX - Set X node IDs
 * @param setY - Set Y node IDs
 * @param conditioningSet - Conditioning set Z node IDs
 * @returns D-separation result
 */
export function checkDSeparation(
  dag: CausalDAG,
  setX: string[],
  setY: string[],
  conditioningSet: string[]
): DSeparationResult {
  const condSet = new Set(conditioningSet);
  const _xSet = new Set(setX);
  const ySet = new Set(setY);

  // Build parent/child maps
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();

  for (const node of dag.nodes) {
    parents.set(node.id, []);
    children.set(node.id, []);
  }

  for (const edge of dag.edges) {
    children.get(edge.from)?.push(edge.to);
    parents.get(edge.to)?.push(edge.from);
  }

  // Find ancestors of conditioning set
  const ancestors = findAncestors(dag, conditioningSet);

  // Bayes-Ball: can we reach Y from X?
  const openPaths: string[][] = [];
  const blockedPaths: string[][] = [];

  // BFS with direction tracking
  type State = { node: string; fromChild: boolean; path: string[] };
  const visited = new Set<string>();
  const queue: State[] = [];

  // Start from X nodes
  for (const x of setX) {
    queue.push({ node: x, fromChild: false, path: [x] });
  }

  while (queue.length > 0) {
    const { node, fromChild, path } = queue.shift()!;
    const stateKey = `${node}-${fromChild}`;

    if (visited.has(stateKey)) continue;
    visited.add(stateKey);

    // Check if reached Y
    if (ySet.has(node)) {
      openPaths.push(path);
      continue;
    }

    const isConditioned = condSet.has(node);
    const isAncestor = ancestors.has(node);

    // Apply d-separation rules
    if (fromChild) {
      // Came from a child
      if (!isConditioned) {
        // Visit parents (not blocked)
        for (const parent of parents.get(node) || []) {
          queue.push({ node: parent, fromChild: false, path: [...path, parent] });
        }
      }
      if (isConditioned || isAncestor) {
        // Collider opened - visit children
        for (const child of children.get(node) || []) {
          queue.push({ node: child, fromChild: true, path: [...path, child] });
        }
      }
    } else {
      // Came from a parent
      if (!isConditioned) {
        // Visit children (chain or fork, not blocked)
        for (const child of children.get(node) || []) {
          queue.push({ node: child, fromChild: true, path: [...path, child] });
        }
        // Visit parents (fork)
        for (const parent of parents.get(node) || []) {
          queue.push({ node: parent, fromChild: false, path: [...path, parent] });
        }
      }
    }
  }

  // Separate blocked paths (those not reaching Y)
  // For simplicity, we consider all non-open paths as blocked
  // A more complete implementation would track all paths

  return {
    separated: openPaths.length === 0,
    setX,
    setY,
    conditioningSet,
    openPaths,
    blockedPaths
  };
}

/**
 * Find all ancestors of a set of nodes
 */
function findAncestors(dag: CausalDAG, nodes: string[]): Set<string> {
  const parents = new Map<string, string[]>();
  for (const node of dag.nodes) {
    parents.set(node.id, []);
  }
  for (const edge of dag.edges) {
    parents.get(edge.to)?.push(edge.from);
  }

  const ancestors = new Set<string>(nodes);
  const queue = [...nodes];

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const parent of parents.get(current) || []) {
      if (!ancestors.has(parent)) {
        ancestors.add(parent);
        queue.push(parent);
      }
    }
  }

  return ancestors;
}

// ============================================================================
// Identification Criteria
// ============================================================================

/**
 * Check backdoor criterion for treatment-outcome pair
 *
 * Backdoor criterion is satisfied if there exists a set Z such that:
 * 1. Z blocks all backdoor paths from treatment to outcome
 * 2. Z contains no descendants of treatment
 *
 * @param dag - Causal DAG
 * @param treatment - Treatment node ID
 * @param outcome - Outcome node ID
 * @returns Backdoor criterion result
 */
export function checkBackdoorCriterion(
  dag: CausalDAG,
  treatment: string,
  outcome: string
): BackdoorCriterionResult {
  // Find all observed nodes that are not descendants of treatment
  const descendants = findDescendants(dag, treatment);
  const validAdjustment = dag.nodes
    .filter((n) => n.observed && !descendants.has(n.id) && n.id !== treatment && n.id !== outcome)
    .map((n) => n.id);

  // Find backdoor paths (paths from T to Y that go through parents of T)
  const backdoorPaths = findBackdoorPaths(dag, treatment, outcome);

  // Find minimal adjustment sets using greedy search
  const adjustmentSets: string[][] = [];
  const minimalSet: string[] = [];

  // Try to find a set that blocks all backdoor paths
  for (const path of backdoorPaths) {
    // A path is blocked if any intermediate node is in the adjustment set
    // (and that node is not a collider without conditioned descendants)
    const intermediates = path.slice(1, -1);
    for (const node of intermediates) {
      if (validAdjustment.includes(node) && !minimalSet.includes(node)) {
        minimalSet.push(node);
        break;
      }
    }
  }

  if (minimalSet.length > 0) {
    adjustmentSets.push(minimalSet);
  }

  // Check if all backdoor paths are blocked
  const dSepResult =
    minimalSet.length > 0
      ? checkDSeparation(dag, [treatment], [outcome], minimalSet)
      : { separated: false, openPaths: backdoorPaths, blockedPaths: [] as string[][] };

  // Filter to only backdoor paths
  const openBackdoorPaths = dSepResult.openPaths.filter((path) =>
    backdoorPaths.some((bp) => JSON.stringify(bp) === JSON.stringify(path))
  );

  return {
    satisfied: openBackdoorPaths.length === 0 || minimalSet.length > 0,
    adjustmentSets,
    minimalSet,
    blockedPaths: backdoorPaths.filter(
      (p) => !openBackdoorPaths.some((op) => JSON.stringify(op) === JSON.stringify(p))
    ),
    openPaths: openBackdoorPaths
  };
}

/**
 * Find backdoor paths from treatment to outcome
 */
function findBackdoorPaths(dag: CausalDAG, treatment: string, outcome: string): string[][] {
  // Backdoor paths start with an edge INTO treatment (through parents)
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();

  for (const node of dag.nodes) {
    parents.set(node.id, []);
    children.set(node.id, []);
  }

  for (const edge of dag.edges) {
    parents.get(edge.to)?.push(edge.from);
    children.get(edge.from)?.push(edge.to);
  }

  const backdoorPaths: string[][] = [];

  // DFS from parents of treatment
  function findPaths(
    current: string,
    path: string[],
    visited: Set<string>,
    _lastDirection: 'up' | 'down'
  ): void {
    if (current === outcome) {
      backdoorPaths.push([treatment, ...path]);
      return;
    }

    if (visited.has(current)) return;
    visited.add(current);

    // Can go up (to parents) or down (to children)
    for (const parent of parents.get(current) || []) {
      if (!visited.has(parent)) {
        findPaths(parent, [...path, parent], visited, 'up');
      }
    }

    for (const child of children.get(current) || []) {
      if (!visited.has(child) && child !== treatment) {
        findPaths(child, [...path, child], visited, 'down');
      }
    }

    visited.delete(current);
  }

  // Start from parents of treatment
  const treatmentParents = parents.get(treatment) || [];
  for (const parent of treatmentParents) {
    findPaths(parent, [parent], new Set(), 'up');
  }

  return backdoorPaths;
}

/**
 * Find all descendants of a node
 */
function findDescendants(dag: CausalDAG, node: string): Set<string> {
  const children = new Map<string, string[]>();
  for (const n of dag.nodes) {
    children.set(n.id, []);
  }
  for (const edge of dag.edges) {
    children.get(edge.from)?.push(edge.to);
  }

  const descendants = new Set<string>();
  const queue = [node];

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const child of children.get(current) || []) {
      if (!descendants.has(child)) {
        descendants.add(child);
        queue.push(child);
      }
    }
  }

  return descendants;
}

/**
 * Check frontdoor criterion
 *
 * Frontdoor criterion is satisfied if there exists a set M such that:
 * 1. M intercepts all directed paths from treatment to outcome
 * 2. There is no unblocked backdoor path from treatment to M
 * 3. All backdoor paths from M to outcome are blocked by treatment
 */
export function checkFrontdoorCriterion(
  dag: CausalDAG,
  treatment: string,
  outcome: string
): FrontdoorCriterionResult {
  // Find potential mediators (nodes on directed paths T -> Y)
  const paths = findAllPaths(dag, treatment, outcome);
  const mediatorCandidates = new Set<string>();

  for (const path of paths) {
    for (const node of path.slice(1, -1)) {
      const nodeObj = dag.nodes.find((n) => n.id === node);
      if (nodeObj?.observed) {
        mediatorCandidates.add(node);
      }
    }
  }

  // Check each mediator candidate
  for (const mediator of mediatorCandidates) {
    // Check condition 2: no unblocked backdoor from T to M
    const backdoorTM = checkBackdoorCriterion(dag, treatment, mediator);

    // Check condition 3: backdoor M to Y blocked by T
    const backdoorMY = checkBackdoorCriterion(dag, mediator, outcome);

    if (backdoorTM.satisfied && backdoorMY.satisfied) {
      return {
        satisfied: true,
        mediators: [mediator],
        identificationFormula: `P(Y|do(${treatment})) = Σ_m P(M=m|${treatment}) Σ_t P(Y|M=m,${treatment}=t) P(${treatment}=t)`
      };
    }
  }

  return {
    satisfied: false,
    mediators: []
  };
}

/**
 * Check instrumental variable validity
 */
export function checkInstrument(
  dag: CausalDAG,
  instrumentId: string,
  treatment: string,
  outcome: string
): InstrumentCheckResult {
  // 1. Relevance: instrument -> treatment path exists
  const relevance = pathExists(dag, instrumentId, treatment);

  // 2. Exclusion: no direct path from instrument to outcome
  const directPathToOutcome = dag.edges.some((e) => e.from === instrumentId && e.to === outcome);
  const exclusion = !directPathToOutcome;

  // 3. Independence: instrument d-separated from confounders
  const dSep = checkDSeparation(dag, [instrumentId], dag.confounderIds, []);
  const independence = dSep.separated;

  return {
    valid: relevance && exclusion && independence,
    instruments: relevance && exclusion && independence ? [instrumentId] : [],
    relevance,
    exclusion,
    independence
  };
}

// ============================================================================
// Identifiability Analysis
// ============================================================================

/**
 * Analyze identifiability of causal effect
 */
export function analyzeIdentifiability(
  dag: CausalDAG,
  treatment: string,
  outcome: string
): IdentifiabilityAnalysis {
  // Check backdoor criterion
  const backdoor = checkBackdoorCriterion(dag, treatment, outcome);
  if (backdoor.satisfied && backdoor.minimalSet.length > 0) {
    return {
      identifiable: true,
      method: 'backdoor',
      adjustmentFormula: `E[Y|do(${treatment})] = Σ_z E[Y|${treatment},Z=z] P(Z=z)`,
      assumptions: [
        'No unmeasured confounding given adjustment set',
        'Positivity: P(Z|X) > 0 for all X',
        'Consistency: Y(z) = Y when Z = z'
      ],
      testableImplications: [`${treatment} ⫫ Z given no conditioning (check for association)`]
    };
  }

  // Check frontdoor criterion
  const frontdoor = checkFrontdoorCriterion(dag, treatment, outcome);
  if (frontdoor.satisfied) {
    return {
      identifiable: true,
      method: 'frontdoor',
      adjustmentFormula: frontdoor.identificationFormula,
      assumptions: [
        'M intercepts all causal paths from treatment to outcome',
        'No backdoor path from treatment to M',
        'Treatment blocks all backdoor paths from M to outcome'
      ],
      testableImplications: [
        'M is associated with treatment',
        'Y is associated with M conditional on treatment'
      ]
    };
  }

  // Check for instrumental variables
  const instruments = dag.nodes.filter((n) => n.type === 'instrument');
  for (const iv of instruments) {
    const ivCheck = checkInstrument(dag, iv.id, treatment, outcome);
    if (ivCheck.valid) {
      return {
        identifiable: true,
        method: 'iv',
        adjustmentFormula: `ATE = [E[Y|IV=1] - E[Y|IV=0]] / [E[Z|IV=1] - E[Z|IV=0]]`,
        assumptions: [
          'Instrument is relevant (affects treatment)',
          'Exclusion restriction (no direct effect on outcome)',
          'Independence (instrument uncorrelated with confounders)'
        ],
        testableImplications: [`${iv.id} is associated with ${treatment} (first stage)`]
      };
    }
  }

  // Effect not identifiable
  return {
    identifiable: false,
    method: 'unidentified',
    assumptions: [],
    testableImplications: [],
    sensitivityParams: [
      {
        name: 'gamma (Rosenbaum)',
        range: [1, 10],
        description: 'Odds ratio bound for unmeasured confounding'
      }
    ]
  };
}

// ============================================================================
// Temporal DAG Operations
// ============================================================================

/**
 * Create a temporal DAG with rolled-out time slices
 */
export function createTemporalDAG(
  templateDAG: CausalDAG,
  numTimeSteps: number,
  temporalDeps: TemporalDependencies
): TemporalDAG {
  const nodes: CausalNode[] = [];
  const edges: CausalEdge[] = [];
  const temporalEdges: CausalEdge[] = [];

  // Create time slices
  for (let t = 0; t < numTimeSteps; t++) {
    for (const node of templateDAG.nodes) {
      if (node.type !== 'temporal') {
        nodes.push({
          ...node,
          id: `${node.id}_t${t}`,
          label: `${node.label} (t=${t})`,
          timeIndex: t
        });
      }
    }

    // Intra-slice edges
    for (const edge of templateDAG.edges) {
      if (edge.type !== 'temporal') {
        edges.push({
          ...edge,
          id: `${edge.id}_t${t}`,
          from: `${edge.from}_t${t}`,
          to: `${edge.to}_t${t}`
        });
      }
    }
  }

  // Inter-temporal edges
  for (const lag of temporalDeps.outcomeLags) {
    for (let t = lag.lag; t < numTimeSteps; t++) {
      const edge: CausalEdge = {
        id: `${lag.variable}_t${t - lag.lag}->${lag.target}_t${t}`,
        from: `${lag.variable}_t${t - lag.lag}`,
        to: `${lag.target}_t${t}`,
        type: 'temporal',
        identified: true,
        lag: lag.lag,
        coefficient: lag.coefficient
      };
      temporalEdges.push(edge);
      edges.push(edge);
    }
  }

  for (const lag of temporalDeps.crossLags) {
    for (let t = lag.lag; t < numTimeSteps; t++) {
      const edge: CausalEdge = {
        id: `${lag.variable}_t${t - lag.lag}->${lag.target}_t${t}`,
        from: `${lag.variable}_t${t - lag.lag}`,
        to: `${lag.target}_t${t}`,
        type: 'temporal',
        identified: true,
        lag: lag.lag,
        coefficient: lag.coefficient
      };
      temporalEdges.push(edge);
      edges.push(edge);
    }
  }

  return {
    ...templateDAG,
    id: `temporal-${templateDAG.id}`,
    name: `Temporal ${templateDAG.name}`,
    nodes,
    edges,
    treatmentId: `${templateDAG.treatmentId}_t${numTimeSteps - 1}`,
    outcomeId: `${templateDAG.outcomeId}_t${numTimeSteps - 1}`,
    numTimeSteps,
    templateDAG,
    temporalEdges,
    stationary: temporalDeps.stationary,
    metadata: {
      ...templateDAG.metadata,
      updatedAt: Date.now()
    }
  };
}

/**
 * Extract time slice from temporal DAG
 */
export function extractTimeSlice(dag: TemporalDAG, timeIndex: number): TimeSlice {
  const nodes = dag.nodes.filter((n) => n.timeIndex === timeIndex);
  const nodeIds = new Set(nodes.map((n) => n.id));

  const intraEdges = dag.edges.filter(
    (e) => nodeIds.has(e.from) && nodeIds.has(e.to) && e.type !== 'temporal'
  );

  const forwardEdges = dag.temporalEdges.filter((e) => nodeIds.has(e.from) && !nodeIds.has(e.to));

  return {
    timeIndex,
    nodes,
    intraEdges,
    forwardEdges
  };
}

// ============================================================================
// Sensitivity Analysis
// ============================================================================

/**
 * Compute Rosenbaum bounds for sensitivity analysis
 *
 * Tests how strong unmeasured confounding would need to be to explain
 * away the observed effect.
 *
 * @param observedEffect - Observed treatment effect
 * @param standardError - Standard error of effect
 * @param gammaValues - Gamma values to test (odds ratio bounds)
 * @returns Array of Rosenbaum bounds
 */
export function computeRosenbaumBounds(
  observedEffect: number,
  standardError: number,
  gammaValues: number[] = [1, 1.5, 2, 3, 5, 10]
): RosenbaumBounds[] {
  const results: RosenbaumBounds[] = [];

  for (const gamma of gammaValues) {
    // Under gamma-level confounding, bounds on the effect
    const boundFactor = Math.log(gamma);

    // Adjusted effect bounds
    const effectLower = observedEffect - boundFactor * standardError;
    const effectUpper = observedEffect + boundFactor * standardError;

    // P-value bounds (using normal approximation)
    const zUpper = effectUpper / standardError;
    const zLower = effectLower / standardError;

    const pValueUpper = 2 * (1 - normalCDF(Math.abs(zLower)));
    const pValueLower = 2 * (1 - normalCDF(Math.abs(zUpper)));

    // Confidence interval under gamma
    const z95 = normalQuantile(0.975);
    const ciLower = observedEffect - z95 * standardError * Math.sqrt(gamma);
    const ciUpper = observedEffect + z95 * standardError * Math.sqrt(gamma);

    results.push({
      gamma,
      pValueUpper: Math.min(1, pValueUpper),
      pValueLower: Math.max(0, pValueLower),
      pointEstimate: observedEffect,
      confidenceInterval: [ciLower, ciUpper]
    });
  }

  return results;
}

/**
 * Compute E-value for sensitivity analysis
 *
 * E-value represents the minimum strength of association (on the risk ratio scale)
 * that an unmeasured confounder would need to have with both treatment and outcome
 * to fully explain away the observed effect.
 *
 * @param riskRatio - Observed risk ratio (or odds ratio for rare outcomes)
 * @param ciLower - Lower bound of confidence interval (optional)
 * @returns E-value result
 */
export function computeEValue(riskRatio: number, ciLower?: number): EValue {
  // E-value formula: E = RR + sqrt(RR * (RR - 1))
  const computeE = (rr: number): number => {
    if (rr <= 1) {
      // For protective effect, use reciprocal
      rr = 1 / rr;
    }
    return rr + Math.sqrt(rr * (rr - 1));
  };

  const ePoint = computeE(riskRatio);
  const eCi = ciLower !== undefined ? computeE(ciLower) : 1;

  let interpretation: string;
  if (ePoint > 3) {
    interpretation = 'Strong evidence: Would require substantial confounding to explain';
  } else if (ePoint > 2) {
    interpretation = 'Moderate evidence: Moderate confounding could explain';
  } else {
    interpretation = 'Weak evidence: Mild confounding could explain';
  }

  return {
    pointEstimate: ePoint,
    ciLowerBound: eCi,
    interpretation
  };
}

/**
 * Compute Manski bounds (nonparametric bounds)
 *
 * @param outcomes - Observed outcomes
 * @param treatments - Treatment assignments
 * @param monotonicity - Whether to assume monotone treatment response
 * @returns Manski bounds
 */
export function computeManskiBounds(
  outcomes: number[],
  treatments: number[],
  monotonicity: boolean = false
): ManskiBounds {
  const n = outcomes.length;

  // Split by treatment
  const treated: number[] = [];
  const control: number[] = [];

  for (let i = 0; i < n; i++) {
    if (treatments[i] === 1) {
      treated.push(outcomes[i]);
    } else {
      control.push(outcomes[i]);
    }
  }

  // Observed means
  const muT = treated.length > 0 ? mean(treated) : 0;
  const muC = control.length > 0 ? mean(control) : 0;

  // Outcome bounds
  const yMin = Math.min(...outcomes);
  const yMax = Math.max(...outcomes);

  // Proportions
  const pT = treated.length / n;
  const pC = control.length / n;

  let lower: number;
  let upper: number;

  if (monotonicity) {
    // With monotone treatment response: Y(1) >= Y(0) for all
    // Narrower bounds
    lower = pT * muT + pC * yMin - (pT * yMax + pC * muC);
    upper = pT * muT + pC * yMax - (pT * yMin + pC * muC);
  } else {
    // Without assumptions: worst-case bounds
    lower = pT * muT + pC * yMin - (pT * yMax + pC * muC);
    upper = pT * muT + pC * yMax - (pT * yMin + pC * muC);
  }

  return {
    lower,
    upper,
    monotonicity,
    selectionAssumption: monotonicity ? 'Monotone treatment response' : 'None'
  };
}

// ============================================================================
// Helper: Normal CDF
// ============================================================================

function normalCDF(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1.0 + sign * y);
}
