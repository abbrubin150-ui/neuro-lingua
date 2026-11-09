/**
 * Numerical audit harness: runs src/math functions on fixed, seeded inputs and writes the inputs
 * and outputs to JSON for scripts/audit/math_reference.py (numpy/scipy reference).
 *
 *   npx tsx scripts/audit/math_harness.ts /tmp/math_audit.json
 *   python3 scripts/audit/math_reference.py /tmp/math_audit.json
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { writeFileSync } from 'node:fs';
import * as cm from '../../src/math/causal_math';
import * as it from '../../src/math/information_theory';
import * as nu from '../../src/math/numerics';
import * as sg from '../../src/math/spectral_graph';
import * as ap from '../../src/math/approximation';

let s = 12345;
const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
const randn = () => {
  const u = rnd() || 1e-12;
  const v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const out: Record<string, unknown> = {};

// ---- causal: propensity (logistic regression), AIPW, IPW, outcome model, tests
const n = 400;
const X = Array.from({ length: n }, () => [randn(), randn()]);
const Z = X.map(([a, b]) => (rnd() < 1 / (1 + Math.exp(-(0.3 + 0.8 * a - 0.5 * b))) ? 1 : 0));
const Y = X.map(([a, b], i) => 1 + 0.5 * a - 0.7 * b + 0.9 * Z[i] + 0.3 * randn());
const prop = cm.estimatePropensityScores(X, Z);
out.propensity = { X, Z, result: prop };
const coef = (prop as any).coefficients ?? (prop as any).params ?? null;
const pis = coef ? cm.predictPropensity(X, coef) : (prop as any).propensities;
out.propensities = pis;
const om = cm.fitOutcomeModel(Y, Z, X);
out.outcomeModel = { Y, om };
const muA = cm.predictOutcome(X, 0, om as any);
const muB = cm.predictOutcome(X, 1, om as any);
out.aipw = { muA, muB, result: cm.computeAIPWEstimate(Y, Z, pis, muA, muB).ate };
out.ipw = cm.computeIPWEstimate(Y, Z, pis);
const ate = (out.aipw as any).result;
out.test = cm.testATESignificance(ate, 0.05);
out.power = [0.1, 0.25, 0.5].map((e) => cm.estimatePower(e, 0.12, 0.05));
out.mde = [0.5, 0.8, 0.9].map((p) => cm.minimumDetectableEffect(p, 0.12, 0.05));
out.variance = {
  x: Y.slice(0, 50),
  v: cm.variance(Y.slice(0, 50)),
  cov: cm.covariance(Y.slice(0, 50), Z.slice(0, 50))
};

// ---- information theory
const rho = 0.8;
const G = Array.from({ length: 1500 }, () => {
  const a = randn(),
    b = randn();
  return [a, rho * a + Math.sqrt(1 - rho * rho) * b];
});
out.ksg = {
  rho,
  G,
  mi: it.ksgMutualInformation(
    G.map((p) => [p[0]]),
    G.map((p) => [p[1]]),
    3
  )
};
const U = Array.from({ length: 2000 }, () => [randn() * 2]);
out.knnEntropy = { sigma: 2, h: it.knnEntropy(U, 3) };

// ---- numerics
const vals = [1e16, 1, -1e16, 1, 3.14159, -2.5e-8];
out.sums = {
  vals,
  kahan: nu.kahanSum(vals),
  neumaier: nu.neumaierSum(vals),
  pairwise: nu.pairwiseSum(vals),
  tenth: nu.kahanSum(new Array(1000000).fill(0.1)),
  tenthNaive: new Array(1000000).fill(0.1).reduce((a, b) => a + b, 0)
};
const M = Array.from({ length: 6 }, () => Array.from({ length: 5 }, () => randn()));
out.norms = {
  M,
  fro: nu.stableFrobeniusNorm(M),
  spec: nu.stableSpectralNorm(M),
  cond: nu.estimateConditionNumber(M.slice(0, 5))
};
const sv = Array.from({ length: 300 }, () => 1e9 + randn());
out.stableVariance = { sv, r: nu.stableVariance(sv) };

// ---- spectral graph
const A = Array.from({ length: 6 }, (_, i) =>
  Array.from({ length: 6 }, (_, j) =>
    i === j ? 0 : Math.abs(i - j) === 1 || i < 3 === j < 3 ? 0.5 + rnd() * 0.5 : 0.01
  )
);
out.graph = {
  A,
  Lnorm: sg.computeLaplacian(A, true),
  L: sg.computeLaplacian(A, false),
  analysis: sg.analyzeAttentionGraph(A)
};

// ---- approximation
const B = Array.from({ length: 12 }, () => Array.from({ length: 8 }, () => randn()));
out.rsvd = { B, r: ap.randomizedSVD(B, 3) };
out.lowrank = ap.analyzeLowRankApproximation(B, 0.9);

writeFileSync(process.argv[2], JSON.stringify(out));
console.log('ok', Object.keys(out).join(','));
