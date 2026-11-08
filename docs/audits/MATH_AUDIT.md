# Numerical audit of `src/math` and the causal-inference statistics

Date: 2026-11-08. Method: every audited function was run on fixed, seeded inputs
(`scripts/audit/math_harness.ts`) and compared with an independent numpy/scipy implementation
(`scripts/audit/math_reference.py`), or with closed-form results. Regression tests for every fix
live in `tests/math/auditFixes.test.ts` and `tests/math/randomizedSVD.test.ts`.

```bash
npx tsx scripts/audit/math_harness.ts /tmp/math_audit.json
python3 scripts/audit/math_reference.py /tmp/math_audit.json   # needs numpy + scipy
```

## Verified correct

| Function | Reference | Max error |
| --- | --- | --- |
| `estimatePropensityScores` (logistic regression) | scipy BFGS maximum likelihood | 1.4e-4 (coefficients) |
| `fitOutcomeModel` (OLS) | `numpy.linalg.lstsq` | 2e-9 |
| `computeAIPWEstimate` (estimate and influence-function SE) | direct numpy formula | 4e-16 |
| `computeIPWEstimate` | Hájek (normalised) estimator | 4e-16 |
| `estimatePower`, `minimumDetectableEffect` | scipy normal distribution | 1e-7 |
| `variance`, `covariance` | numpy (ddof = 1) | 6e-17 |
| `ksgMutualInformation` (Kraskov algorithm 1) | independent cKDTree implementation on the same samples | 2e-9 |
| `neumaierSum`, `kahanSum` (0.1 × 10⁶) | `math.fsum` | 0 |
| `stableFrobeniusNorm`, `stableSpectralNorm` | `numpy.linalg.norm` | 1e-10 |
| `stableVariance` (values ≈ 1e9) | numpy | 1.5e-8 |

## Bugs found and fixed

| Function | Problem | Effect |
| --- | --- | --- |
| `knnEntropy` (Kozachenko–Leonenko) | Added both log V_d (radius form) and d·log 2 (diameter form) | Entropy overestimated by exactly d·ln 2 (0.69 nats in 1-D) |
| `computeLaplacian` / `analyzeAttentionGraph` | Used the directed attention matrix and counted self-loops in the degree but not in L | L was not symmetric and L·1 ≠ 0, so λ₁ ≠ 0 and the symmetric eigen-solver ran on a non-symmetric matrix |
| attention-graph eigen-solver | Power iteration with deflation on −L converges to the largest-magnitude eigenvalues, only 10 were computed, eigenvectors were not re-ordered with the sorted eigenvalues, random start | For graphs with more than 10 nodes λ₂ was never computed; the "Fiedler vector" belonged to another eigenvalue; results were not reproducible. Replaced by cyclic Jacobi (exact, deterministic). |
| `estimateConditionNumber` | The "Hager" iteration multiplied by A instead of solving with A | Returned ≈ ‖A‖₁ (5% of κ in the audit case). Now exact κ₁ via LU with partial pivoting. |
| `randomizedSVD` | `U = Q·U_B` multiplied by the list of vectors instead of the matrix of columns; the error bound used a singular value that was never computed; no re-orthonormalisation between power iterations | Threw on every input; after the shape fix accuracy was only ~1e-6. Now exact to ~1e-15 on matrices with known spectra. |
| `computeEValue` (CI) | Took the reciprocal of the lower limit even when the interval contained 1; ignored the upper limit for protective effects | Reported robustness to confounding for non-significant effects |
| Causal panel E-value | Used RR = exp(ATE) for a continuous outcome | Depended on the outcome's units; now RR ≈ exp(0.91·ATE/σ_Y) (VanderWeele, 2017) |
| Causal panel identifiability | Positivity, propensity range etc. were constants | Now from `CausalInferenceEngine.checkIdentifiability()` |

## Known limitations (documented, not bugs)

- `kahanSum` and `pairwiseSum` lose the `1` in `[1e16, 1, −1e16, 1]`; this is a property of those
  algorithms. Use `neumaierSum` for sums with heavy cancellation.
- `ksgMutualInformation` reports a standard error from a heuristic formula; KSG has no simple
  closed-form variance, so treat its confidence interval as indicative.
- `bootstrapATEConfidenceInterval` and some estimators use `Math.random` and are not seeded.
- The cyclic Jacobi solver is O(n³) per sweep and intended for the small attention graphs used here
  (n up to a few hundred).
- `convergence.ts` encodes published rate bounds as formulas; they are not machine-checked proofs.

## Not audited

`bias_verification.ts` (heuristic checks), `ntk_analysis.ts`, `sampling_analysis.ts`,
`statistics.ts`, `convergence.ts`, and the remaining `dag_operations.ts` functions (the E-value is
covered). They have unit tests in `tests/math`, but were not compared with an independent
reference.
