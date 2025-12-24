# Generalization Diagnostics: Bias-Variance, PAC-Bounds, and Fisher Information

The Neuro-Lingua toolkit now exposes mathematical primitives for rigorous
analysis of training dynamics. This note complements the implementation in
`src/math/analysis.ts` by summarizing the statistical assumptions and linking
them to classical generalization theory.

## Bias-Variance Trade-off

- **Model bias** quantifies the systematic error incurred when the hypothesis
  class cannot represent the true data-generating process. In practice we
  diagnose bias via residual structure in the training loss or persistent
  mis-calibration of predicted probabilities.
- **Model variance** tracks the sensitivity of the trained parameters to random
  perturbations of the dataset or initialization. The Lyapunov exponent exposed
  by `analyzeLyapunov` approximates this sensitivity by measuring how small
  perturbations grow under repeated application of the Jacobian.
- Numerical stability routines rely on the diagonal of the empirical Fisher
  matrix (gradient outer products) to identify directions with excessive
  variance. Dampening the Newton step (`damping` hyperparameter) directly
  regularizes these directions.

## PAC-Bayesian and Uniform Convergence Bounds

The spectral radius of the Jacobian acts as a proxy for the Lipschitz constant
of the update map. Bounding this quantity allows us to translate standard
Probably Approximately Correct (PAC) guarantees into concrete statements about
Neuro-Lingua:

1. A contraction mapping (spectral radius `< 1`) implies uniform stability of
   the training algorithm, which tightens PAC bounds on the generalization gap.
2. The exported `ANALYSIS_ASSUMPTIONS` enumerate the precise hypotheses under
   which these bounds hold: linearization around fixed points, dominant
   eigenvalues for power iteration, and the absence of exogenous noise.
3. Deviations from these assumptions (e.g., heavy-tailed gradient noise) should
   be flagged during review because they weaken the theoretical guarantees.

## Fisher Information and Second-Order Optimizers

- The empirical Fisher diagonal implemented in `computeDiagonalHessian` serves
  as a curvature-aware pre-conditioner. Its link to Fisher information provides
  a principled interpretation: directions with large gradient variance receive
  smaller Newton updates to maintain numerical stability.
- The limited-memory BFGS routine maintains a curvature history that mimics the
  inverse Fisher metric. By combining it with Lyapunov diagnostics we can audit
  whether the optimizer keeps the training trajectory in a stable basin.
- When exporting models we recommend documenting the effective damping and
  curvature statistics so that downstream audits can replicate the stability
  analysis without retraining.

## Practical Workflow

1. **Inspect stability** using `analyzeLyapunov` before and after major
   optimizer changes. The `stable` flag and `stabilityMargin` highlight whether
   updates remain contractive.
2. **Tune damping** for Newton updates by monitoring the diagonal Hessian: large
   entries indicate directions that benefit from stronger regularization.
3. **Report assumptions** from `ANALYSIS_ASSUMPTIONS` alongside experimental
   results to keep theoretical guarantees explicit.
