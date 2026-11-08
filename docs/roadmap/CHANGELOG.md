# Changelog

## [Unreleased]

An audit of what actually runs versus what the UI and documentation claimed. Every model change is
covered by finite-difference gradient tests; quality changes are measured on held-out data
(`pnpm benchmark:models`, see [MODEL_BENCHMARK.md](../benchmarks/MODEL_BENCHMARK.md)).

### Models

- **TransformerLM is trained for real.** Attention and feed-forward weights used to be moved by a
  uniform "pseudo-gradient"; they now receive exact gradients (`src/models/transformer_backprop.ts`).
- **Causal language-model objective** (default for new models): causal attention, per-head RoPE,
  final RMSNorm and a next-token loss at every position. Held-out results: 2.88 bpc vs 3.22–3.26 for
  the previous objective on character-level `corpus.txt`; 100% vs 55–81% on an order-dependent task.
  Saved models without an objective load as `pooled` and keep their predictions.
- Per-head RoPE: the legacy full-row layout gave GQA query/key pairs different frequencies.
- Attention dropout and DropConnect act in training with exact gradients (never at inference).
- Mini-batches (`batchSize`, default 8) with a fused optimizer pass: ≈ 2.7× faster per window with the
  UI defaults.
- The base-class inference paths (`generate` with all sampling options, `getLogitsForContext`,
  distillation, `evaluate`) run the transformer instead of the base MLP.
- **AdvancedNeuralLM options act in the UI training path**: activation (GELU/ELU/LeakyReLU), LayerNorm
  with exact backward, LR schedules across one-epoch-per-call training (`setTrainingHorizon`), weight
  decay and gradient clipping. Advanced models are saved and restored as AdvancedNeuralLM.
- Fixed: a token repeated in the context window received one optimizer step per occurrence (Adam
  moved it ~3.9× too far for three copies).
- Fixed: `RMSNorm.backward` input gradient.
- `'newton'` uses the exact Gauss-Newton diagonal (EMA + damping) instead of `g / (g² + λ)`.
- New `model.evaluate(text)` (held-out loss, perplexity, accuracy, OOV rate).

### Cerebro neuron injection

- Fixed: injection broke every architecture (MLP training threw; transformer GQA produced NaN).
  MLP expansion is now function-preserving; transformer heads grow equally (k must be a multiple of
  `numHeads`).
- Fixed: rollback reshaped snapshots with wrong strides (wHidden, transposed wOutput, half-width
  ff1, square K/V) and corrupted the model; it now restores predictions exactly.

### Honest UI

- Explainability: SHAP used the token id as the "prediction", integrated gradients differentiated
  the id, and attention rollout was a uniform placeholder. All three now use the model (exact
  Shapley values, embedding-space IG with exact gradients, rollout over real attention maps).
  `attentionRollout()` also composed layers in the wrong order and ignored `addIdentity`.
- Information bottleneck: the plotted I(X;Z), I(Z;Y) and H(Z) were formulas of the epoch number and
  the advertised IB loss was never applied. Replaced by binning estimates from hidden
  representations; the panel is labelled as monitoring only.
- Edge-learning diagnostics: "Fisher information" was the final loss, "entropy" log(#params)·(1+convergence),
  "variance" the final loss, and "efficiency" a ratio of those, shown with a "COMPUTED" badge. They are now
  measured on the model (empirical output-layer Fisher from exact per-window gradients, predictive entropy,
  loss variance, diagonal Cramér-Rao bound); "efficiency" was removed.
- Causal analysis: identifiability diagnostics (positivity, propensity range, ...) were hard-coded;
  they now come from the engine, and the panel says it runs on simulated data.
- Training stats show the current epoch, not the running mean over all epochs.
- Sophia's β₁, β₂, ρ and Hessian interval from the UI reach the optimizer.
- Onboarding tip: LR ~0.003 with Adam (the old 0.05–0.1 tip prevents Adam from learning).
- WebGPU messages no longer promise a 2–5× speedup (none was measured; the Transformer is CPU-only).

### Tokenizer and data

- Character mode keeps word boundaries (one `' '` token per whitespace run) and generation joins
  characters without spaces; `tokenizeWithOffsets()` gives exact character spans.
- Fixed: the custom-RegExp loss mask drifted off the matching tokens on any text with spaces.
- Fixed: importing a trace-format export always failed silently (`src/lib/modelImport.ts`).

### Math and statistics ([audit](../audits/MATH_AUDIT.md))

- Fixed: `knnEntropy` overestimated entropy by d·ln 2; `computeLaplacian` used directed weights
  with inconsistent self-loops; the attention-graph eigen-solver missed λ₂ for graphs with more than
  10 nodes and mismatched the Fiedler vector (now cyclic Jacobi); `estimateConditionNumber`
  returned ≈ ‖A‖₁ (now exact κ₁ via LU); `randomizedSVD` threw on every input; `computeEValue`
  overstated robustness for intervals containing 1; the causal panel's E-value depended on outcome
  units.
- Verified against numpy/scipy: propensity MLE, OLS outcome model, AIPW/IPW, power/MDE, KSG mutual
  information, compensated sums and norms (`scripts/audit/`).

### Documentation

- README implementation-status table (what runs, what is a library module, what is simulated),
  without unmeasured performance claims; benchmark document; updated CLAUDE.md and GPU guide.
