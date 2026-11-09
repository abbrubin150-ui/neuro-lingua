# Changelog

## [Unreleased]

An audit of what actually runs versus what the UI and documentation claimed. Every model change is
covered by finite-difference gradient tests; quality changes are measured on held-out data
(`pnpm benchmark:models`, see [MODEL_BENCHMARK.md](../benchmarks/MODEL_BENCHMARK.md)).

### Offline model production ([guide](../guides/OFFLINE_MODEL_FACTORY.md))

Everything needed to produce, package and run a model from local files, with no external AI model,
API or network service.

- **Model factory CLI** (`pnpm lm train | eval | generate | serve | info`, `src/factory/`): local
  documents (.txt .md .jsonl .json .csv) → cleaned, de-duplicated corpus with a content-hash
  train/validation split → subword tokenizer → transformer trained with validation every epoch,
  early stopping and the best epoch kept → package + model card + training log. `--init`
  continues an existing model. Held-out quality is reported in bits per character, which does not
  depend on the tokenizer.
- **Subword (BPE) tokenizer** wired into all three architectures (`src/tokenizer/subword.ts`):
  deterministic incremental training (≈ 0.2 s for 200 K characters), Hebrew niqqud kept with its
  letter, exact character offsets, spaces restored on decoding. Selectable in the tokenizer panel.
- **Transformer training for larger corpora**: chunked windows (`trainingStride`) with the
  alignment re-drawn every epoch, per-step warm-up + cosine learning rate, decoupled weight decay.
- **Model packages** (`neuro-lingua-model` v1, `src/lib/modelPackageCodec.ts`): base64
  float32/float16 tensors, optimizer state dropped, FNV-1a 64-bit checksum verified on load, model
  card with provenance (`offline: true | false | null`). The UI-default transformer: 23 MB as plain
  JSON → 5.7 MB (float32) → 2.9 MB (float16).
- **Local inference server** (`pnpm lm serve`): completions / chat-completions JSON shapes,
  `127.0.0.1` by default, CORS only on request, request and token limits, `seed` for
  reproducible samples; `finish_reason` and token counts come from the sampler
  (`generateDetailed`), not from re-tokenizing the output.
- **Browser app**: 📦 Package export with an honest model card (a model that started from an
  imported file keeps that file's provenance), Import of packages, and localStorage saves that fall
  back to float32 / float16 packages — the default transformer did not fit before, so saving
  failed silently.
- **Offline web app**: a build-time service worker pre-caches every file (network-first
  navigations, cached fallback, old caches removed on update) and a web app manifest; the desktop
  app needs none.
- **Responsive in-browser training**: training yields to the page every 50 ms (progress within the
  epoch, Stop acts within one window and the interrupted epoch is not recorded). Fixed: strided
  epochs reported accuracy over all targets instead of the windows trained.
- **1.56× faster transformer training** (output layer and softmax as monomorphic kernels;
  bit-identical results).
- Fixed: the embedding-visualization panel ran t-SNE over the whole vocabulary on the main thread
  whenever the model changed (≈ 12 s for 300 tokens, minutes for a 2,000-token subword
  vocabulary, freezing the page). Projections now run on request over a bounded selection —
  words and subwords first, specials excluded — with UMAP as the default method.
- **No-external-AI guard** (`tests/offline/noExternalServices.test.ts`).
- **Multi-core training and evaluation** (`pnpm lm train|eval --workers <n|auto>`): every
  mini-batch is split over worker threads holding replicas of the transformer; gradients and
  parameters travel through shared memory and the optimizer step stays on the main thread. Same
  windows, batches and LR steps as single-thread training; without dropout the same result up to
  summation order, with dropout reproducible per seed and worker count. 2.6× faster per epoch on
  4 cores (99.4 → 38.6 s on the docs corpus); evaluation 31.4 → 12.5 s with the same result.
- **Exact resume** (`pnpm lm train --resume <dir>`): a lossless checkpoint after every epoch
  (float64 weights, optimizer moments, schedule position, random streams, best model, run
  settings), written atomically; a resumed run is bit-identical to an uninterrupted one. Before,
  the transformer's optimizer state and schedule position were not saved anywhere.

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
