# Neuro-Lingua Development Roadmap

**Generated:** 2025-11-07  
**Project:** Neuro-Lingua DOMESTICA v3.2.4 (Projects/Runs + Σ-SIG live)

## Executive Summary
Neuro-Lingua now ships a fully bilingual React UI with ProNeuralLM, AdvancedNeuralLM, TransformerLM, WebGPU acceleration, Σ-SIG governance (Projects/Runs/Decision Ledger), scenario testing, explainability panels, and t-SNE/UMAP visualizations. The next milestones emphasize **stability, reproducibility, and UX clarity** rather than net-new architectures.

## Current State Assessment
### Strengths
- GPU acceleration, transformer training, and bilingual UI are live in `App.tsx`.
- Project/Run management with Decision Ledger + Scenario Manager ships in the main UI.
- Trace export/import includes projectMeta, decisionLedger, and trainingTrace fields.
- Tests cover core models and numerics; lint/format/test scripts are wired in CI.
- Browser-first experience with persisted settings, tokenizer presets, and downloadable history CSV.

### Key Gaps
- Limited automated coverage for governance flows (Projects/Runs/Scenarios) and GPU/CPU parity.
- Documentation drift between README and changelog vs. the actual UI labels/version (3.2.4 runtime, 3.3 changelog notes).
- Export/import round-trip is not routinely validated in tests.
- GPU support messaging is generic; observed compatibility/perf data is missing.
- Data pipeline notebooks exist but are not linked from the UI or docs for repeatable corpus prep.

---

## Phase 1: Reliability & Traceability (1-2 weeks)
**Goal:** Prove that the shipping UI, exports, and governance rules behave predictably.

- Add Testing Library specs for ProjectManager, DecisionLedgerEditor, and ScenarioManager to ensure training blocks on HOLD/ESCALATE and scenarios persist into run history.
- Create a parity test that trains ProNeuralLM on CPU vs. GPU tensors (where available) and asserts loss/perplexity are within tolerance.
- Exercise trace export/import in tests to catch schema drift (projectMeta + decisionLedger + trainingTrace).
- Update README + CHANGELOG_v3.3 to reference the same field names and to clarify the current runtime version (v3.2.4) while documenting v3.3 governance spec.
- Document observed WebGPU compatibility (browsers/driver notes) without weakening the 2-5x speedup claim.

**Exit Criteria:**
- Governance + scenario flows covered by automated tests.
- Export/import round-trip passes in CI.
- README/changelog aligned with the live UI and version identifiers.
- GPU parity test passes on CPU fallback and locally on a GPU-capable browser.

## Phase 2: Dataset & Experiment Ops (2-3 weeks)
**Goal:** Make repeatable experiments easy for contributors.

- Link corpus preparation notebooks from docs and surface recommended dataset sizes in the UI helper text.
- Add CLI flags to `scripts/train.ts` for tokenizer presets and project/run metadata to mirror the browser configuration.
- Provide sample scenario suites in `data/` that match the bilingual UI (EN/HE) for quick validation.
- Expand benchmarking script to log CPU vs. GPU metrics to `logs/` for comparison.

**Exit Criteria:**
- Contributors can reproduce browser runs via `pnpm train` with matching tokenizer and governance metadata.
- Sample scenario suites load without manual edits.
- Benchmark logs show consistent metric formats for CPU and GPU.

## Phase 3: UX Polish & Observability (ongoing)
**Goal:** Keep the UI approachable and diagnosable.

- Add inline explainer text for Decision Ledger statuses and scenario scoring results.
- Surface GPU fallback reasons in ModelMetrics when WebGPU is unavailable.
- Provide lightweight telemetry hooks (console/debug logs) gated behind a dev toggle for performance investigations.
- Continue improving localization coverage beyond the current translation map in `src/App.tsx`.

**Exit Criteria:**
- Users understand why training is blocked or why GPU is unavailable without reading source code.
- Localization coverage increases (at least key advanced settings translated).
- Debug hooks reduce time-to-diagnose for performance regressions.
