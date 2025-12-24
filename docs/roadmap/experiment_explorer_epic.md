# Σ-SIG Experiment Explorer: Epic Brief

This document outlines a low-risk, high-leverage epic to extend **ProjectManager** and **ModelMetrics** into a full experiment comparison and export surface. The focus is on improving transparency, reproducibility, and governance without altering the core model implementations.

## Goals
- Compare runs across projects with unified views of hyperparameters, metrics, and outputs.
- Capture decision context (KPI, alternatives considered, affected runs) in a structured ledger.
- Enable export of experiment data to JSON/CSV for offline analysis and reporting.
- Provide diff-style insights for quick evaluation of changes between runs.

## Non-Goals
- Changing core training logic or model architectures.
- Introducing new compression or tokenizer algorithms (tracked separately).
- Replacing existing persistence mechanisms; initial scope assumes current storage/local mechanisms.

## Functional Scope
1. **Run Comparison Panel**
   - Select multiple runs (2–3) from ProjectManager and view a side-by-side diff of hyperparameters and key metrics (loss, perplexity, runtime, model size).
   - Persist comparison “scenarios” so users can re-run or revisit the same run sets.

2. **Decision Ledger 2.0 Fields**
   - Structured fields: Problem, Alternatives Considered, KPI Used for Decision, Runs Applied.
   - Filters: e.g., only decisions affecting compression, or optimizer-related decisions.

3. **Data Export**
   - Button to export ProjectManager state (projects, runs, decisions, scenarios) as JSON and CSV.
   - Ensure exported schema is documented and versioned for backward compatibility.

4. **ModelMetrics Enhancements**
   - Display diffs for chosen runs: hyperparameter changes, loss/perplexity deltas, runtime deltas.
   - Hooks for future integration of GPU/CPU timing and fallback counts (from WebGPU profiler work).

## UX Notes
- Integrate comparison and export actions into existing ProjectManager UI to avoid new navigation complexity.
- Provide safe defaults: when no runs are selected, show guidance on how to create comparisons.
- Keep Expert vs. Beginner modes in mind: expose full diff and export options in Expert mode first.

## Technical Notes
- Add typed structures for `ExperimentScenario`, `DecisionEntry`, and `RunDiff` in the shared types module.
- Reuse existing run metadata wherever possible; avoid duplicating storage.
- Keep the export path pure-TS/JSON to allow future Python ingestion (e.g., `edge_formalism/`).
- Prepare tests for diff generation (hyperparameter and metric comparison) and schema validation for export payloads.

## Milestones
1. **Data Layer**: types, diff utilities, and export format definition (JSON/CSV).
2. **UI Layer**: run selection UI, diff view in ModelMetrics, and comparison scenario save/load.
3. **Ledger Upgrade**: structured decision fields and filtering in ProjectManager.
4. **Export Surface**: JSON/CSV export with schema docs and sample artifacts.

## Risks & Mitigations
- **Scope creep**: keep compression/tokenizer work out of scope; focus on comparison/export.
- **Performance**: defer heavy computations; pre-compute diffs when selections change.
- **Compatibility**: version exported schema and ensure older saved scenarios still load.

## Follow-On Work (Out of Scope for This Epic)
- GPU profiler integration, fused ops benchmarking, and worker-based training orchestration.
- Compression Playground A/B simulator and presets.
- Tokenizer training wizard and multilingual presets.
- Edge Formalism dashboard and auto-export.
