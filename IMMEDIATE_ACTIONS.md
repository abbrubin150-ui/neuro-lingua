# Immediate Action Plan

**Priority:** Ship stability & traceability improvements this week  
**Updated:** 2025-11-07

## Quick Wins (Next 3-5 Days)

### 1) Harden GPU + CPU parity
- **Goal:** Confirm WebGPU and CPU execution paths stay in sync.
- **Actions:**
  - Add regression tests that train `ProNeuralLM` with and without GPU tensors and compare loss/perplexity within tolerance.
  - Smoke test on a WebGPU-capable browser and note observed driver/browser coverage.
  - Keep the 2-5x speedup claim in docs but pair it with the observed compatibility list.
- **Files:** `src/backend/webgpu.ts`, `src/backend/gpu_neural_ops.ts`, `tests/` (new parity spec).
- **Success:** Parity test passes in CI (CPU fallback) and locally when GPU is present; docs call out current GPU compatibility.

### 2) Project/Run UX validation
- **Goal:** Make Σ-SIG governance flows obvious and reliable in the live UI.
- **Actions:**
  - Add Testing Library coverage for `ProjectManager`, `DecisionLedgerEditor`, and `ScenarioManager` to prove training blocks on HOLD/ESCALATE statuses.
  - Verify scenario results persist into run history after training in `App.tsx`.
  - Document the workflow in the README so it matches the shipping UI.
- **Files:** `src/components/ProjectManager.tsx`, `src/components/DecisionLedgerEditor.tsx`, `src/components/ScenarioManager.tsx`, `src/App.tsx`, `tests/components/` (new specs).
- **Success:** Tests cover governance guardrails and scenario persistence; README text mirrors the UI labels and flow.

### 3) Export/Import audit trail check
- **Goal:** Keep trace exports aligned with live data shapes.
- **Actions:**
  - Generate a trace export from the running app and re-import it to confirm `projectMeta`, `decisionLedger`, and `trainingTrace` fields hydrate correctly.
  - Reconcile README trace/export notes with the v3.3 changelog while keeping the current model version (v3.2.4) visible.
  - Note any discrepancies for a follow-up patch instead of silently mutating data shapes.
- **Files:** `src/lib/traceExport.ts`, `src/types/project.ts`, `README.md`, `CHANGELOG_v3.3.md` (cross-reference only).
- **Success:** Round-trip works without manual edits; docs reference the same field names and version numbers used in the app.

## Validation Checklist
- [ ] GPU parity test added and green in CI (CPU) and locally (GPU-capable)
- [ ] Governance flow covered by component tests and blocks training on HOLD/ESCALATE
- [ ] Scenario scores persist into run history after training
- [ ] Trace export/import round-trips with Σ-SIG metadata intact
- [ ] README matches the in-app labels and version identifiers

## Risks & Mitigations
- **WebGPU availability varies:** Maintain CPU parity tests and keep the GPU toggle optional with clear messaging.
- **Schema drift:** Treat README + CHANGELOG as the contract and adjust code/tests together when fields change.
- **Test fragility:** Prefer data-testid hooks over brittle text queries in new component specs.
