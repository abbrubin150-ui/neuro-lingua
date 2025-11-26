# Next Coding Tasks for Neuro-Lingua

The core bilingual UI, Projects/Runs with Σ-SIG governance, transformer support, GPU acceleration, and explainability/visualization panels are already shipped. The next milestones focus on **testing, documentation alignment, and reliable exports**.

## 1. Strengthen governance + scenario coverage
- Add Testing Library specs that create a project/run, set HOLD/ESCALATE statuses in `DecisionLedgerEditor`, and confirm training is blocked until status is EXECUTE.
- Verify scenario prompts/results persist into run history after training in `App.tsx`.
- Add data-testid hooks if needed to keep selectors stable.

## 2. GPU/CPU parity and compatibility notes
- Write a regression test that trains `ProNeuralLM` with GPU tensors (when available) and on CPU, asserting loss/perplexity are within tolerance.
- Capture observed browser/driver coverage for WebGPU and reference it in the README next to the 2-5x speedup claim.

## 3. Trace export/import round-trip
- Use the live UI to export a model with projectMeta + decisionLedger + trainingTrace and re-import it.
- Update README + CHANGELOG_v3.3 so field names and version identifiers match the current runtime (v3.2.4) and the governance spec.
- Add a lightweight automated round-trip check to catch schema drift early.

## 4. Developer experience polish
- Surface links to corpus preparation notebooks and sample scenario suites from docs for quicker onboarding.
- Expand `scripts/train.ts` CLI flags to mirror tokenizer presets and governance metadata exposed in the UI.
- Keep lint/format/test scripts green (pnpm run check) and broaden component test coverage where practical.

## 5. Backlog ideas
- Clarify GPU fallback reasons inside `ModelMetrics` when WebGPU is unavailable.
- Add inline explainer text for decision statuses and scenario scores to reduce documentation hunting.
- Extend localization coverage for advanced settings beyond the existing translation map in `src/App.tsx`.
