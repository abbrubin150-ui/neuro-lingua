# Neuro-Lingua Studio (`studio.html`)

A second, focused interface next to the classic UI (`index.html`, unchanged). Open
`/studio.html` (dev: `http://localhost:5173/studio.html`; Pages: `…/neuro-lingua/studio.html`).

Five steps in a side navigation: **Data → Model → Train → Generate → Models**. English / Hebrew
(RTL), light / dark, works at phone width. Shortcuts: Ctrl/⌘+Enter train or stop, Ctrl/⌘+G
generate.

## What it keeps (from the classic UI)

- The three architectures, BPE / character / word tokenizers, Adam or momentum SGD.
- Presets: transformer `tiny` / `small` (the `pnpm lm train` sizes, docs/benchmarks/MODEL_SIZE.md)
  and a fast MLP.
- Held-out validation, early stopping and best-epoch restore (`splitForValidation`,
  `evaluateInSlices`), with the exact held-out bits per character reported at the end.
- "Continue training" on the current model (refused when a word/character text brings tokens the
  model does not know).
- Sampling: greedy, top-k, top-p, typical, with the no-repeat n-gram default
  (`recommendedNoRepeatNgram`).
- Save / load in the same browser storage slots as the classic UI, `.nlm.json` packages with a
  model card, full JSON export, import of packages and JSON exports.

## What it leaves out

Options that are experimental, simulated or unmeasured in this repository: projects and the
decision ledger, Cerebro injection, brain / triadic / causal panels, information-plane monitoring,
EMA weights, WebGPU (no measured speedup), focal / label-smoothing / SCE losses, loss masks, Lion /
Sophia / Newton / BFGS, Mirostat, contrastive and beam search, and the Monte-Carlo "confidence"
(sample diversity is not a calibrated confidence). They remain available in the classic UI.

## Code

`src/studio/`: `trainer.ts` (training loop without React), `modelIO.ts` (storage, import /
export, generation), `settings.ts` (options, presets, validation), `strings.ts` (EN/HE),
`StudioApp.tsx`, `components.tsx`, `studio.css`. Tests: `tests/studio/`. Vite builds both pages
(`rollupOptions.input`), and the offline service worker precaches `studio.html`.
