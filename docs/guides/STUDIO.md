# Neuro-Lingua Studio (the main page)

The Studio is the app's main page (`index.html`; dev: `http://localhost:5173/`; Pages:
`https://abbrubin150-ui.github.io/neuro-lingua/`). The previous interface is unchanged and lives
at `classic.html` (linked from the Studio's top bar). `studio.html` redirects old links to the
main page. The desktop app opens the Studio too (`dist/index.html`).

Five steps in a side navigation: **Data → Model → Train → Generate → Models**. English / Hebrew
(RTL), light / dark, works at phone width. Shortcuts: Ctrl/⌘+Enter train or stop, Ctrl/⌘+G
generate.

## What it keeps (from the classic UI)

- The three architectures, BPE / character tokenizers, Adam or momentum SGD. Default model: the
  `tiny` transformer (13x faster than `small` for 4.4% worse held-out bpc; it fits the browser
  storage at full precision, `small` only at half precision).
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

Word tokens: 41% (English) to 73% (Hebrew) of held-out words were unknown in the repository's
corpora, which made their bits per character look better than BPE while the model was worse.
Word-level models from files or the classic UI still load and run.


Options that are experimental, simulated or unmeasured in this repository: projects and the
decision ledger, Cerebro injection, brain / triadic / causal panels, information-plane monitoring,
EMA weights, WebGPU (no measured speedup), focal / label-smoothing / SCE losses, loss masks, Lion /
Sophia / Newton / BFGS, Mirostat, contrastive and beam search, and the Monte-Carlo "confidence"
(sample diversity is not a calibrated confidence). They remain available in the classic UI.

## Code

`src/studio/`: `trainer.ts` (training loop without React), `modelIO.ts` (storage, import /
export, generation), `settings.ts` (options, presets, validation), `strings.ts` (EN/HE),
`StudioApp.tsx`, `components.tsx`, `studio.css`. Tests: `tests/studio/`. Vite builds all three pages
(`rollupOptions.input`), and the offline service worker precaches them.
