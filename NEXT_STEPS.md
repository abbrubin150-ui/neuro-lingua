# Next Coding Tasks for Neuro-Lingua

The core scaffolding, training script, and neural LM implementation now exist. The next milestones should bring the implementation in line with the README promises and polish the developer experience.

## 1. Align the React UI with the README
- Translate all Hebrew copy in `src/App.tsx` to English so the in-browser experience matches the English-only promise in the README.
- Replace the hard-coded Hebrew demo corpus with a concise English example and add inline guidance in English for new users.
- Double-check that UI labels, notifications, and the training chart use consistent English terminology.

## 2. Improve persistence & onboarding
- Persist the last-used hyperparameters and training corpus to `localStorage` so that reloading the page restores the session fully.
- Add a lightweight onboarding panel (or tooltip) that explains how to load/import/export models and how pause/resume behaves.
- Document within the UI when the stored model was last updated (timestamp + vocab size) to help users decide whether to retrain.

## 3. GitHub Action automation
- Replace the placeholder `.github/workflows/jekyll-docker.yml` with a `train-model.yml` workflow that installs dependencies, runs `pnpm train`, and commits the updated artifact back to the repo.
- Ensure the workflow exposes inputs for key hyperparameters (epochs, optimizer, dropout) so retraining can be tuned without modifying the repo.
- Update the README with any workflow usage notes (e.g., required repository permissions) once the workflow exists.

## 4. Developer experience enhancements
- Keep the new ESLint + Prettier setup green by running `pnpm lint`, `pnpm format:check`, or the aggregate `pnpm check` locally before opening a PR. The CI workflow now calls the same script.
- Continue expanding Vitest coverage for the React UI (new specs live in `tests/components`). Prioritize complex flows like the Scenario Manager, Decision Ledger, and GPU toggles.
- When modifying `data/corpus.txt`, maintain the multi-paragraph English sample so `scripts/train.ts` continues to emit noticeable history/perplexity shifts for demos.

## 5. Future ideas
- Surface perplexity/accuracy history as a downloadable CSV to compare runs.
- Explore exporting/importing tokenizer configurations so ASCII vs Unicode modes can be swapped in the UI.
