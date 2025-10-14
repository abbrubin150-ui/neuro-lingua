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
- Add ESLint + Prettier (or Biome) with npm scripts and CI checks to keep the TypeScript codebase consistent.
- Introduce Vitest-based unit tests for the React UI (current tests only cover `ProNeuralLM`).
- Provide a sample corpus in `data/corpus.txt` that is longer than a single sentence so the training script demonstrates history and perplexity changes.

## 5. Future ideas
- Surface perplexity/accuracy history as a downloadable CSV to compare runs.
- Explore exporting/importing tokenizer configurations so ASCII vs Unicode modes can be swapped in the UI.
