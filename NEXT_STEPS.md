# Next Coding Tasks for Neuro-Lingua

The repository currently only contains the project README, so the next steps focus on scaffolding the actual application that the README describes.

## 1. Project Scaffolding
- Initialize a Vite React + TypeScript project structure (`pnpm create vite` or equivalent) to provide the `/src`, `/public`, and configuration files referenced in the README.
- Add base dependencies to `package.json`, including React, TypeScript, Vite, and state management/testing utilities as needed.

## 2. Core Library Implementation
- Create `src/lib/ProNeuralLM.ts` that houses the framework-independent neural language model logic described in the README (supporting SGD with Momentum/Adam, dropout, top-p/top-k sampling, pause/resume, and tokenizer customization).
- Write accompanying unit tests to verify training steps, sampling strategies, and serialization routines.

## 3. React UI
- Implement `src/App.tsx` and `src/main.tsx` to render the browser interface for loading corpora, kicking off training, monitoring history, and interacting with the model artifact in English as promised by the README.
- Wire the UI to browser storage (localStorage) so sessions can be saved/loaded.

## 4. Training Script
- Add the `scripts/train.ts` Node script that reads `data/corpus.txt`, trains the model headlessly, and outputs the JSON artifact under `models/`.
- Ensure the script respects environment variables such as `USE_ASCII_TOKENIZER` and exposes CLI flags for key hyperparameters (epochs, optimizer, dropout rate).

## 5. CI Workflow and Artifacts
- Populate `data/corpus.txt`, `models/.gitkeep`, and create the GitHub Action defined in the README that runs the training pipeline and commits updated artifacts.
- Document the workflow configuration and provide guidance on enabling "Read and write" permissions so the action can push commits.

## 6. Developer Experience Enhancements
- Add formatting and linting (Prettier/ESLint) plus a testing setup (Vitest or Jest) to keep the project maintainable.
- Provide scripts in `package.json` for development, build, test, lint, and training to match the commands advertised in the README.

Completing these steps will bring the repository in line with the capabilities outlined in the README and establish a solid foundation for future iterations of the Neuro-Lingua project.
