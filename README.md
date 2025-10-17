# Neuro‑Lingua DOMESTICA — v3.2.4 (EN)

**Browser‑native neural language model** built in React + TypeScript.

- SGD with **Momentum** or **Adam**
- **Dropout** (train‑only)
- **Top‑p** (nucleus) and **Top‑k** sampling with temperature
- **Session persistence**, onboarding tips, and downloadable **training-history CSVs**
- **Tokenizer presets** (Unicode/ASCII/custom) with import/export support
- **Agent** workflow: a single GitHub Action retrains the model and commits the updated JSON artifact

> This repository is intentionally simple: the only thing your agent does is **train** and **update** the model JSON.  
> UI and code are entirely in English. The tokenizer is language‑agnostic.

---

## Quickstart

```bash
# 1) Install
pnpm i  # or: npm i / yarn

# 2) Dev
pnpm dev

# 3) Quality
pnpm lint && pnpm test

# 4) Build
pnpm build && pnpm preview

# 5) Train (Node script, no browser needed)
pnpm train
```

The browser UI allows you to paste a training corpus and interact with the model.  
The Node training script (`scripts/train.ts`) reads from `data/corpus.txt` and writes the artifact to `models/neuro‑lingua‑v324.json`.

---

## Repo layout

```
.
├── data/
│   └── corpus.txt                  # training corpus for the agent
├── models/
│   └── neuro-lingua-v324.json       # latest trained model artifact
├── scripts/
│   └── train.ts                    # Node training script (ts-node)
├── src/
│   ├── lib/ProNeuralLM.ts          # the neural LM (framework-free)
│   ├── App.tsx                     # React UI (English)
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── README.md
├── LICENSE
└── .github/workflows/train-model.yml
```

---

## Agent / CI

- **Workflow**: `.github/workflows/train-model.yml`
- **Triggers**:
  - `workflow_dispatch` with inputs `epochs`, `optimizer`, `dropout`
  - `push` events that touch `data/corpus.txt`
- **Action**: install dependencies, run `pnpm train`, commit the updated JSON artifact (if changed) with the built-in `GITHUB_TOKEN`

Example manual dispatch:

```bash
gh workflow run train-model.yml \
  -f epochs=40 \
  -f optimizer=adam \
  -f dropout=0.15
```

Grant the workflow write access:

- Repository → Settings → Actions → General → Workflow permissions → **Read and write permissions**

---

## Notes

- The LM is educational and runs fully in the browser. It is **not** optimized for long texts.
- Browser sessions persist hyperparameters, tokenizer configuration, and training corpora via `localStorage`. Use the onboarding card in the UI to review import/export and pause/resume behaviour.
- Download the training-history CSV from the statistics panel to compare runs.
- The tokenizer uses a Unicode-aware rule by default. Override via environment when training headlessly: set `TOKENIZER_MODE=ascii` or provide `TOKENIZER_MODE=custom` with `TOKENIZER_PATTERN="[^a-z]+"`. The UI exposes the same presets and allows exporting/importing tokenizer JSON files.
- The Node training script now adds `<PAD>` to the vocabulary to match the browser experience.

Enjoy!
