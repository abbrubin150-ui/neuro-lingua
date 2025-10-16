# Neuro‑Lingua DOMESTICA — v3.2.4 (EN)

**Browser‑native neural language model** built in React + TypeScript.
- SGD with **Momentum** or **Adam**
- **Dropout** (train‑only)
- **Top‑p** (nucleus) and **Top‑k** sampling with temperature
- **Pause/Resume**, **training history** and localStorage **save/load**
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

# 3) Build
pnpm build && pnpm preview

# 4) Train (Node script, no browser needed)
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

- **Trigger**: `workflow_dispatch` (with configurable inputs) or on changes to `data/corpus.txt`
- **Action**: run `pnpm train` → write `models/neuro‑lingua‑v324.json` → commit back using the default `GITHUB_TOKEN`
- **Manual runs**: provide optional `epochs`, `optimizer` (`momentum`/`adam`) and `dropout` values when dispatching to override defaults without editing the repo.

The workflow pushes directly to the repository, so ensure it has permission to write:
- Repo → Settings → Actions → General → Workflow permissions → **Read and write permissions**

---

## Notes

- The LM is educational and runs fully in the browser. It is **not** optimized for long texts.
- The tokenizer uses a Unicode‑aware rule by default. If you want ASCII‑only, set `USE_ASCII_TOKENIZER=true` in env (for the Node script) or tweak the regex in `ProNeuralLM.ts`.

Enjoy!
