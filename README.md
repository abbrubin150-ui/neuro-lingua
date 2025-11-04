# Neuro‑Lingua DOMESTICA — v3.2.4 (EN)

**Browser‑native neural language model** built in React + TypeScript.

🌐 **[Try the live demo →](https://abbrubin150-ui.github.io/neuro-lingua/)**

## Core Features

- **Multiple Architectures**: Standard ProNeuralLM, AdvancedNeuralLM, and Transformer models
- **WebGPU Acceleration**: 2-5x faster training on compatible hardware with automatic CPU fallback
- SGD with **Momentum**, **Adam**, **Damped Newton**, or **L-BFGS** optimization
- **Dropout** (train‑only)
- **Top‑p** (nucleus) and **Top‑k** sampling with temperature
- **Session persistence**, onboarding tips, and downloadable **training-history CSVs**
- **Tokenizer presets** (Unicode/ASCII/custom) with import/export support
- **Agent** workflow: a single GitHub Action retrains the model and commits the updated JSON artifact

## 🚀 Advanced Features

### Neural Network Architectures

- **🔮 Transformer**: Multi-head self-attention with position embeddings (2 layers, 4 heads)
- **🚀 AdvancedNeuralLM**: State-of-the-art feedforward architecture
- **📊 ProNeuralLM**: Standard baseline model

### Mathematical Enhancements

- ✅ **He/Xavier Initialization** - Faster convergence with proper weight init
- ✅ **Advanced Activations** - LeakyReLU, ELU, GELU, Swish
- ✅ **Learning Rate Scheduling** - Cosine annealing, exponential decay, warmup
- ✅ **L2 Regularization** - Weight decay for better generalization
- ✅ **Layer Normalization** - Training stability
- ✅ **Beam Search** - Higher quality text generation
- ✅ **Numerical Stability** - Log-sum-exp, stable softmax
- ✅ **Perplexity Calculation** - Model evaluation metric

### Performance Optimization

- ✅ **WebGPU Acceleration** - Hardware-accelerated training with GPU
- ✅ **GPU Metrics Dashboard** - Real-time performance monitoring
- ✅ **Automatic Fallback** - Seamless CPU fallback when GPU unavailable

📚 **[See full mathematical documentation →](./MATHEMATICAL_ENHANCEMENTS.md)**

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

## ⚠️ Important Warnings

### Privacy & Sensitive Data

**DO NOT use this application with:**

- Personally Identifiable Information (PII)
- Sensitive personal data
- Confidential business information
- Medical records or health data
- Financial information
- Authentication credentials or secrets

**Why?** This application stores training data and models in browser localStorage, which:

- Is not encrypted
- Persists across sessions
- Could be accessed by browser extensions or malicious scripts
- May be included in browser sync/backup

**Recommendation:** Use only public, non-sensitive text for training and experimentation.

---

## Notes

- The LM is educational and runs fully in the browser. It is **not** optimized for long texts.
- Browser sessions persist hyperparameters, tokenizer configuration, and training corpora via `localStorage`. Use the onboarding card in the UI to review import/export and pause/resume behaviour.
- Download the training-history CSV from the statistics panel to compare runs.
- The tokenizer uses a Unicode-aware rule by default. Override via environment when training headlessly: set `TOKENIZER_MODE=ascii` or provide `TOKENIZER_MODE=custom` with `TOKENIZER_PATTERN="[^a-z]+"`. The UI exposes the same presets and allows exporting/importing tokenizer JSON files.
- The Node training script now adds `<PAD>` to the vocabulary to match the browser experience.

Enjoy!
