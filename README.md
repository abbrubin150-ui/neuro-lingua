# Neuro‑Lingua DOMESTICA — v3.2.4 (EN)

**Browser‑native neural language model** built in React + TypeScript.

🌐 **[Try the live demo →](https://abbrubin150-ui.github.io/neuro-lingua/)** — includes an English ↔ Hebrew toggle for the UI

## Core Features

- **Multiple Architectures**: Standard ProNeuralLM, AdvancedNeuralLM, and fully-functional Transformer models with multi-head attention
- **WebGPU Acceleration**: 2-5x faster training on compatible hardware with automatic CPU fallback
- SGD with **Momentum**, **Adam**, **Damped Newton**, or **L-BFGS** optimization
- **Dropout** (train‑only)
- **Advanced Text Generation**: Greedy, Sampling (Top-p/Top-k), Beam Search, and Contrastive decoding
- **Session persistence**, onboarding tips, and downloadable **training-history CSVs** (localized labels)
- **Tokenizer presets** (Unicode/ASCII/custom) with import/export support
- **Agent** workflow: a single GitHub Action retrains the model and commits the updated JSON artifact

## 🚀 Advanced Features

### Neural Network Architectures

- **🔮 Transformer**: Fully-implemented multi-head self-attention with position embeddings, residual connections, and layer normalization (configurable layers and heads)
- **🚀 AdvancedNeuralLM**: State-of-the-art feedforward architecture
- **📊 ProNeuralLM**: Standard baseline model

### Mathematical Enhancements

- ✅ **He/Xavier Initialization** - Faster convergence with proper weight init
- ✅ **Advanced Activations** - LeakyReLU, ELU, GELU, Swish
- ✅ **Learning Rate Scheduling** - Cosine annealing, exponential decay, warmup
- ✅ **L2 Regularization** - Weight decay for better generalization
- ✅ **Layer Normalization** - Training stability
- ✅ **Numerical Stability** - Log-sum-exp, stable softmax
- ✅ **Perplexity Calculation** - Model evaluation metric

### Text Generation Methods

- ✅ **Greedy Decoding** - Deterministic selection of most likely token (argmax)
- ✅ **Temperature Sampling** - Controlled randomness in generation
- ✅ **Top-k Sampling** - Sample from k most likely tokens
- ✅ **Nucleus (Top-p) Sampling** - Sample from smallest set with cumulative probability p
- ✅ **Beam Search** - Maintain multiple hypotheses for higher quality output
- ✅ **Contrastive Search** - Balance model confidence with diversity to reduce repetition

### Performance Optimization

- ✅ **WebGPU Acceleration** - Hardware-accelerated training with GPU
- ✅ **GPU Metrics Dashboard** - Real-time performance monitoring
- ✅ **Automatic Fallback** - Seamless CPU fallback when GPU unavailable

### 🎯 Experiment Management (Σ-SIG Compliance)

- ✅ **Project & Run Architecture** - Full experiment tracking with frozen configurations
- ✅ **Decision Ledger** - Governance framework with rationale, witness, and expiry tracking
- ✅ **Scenario Testing** - Define and evaluate test scenarios across multiple runs
- ✅ **Execution Status** - EXECUTE/HOLD/ESCALATE compliance checks
- ✅ **Trace Export** - Complete audit trail with model weights, configs, and metadata

### 🔍 Model Interpretability

- ✅ **SHAP Values** - Estimate feature importance using Shapley values
- ✅ **Integrated Gradients** - Attribution method for input token contributions
- ✅ **Attention Rollout** - Visualize attention flow in Transformer models
- ✅ **Explainability Panel** - Interactive UI for model interpretation

### 📊 Advanced Visualization

- ✅ **Embedding Visualization** - Interactive t-SNE and UMAP projections
- ✅ **Information Theory Panel** - I(X;Z) vs I(Z;Y) information plane
- ✅ **Information Bottleneck** - Compression-prediction trade-off analysis
- ✅ **Canvas Interaction** - Pan, zoom, and explore embedding spaces

📚 **[See full mathematical documentation →](./MATHEMATICAL_ENHANCEMENTS.md)**
📖 **[Transformer architecture guide →](./TRANSFORMER_GUIDE.md)**
⚡ **[GPU acceleration setup →](./GPU_ACCELERATION_GUIDE.md)**

> This repository is intentionally simple: the only thing your agent does is **train** and **update** the model JSON.
> The tokenizer is language‑agnostic, while the UI strings are available in English and Hebrew.

### Localization & translations

- Use the language toggle in the navbar to switch between English (LTR) and Hebrew (RTL) for the training editor, onboarding card, info cards, and chat console labels.
- Advanced configuration options, chart axes, and logs are currently English-only. Contributions that extend localization to those areas are welcome.
- Translations live directly in [`src/App.tsx`](./src/App.tsx) inside the `TRANSLATIONS` map so you can add new locales without touching other components.

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

# 6) GPU Benchmarks (optional)
pnpm benchmark:gpu
```

The browser UI allows you to paste a training corpus and interact with the model.  
The Node training script (`scripts/train.ts`) reads from `data/corpus.txt` and writes the artifact to `models/neuro‑lingua‑v324.json`.

---

## Repo layout

```
.
├── .github/workflows/
│   ├── ci.yml                      # Continuous integration (test, lint, build)
│   ├── train-model.yml             # Automated model retraining
│   └── deploy-pages.yml            # GitHub Pages deployment
├── data/
│   ├── corpus.txt                  # training corpus for the agent
│   ├── raw/                        # raw datasets (wikitext, hebrew_news)
│   └── processed/                  # preprocessed train/val/test splits
├── docs/
│   ├── experiments/                # experiment results and summaries
│   ├── theory/                     # theoretical documentation
│   └── visuals/                    # embedding visualization exports
├── models/
│   └── neuro-lingua-v324.json      # latest trained model artifact (3MB)
├── scripts/
│   ├── train.ts                    # Node training script (ts-node)
│   ├── benchmark_gpu.ts            # GPU performance benchmarks
│   └── visualize_embeddings.ts     # Generate t-SNE/UMAP visualizations
├── src/
│   ├── backend/
│   │   ├── webgpu.ts               # WebGPU backend and tensor operations
│   │   └── gpu_neural_ops.ts      # High-level neural operations on GPU
│   ├── components/
│   │   ├── TrainingPanel.tsx       # Main training configuration panel
│   │   ├── ModelMetrics.tsx        # Performance metrics dashboard
│   │   ├── ProjectManager.tsx      # Project/run management (Σ-SIG)
│   │   ├── ScenarioManager.tsx     # Test scenario editor
│   │   ├── DecisionLedgerEditor.tsx # Governance/decision tracking
│   │   ├── ExplainabilityPanel.tsx # SHAP/gradients/attention visualization
│   │   ├── EmbeddingVisualizationPanel.tsx # t-SNE/UMAP interactive canvas
│   │   ├── InformationTheoryPanel.tsx # Information bottleneck metrics
│   │   ├── ChatInterface.tsx       # Chat-style generation UI
│   │   └── TokenizerConfig.tsx     # Tokenizer settings
│   ├── contexts/
│   │   └── ProjectContext.tsx      # Project/run state management
│   ├── explainability/
│   │   ├── shap.ts                 # SHAP values
│   │   ├── integrated_gradients.ts # Integrated gradients
│   │   └── attention_rollout.ts    # Attention visualization
│   ├── generation/
│   │   ├── sampler.ts              # Top-k, top-p, temperature sampling
│   │   ├── beam_search.ts          # Beam search implementation
│   │   └── contrastive_search.ts   # Contrastive decoding
│   ├── lib/
│   │   ├── ProNeuralLM.ts          # Base feedforward LM
│   │   ├── AdvancedNeuralLM.ts     # Enhanced LM with advanced features
│   │   ├── TransformerLM.ts        # Transformer architecture
│   │   ├── MathUtils.ts            # Numerical stability utilities
│   │   ├── storage.ts              # localStorage abstraction
│   │   ├── utils.ts                # Tokenizer and CSV utilities
│   │   └── traceExport.ts          # Σ-SIG compliant experiment tracing
│   ├── losses/                     # Advanced loss functions
│   ├── training/                   # Optimization algorithms
│   ├── types/                      # TypeScript type definitions
│   ├── visualization/              # Embedding visualization (t-SNE, UMAP)
│   ├── App.tsx                     # Main React application
│   └── main.tsx                    # Application entry point
├── tests/
│   ├── ProNeuralLM.test.ts         # Core model tests
│   ├── AdvancedNeuralLM.test.ts    # Advanced features tests
│   ├── TransformerLM.test.ts       # Transformer tests
│   └── numerics/                   # Numerical correctness tests
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── CLAUDE.md                       # AI assistant development guide
├── MATHEMATICAL_ENHANCEMENTS.md    # Detailed math formulations
├── TRANSFORMER_GUIDE.md            # Transformer architecture explanation
├── GPU_ACCELERATION_GUIDE.md       # WebGPU setup and usage
├── README.md
└── LICENSE
```

---

## Agent / CI

- **Workflow**: `.github/workflows/train-model.yml`
- **Triggers**:
  - Manual `workflow_dispatch` inputs for `epochs`, `optimizer`, `dropout`
  - `push` events that touch `data/corpus.txt` or `scripts/train.ts`
- **Steps**:
  1. Checkout the repo with history so model diffs can be detected.
  2. Install dependencies via `pnpm install --frozen-lockfile`.
  3. Run `pnpm train` with those inputs provided as `EPOCHS`, `OPTIMIZER`, `DROPOUT` environment variables.
  4. Commit and push `models/neuro-lingua-v324.json` if the artifact changed.

Because the workflow pushes commits, the token it runs with must have `contents: write` access:

- Repository → Settings → Actions → General → Workflow permissions → **Read and write permissions**
- For forks, supply a PAT (for example `secrets.WORKFLOW_TOKEN`) and configure the job to use it before `git push`.

Example manual dispatch:

```bash
gh workflow run train-model.yml \
  -f epochs=40 \
  -f optimizer=adam \
  -f dropout=0.15
```

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

## 🎯 Experiment Management with Σ-SIG Compliance

Neuro-Lingua implements the **Σ-SIG (Scientific Infrastructure for Governance)** framework for reproducible experiment tracking:

### Projects & Runs

- **Projects** organize related training experiments with shared goals
- **Runs** capture frozen training configurations with complete snapshots:
  - All hyperparameters (frozen and immutable after creation)
  - Architecture configuration (ProNeuralLM/AdvancedNeuralLM/TransformerLM)
  - Tokenizer settings
  - Training corpus with checksum
  - Complete training history and results
  - Serialized model weights

### Decision Ledger

Every run includes a governance layer with:

- **Rationale**: Why this training run is necessary
- **Witness**: Who authorized the training (e.g., "local-user")
- **Expiry**: Optional expiration date (ISO 8601)
- **Rollback**: Action after expiry (keep/delete-after-expiry/archive)
- **Execution Status**: EXECUTE ✅ / HOLD ⏸️ / ESCALATE 🚨

### Scenario Testing

Define test scenarios per project:

- **Prompt**: Input text for generation
- **Expected Response**: Optional reference output
- **Scoring**: Track performance across multiple runs
- **Comparison**: Evaluate which configuration performs best

### Trace Export

Export models with complete audit trail including project metadata, decision ledger, training trace, and full reproducibility information. See `CHANGELOG_v3.3.md` for the complete export format specification.

---

## 🔍 Model Interpretability & Visualization

### Explainability Methods

- **SHAP Values**: Estimate token importance using Shapley value approximation
- **Integrated Gradients**: Attribution method measuring token contributions
- **Attention Rollout**: Visualize attention flow in Transformer models (multi-head support)

### Embedding Visualization

- **t-SNE Projection**: Interactive 2D visualization of token embeddings
- **UMAP Projection**: Alternative dimensionality reduction with configurable parameters
- **Canvas Interaction**: Pan, zoom, and explore embedding spaces
- **Export**: Save visualizations for documentation

### Information Theory

- **Information Bottleneck**: I(X;Z) vs I(Z;Y) information plane visualization
- **Compression-Prediction Trade-off**: Balance between model compression and prediction accuracy
- **Entropy Metrics**: H(Z), H(Z|X) tracking during training

---

## 📚 Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive AI assistant development guide
- **[MATHEMATICAL_ENHANCEMENTS.md](./MATHEMATICAL_ENHANCEMENTS.md)** - Detailed mathematical formulations
- **[TRANSFORMER_GUIDE.md](./TRANSFORMER_GUIDE.md)** - Transformer architecture deep dive
- **[TRANSFORMER_IMPLEMENTATION.md](./TRANSFORMER_IMPLEMENTATION.md)** - Implementation details
- **[GPU_ACCELERATION_GUIDE.md](./GPU_ACCELERATION_GUIDE.md)** - WebGPU setup and benchmarking
- **[DEVELOPMENT_SETUP_GUIDE.md](./DEVELOPMENT_SETUP_GUIDE.md)** - Development environment setup
- **[CHANGELOG_v3.3.md](./CHANGELOG_v3.3.md)** - Project & Run management release notes

---

## Notes

- The LM is educational and runs fully in the browser. It is **not** optimized for long texts.
- Browser sessions persist hyperparameters, tokenizer configuration, and training corpora via `localStorage`. Use the onboarding card in the UI to review import/export and pause/resume behaviour.
- Download the training-history CSV from the statistics panel to compare runs.
- The tokenizer uses a Unicode-aware rule by default. Override via environment when training headlessly: set `TOKENIZER_MODE=ascii` or provide `TOKENIZER_MODE=custom` with `TOKENIZER_PATTERN="[^a-z]+"`. The UI exposes the same presets and allows exporting/importing tokenizer JSON files.
- The Node training script now adds `<PAD>` to the vocabulary to match the browser experience.

Enjoy!
