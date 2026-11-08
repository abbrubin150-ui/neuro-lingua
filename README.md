# Neuro‑Lingua DOMESTICA — v4.4.0 (EN)

**Browser‑native neural language model** built in React + TypeScript.

> Runtime version: **v4.4.0** (UI + model) · Governance/export spec documented in **CHANGELOG_v3.3.md**. The bilingual UI, project/run management, and WebGPU acceleration in `App.tsx` match this runtime.

🌐 **[Try the live demo →](https://abbrubin150-ui.github.io/neuro-lingua/)** — includes an English ↔ Hebrew toggle for the UI

## Core Features

- **Multiple Architectures**: Standard ProNeuralLM, AdvancedNeuralLM, and fully-functional Transformer models with multi-head attention
- **Optional WebGPU offload** of the MLP's matrix-vector products with automatic CPU fallback. Each product uploads the weights and waits for the result, so for the small models used here the CPU is often as fast; the Transformer always trains on the CPU.
- **6 Optimizers**: SGD with Momentum, Adam, **Lion (v4.0)**, **Sophia (v4.3)**, Damped Newton (diagonal Gauss-Newton, experimental), L-BFGS
- **Dropout** (train‑only)
- **Advanced Text Generation**: Greedy, Sampling (Top-p/Top-k/Typical/Mirostat v2), Beam Search, and Contrastive decoding
- **Session persistence**, onboarding tips, and downloadable **training-history CSVs** (localized labels)
- **Tokenizer presets** (Unicode / ASCII / custom regex / character-level) with import/export support; a BPE tokenizer exists as a library module (`src/tokenizer/BPETokenizer.ts`) but is not wired into the models
- **Dataset abstraction** and **reproducibility layer** (v4.4): library modules with tests (splits, hashing, replay verification); not yet used by the UI training loop
- **Held-out evaluation**: `model.evaluate(text)` and `pnpm benchmark:models` ([results](./docs/benchmarks/MODEL_BENCHMARK.md))
- **Desktop app** (v4.4): Electron packaging for Windows with full feature parity
- **Agent** workflow: a single GitHub Action retrains the model and commits the updated JSON artifact

## ✅ Implementation status (verified)

What runs in the app, what is a standalone library, and what is simulated. Gradients of all three
architectures are checked against finite differences in the test suite.

| Area | Status |
| --- | --- |
| ProNeuralLM (MLP) | Trained with exact backprop. The context is the **mean** of the token embeddings, so the model is order-blind (see the benchmark). |
| AdvancedNeuralLM | Activation (ReLU/LeakyReLU/ELU/GELU), LayerNorm, LR schedules, weight decay and gradient clipping all act in the regular training path. |
| TransformerLM | Decoder-only causal transformer: causal MHA/GQA/MQA with per-head RoPE, pre-norm RMSNorm, SwiGLU, final RMSNorm, attention dropout + DropConnect, exact backprop, mini-batches. Models saved with the older objective load as `pooled` and keep their predictions. |
| Optimizers | Adam, momentum SGD, Lion, Sophia, L-BFGS and damped Newton act on the MLP models; the Transformer uses Adam / momentum / Lion (second-order choices fall back to Adam). |
| Explainability panel | Shapley values (exact for windows ≤ 10 tokens), integrated gradients on embeddings and attention rollout, all computed from the model. |
| Information-plane panel | Binning estimates of I(X;Z) and I(Z;Y) from hidden representations; monitoring only (the loss stays cross-entropy). |
| Edge-learning diagnostics | Empirical Fisher information of the output layer (exact per-window gradients), predictive entropy, loss variance and the diagonal Cramér-Rao bound, measured on the trained model. |
| Cerebro neuron injection | Works for all architectures; function-preserving for the MLP; rollback restores the exact model. |
| WebGPU | Offloads the MLP's matrix-vector products only; no measured speedup is claimed. |
| Causal analysis panel | Runs on data from the engine's built-in simulator (labelled in the UI). |
| Library modules not wired into the app | BPE tokenizer, dataset abstraction, reproducibility layer, sparse / linearized attention, K-FAC, mixed precision, `src/integrations` (simulated HuggingFace / Slack / Prometheus services), `src/autodiff`, `GovernanceBoard` component. |

## 🚀 Advanced Features

### Neural Network Architectures

- **🔮 Transformer**: decoder-only causal transformer trained with exact backpropagation — causal multi-head attention with per-head **RoPE**, **pre-attention/pre-FFN RMSNorm**, SwiGLU feed-forward, residual connections, **Grouped-Query Attention (GQA/MQA)**, final RMSNorm and next-token loss at every position (configurable layers and heads)
- **🚀 AdvancedNeuralLM**: feed-forward model with selectable activation, LayerNorm, LR schedules, weight decay and gradient clipping
- **📊 ProNeuralLM**: Standard baseline model

### Mathematical Enhancements

- ✅ **He/Xavier Initialization** - Faster convergence with proper weight init
- ✅ **Advanced Activations** - LeakyReLU, ELU, GELU
- ✅ **Learning Rate Scheduling** - Cosine annealing, exponential decay, warmup
- ✅ **L2 Regularization** - Weight decay for better generalization
- ✅ **Layer Normalization** - Training stability
- ✅ **RMSNorm (v4.0)** - Normalization without mean subtraction or bias (transformer blocks)
- ✅ **Lion Optimizer (v4.0)** - Sign-based optimizer with a single momentum buffer
- ✅ **Sophia Optimizer (v4.3)** - Diagonal-curvature optimizer with update clipping (β₁, β₂, ρ configurable in the UI)
- 🧩 **Mixed precision utilities (v4.3)** - FP16/FP32 helpers in `src/backend/mixed_precision.ts`; not used by the training loop
- ✅ **Numerical Stability** - Log-sum-exp, stable softmax
- ✅ **Perplexity Calculation** - Model evaluation metric

### 🔬 Mathematical Analysis Module (v4.3)

Rigorous mathematical analysis tools with formal guarantees:

- ✅ **Numerical Stability** - Kahan/Neumaier summation, stable norms, condition number estimation
- ✅ **Convergence Theorems** - Formal proofs for Sophia, Lion, and SGD optimizers with rate bounds
- ✅ **Information Theory** - KSG mutual information, k-NN entropy, rate-distortion curves
- ✅ **Sampling Analysis** - Entropy distribution tests, Mirostat convergence, temperature calibration
- ✅ **Spectral Graph Analysis** - Laplacian spectrum, Fiedler vectors, expander properties for attention patterns
- ✅ **Approximation Theory** - Wedin bounds, Nyström approximation, randomized SVD with error guarantees

### Text Generation Methods

- ✅ **Greedy Decoding** - Deterministic selection of most likely token (argmax)
- ✅ **Temperature Sampling** - Controlled randomness in generation
- ✅ **Top-k Sampling** - Sample from k most likely tokens
- ✅ **Nucleus (Top-p) Sampling** - Sample from smallest set with cumulative probability p
- ✅ **Typical Sampling** - Entropy-based token selection for natural coherence
- ✅ **Mirostat v2 Sampling** - Adaptive sampling with dynamic perplexity control
- ✅ **Beam Search** - Maintain multiple hypotheses for higher quality output
- ✅ **Contrastive Search** - Balance model confidence with diversity to reduce repetition

### Performance Optimization

- ✅ **WebGPU offload** - MLP matrix-vector products on the GPU (measure with the GPU metrics; small models are often as fast on the CPU)
- ✅ **GPU Metrics Dashboard** - Real-time performance monitoring
- ✅ **Automatic Fallback** - Seamless CPU fallback when GPU unavailable
- ✅ **Mini-batch transformer training** - gradient accumulation with one fused optimizer step per batch

### 🔀 Sparse Attention (v4.3)

Library module (`src/models/sparse_attention.ts`, with tests); TransformerLM uses dense causal
attention over its short context window. Patterns that reduce O(n²) attention cost:

- ✅ **Local (Sliding Window)** - Each token attends to w neighbors
- ✅ **BigBird** - Local + Global + Random attention (Google Research)
- ✅ **Longformer** - Local + Global with task-specific tokens
- ✅ **Block Sparse** - Block-diagonal for hardware efficiency
- ✅ **Strided/Dilated** - Hierarchical attention patterns
- ✅ **Axial** - Factorized 2D attention for image-like data

**Cost**: O(n·w) instead of O(n²) for window/block size w (not benchmarked in this repository).

#### WebGPU Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 113+ | ✅ Full support |
| Edge | 113+ | ✅ Full support |
| Opera | 99+ | ✅ Full support |
| Firefox | 127+ | ⚠️ Behind flag (`dom.webgpu.enabled`) |
| Safari | — | ❌ Not yet supported |

> **Note:** WebGPU requires a compatible GPU and up-to-date drivers. If GPU acceleration is unavailable, Neuro-Lingua automatically falls back to CPU with full functionality.

### 🎯 Experiment Management (Σ-SIG Compliance)

- ✅ **Project & Run Architecture** - Full experiment tracking with frozen configurations
- ✅ **Decision Ledger** - Governance framework with rationale, witness, and expiry tracking
- ✅ **Scenario Testing** - Define and evaluate test scenarios across multiple runs
- ✅ **Execution Status** - EXECUTE/HOLD/ESCALATE compliance checks
- ✅ **Trace Export** - Complete audit trail with model weights, configs, and metadata

### 🔍 Model Interpretability

- ✅ **Shapley Values** - Exact Shapley values of the context tokens for the predicted next token
- ✅ **Integrated Gradients** - Embedding-space attributions with the model's exact gradient
- ✅ **Attention Rollout** - Rollout over the Transformer's attention maps (residual-aware)
- ✅ **Explainability Panel** - Interactive UI for model interpretation

### 📊 Advanced Visualization

- ✅ **Embedding Visualization** - Interactive t-SNE and UMAP projections
- ✅ **Information Theory Panel** - I(X;Z) vs I(Z;Y) information plane
- ✅ **Information Bottleneck monitoring** - Binning estimates of the information plane (no IB loss term)
- ✅ **Canvas Interaction** - Pan, zoom, and explore embedding spaces

### 🧠 Cerebro: Adaptive Neuron Injection

- ✅ **Bubble Detection** - Identify residual clusters in embedding space
- ✅ **Neuron Injection** - Dynamically grow hidden layer to capture missed patterns
- ✅ **Architecture Adapters** - Works with ProNeuralLM, AdvancedNeuralLM, and TransformerLM
- ✅ **Rollback Support** - Undo an injection and restore the exact pre-injection weights
- ✅ **Decision Ledger Integration** - All injections logged for Σ-SIG compliance

### 🗜️ Model Compression

- ✅ **Int8 Quantization** - 4x size reduction with minimal accuracy loss
- ✅ **Knowledge Distillation** - Train smaller student models from teacher
- ✅ **Low-Rank Approximation** - SVD-based weight compression with configurable rank
- ✅ **Compression Panel** - Interactive UI for compression operations

### 🧬 Brain Vitals System

- ✅ **Mood Tracking** - CALM, FOCUSED, AGITATED, DREAMY, BURNT_OUT states
- ✅ **Creativity & Stability Gauges** - 0-100 vitals based on usage
- ✅ **Event Diary** - Logs training, generation, and feeding events
- ✅ **Pet Naming** - Assign friendly labels to model instances
- ✅ **Telemetry Panel** - Detailed brain analytics and trends

### 📊 DAG-Based Causal Inference (v4.2)

- ✅ **CausalInferenceEngine** - Probabilistic dynamic causal inference system
- ✅ **Directed Acyclic Graph (DAG)** - Explicit causal structure representation
- ✅ **Temporal Dependencies** - Y_{t-1} → Y_t autoregressive modeling
- ✅ **Unmeasured Confounders** - U-variable handling with sensitivity analysis
- ✅ **AIPW Estimator** - Augmented Inverse Probability Weighting for robust ATE
- ✅ **Adaptive Quantization** - Entropy-based dynamic discretization
- ✅ **Bias Verification** - Neutrality axiom and differential privacy checks
- ✅ **Three-Phase Pipeline** - Offline learning → Online selection → Statistical testing

### 🔤 BPE Tokenizer (v4.4)

- ✅ **Byte Pair Encoding** - Subword tokenization for better vocabulary coverage
- ✅ **Configurable Vocab Size** - Train tokenizers with custom vocabulary limits
- ✅ **Merge Rules** - Priority-based merge operations with full history
- ✅ **Tokenizer Metrics** - Coverage, entropy, fertility, compression ratio tracking
- ✅ **Versioned Artifacts** - Hash-verified tokenizer exports with corpus stats
- ✅ **Special Tokens** - Configurable PAD, BOS, EOS, UNK tokens

### 📦 Dataset Abstraction (v4.4)

- ✅ **Formal Dataset Schema** - Name, version, description, language, license metadata
- ✅ **Train/Val/Test Splits** - Configurable ratios with reproducible shuffling
- ✅ **Dataset Statistics** - Token counts, sample lengths, vocabulary analysis
- ✅ **Batch Iteration** - Configurable batch size with shuffle and drop-last options
- ✅ **Multi-Format Support** - Text, JSONL, CSV with UTF-8/ASCII encoding
- ✅ **Progress Callbacks** - Real-time loading, tokenizing, splitting, hashing feedback

### 🔁 Reproducibility Layer (v4.4)

- ✅ **Deterministic PRNG** - Seeded random number generator with state export
- ✅ **Config Hashing** - SHA-256 hashes for hyperparameters, architecture, tokenizer
- ✅ **Weight Hashing** - Layer-wise and global weight checksums
- ✅ **Reproducibility Manifest** - Complete training environment snapshot
- ✅ **Replay Verification** - Validate training runs match expected loss trajectories
- ✅ **Tolerance Levels** - Strict (1e-10) to loose (1e-2) numerical tolerance options

### 🖥️ Desktop Application (v4.4)

- ✅ **Electron Packaging** - Native Windows desktop application
- ✅ **Development Mode** - Hot-reload with `pnpm electron:dev`
- ✅ **Production Build** - NSIS installer with `pnpm electron:build`
- ✅ **Full Feature Parity** - All browser features available in desktop app

### 📄 Neuro-Lingua Lite

- ✅ **Single-File HTML** - Complete neural LM in one portable HTML file
- ✅ **Zero Dependencies** - No build tools, frameworks, or external files
- ✅ **Mobile Friendly** - Responsive design for all screen sizes
- ✅ **Quick Start** - Just open `neuro-lingua-lite.html` in any browser

📚 **[See full mathematical documentation →](./MATHEMATICAL_ENHANCEMENTS.md)**
🚀 **[v4.0 Mathematical & Architectural Upgrades (Hebrew) →](./NEURO_LINGUA_V4_UPGRADES.md)**
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

# 7) Desktop App (Windows)
pnpm electron:dev    # Development with hot-reload
pnpm electron:build  # Build NSIS installer
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
│   │   ├── gpu_neural_ops.ts       # High-level neural operations on GPU
│   │   └── mixed_precision.ts      # FP16/FP32 mixed precision training
│   ├── compression/
│   │   ├── compress.ts             # Unified compression interface
│   │   ├── quantization.ts         # Int8 weight quantization
│   │   ├── distillation.ts         # Knowledge distillation
│   │   └── lowrank.ts              # SVD-based low-rank approximation
│   ├── components/
│   │   ├── TrainingPanel.tsx       # Main training configuration panel
│   │   ├── ModelMetrics.tsx        # Performance metrics dashboard
│   │   ├── ProjectManager.tsx      # Project/run management (Σ-SIG)
│   │   ├── ScenarioManager.tsx     # Test scenario editor
│   │   ├── DecisionLedgerEditor.tsx # Governance/decision tracking
│   │   ├── ExplainabilityPanel.tsx # SHAP/gradients/attention visualization
│   │   ├── EmbeddingVisualizationPanel.tsx # t-SNE/UMAP interactive canvas
│   │   ├── InformationTheoryPanel.tsx # Information bottleneck metrics
│   │   ├── CompressionPanel.tsx    # Model compression UI
│   │   ├── BrainPanel.tsx          # Brain vitals and mood tracking
│   │   ├── CerebroPanel.tsx        # Cerebro neuron injection UI
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
│   │   ├── BrainEngine.ts          # Brain vitals and mood system
│   │   ├── GovernanceEngine.ts     # Autonomous parameter calibration
│   │   ├── CausalInferenceEngine.ts # DAG-based causal inference system
│   │   ├── RMSNorm.ts              # Root Mean Square Normalization
│   │   ├── storage.ts              # localStorage abstraction
│   │   ├── utils.ts                # Tokenizer and CSV utilities
│   │   ├── traceExport.ts          # Σ-SIG compliant experiment tracing
│   │   └── expandable/             # Cerebro neuron injection system
│   │       ├── InjectionEngine.ts  # Diagnostics, proposals, execution
│   │       ├── InjectableLayer.ts  # Minimal interface for injection
│   │       ├── ProNeuralLMAdapter.ts
│   │       ├── AdvancedNeuralLMAdapter.ts
│   │       ├── TransformerLMAdapter.ts
│   │       ├── bubbleExtractor.ts  # Extract bubbles from embeddings
│   │       └── injection_math.ts   # Residual analysis, eigenvectors
│   ├── losses/                     # Advanced loss functions
│   ├── math/
│   │   ├── index.ts                # Unified exports for math module
│   │   ├── numerics.ts             # Kahan summation, stable norms
│   │   ├── convergence.ts          # Optimizer convergence theorems
│   │   ├── information_theory.ts   # KSG MI, entropy, rate-distortion
│   │   ├── sampling_analysis.ts    # Mirostat analysis, temperature calibration
│   │   ├── spectral_graph.ts       # Laplacian spectrum, expander properties
│   │   ├── approximation.ts        # Wedin bounds, Nyström, randomized SVD
│   │   ├── analysis.ts             # Spectral/Lyapunov analysis
│   │   ├── statistics.ts           # Fisher information statistics
│   │   ├── causal_math.ts          # Causal inference mathematics
│   │   ├── dag_operations.ts       # DAG graph operations
│   │   └── bias_verification.ts    # Neutrality axiom verification
│   ├── models/
│   │   └── sparse_attention.ts     # Sparse attention patterns (BigBird, Longformer, etc.)
│   ├── training/
│   │   ├── LionOptimizer.ts        # Lion optimizer (v4.0)
│   │   └── SophiaOptimizer.ts      # Sophia second-order optimizer (v4.3)
│   ├── types/
│   │   ├── causal.ts               # Causal inference types
│   │   ├── dag.ts                  # DAG structure types
│   │   ├── tokenizer.ts            # BPE tokenizer types (v4.4)
│   │   ├── dataset.ts              # Dataset abstraction types (v4.4)
│   │   └── reproducibility.ts      # Reproducibility layer types (v4.4)
│   ├── visualization/              # Embedding visualization (t-SNE, UMAP)
│   ├── App.tsx                     # Main React application
│   └── main.tsx                    # Application entry point
├── electron/                       # Electron desktop app (v4.4)
│   ├── main.ts                     # Electron main process
│   ├── preload.ts                  # Preload script
│   └── tsconfig.json               # Electron TypeScript config
├── tests/
│   ├── ProNeuralLM.test.ts         # Core model tests
│   ├── AdvancedNeuralLM.test.ts    # Advanced features tests
│   ├── TransformerLM.test.ts       # Transformer tests
│   └── numerics/                   # Numerical correctness tests
├── index.html
├── neuro-lingua-lite.html          # Standalone lightweight version
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

Export models with complete audit trail including project metadata, decision ledger, training trace, and full reproducibility information. The current runtime (v4.4.0) emits the v3.3 governance/export fields described in `CHANGELOG_v3.3.md`.

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
- **[NEURO_LINGUA_V4_UPGRADES.md](./NEURO_LINGUA_V4_UPGRADES.md)** - v4.0 Mathematical & Architectural Upgrades (Hebrew)
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
