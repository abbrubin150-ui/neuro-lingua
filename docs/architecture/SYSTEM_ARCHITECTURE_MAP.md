# System Architecture Map

> Neuro-Lingua DOMESTICA v4.4.0 — Module map, responsibilities, and dependency graph.

---

## High-Level Overview

```
                        +------------------+
                        |    index.html    |
                        |    main.tsx      |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |     App.tsx      |   <-- Central orchestrator (2,629 lines)
                        +--------+---------+
                                 |
         +-----------+-----------+-----------+-----------+
         |           |           |           |           |
    components/  contexts/    lib/       backend/    types/
         |           |           |           |           |
         +-----------+-----------+-----------+-----------+
                                 |
         +-----------+-----------+-----------+-----------+
         |           |           |           |           |
      models/    training/  generation/   losses/   tokenizer/
         |           |           |           |           |
         +-----------+-----------+-----------+-----------+
                                 |
         +-----------+-----------+-----------+-----------+
         |           |           |           |           |
       math/   compression/ explainability/ integrations/ visualization/
         |           |           |           |           |
         +-----------+-----------+-----------+-----------+
                                 |
                  +--------------+--------------+
                  |              |              |
               autodiff/       data/        config/
```

---

## Module Directory

### 1. `src/App.tsx` — Application Orchestrator

| Attribute | Value |
|-----------|-------|
| Lines | ~2,629 |
| Role | Central hub wiring all panels, contexts, and models |

- Instantiates `ProNeuralLM`, `TransformerLM`, `AdvancedNeuralLM`
- Manages global UI state (active tab, language, theme)
- Delegates to 26+ component panels
- Wraps content with `BrainContext` and `ProjectContext` providers

---

### 2. `src/lib/` — Core Neural Network Library (24 files)

The heart of the system. Contains model implementations, governance logic, and brain state management.

| File | Responsibility |
|------|----------------|
| **ProNeuralLM.ts** | Baseline feedforward LM: embedding → hidden (ReLU) → output. Supports multiple optimizers (SGD, Adam, Newton, BFGS, Lion, Sophia). GPU acceleration via `GPUNeuralOps`. |
| **TransformerLM.ts** | Transformer architecture extending `ProNeuralLM`. Multi-head self-attention, GQA, positional encoding, RMSNorm, SwiGLU FFN. |
| **AdvancedNeuralLM.ts** | Enhanced feedforward extending `ProNeuralLM`. Advanced activations (GELU, ELU), LR scheduling, beam search, contrastive decoding. |
| **GovernanceEngine.ts** | Autonomous parameter calibration with Sigma-SIG compliance. Detects plateau/overfitting/divergence. Generates calibration actions with audit trail. |
| **BrainEngine.ts** | Model state tracking: mood (CALM/FOCUSED/AGITATED/DREAMY/BURNT_OUT), vitals (creativity, stability), event reducer pattern, diary. |
| **BrainGovernanceBridge.ts** | Connects brain mood/vitals with governance parameter calibration. Mood-modulated governance with action queuing. |
| **CausalInferenceEngine.ts** | Probabilistic causal inference: DAG modeling, propensity scoring, AIPW estimation, hypothesis testing. |
| **GovernanceAndTelemetry.ts** | Sigma-SIG compliance: R_ANCHOR_GATE, CANON5 bundles, C68 silence windows, presence stamps, mirror veto. |
| **KernelPrimitives.ts** | Low-level kernel operations for governance decision state machines. |
| **MathUtils.ts** | Pure math: Xavier/He init, activations, stable softmax, log softmax, cosine annealing, beam search. |
| **RMSNorm.ts** | Root Mean Square normalization (20% less memory than LayerNorm). Forward/backward with gradient support. |
| **Conv2DMatrix.ts** | 2D convolution operations on weight matrices. |
| **MatrixConvolution.ts** | Matrix-based convolution utilities. |
| **storage.ts** | Type-safe `localStorage` abstraction (`StorageManager`). |
| **utils.ts** | Utility functions: tokenizer config parsing, timestamps, CSV generation, file downloads. |
| **Reproducibility.ts** | Deterministic execution: seeded RNG, reproducibility tracking. |
| **diffUtils.ts** | Difference/delta computation between model states. |
| **exportUtils.ts** | Model export utilities (JSON serialization). |
| **experimentComparison.ts** | Run comparison and hyperparameter diff computation. |
| **traceExport.ts** | Execution trace/history export. |
| **triadicOperator.ts** | Triadic operator logic (3-way domain relationships). |
| **EvidenceAndMetrics.ts** | Evidence collection and metric computation. |
| **GenerativityTests.ts** | Test generation for output quality analysis. |
| **expandable/** | Neuron injection system: `InjectableLayer`, `InjectionEngine`, model adapters (Pro/Advanced/Transformer), `bubbleExtractor`. |

**Inheritance hierarchy:**
```
ProNeuralLM
├── TransformerLM   (extends)
└── AdvancedNeuralLM (extends)
```

---

### 3. `src/models/` — Attention Mechanisms & Regularization (6 files)

| File | Responsibility |
|------|----------------|
| **attention.ts** | Multi-head attention with GQA, RoPE positional embeddings, optional dropout. GPU/CPU fallback. |
| **sparse_attention.ts** | 7 sparse attention patterns: Local, Strided, Dilated, BigBird, Longformer, Block Sparse, Axial. O(n) vs O(n^2). |
| **linearized_attention.ts** | O(n) kernel-based attention: ELU, RFF (Random Fourier Features), FAVOR+, Chebyshev. |
| **mini_transformer.ts** | Single transformer block: pre-norm RMSNorm, multi-head attention, SwiGLU FFN, residual connections. |
| **regularizers.ts** | DropConnect (weight masking) and Batch Renormalization. |
| **index.ts** | Barrel re-exports. |

---

### 4. `src/training/` — Optimizers & Training Utilities (7 files)

| File | Responsibility |
|------|----------------|
| **optimizer.ts** | Second-order optimization: L-BFGS, damped Newton step, diagonal Hessian, parameter flattening. |
| **LionOptimizer.ts** | Sign-based momentum optimizer. 50% less memory than Adam. |
| **SophiaOptimizer.ts** | Second-order stochastic optimizer. Diagonal Gauss-Newton Hessian. 2x faster convergence. |
| **KFACOptimizer.ts** | Kronecker-factored approximate curvature (natural gradient). Cholesky-based curvature inversion. |
| **GradientClipping.ts** | Global norm and per-parameter gradient clipping. Returns statistics. |
| **EMAWeights.ts** | Exponential moving average of training weights (Polyak averaging). |
| **injection_hooks.ts** | Weight injection session management. Snapshot, propose, execute, undo. |

---

### 5. `src/backend/` — GPU Acceleration Layer (4 files)

| File | Responsibility |
|------|----------------|
| **webgpu.ts** | Low-level WebGPU: device init, buffer management, WGSL compute shaders (matmul, softmax, elementwise). |
| **gpu_neural_ops.ts** | High-level GPU ops singleton: matrix-vector multiplication, automatic CPU fallback, metrics tracking. |
| **mixed_precision.ts** | FP16/FP32 training: bit-level conversion, dynamic loss scaling, master weights strategy. |
| **edgeLearning.ts** | Information-theoretic diagnostics: Fisher Information, entropy, Cramer-Rao bound, learning efficiency. |

---

### 6. `src/generation/` — Text Generation (1 file)

| File | Responsibility |
|------|----------------|
| **sampling.ts** | 8+ decoding strategies: greedy, temperature, top-k, nucleus, Mirostat v2, typical sampling, beam search, contrastive search, Monte Carlo. Repetition penalties. |

---

### 7. `src/losses/` — Loss Functions (4 files)

| File | Responsibility |
|------|----------------|
| **advanced.ts** | Focal loss, label smoothing CE, symmetric CE, cosine embedding loss. |
| **lossMask.ts** | Answer-only training: mask tokens before delimiter so loss only applies to answers. |
| **information_bottleneck.ts** | IB loss: L_IB = -I(Z;Y) + beta * I(X;Z). Beta scheduling. MI estimation. |
| **variational_ib.ts** | Variational IB: reparameterization trick, KL divergence, MINE/InfoNCE/NWJ MI estimators, rate-distortion curves. |

---

### 8. `src/tokenizer/` — Byte Pair Encoding (2 files)

| File | Responsibility |
|------|----------------|
| **BPETokenizer.ts** | BPE implementation: train from corpus, encode/decode, metrics (coverage, entropy, fertility, compression), artifact versioning with SHA-256. |
| **index.ts** | Barrel exports with types. |

---

### 9. `src/components/` — React UI Panels (26 files)

| Component | Panel/Feature |
|-----------|---------------|
| **TrainingPanel.tsx** | Main training interface (2,277 lines). Hyperparameter controls, training loop, metrics display. |
| **BrainPanel.tsx** | Brain vitals dashboard: creativity, stability, mood, feeding, diary. |
| **BrainTelemetryPanel.tsx** | Brain telemetry dashboard with autonomous ticker. |
| **ChatInterface.tsx** | Conversational UI with generation controls, bilingual (en/he). |
| **ProjectManager.tsx** | Project CRUD, run management, scenario management. |
| **CausalAnalysisPanel.tsx** | DAG visualization, AIPW estimation, sensitivity analysis. |
| **ModelMetrics.tsx** | Real-time training metrics: loss, accuracy, GPU stats. |
| **GovernanceBoard.tsx** | Alerts, calibration history, governance ledger. |
| **ExplainabilityPanel.tsx** | Token attribution: SHAP, integrated gradients, attention rollout. |
| **CompressionPanel.tsx** | Int8 quantization, low-rank SVD, knowledge distillation. |
| **CerebroPanel.tsx** | Layer injection: propose, inject, undo neurons. |
| **CerebroBubbleGraph.tsx** | 2D/3D concept bubble visualization. |
| **EmbeddingVisualizationPanel.tsx** | t-SNE/UMAP embedding projections. |
| **InformationTheoryPanel.tsx** | Information bottleneck curves and entropy. |
| **ExportPanel.tsx** | JSON/CSV export for projects, runs, comparisons. |
| **RunComparisonPanel.tsx** | Side-by-side hyperparameter and metrics diff. |
| **DecisionEntry2Panel.tsx** | Decision logging: alternatives, KPIs, affected runs. |
| **DecisionLedgerEditor.tsx** | Sigma-SIG editor: rationale, witness, expiry. |
| **ScenarioManager.tsx** | Test scenarios: add/delete/run with expected responses. |
| **TokenizerConfig.tsx** | Tokenizer mode selection, regex validation, import/export. |
| **TriadicOperatorPanel.tsx** | Triadic operator table visualization. |
| **ModelSnapshot.tsx** | Frozen model state card. |
| **OnboardingCard.tsx** | Dismissible tutorial card. |
| **OnboardingTooltip.tsx** | Context-sensitive help tooltip. |
| **ErrorBoundary.tsx** | React error boundary wrapper. |

---

### 10. `src/contexts/` — React State Management (2 files)

| Context | State Managed |
|---------|---------------|
| **BrainContext.tsx** | Brain vitals, mood, suggestions, autopilot level, health score, pending actions. Hooks: `useBrain()`, `useBrainTraining()`, `useBrainFeed()`, `useBrainAutoPilot()`, `useBrainHealth()`. |
| **ProjectContext.tsx** | Projects, runs, comparisons, decisions, governance state. GovernanceEngine integration. Hooks: `useProjects()`, `useCreateScenario()`. Persistence to localStorage. |

---

### 11. `src/types/` — Type Definitions (14 files)

| File | Key Types |
|------|-----------|
| **project.ts** | `Project`, `Run`, `TrainingConfig`, `DecisionLedger`, `ExecutionStatus`, `Scenario`, `LossMaskConfig` |
| **governance.ts** | `BoardAlert`, `CalibrationAction`, `GovernanceLedgerEntry`, `GovernorConfig`, `MetricSnapshot` |
| **experiment.ts** | `ExperimentComparison`, `DecisionEntry`, `HyperparameterDiff`, `FieldDiff<T>` |
| **injection.ts** | `InjectionTarget`, `InjectionProposal`, `InjectionEvent`, `CerebroBubble` |
| **causal.ts** | `FeatureVector`, `PolicySelection`, `CausalModelConfig`, `GroupEffect` |
| **kernel.ts** | Kernel operation types (44KB) |
| **dag.ts** | DAG node/edge types, identifiability analysis |
| **triadic.ts** | `TriadicVector`, `TriadicCell`, `TriadicDomain`, `TriadicTable` |
| **dataset.ts** | Dataset loading and preprocessing |
| **tokenizer.ts** | BPE config, artifact, merge rules, metrics |
| **modelMeta.ts** | Model metadata and comparison |
| **reproducibility.ts** | Seed management |
| **conv2d.ts** | 2D convolution types |

---

### 12. `src/math/` — Mathematical Analysis (14 files, ~258KB)

| File | Responsibility |
|------|----------------|
| **numerics.ts** | Kahan/Neumaier summation, matrix norms, condition number, stable variance. |
| **convergence.ts** | Formal convergence proofs: Sophia, Lion, SGD. Lipschitz/strong convexity bounds. |
| **information_theory.ts** | KSG mutual information, K-NN entropy, rate-distortion, information plane, Fisher information. |
| **causal_math.ts** | Statistical utilities for causal inference (35KB). |
| **dag_operations.ts** | DAG validation, Rosenbaum bounds, E-values (33KB). |
| **ntk_analysis.ts** | Neural tangent kernel theory (20KB). |
| **spectral_graph.ts** | Graph Laplacian, attention spectral analysis, expander properties. |
| **sampling_analysis.ts** | Entropy, KL divergence, Mirostat convergence, temperature calibration. |
| **approximation.ts** | Wedin bounds, Nystrom approximation, randomized SVD. |
| **analysis.ts** | Spectral radius, Lyapunov stability analysis. |
| **statistics.ts** | Empirical Fisher, Fisher-Hessian statistics, diagonal scaling. |
| **bias_verification.ts** | Fairness and bias verification. |

---

### 13. `src/compression/` — Model Compression (5 files)

| File | Responsibility |
|------|----------------|
| **quantization.ts** | Symmetric int8 quantization (4x size reduction). |
| **distillation.ts** | Knowledge distillation: teacher soft targets + student training. |
| **lowrank.ts** | SVD-based weight factorization: W ~ U * Sigma * V^T. |
| **compress.ts** | Unified compression API (quantization + distillation + low-rank + hybrid). |
| **index.ts** | Barrel exports. |

---

### 14. `src/explainability/` — Interpretability Tools (5 files)

| File | Responsibility |
|------|----------------|
| **integratedGradients.ts** | Path-integral feature attribution. |
| **shap.ts** | Permutation-based SHAP values (Monte Carlo). |
| **attentionRollout.ts** | Multi-layer attention aggregation (mean/max). |
| **conformal_prediction.ts** | Uncertainty quantification with guaranteed coverage (APS, RAPS). |
| **index.ts** | Barrel exports. |

---

### 15. `src/integrations/` — External Services (7 files)

All services are mock/browser-local implementations.

| File | Responsibility |
|------|----------------|
| **MonitoringService.ts** | Prometheus/Datadog metrics push (mock). |
| **StorageService.ts** | HuggingFace/ModelZoo model storage (mock). |
| **VisualizationService.ts** | Plotly chart generation with 5 color palettes. |
| **WebhookService.ts** | Slack/Discord webhook delivery (mock). |
| **WebSocketService.ts** | Cross-tab sync via BroadcastChannel API. |
| **types.ts** | Shared types for all integrations (436 lines). |
| **index.ts** | React context: `IntegrationProvider`, hooks (`useMonitoring()`, `useModelStorage()`, etc.). |

---

### 16. `src/visualization/` — Embedding Projection (1 file)

| File | Responsibility |
|------|----------------|
| **embeddings.ts** | t-SNE and UMAP dimensionality reduction, L2/z-score normalization. |

---

### 17. `src/autodiff/` — Automatic Differentiation (1 file)

| File | Responsibility |
|------|----------------|
| **graph.ts** | Reverse-mode autodiff for scalars. `Variable` class with computation graph and `backward()`. Supports +, -, *, /, pow, tanh, exp, log. |

---

### 18. `src/data/` — Dataset Management (3 files)

| File | Responsibility |
|------|----------------|
| **Dataset.ts** | `DatasetBuilder` fluent API, `BatchIterator`, train/val/test splits, SHA-256 verification. |
| **conv2dData.ts** | 14x14 conceptual framework data (bilingual English/Hebrew). |
| **triadicTable.ts** | Triadic operator table data. |

---

### 19. `src/config/` — Configuration (1 file)

| File | Responsibility |
|------|----------------|
| **constants.ts** | Application constants, special tokens, default configurations. |

---

### 20. `src/experiments/` — Experimental Features (1 file)

| File | Responsibility |
|------|----------------|
| **bayesian.ts** | Bayesian inference for experimental analysis. |

---

## Dependency Graph

### Layer 0 — Foundations (no internal dependencies)

```
MathUtils.ts          (pure math)
RMSNorm.ts            (pure normalization)
storage.ts            (localStorage wrapper)
autodiff/graph.ts     (scalar autodiff)
types/*               (TypeScript interfaces)
config/constants.ts   (app constants)
```

### Layer 1 — Backend & Core Algorithms

```
backend/webgpu.ts          ← MathUtils
backend/gpu_neural_ops.ts  ← backend/webgpu
backend/mixed_precision.ts  (standalone)
backend/edgeLearning.ts     (standalone)

generation/sampling.ts     ← MathUtils
losses/advanced.ts         ← MathUtils
losses/lossMask.ts         ← types/project
losses/information_bottleneck.ts ← MathUtils
losses/variational_ib.ts    (standalone)

tokenizer/BPETokenizer.ts  ← types/tokenizer
math/*                     ← math/numerics (internal)

models/attention.ts        ← MathUtils, backend/gpu_neural_ops
models/sparse_attention.ts ← MathUtils
models/linearized_attention.ts ← models/attention (Matrix type)
models/regularizers.ts      (standalone)
models/mini_transformer.ts ← models/attention, models/regularizers, RMSNorm, backend/gpu_neural_ops
```

### Layer 2 — Neural Network Models

```
ProNeuralLM.ts    ← MathUtils, generation/sampling, backend/gpu_neural_ops,
                     training/optimizer, training/SophiaOptimizer, types/project

TransformerLM.ts  ← ProNeuralLM, models/mini_transformer, models/attention,
                     MathUtils, RMSNorm, backend/gpu_neural_ops, types/project

AdvancedNeuralLM.ts ← ProNeuralLM, MathUtils, generation/sampling, types/project
```

### Layer 3 — Governance & Brain

```
GovernanceEngine.ts       ← types/governance
BrainEngine.ts            ← storage.ts
BrainGovernanceBridge.ts  ← BrainEngine, types/governance
GovernanceAndTelemetry.ts ← types/kernel
CausalInferenceEngine.ts  ← types/causal, math/causal_math
```

### Layer 4 — Support Systems

```
compression/distillation.ts ← ProNeuralLM, AdvancedNeuralLM, TransformerLM
compression/quantization.ts  (standalone)
compression/lowrank.ts        (standalone)

explainability/*              (mostly standalone)
visualization/embeddings.ts  ← external: umap-js, tsne-js
data/Dataset.ts              ← lib/Reproducibility

training/LionOptimizer.ts     (standalone)
training/SophiaOptimizer.ts    (standalone)
training/KFACOptimizer.ts      (standalone)
training/GradientClipping.ts   (standalone)
training/EMAWeights.ts         (standalone)
training/injection_hooks.ts   ← lib/expandable/*
```

### Layer 5 — Integration & Context

```
integrations/MonitoringService.ts  ← BrainEngine, types/governance, storage
integrations/StorageService.ts     ← storage
integrations/WebhookService.ts     ← BrainEngine, types/governance, storage
integrations/WebSocketService.ts   ← BrainEngine

contexts/BrainContext.tsx     ← BrainEngine, BrainGovernanceBridge
contexts/ProjectContext.tsx   ← GovernanceEngine, types/project, types/governance
```

### Layer 6 — UI Components

```
components/* ← contexts/*, lib/*, models/*, types/*, training/*,
               compression/*, explainability/*, visualization/*,
               math/*, backend/*
```

### Layer 7 — Application Shell

```
App.tsx  ← components/*, contexts/*, lib/ProNeuralLM, lib/TransformerLM,
           lib/AdvancedNeuralLM, types/*
main.tsx ← App.tsx, components/ErrorBoundary
```

---

## Dependency Diagram (Simplified)

```
main.tsx
  └── App.tsx
        ├── contexts/BrainContext ──────── lib/BrainEngine ── lib/storage
        │                         └────── lib/BrainGovernanceBridge
        ├── contexts/ProjectContext ────── lib/GovernanceEngine
        │
        ├── components/TrainingPanel ───── lib/ProNeuralLM ─── backend/gpu_neural_ops
        │                            │                    ├── training/optimizer
        │                            │                    ├── training/SophiaOptimizer
        │                            │                    ├── generation/sampling
        │                            │                    └── lib/MathUtils
        │                            │
        │                            ├── lib/TransformerLM ── models/mini_transformer
        │                            │                    ├── models/attention
        │                            │                    └── lib/RMSNorm
        │                            │
        │                            └── lib/AdvancedNeuralLM
        │
        ├── components/CausalAnalysis ─── lib/CausalInferenceEngine
        │                            └── math/dag_operations
        │
        ├── components/CompressionPanel ─ compression/*
        │                            └── lib/ProNeuralLM (distillation)
        │
        ├── components/ExplainabilityPanel ── explainability/*
        │
        ├── components/EmbeddingViz ──── visualization/embeddings
        │
        ├── components/CerebroPanel ──── lib/expandable/*
        │                          └── training/injection_hooks
        │
        ├── components/InformationTheory ── losses/information_bottleneck
        │                              └── losses/variational_ib
        │
        └── components/GovernanceBoard ─── lib/GovernanceEngine
                                      └── lib/GovernanceAndTelemetry
```

---

## External Infrastructure

### `tests/` — Test Suite (135+ files)

Mirrors `src/` structure exactly. Includes:
- Unit tests for all modules
- Parity tests (`*.parity.test.ts`) — CPU vs GPU numerical equivalence
- Device tests (`*.device.test.ts`) — WebGPU-specific tests
- Component tests via `@testing-library/react`
- Integration tests for cross-module workflows

### `electron/` — Desktop App

- `main.ts` — Electron main process, wraps the Vite dev server / production build

### `scripts/` — Training & Benchmarking

- `train.ts` — Node-based training script
- `benchmark_gpu.ts` — GPU performance measurement
- `train_experiment.py` — Python training alternative

### `configs/` — Baseline Configurations

- `wikitext_baseline.json`, `wikitext_dropout.json`, `hebrew_news_baseline.json`

### `data/` — Training Data

- `neuro-lingua-v324.json` (7.2MB trained model)
- `corpus.txt` — Training corpus
- `processed/`, `raw/`, `scenarios/` — Dataset directories

### Python Modules (top-level)

| Directory | Responsibility |
|-----------|----------------|
| **edge_formalism/** | Edge learning with synonym support (en/he) |
| **symmetry_coupling/** | Symmetry-based learning module |
| **neurosync/** | NeuroSync agent (35KB main module) |

### `.github/workflows/` — CI/CD

| Workflow | Trigger |
|----------|---------|
| **ci.yml** | Push/PR: type check + lint + test + build |
| **deploy-pages.yml** | Push to main: GitHub Pages deployment |
| **train-model.yml** | Manual/scheduled: model retraining |
| **build-desktop.yml** | Manual: Electron Windows installer |

---

## Key Architectural Patterns

1. **Model Inheritance**: `ProNeuralLM` as base class, extended by `TransformerLM` and `AdvancedNeuralLM`
2. **Event Reducer**: `BrainEngine` uses pure reducer pattern (`reduceBrain()`) for deterministic state transitions
3. **Singleton Services**: Integration services use factory pattern (`getMonitoringService()`)
4. **Context + Hooks**: Two React contexts (`BrainContext`, `ProjectContext`) expose domain-specific hooks
5. **GPU Fallback**: All GPU operations have automatic CPU fallback paths
6. **Barrel Exports**: `index.ts` files centralize module APIs
7. **Mock Integrations**: External services (Prometheus, Slack, HuggingFace) are browser-local mocks
8. **Governance Audit Trail**: All calibration decisions logged with rationale, witness, and expiry (Sigma-SIG compliance)
9. **Bilingual UI**: Components support English and Hebrew with RTL layout
10. **localStorage Persistence**: Model weights, brain state, projects, and governance state all persist to browser storage
