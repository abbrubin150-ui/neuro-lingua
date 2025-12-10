# CLAUDE.md - AI Assistant Guide for Neuro-Lingua

> **Last Updated**: 2025-12-10
> **Version**: 4.2.0
> **Purpose**: Comprehensive guide for AI assistants working on the Neuro-Lingua codebase

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Repository Structure](#repository-structure)
4. [Key Components](#key-components)
5. [Development Workflows](#development-workflows)
6. [Testing Guidelines](#testing-guidelines)
7. [Code Conventions](#code-conventions)
8. [Neural Network Architectures](#neural-network-architectures)
9. [Common Tasks](#common-tasks)
10. [Important Files Reference](#important-files-reference)
11. [Integration Points](#integration-points)
12. [Project Management](#project-management)
13. [Watch Out For](#watch-out-for)
14. [Deployment & CI/CD](#deployment--cicd)

---

## Project Overview

**Neuro-Lingua DOMESTICA** is a browser-native neural language model built entirely in React + TypeScript. It runs completely client-side without requiring a server, featuring WebGPU acceleration for 2-5x faster training on compatible hardware.

### Core Purpose

- Educational demonstration of neural language models
- Browser-native implementation (no server required)
- Multiple neural architectures (Feedforward, Advanced, Transformer)
- Real-time training and text generation
- WebGPU hardware acceleration with automatic CPU fallback

### Live Demo

🌐 **[https://abbrubin150-ui.github.io/neuro-lingua/](https://abbrubin150-ui.github.io/neuro-lingua/)**

### Key Features

- **3 Neural Architectures**: ProNeuralLM (baseline), AdvancedNeuralLM (enhanced), TransformerLM (attention-based)
- **Architecture Presets**: Automatic configuration when switching between architectures
- **WebGPU Acceleration**: Automatic GPU detection with graceful CPU fallback
- **5 Optimizers**: SGD with momentum, Adam, Lion (v4.0), Damped Newton, L-BFGS
- **7+ Generation Methods**: Greedy, Top-k, Top-p (nucleus), Typical, Mirostat v2, Beam Search, Contrastive Search
- **Grouped-Query Attention (GQA)**: Efficient attention mechanism with configurable KV heads
- **Model Compression**: Int8 quantization, knowledge distillation, low-rank approximation (SVD)
- **Advanced Features**: Dropout, layer normalization, RMSNorm, learning rate scheduling, weight decay
- **Cerebro System**: Concept injection with bubble visualization and governance
- **Σ-SIG Compliance**: Experiment tracking with Decision Ledger governance
- **Autonomous Governance**: GovernanceEngine for automatic parameter calibration
- **Brain Vitals System**: BrainEngine tracks model "mood" and provides behavioral insights
- **Model Interpretability**: SHAP values, integrated gradients, attention rollout
- **Information Theory**: Information plane analysis (I(X;Z) vs I(Z;Y))
- **Session Persistence**: localStorage-based state management
- **GitHub Actions**: Automated model retraining workflow
- **Bilingual UI**: English/Hebrew with RTL support

---

## Technology Stack

### Frontend

- **React** 18.2 - UI framework
- **TypeScript** 5.2 - Type-safe development
- **Vite** 4.4 - Build tool and dev server
- **WebGPU** - Hardware acceleration (via @webgpu/types)

### Testing

- **Vitest** 1.5 - Test runner (Vite-native)
- **@testing-library/react** 14.2 - React component testing
- **jsdom** 24.0 - DOM environment for tests

### Code Quality

- **ESLint** 8.57 - Linting with TypeScript, React, JSX-a11y plugins
- **Prettier** 3.2 - Code formatting
- **TypeScript** strict mode - Enhanced type safety

### Dependencies

- **tsne-js** ^1.0.3 - t-SNE embedding visualization
- **umap-js** ^1.3.3 - UMAP dimensionality reduction

### Build & Package Management

- **pnpm** (preferred) or npm/yarn
- **tsx** 3.12 - TypeScript execution for scripts
- **GitHub Actions** - CI/CD automation

---

## Repository Structure

```
/home/user/neuro-lingua/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Continuous integration (test, lint, build)
│   │   ├── train-model.yml           # Automated model retraining
│   │   └── deploy-pages.yml          # GitHub Pages deployment
│   └── ISSUE_TEMPLATE/               # Issue and PR templates
│
├── configs/                          # Training experiment configurations
│   ├── wikitext_baseline.json
│   ├── wikitext_dropout.json
│   └── hebrew_news_baseline.json
│
├── data/                             # Training datasets
│   ├── corpus.txt                    # Main training corpus (used by CI)
│   ├── raw/                          # Raw datasets (wikitext, hebrew_news)
│   ├── processed/                    # Preprocessed train/val/test splits
│   └── edge_formalism/               # Edge learning experimental data
│
├── docs/                             # Documentation
│   ├── experiments/                  # Experiment results and summaries
│   └── visuals/                      # Embedding visualization exports
│
├── edge_formalism/                   # Edge learning formalism (Python)
│   ├── __init__.py
│   ├── edge_analyzer.py
│   └── recursive_optimizer.py
│
├── logs/                             # Training logs
│
├── models/                           # Trained model artifacts
│   ├── neuro-lingua-v324.json        # Latest production model (3MB)
│   └── experiments/                  # Experimental checkpoints
│
├── public/                           # Static assets
│
├── scripts/                          # Utility scripts
│   ├── train.ts                      # Node.js headless training
│   ├── benchmark_gpu.ts              # GPU performance benchmarks
│   ├── visualize_embeddings.ts       # Generate t-SNE/UMAP visualizations
│   ├── experiments/                  # Experiment scripts
│   └── data/                         # Data processing scripts
│
├── src/                              # Main application source
│   ├── autodiff/                     # Automatic differentiation (experimental)
│   ├── backend/                      # WebGPU integration
│   │   ├── webgpu.ts                 # WebGPU backend and tensor operations
│   │   ├── gpu_neural_ops.ts        # High-level neural operations on GPU
│   │   └── edgeLearning.ts          # Edge learning integration
│   ├── components/                   # React UI components
│   │   ├── TrainingPanel.tsx         # Main training configuration panel (1671 lines)
│   │   ├── DecisionEntry2Panel.tsx   # Enhanced decision entry UI (841 lines)
│   │   ├── BrainPanel.tsx            # Brain vitals and mood system (731 lines)
│   │   ├── ModelMetrics.tsx          # Performance metrics dashboard (712 lines)
│   │   ├── EmbeddingVisualizationPanel.tsx # t-SNE/UMAP visualization (660 lines)
│   │   ├── ChatInterface.tsx         # Chat-style generation UI (644 lines)
│   │   ├── GovernanceBoard.tsx       # Governance monitoring dashboard (590 lines)
│   │   ├── RunComparisonPanel.tsx    # Compare training runs (558 lines)
│   │   ├── CompressionPanel.tsx      # Model compression UI (513 lines)
│   │   ├── ExportPanel.tsx           # Model/data export interface (505 lines)
│   │   ├── ProjectManager.tsx        # Project/run management (485 lines)
│   │   ├── ExplainabilityPanel.tsx   # SHAP/gradients/attention (412 lines)
│   │   ├── BrainTelemetryPanel.tsx   # Brain telemetry display (373 lines)
│   │   ├── InformationTheoryPanel.tsx # Information plane analysis (365 lines)
│   │   ├── ScenarioManager.tsx       # Test scenario editor (285 lines)
│   │   ├── DecisionLedgerEditor.tsx  # Governance/decision tracking (254 lines)
│   │   ├── TokenizerConfig.tsx       # Tokenizer settings (196 lines)
│   │   ├── CerebroPanel.tsx          # Cerebro concept injection UI (194 lines)
│   │   ├── ErrorBoundary.tsx         # React error boundary (107 lines)
│   │   ├── OnboardingCard.tsx        # First-time user guide (106 lines)
│   │   ├── CerebroBubbleGraph.tsx    # Cerebro bubble visualization (86 lines)
│   │   ├── OnboardingTooltip.tsx     # Interactive onboarding tips (78 lines)
│   │   └── ModelSnapshot.tsx         # Model metadata snapshot display (77 lines)
│   ├── compression/                  # Model compression system
│   │   ├── compress.ts               # Unified compression interface
│   │   ├── quantization.ts           # Int8/16 quantization
│   │   ├── distillation.ts           # Knowledge distillation
│   │   ├── lowrank.ts                # SVD-based low-rank approximation
│   │   └── index.ts                  # Module exports
│   ├── config/                       # Application configuration
│   │   └── constants.ts              # Default hyperparameters and constraints
│   ├── contexts/                     # React Context providers
│   │   └── ProjectContext.tsx        # Project/run state management
│   ├── experiments/                  # Experimental features
│   │   └── bayesian.ts               # Bayesian optimization
│   ├── explainability/               # Model interpretability
│   │   ├── shap.ts                   # SHAP values
│   │   ├── integrated_gradients.ts   # Integrated gradients
│   │   └── attention_rollout.ts      # Attention visualization
│   ├── generation/                   # Text generation algorithms
│   │   ├── sampler.ts                # Top-k, top-p, temperature sampling
│   │   ├── beam_search.ts            # Beam search implementation
│   │   └── contrastive_search.ts     # Contrastive decoding
│   ├── lib/                          # Core neural network implementations
│   │   ├── ProNeuralLM.ts            # Base feedforward LM (25KB)
│   │   ├── AdvancedNeuralLM.ts       # Enhanced LM with advanced features (21KB)
│   │   ├── TransformerLM.ts          # Transformer architecture (15KB)
│   │   ├── MathUtils.ts              # Numerical stability utilities (16KB)
│   │   ├── BrainEngine.ts            # Brain vitals and mood system
│   │   ├── GovernanceEngine.ts       # Autonomous parameter calibration
│   │   ├── RMSNorm.ts                # Root Mean Square Normalization (v4.0)
│   │   ├── storage.ts                # localStorage abstraction
│   │   ├── utils.ts                  # Tokenizer and CSV utilities
│   │   ├── diffUtils.ts              # Diff computation utilities
│   │   ├── exportUtils.ts            # Model export utilities
│   │   ├── experimentComparison.ts   # Run comparison utilities
│   │   ├── traceExport.ts            # Σ-SIG compliant experiment tracing
│   │   └── expandable/               # Cerebro injection system
│   │       ├── InjectionEngine.ts    # Core injection logic (169 lines)
│   │       ├── InjectableLayer.ts    # Layer interface (19 lines)
│   │       ├── injection_math.ts     # Linear algebra operations (205 lines)
│   │       ├── bubbleExtractor.ts    # Semantic bubble extraction (181 lines)
│   │       ├── injection_hooks.ts    # Training integration (86 lines)
│   │       ├── ProNeuralLMAdapter.ts # Feedforward adapter
│   │       ├── AdvancedNeuralLMAdapter.ts # Enhanced adapter
│   │       └── TransformerLMAdapter.ts # Transformer adapter
│   ├── losses/                       # Advanced loss functions
│   │   ├── advanced.ts               # Focal, label smoothing, SCE (76 lines)
│   │   └── information_bottleneck.ts # IB loss implementation (316 lines)
│   ├── math/                         # Mathematical utilities
│   │   ├── analysis.ts               # Spectral/Lyapunov analysis (234 lines)
│   │   └── statistics.ts             # Fisher information stats (197 lines)
│   ├── models/                       # Architecture-specific implementations
│   │   ├── mini_transformer.ts       # Compact transformer
│   │   └── attention.ts              # Multi-head attention
│   ├── training/                     # Optimization algorithms
│   ├── types/                        # TypeScript type definitions
│   └── visualization/                # Embedding visualization (t-SNE, UMAP)
│
├── symmetry_coupling/                # Symmetry coupling modules (Python)
│
├── tests/                            # Test suites
│   ├── ProNeuralLM.test.ts           # Base model tests
│   ├── ProNeuralLM.device.test.ts    # Device-specific model tests
│   ├── AdvancedNeuralLM.test.ts      # Advanced features tests
│   ├── TransformerLM.test.ts         # Transformer tests
│   ├── GovernanceEngine.test.ts      # Autonomous governance tests
│   ├── MathUtils.test.ts             # Math utilities tests
│   ├── tokenizer.test.ts             # Tokenizer tests
│   ├── sampler.test.ts               # Generation tests
│   ├── gqa.test.ts                   # Grouped-Query Attention tests
│   ├── embedding_extraction.test.ts  # Embedding extraction tests
│   ├── App.test.tsx                  # Main app component tests
│   ├── backend/                      # Backend tests (WebGPU, edge learning)
│   ├── components/                   # Component tests (UI panels)
│   ├── compression/                  # Compression system tests
│   ├── contexts/                     # Context provider tests
│   ├── lib/                          # Library module tests
│   │   ├── bubbleExtractor.test.ts   # Bubble extraction tests
│   │   └── *Adapter.test.ts          # Injection adapter tests
│   ├── losses/                       # Loss function tests
│   │   └── information_bottleneck.test.ts
│   ├── math/                         # Mathematical analysis tests
│   │   └── analysis.test.ts          # Spectral/Lyapunov tests
│   ├── numerics/                     # Numerical correctness tests
│   │   ├── sampling.test.ts
│   │   └── bayesian.test.ts
│   ├── training/                     # Training module tests
│   │   └── LionOptimizer.test.ts
│   └── setup.ts                      # Test environment setup
│
├── index.html                        # HTML entry point
├── package.json                      # Package dependencies and scripts
├── tsconfig.json                     # TypeScript configuration
├── vite.config.ts                    # Vite build configuration
├── .eslintrc.cjs                     # ESLint configuration
├── .prettierrc                       # Prettier configuration
└── README.md                         # Project README
```

---

## Key Components

### 1. Neural Network Core (`src/lib/`)

#### **ProNeuralLM.ts** - Base Feedforward Model

**Location**: `/home/user/neuro-lingua/src/lib/ProNeuralLM.ts`

**Purpose**: Foundation neural language model with basic feedforward architecture

**Key Features**:

- Character-level language modeling
- Embedding layer → Hidden layer → Output layer
- ReLU activation
- 5 optimizers: Momentum SGD, Adam, Lion (v4.0), Damped Newton, L-BFGS
- Dropout during training (disabled during inference)
- Training history tracking
- Import/export with optimizer state serialization

**Interface**:

```typescript
class ProNeuralLM {
  constructor(
    vocabSize: number,
    hiddenSize: number,
    contextSize: number,
    tokenizerConfig?: TokenizerConfig
  )

  train(corpus: string, epochs: number, lr: number, ...) => void
  generate(prompt: string, maxTokens: number, temperature: number, ...) => string
  exportModel() => SerializedModel
  static loadModel(data: SerializedModel) => ProNeuralLM
}
```

**When to Use**:

- Default/baseline implementation
- Educational purposes
- When simplicity is preferred

#### **AdvancedNeuralLM.ts** - Enhanced Feedforward Model

**Location**: `/home/user/neuro-lingua/src/lib/AdvancedNeuralLM.ts`

**Purpose**: Extends ProNeuralLM with state-of-the-art training enhancements

**Key Features**:

- Advanced activations: LeakyReLU, ELU, GELU, Swish
- He/Xavier weight initialization
- Learning rate scheduling: cosine annealing, exponential decay, warmup
- L2 regularization (weight decay)
- Layer normalization option
- Gradient clipping
- Beam search generation
- Perplexity calculation

**Interface**: Extends ProNeuralLM with additional config options

**When to Use**:

- Production use cases
- When better convergence is needed
- For comparing advanced techniques

#### **TransformerLM.ts** - Transformer Architecture

**Location**: `/home/user/neuro-lingua/src/lib/TransformerLM.ts`

**Purpose**: Full transformer implementation with multi-head self-attention

**Key Features**:

- Multi-head self-attention (1-16 heads configurable)
- Position embeddings
- Residual connections
- Batch renormalization
- Feed-forward layers
- Configurable 1-8 layers
- Compatible with ProNeuralLM interface

**When to Use**:

- When attention mechanisms are needed
- For sequence modeling tasks
- Research and experimentation

### 2. WebGPU Backend (`src/backend/`)

#### **webgpu.ts** - GPU Backend

**Location**: `/home/user/neuro-lingua/src/backend/webgpu.ts`

**Purpose**: WebGPU device management and tensor operations

**Key Classes**:

- `WebGPUBackend`: Device initialization and management
- `WebGPUTensor`: GPU tensor with operations (matmul, add, ReLU)

**Features**:

- Automatic GPU detection
- Shader compilation for operations
- Memory management
- Error handling with graceful fallback

#### **gpu_neural_ops.ts** - Neural Operations

**Location**: `/home/user/neuro-lingua/src/backend/gpu_neural_ops.ts`

**Purpose**: High-level neural network operations on GPU

**Key Features**:

- Matrix-vector multiplication
- Matrix multiplication
- ReLU activation
- Performance metrics tracking
- Automatic CPU fallback

**Usage Pattern**:

```typescript
const gpuOps = new GPUNeuralOps(webgpuBackend);
model.setGPUOps(gpuOps);
// Training automatically uses GPU when available
```

### 3. React Components (`src/components/`)

#### **TrainingPanel.tsx** - Main Training UI

**Location**: `/home/user/neuro-lingua/src/components/TrainingPanel.tsx` (40KB)

**Purpose**: Primary interface for model configuration and training

**Features**:

- Architecture selection (ProNeuralLM, AdvancedNeuralLM, TransformerLM)
- **Architecture Presets**: Automatic optimal settings when switching architectures
  - Transformer: Enables Adam optimizer, LayerNorm, and attention defaults
  - AdvancedNeuralLM: Applies LayerNorm and deep-optimization defaults
  - ProNeuralLM: Standard baseline configuration
- Hyperparameter controls
- Corpus input
- Training progress visualization
- Real-time loss/perplexity charts
- GPU acceleration toggle
- Session persistence

#### **ProjectManager.tsx** - Experiment Tracking

**Location**: `/home/user/neuro-lingua/src/components/ProjectManager.tsx`

**Purpose**: Σ-SIG compliant project and run management

**Features**:

- Create/edit/delete projects
- Track training runs with frozen configs
- Run comparison and analysis
- Export experiment metadata
- Decision Ledger integration

#### **ChatInterface.tsx** - Generation UI

**Location**: `/home/user/neuro-lingua/src/components/ChatInterface.tsx`

**Purpose**: Chat-style text generation interface

**Features**:

- Multi-turn conversation
- Temperature, top-k, top-p controls
- Beam search toggle
- Copy/export chat history

#### **CompressionPanel.tsx** - Model Compression UI

**Location**: `/home/user/neuro-lingua/src/components/CompressionPanel.tsx` (513 lines)

**Purpose**: User interface for model compression operations

**Features**:

- Three compression methods: Quantization, Distillation, Low-rank approximation
- Int8 quantization with 4x size reduction
- Knowledge distillation configuration (temperature, alpha, student size)
- SVD-based low-rank approximation with automatic rank selection
- Compression results display (original size, compressed size, ratio)
- Export compressed models
- Real-time compression progress feedback

**When to Use**:

- Reducing model size for deployment
- Faster model loading and inference
- Experimenting with compression trade-offs
- Creating student models from larger teachers

#### **GovernanceBoard.tsx** - Governance Monitoring Dashboard

**Location**: `/home/user/neuro-lingua/src/components/GovernanceBoard.tsx` (590 lines)

**Purpose**: Real-time monitoring dashboard for autonomous governance

**Features**:

- Live metric visualization (loss, accuracy, perplexity trends)
- Alert feed with severity indicators (info, warning, critical)
- Calibration history timeline
- Parameter change tracking
- Governor configuration controls
- Session-by-session analysis
- Alert acknowledgment system

**When to Use**:

- Monitoring autonomous governance in action
- Reviewing parameter calibration decisions
- Debugging training issues detected by governor
- Compliance reporting (Σ-SIG)

#### **BrainPanel.tsx** - Brain Vitals and Mood System

**Location**: `/home/user/neuro-lingua/src/components/BrainPanel.tsx` (731 lines)

**Purpose**: Interactive UI for brain vitals and mood tracking

**Features**:

- Mood indicator with visual representation
- Creativity and stability gauges (0-100)
- Pet name editing
- Event diary with timeline
- Training/generation statistics
- Mood history chart
- Suggestion system (non-intrusive)
- Manual mood override controls

**When to Use**:

- Gamifying the training experience
- Understanding model lifecycle
- Educational demonstrations
- Tracking long-term model behavior

#### **BrainTelemetryPanel.tsx** - Brain Telemetry Display

**Location**: `/home/user/neuro-lingua/src/components/BrainTelemetryPanel.tsx` (373 lines)

**Purpose**: Detailed telemetry and analytics for brain system

**Features**:

- Real-time vitals monitoring
- Mood transition graph
- Event frequency analysis
- Diary entry search and filter
- Export brain state data
- Historical trend visualization

**When to Use**:

- Deep-dive analysis of brain behavior
- Debugging mood calculation
- Exporting brain data for research
- Long-term tracking

#### **InformationTheoryPanel.tsx** - Information Plane Analysis

**Location**: `/home/user/neuro-lingua/src/components/InformationTheoryPanel.tsx` (365 lines)

**Purpose**: Information-theoretic analysis and visualization

**Features**:

- I(X;Z) vs I(Z;Y) information plane plot
- Compression-prediction trade-off visualization
- Information bottleneck analysis
- Layer-wise information flow
- Interactive canvas with zoom/pan
- Epoch-by-epoch animation
- Export plots as PNG/SVG

**When to Use**:

- Understanding representation learning
- Analyzing information bottleneck dynamics
- Research on deep learning theory
- Visualizing compression during training

#### **ExplainabilityPanel.tsx** - Model Interpretability

**Location**: `/home/user/neuro-lingua/src/components/ExplainabilityPanel.tsx` (412 lines)

**Purpose**: Model interpretation and explanation tools

**Features**:

- SHAP value calculation and visualization
- Integrated gradients attribution
- Attention rollout (for Transformers)
- Token-level importance heatmaps
- Layer activation visualization
- Export attribution maps

**When to Use**:

- Understanding model predictions
- Debugging unexpected outputs
- Research on model interpretability
- Compliance requirements (explainable AI)

#### **EmbeddingVisualizationPanel.tsx** - Embedding Visualization

**Location**: `/home/user/neuro-lingua/src/components/EmbeddingVisualizationPanel.tsx` (660 lines)

**Purpose**: Interactive embedding space visualization

**Features**:

- t-SNE projection with configurable perplexity
- UMAP projection with neighbor parameters
- Interactive canvas (pan, zoom, hover)
- Token label display
- Cluster highlighting
- 2D/3D view toggle
- Export visualizations
- Snapshot saving

**When to Use**:

- Exploring learned embeddings
- Understanding token relationships
- Visualizing semantic clusters
- Research presentations

#### **RunComparisonPanel.tsx** - Training Run Comparison

**Location**: `/home/user/neuro-lingua/src/components/RunComparisonPanel.tsx` (558 lines)

**Purpose**: Side-by-side comparison of training runs

**Features**:

- Multi-run selection
- Loss curve overlay
- Hyperparameter diff view
- Metric comparison table
- Statistical significance testing
- Best run highlighting
- Export comparison reports

**When to Use**:

- Hyperparameter tuning experiments
- A/B testing different architectures
- Finding optimal configurations
- Reporting experiment results

#### **DecisionEntry2Panel.tsx** - Enhanced Decision Entry

**Location**: `/home/user/neuro-lingua/src/components/DecisionEntry2Panel.tsx` (841 lines)

**Purpose**: Advanced decision ledger entry interface

**Features**:

- Rich text decision rationale
- Alternative options tracking
- Witness/approval signatures
- Expiry date management
- Linked run references
- Decision impact tracking
- Compliance status indicators

**When to Use**:

- Formal governance decisions
- Σ-SIG compliance requirements
- Multi-stakeholder approvals
- Audit trail maintenance

#### **ExportPanel.tsx** - Model and Data Export

**Location**: `/home/user/neuro-lingua/src/components/ExportPanel.tsx` (505 lines)

**Purpose**: Comprehensive export functionality

**Features**:

- Model export (JSON, optimized formats)
- Training history CSV export
- Experiment metadata export
- Σ-SIG trace bundles
- Embedding exports
- Batch export operations
- Export configuration presets

**When to Use**:

- Saving trained models
- Data analysis in external tools
- Compliance reporting
- Archiving experiments

#### **OnboardingTooltip.tsx** - Interactive Onboarding

**Location**: `/home/user/neuro-lingua/src/components/OnboardingTooltip.tsx` (78 lines)

**Purpose**: Context-sensitive onboarding tooltips

**Features**:

- Step-by-step guided tours
- Context-aware hints
- Dismissible tooltips
- Progress tracking
- Multi-language support
- Keyboard navigation

**When to Use**:

- First-time user experience
- Feature discovery
- UI element explanations
- Reducing learning curve

#### **CerebroPanel.tsx** - Cerebro Concept Injection System

**Location**: `/home/user/neuro-lingua/src/components/CerebroPanel.tsx` (194 lines)

**Purpose**: Interactive UI for concept injection into neural model layers

**Features**:

- Concept bubble management with activation levels
- Injection proposal and execution workflow
- Undo/redo support for injections
- Tagged bubbles: body, desire, risk, value, action
- Integration with InjectionEngine and InjectableLayer
- Ledger tracking for all injection events
- Advanced mode toggle for detailed controls

**Key Concepts**:

- **Bubbles**: Concept representations with embeddings and activation levels
- **Injection**: Adding concept bubbles to model layer
- **Proposal**: Suggested injection with rationale
- **Tags**: Categorization system for concept types

**When to Use**:

- Experimenting with concept injection
- Steering model behavior via external concepts
- Research on controllable generation
- Governance-aware model modification

**Example Usage**:

```typescript
import { CerebroPanel } from './components/CerebroPanel';
import { InjectionEngine } from './lib/expandable/InjectionEngine';

// In your component
<CerebroPanel
  layer={injectableLayer}
  bubbles={conceptBubbles}
  engine={new InjectionEngine()}
/>
```

#### **CerebroBubbleGraph.tsx** - Bubble Visualization

**Location**: `/home/user/neuro-lingua/src/components/CerebroBubbleGraph.tsx` (86 lines)

**Purpose**: SVG-based visualization of concept bubbles in circular layout

**Features**:

- Circular bubble arrangement
- Size scaled by activation level
- Color-coded by tag type
- Interactive selection
- Responsive SVG rendering

**Tag Colors**:

- Risk: Red (#ef4444)
- Value: Green (#10b981)
- Action: Blue (#3b82f6)
- Desire: Purple (#a855f7)
- Body: Gray (#6b7280)

**When to Use**:

- Visualizing concept space
- Understanding bubble relationships
- Interactive bubble selection
- Educational demonstrations

#### **ModelSnapshot.tsx** - Model Metadata Display

**Location**: `/home/user/neuro-lingua/src/components/ModelSnapshot.tsx` (77 lines)

**Purpose**: Display model metadata snapshot with architecture badge

**Features**:

- Architecture-specific badges (ProNeural, Advanced, Transformer)
- Timestamp formatting
- Vocabulary size display
- Empty state handling
- Consistent styling across architectures

**When to Use**:

- Displaying saved model information
- Comparing model versions
- Model loading UI
- Checkpoint visualization

### 4. Model Compression System (`src/compression/`)

#### **compress.ts** - Unified Compression Interface

**Location**: `/home/user/neuro-lingua/src/compression/compress.ts` (~324 lines)

**Purpose**: High-level API for model compression

**Key Functions**:

- `compressWithQuantization()`: Int8 quantization with 4x size reduction
- `compressWithDistillation()`: Train smaller student model from teacher
- `compressWithLowRank()`: SVD-based weight matrix approximation
- `exportCompressedModel()`: Serialize compressed model to file

**Example Usage**:

```typescript
import { compressWithQuantization } from '../compression/compress';

const result = await compressWithQuantization(model);
console.log(`Compressed from ${result.originalSize} to ${result.compressedSize} bytes`);
console.log(`Compression ratio: ${result.compressionRatio.toFixed(2)}x`);
```

#### **quantization.ts** - Weight Quantization

**Location**: `/home/user/neuro-lingua/src/compression/quantization.ts` (~181 lines)

**Purpose**: Convert float32 weights to int8 for 4x size reduction

**Key Features**:

- Symmetric and asymmetric quantization
- Per-tensor and per-channel scaling
- Minimal accuracy loss (<2% typically)
- Fast dequantization for inference

**Formula**:

```
quantized = round((value - min) / (max - min) * 255)
dequantized = quantized / 255 * (max - min) + min
```

#### **distillation.ts** - Knowledge Distillation

**Location**: `/home/user/neuro-lingua/src/compression/distillation.ts` (~272 lines)

**Purpose**: Train smaller student model to mimic larger teacher

**Key Features**:

- Temperature-scaled soft targets
- Combined soft + hard label loss
- Configurable student architecture
- Progress tracking and validation

**Default Configuration**:

```typescript
{
  temperature: 3.0,
  alpha: 0.7,
  studentHiddenSize: 32,
  epochs: 30,
  learningRate: 0.1,
  useHardLabels: true
}
```

**Reference**: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"

#### **lowrank.ts** - Low-Rank Approximation

**Location**: `/home/user/neuro-lingua/src/compression/lowrank.ts` (~330 lines)

**Purpose**: SVD-based weight matrix compression

**Key Features**:

- Automatic optimal rank selection
- Target compression ratio support
- Frobenius norm error estimation
- Preserves most important singular values

**Compression Process**:

```
W ≈ U Σ V^T  (SVD decomposition)
W_compressed = U_k Σ_k V_k^T  (keep top k singular values)
```

### 5. Generation Methods (`src/generation/`)

#### **sampling.ts** - Advanced Sampling Algorithms

**Location**: `/home/user/neuro-lingua/src/generation/sampling.ts`

**Purpose**: Advanced text generation sampling methods beyond basic greedy/temperature

**Key Functions**:

##### **Typical Sampling** (`typicalSample`)

**Purpose**: Entropy-based token selection that balances probability and surprise

**Parameters**:
- `logits`: Raw model outputs
- `tau`: Typicality threshold (0-1, default 0.9)
- `options`: SamplingOptions (temperature, etc.)

**How It Works**:
1. Compute probabilities and entropy
2. Calculate surprise for each token: `-log(p(token))`
3. Select tokens with surprise close to expected entropy
4. Sample from filtered distribution

**When to Use**:
- More coherent than pure temperature sampling
- Less repetitive than top-k/top-p
- Better for creative generation

**Example**:
```typescript
import { typicalSample } from '../generation/sampling';

const nextToken = typicalSample(logits, 0.9, { temperature: 0.8 });
```

##### **Mirostat v2 Sampling** (`mirostatV2Sample`)

**Purpose**: Adaptive sampling with dynamic temperature control based on target entropy

**Parameters**:
- `logits`: Raw model outputs
- `options.targetEntropy`: Desired surprise level (default 5)
- `options.learningRate`: Adaptation rate (default 0.1)
- `options.state`: Stateful mu parameter for continuity

**How It Works**:
1. Maintain running estimate of optimal threshold (mu)
2. Filter tokens above threshold
3. Calculate actual surprise
4. Adapt mu based on error: `mu += learningRate * (surprise - targetEntropy)`

**Key Features**:
- Stateful: maintains `mu` across generations
- Self-adjusting: adapts to model behavior
- Controlled surprise: maintains consistent entropy

**Reference**: Basu et al. (2021) "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity"

**When to Use**:
- Long-form generation requiring consistency
- When you want controlled perplexity
- Multi-turn conversations

**Example**:
```typescript
import { mirostatV2Sample } from '../generation/sampling';

let state = { mu: 5.0 };
for (let i = 0; i < 100; i++) {
  const result = mirostatV2Sample(logits, {
    targetEntropy: 5.0,
    learningRate: 0.1,
    state
  });
  const nextToken = result.index;
  state = result.state;  // Maintain state across steps
  console.log(`Surprise: ${result.surprise}`);
}
```

##### **Beam Search** (`beamSearch`)

**Purpose**: Find high-probability sequences through parallel search

**Key Features**:
- Maintains top-k beams (partial sequences)
- Scores sequences by log probability
- Length normalization
- EOS token handling

**When to Use**:
- Translation tasks
- Summarization
- When quality > diversity

##### **Contrastive Search** (`contrastiveSearch`)

**Purpose**: Balance likelihood and diversity through contrastive scoring

**Key Features**:
- Penalizes similarity to previous tokens
- Alpha parameter controls likelihood vs diversity trade-off
- Uses embedding similarity

**When to Use**:
- Open-ended generation
- Avoiding repetition
- Creative writing

**Comparison of Sampling Methods**:

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **Greedy** | Fast, deterministic | Repetitive, no diversity | Debugging |
| **Top-k** | Simple, controllable | Arbitrary cutoff | General use |
| **Top-p (Nucleus)** | Dynamic cutoff | Complex tuning | Balanced generation |
| **Typical** | Natural coherence | Requires tuning tau | Creative writing |
| **Mirostat v2** | Adaptive, consistent | Stateful complexity | Long-form text |
| **Beam Search** | High quality | Slow, less diverse | Translation |
| **Contrastive** | Avoids repetition | Requires embeddings | Open-ended |

### 6. Context Providers (`src/contexts/`)

#### **ProjectContext.tsx** - State Management

**Location**: `/home/user/neuro-lingua/src/contexts/ProjectContext.tsx`

**Purpose**: Centralized project and run state management

**Provides**:

- Project CRUD operations
- Run tracking and management
- Scenario testing framework
- localStorage persistence
- Decision Ledger governance

**Usage**:

```typescript
import { useProjects } from '../contexts/ProjectContext';

function MyComponent() {
  const { projects, createProject, addRun } = useProjects();
  // ...
}
```

### 7. Configuration (`src/config/`)

#### **constants.ts** - Application Defaults

**Location**: `/home/user/neuro-lingua/src/config/constants.ts`

**Contains**:

- `DEFAULT_HYPERPARAMETERS`: hiddenSize=64, epochs=20, lr=0.08, etc.
- `DEFAULT_GENERATION`: temperature=0.8, topK=20, topP=0.9
- `DEFAULT_ADVANCED_CONFIG`: activation, initialization, scheduling
- `HYPERPARAMETER_CONSTRAINTS`: Min/max values for validation
- `STORAGE_KEYS`: localStorage key names
- `SPECIAL_TOKENS`: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`

**When to Modify**:

- Changing default hyperparameters
- Adding new configuration options
- Adjusting validation constraints

### 8. Autonomous Governance System (`src/lib/GovernanceEngine.ts`)

#### **GovernanceEngine** - Automatic Parameter Calibration

**Location**: `/home/user/neuro-lingua/src/lib/GovernanceEngine.ts`

**Purpose**: Autonomous parameter calibration with Σ-SIG compliance

**Key Features**:

- **Metric Monitoring**: Tracks training/validation loss, accuracy, perplexity
- **Issue Detection**: Identifies plateau, overfitting, underfitting, divergence
- **Automatic Calibration**: Adjusts learning rate and dropout based on metrics
- **Decision Ledger**: Full audit trail of all parameter changes
- **Alert System**: Generates warnings for training issues
- **Probabilistic Activation**: Configurable check intervals and activation probability

**Core Components**:

```typescript
interface GovernorConfig {
  enabled: boolean;
  checkInterval: number; // Check every N sessions
  activationProbability: number; // 0-1 chance of activation
  improvementThreshold: number; // Minimum % improvement
  learningRate: {
    min: number;
    max: number;
    decreaseFactor: number;
    increaseFactor: number;
  };
  dropout: {
    min: number;
    max: number;
    increaseStep: number;
    decreaseStep: number;
  };
  overfittingThreshold: number; // Train/val gap %
  underfittingThreshold: number; // Absolute loss threshold
  plateauWindow: number; // Look back N sessions
}
```

**Usage**:

```typescript
import { GovernanceEngine } from '../lib/GovernanceEngine';

// Initialize
const governor = new GovernanceEngine({
  enabled: true,
  checkInterval: 2,
  activationProbability: 0.5
});

// Record metrics after training
governor.recordMetrics({
  sessionId: 'run-123',
  epoch: 10,
  trainLoss: 1.234,
  valLoss: 1.456,
  trainAccuracy: 0.75,
  perplexity: 3.45
});

// Check if should activate
if (governor.shouldActivate()) {
  const analysis = governor.analyze();
  const action = governor.calibrate();

  if (action) {
    console.log(`Adjusting ${action.parameter}: ${action.previousValue} → ${action.newValue}`);
    console.log(`Reason: ${action.reason}`);
  }
}
```

**Alert Types**:

- `plateau`: No improvement detected over N sessions
- `overfitting`: Train/validation gap exceeding threshold
- `underfitting`: Both losses above threshold
- `divergence`: Loss increasing trend
- `oscillation`: High variance in metrics
- `calibration`: Parameter adjusted by governor

**When to Use**:

- Automated hyperparameter tuning during long training runs
- Detecting and responding to training issues
- Maintaining audit trail for experiments
- Governance compliance (Σ-SIG)

### 9. Brain Vitals System (`src/lib/BrainEngine.ts`)

#### **BrainEngine** - Model Mood and Behavioral Tracking

**Location**: `/home/user/neuro-lingua/src/lib/BrainEngine.ts`

**Purpose**: Track the "life" of the neural model through mood and vitals

**Key Features**:

- **Mood System**: CALM, FOCUSED, AGITATED, DREAMY, BURNT_OUT
- **Brain Vitals**: Creativity (0-100), Stability (0-100)
- **Event Tracking**: Training runs, generation, feeding corpus, idle time
- **Diary System**: Log of major events and mood shifts
- **Autonomous Suggestions**: Non-intrusive recommendations (never auto-executes)
- **Pet Naming**: Assign friendly labels to model instances

**Mood States**:

```typescript
type Mood =
  | 'CALM'        // Balanced, stable state
  | 'FOCUSED'     // High stability + good creativity
  | 'AGITATED'    // Low stability, needs training
  | 'DREAMY'      // High creativity, low stability
  | 'BURNT_OUT';  // Low creativity + high stability (overtrained)
```

**BrainStats Interface**:

```typescript
interface BrainStats {
  id: string;                    // Linked to model artifact
  label: string;                 // Pet name (e.g., "Avi-LM-Pet #1")
  createdAt: string;             // ISO timestamp
  updatedAt: string;             // ISO timestamp

  // Training metrics
  totalTrainSteps: number;       // Total epochs completed
  totalTokensSeen: number;       // Total tokens processed
  vocabSize: number;             // Current vocabulary size

  // Brain vitals (0-100)
  creativity: number;            // Generation diversity
  stability: number;             // Training stability

  // State
  mood: Mood;                    // Current mood
  lastFeedSummary: string | null; // Last corpus summary
  autonomyEnabled: boolean;      // Autonomous mode active

  // Memory
  diary: DiaryEntry[];           // Event log
}
```

**Event Types**:

- `TRAIN_RUN`: Training epoch completed → increases stability
- `GEN_RUN`: Text generation performed → affects creativity
- `FEED`: New corpus fed → adds vocabulary, changes mood
- `IDLE_TICK`: Time passing without activity → natural decay
- `MOOD_OVERRIDE`: Manual mood change by user

**Usage**:

```typescript
import { BrainEngine } from '../lib/BrainEngine';

// Initialize or load
const brain = BrainEngine.initialize('model-v1', 'My Neural Pet');

// Record training event
brain.recordEvent({
  type: 'TRAIN_RUN',
  timestamp: Date.now(),
  payload: { epochs: 10, finalLoss: 1.234 }
});

// Record generation event
brain.recordEvent({
  type: 'GEN_RUN',
  timestamp: Date.now(),
  payload: { tokensGenerated: 50, temperature: 0.8 }
});

// Check mood and get suggestions
const stats = brain.getStats();
console.log(`Mood: ${stats.mood}`);
console.log(`Creativity: ${stats.creativity}, Stability: ${stats.stability}`);

if (stats.mood === 'AGITATED') {
  console.log('Suggestion: Model needs more training for stability');
}

// Save state
brain.save();
```

**Mood Determination Logic**:

- **FOCUSED**: stability ≥ 70 AND creativity ≥ 60
- **DREAMY**: creativity ≥ 80 AND stability < 50
- **AGITATED**: stability < 30
- **BURNT_OUT**: creativity < 20 AND stability ≥ 60
- **CALM**: Default balanced state

**When to Use**:

- Gamifying model training experience
- Providing intuitive feedback to users
- Tracking model lifecycle and usage patterns
- Educational demonstrations of model behavior
- Experiment logging with personality

**Important Note**: BrainEngine NEVER performs heavy operations autonomously. It only suggests actions via UI.

### 10. RMSNorm Layer (`src/lib/RMSNorm.ts`)

#### **RMSNorm** - Root Mean Square Normalization

**Location**: `/home/user/neuro-lingua/src/lib/RMSNorm.ts`

**Purpose**: Efficient normalization layer for neural networks (v4.0 feature)

**Key Benefits**:

- **20% less memory**: Only γ parameters (no β like LayerNorm)
- **2x faster**: No mean calculation required
- **Equivalent performance**: Matches LayerNorm in practice
- **Simpler**: Fewer parameters to learn

**Formula**:

```
RMSNorm(x) = (x / RMS(x)) ⊙ γ
where RMS(x) = sqrt(mean(x²) + ε)
```

**Reference**: Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"

**Used in**: T5, LLaMA, PaLM architectures

**Interface**:

```typescript
class RMSNorm {
  constructor(dimension: number, epsilon?: number);

  forward(x: number[]): number[];
  backward(gradOutput: number[]): number[];
  updateWeights(learningRate: number): void;

  getState(): RMSNormState;
  loadState(state: RMSNormState): void;
}
```

**Usage**:

```typescript
import { RMSNorm } from '../lib/RMSNorm';

// Create RMSNorm layer for 128-dim vectors
const rmsNorm = new RMSNorm(128, 1e-6);

// Forward pass
const normalized = rmsNorm.forward(hiddenState);

// Backward pass during training
const gradInput = rmsNorm.backward(gradOutput);

// Update parameters
rmsNorm.updateWeights(learningRate);
```

**When to Use**:

- As replacement for LayerNorm in attention layers
- In transformer architectures (v4.0 upgrade path)
- When memory efficiency is critical
- For faster training/inference

### 11. Grouped-Query Attention (GQA) - v4.1 Feature

#### **Overview**

**Purpose**: Efficient attention mechanism that reduces KV cache size while maintaining quality

**Key Concept**: Share key-value heads across multiple query heads

**Benefits**:
- **2-4x faster inference**: Smaller KV cache = faster memory access
- **50% memory savings**: Fewer KV parameters to store
- **Minimal quality loss**: <1% performance drop vs full MHA
- **Scalable**: Works well at larger model sizes

#### **Architecture Comparison**

**Multi-Head Attention (MHA)**: Standard attention
```
numHeads = 8
numKVHeads = 8
Each head has its own Q, K, V
Total KV heads: 8
```

**Grouped-Query Attention (GQA)**: Efficient variant
```
numHeads = 8
numKVHeads = 2  (or 4)
8 query heads share 2 KV heads
Query heads grouped: [Q0,Q1,Q2,Q3] → KV0, [Q4,Q5,Q6,Q7] → KV1
Total KV heads: 2 (4x reduction)
```

**Multi-Query Attention (MQA)**: Extreme case
```
numHeads = 8
numKVHeads = 1
All query heads share 1 KV head
Maximum efficiency, slight quality loss
```

#### **Configuration**

**In TransformerLM**:

```typescript
// Standard MHA (baseline)
const model = new TransformerLM({
  numHeads: 8,
  numKVHeads: 8  // Same as numHeads
});

// GQA (recommended)
const model = new TransformerLM({
  numHeads: 8,
  numKVHeads: 2  // 4x fewer KV heads
});

// MQA (maximum efficiency)
const model = new TransformerLM({
  numHeads: 8,
  numKVHeads: 1  // Single shared KV head
});
```

**In UI (App.tsx)**:

The training panel exposes `numHeads` and `numKVHeads` sliders for transformer architecture. Constraints:
- `numKVHeads` must divide `numHeads` evenly
- `1 ≤ numKVHeads ≤ numHeads`
- Common ratios: 1:1 (MHA), 2:1, 4:1, 8:1 (MQA)

#### **Mathematical Details**

**Standard MHA**:
```
For each head i:
  Q_i = X W_q^i
  K_i = X W_k^i
  V_i = X W_v^i
  Attention_i = softmax(Q_i K_i^T / √d) V_i
```

**GQA**:
```
For each KV group g with heads [i_start, ..., i_end]:
  K_g = X W_k^g  (shared across group)
  V_g = X W_v^g  (shared across group)

For each query head i in group g:
  Q_i = X W_q^i  (unique)
  Attention_i = softmax(Q_i K_g^T / √d) V_g  (uses shared KV)
```

#### **Performance Characteristics**

| Configuration | KV Cache Size | Speed | Quality | Use Case |
|---------------|---------------|-------|---------|----------|
| **MHA** (8:8) | 100% | 1.0x | Best | Small models, quality critical |
| **GQA** (8:4) | 50% | 1.5x | 99% | Balanced efficiency |
| **GQA** (8:2) | 25% | 2.0x | 98% | Production deployment |
| **MQA** (8:1) | 12.5% | 4.0x | 95% | Resource constrained |

#### **Implementation Details**

**Location**: `/home/user/neuro-lingua/src/App.tsx` (configuration)

The transformer model internally handles GQA by:
1. Creating `numKVHeads` key/value projections
2. Creating `numHeads` query projections
3. Grouping `numHeads / numKVHeads` queries per KV head
4. Computing attention within each group
5. Concatenating results across all heads

**Reference**: Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"

**When to Use GQA**:

- **Production deployment**: Reduce serving costs
- **Long context**: KV cache grows with sequence length
- **Large models**: Memory savings more significant
- **Inference-heavy**: Benefits compound over many generations

**When to Use MHA**:

- **Research**: Baseline comparisons
- **Small models**: Overhead not significant
- **Quality critical**: Every % matters
- **Training**: GQA benefits mainly at inference

### 12. Lion Optimizer - v4.0 Feature

#### **Overview**

**Location**: `/home/user/neuro-lingua/src/training/LionOptimizer.ts`

**Purpose**: Memory-efficient optimizer with faster convergence than Adam

**Key Concept**: Uses sign of momentum instead of actual gradient magnitude

**Benefits**:
- **50% less memory**: Only one momentum buffer (vs two for Adam)
- **1.5-2× faster convergence**: Reaches lower loss in fewer steps
- **More stable training**: Less sensitive to learning rate choice
- **Simpler**: Fewer hyperparameters to tune

#### **Algorithm**

```
update = sign(β₁ × m + (1 - β₁) × g)
θ ← θ - η × update - η × λ × θ  (with weight decay)
m ← β₂ × m + (1 - β₂) × g
```

Where:
- `g` = gradient
- `m` = momentum buffer
- `η` = learning rate (lower than Adam, ~3e-4)
- `β₁` = 0.9 (update momentum)
- `β₂` = 0.99 (state momentum)
- `λ` = weight decay (0.01)

**Reference**: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"

#### **Comparison: Adam vs Lion**

| Feature | Adam | Lion |
|---------|------|------|
| Memory | 2× parameters | **1× parameters** |
| Convergence | baseline | **1.5-2× faster** |
| Learning rate | ~1e-3 | ~3e-4 (lower) |
| Momentum buffers | 2 (m, v) | 1 (m only) |
| Final perplexity | baseline | -3% to -8% better |

#### **Usage**

```typescript
// In model constructor
const model = new ProNeuralLM(vocab, 64, 0.0003, 3, 'lion', 0.9, 0.1, 42);

// Or in UI: Select "Lion (v4.0)" from optimizer dropdown
```

#### **Standalone Usage**

```typescript
import { LionOptimizer, LION_DEFAULTS } from '../training/LionOptimizer';

const optimizer = new LionOptimizer({
  lr: 3e-4,
  beta1: 0.9,
  beta2: 0.99,
  weightDecay: 0.01
});

// Update weights
optimizer.updateMatrix(weights, gradients, 'layer1');
optimizer.updateVector(biases, biasGradients, 'bias1');
```

#### **When to Use Lion**

✅ **Recommended for**:
- Small to medium models (10M - 3B parameters)
- Memory-constrained environments (browser, mobile)
- Faster experimentation cycles
- Transformer training

❌ **Consider alternatives when**:
- Very high learning rates are needed
- Existing Adam pipelines work well
- Second-order methods (Newton, L-BFGS) provide better results

### 13. Cerebro Injection System (`src/lib/expandable/`)

The Cerebro system enables dynamic concept injection into neural network layers using mathematical projections and residual analysis.

#### **Architecture Overview**

```
CerebroPanel (UI)
    ↓
InjectionEngine (Core Logic)
    ↓
InjectableLayer (Interface)
    ↓
Model Adapters (ProNeuralLM, AdvancedNeuralLM, TransformerLM)
    ↓
injection_math.ts (Mathematical Operations)
```

#### **InjectionEngine.ts** - Core Injection Engine

**Location**: `/home/user/neuro-lingua/src/lib/expandable/InjectionEngine.ts` (169 lines)

**Purpose**: Orchestrates the diagnosis, proposal, and execution of concept injections

**Key Methods**:

```typescript
class InjectionEngine {
  // Analyze layer capacity for new concepts
  diagnose(bubbles: CerebroBubble[], layer: InjectableLayer): InjectionDiagnostics;

  // Create injection proposal based on diagnostics
  propose(diag: InjectionDiagnostics, target: InjectionTarget): InjectionProposal;

  // Execute injection with rollback support
  execute(proposal: InjectionProposal, layer: InjectableLayer, bubbles?: CerebroBubble[]): InjectionEvent;

  // Extract top eigenvectors for concept representation
  materialiseVectors(bubbles: CerebroBubble[], layer: InjectableLayer, components: number): number[][];
}
```

**Diagnostics Interface**:

```typescript
interface InjectionDiagnostics {
  meanResidual: number;    // Average energy in residual space
  tracePerp: number;       // Trace of orthogonal projection
  estimatedGain: number;   // Expected improvement from injection
  suggestedK: number;      // Recommended neurons to add
}
```

**Injection Methods**:

- `residual_eig`: Use top eigenvectors of residual covariance (default when meanResidual > epsilon)
- `random_he`: He initialization for random expansion
- `svd_local`: Local SVD-based initialization

#### **injection_math.ts** - Mathematical Operations

**Location**: `/home/user/neuro-lingua/src/lib/expandable/injection_math.ts` (205 lines)

**Purpose**: Linear algebra operations for concept injection

**Key Functions**:

```typescript
// Gram-Schmidt orthogonalization
gramSchmidt(vectors: number[][]): number[][];

// Analyze residuals outside current basis
analyseResiduals(bubbles: CerebroBubble[], basis: number[][]): ResidualAnalysis;

// Activation-weighted covariance matrix
weightedCovariance(bubbles: CerebroBubble[], dimension: number): number[][];

// Orthogonal projector construction (I - BB^T)
orthogonalProjector(basis: number[][], dimension: number): number[][];

// Top eigenvectors via power iteration with deflation
topEigenvectors(matrix: number[][], k: number): number[][];

// Suggest optimal expansion size
suggestKFromEnergy(meanResidual: number, residualTrace: number, hiddenSize: number): number;
```

**Mathematical Concepts**:

1. **Residual Analysis**: Project concept embeddings onto existing weight basis, measure orthogonal energy
2. **Weighted Covariance**: Σ = Σ_i a_i (x_i - μ)(x_i - μ)^T where a_i is activation
3. **Orthogonal Projector**: P^⊥ = I - BB^T projects onto complement of basis B
4. **Energy Gain**: Estimated improvement from capturing residual variance

#### **InjectableLayer Interface**

**Location**: `/home/user/neuro-lingua/src/lib/expandable/InjectableLayer.ts` (19 lines)

```typescript
interface InjectableLayer {
  getTarget(): InjectionTarget;
  canInject(k: number): boolean;
  inject(k: number, method: string): void;
  exportWeights(): [Float32Array | null, Float32Array | null];
  importWeights(snapshot: [Float32Array | null, Float32Array | null]): void;
}
```

#### **CerebroBubble Types**

**Location**: `/home/user/neuro-lingua/src/types/injection.ts`

```typescript
type CerebroBubbleTag = 'body' | 'desire' | 'risk' | 'value' | 'action' | 'other';

interface CerebroBubble {
  id: string;
  label: string;
  embedding: number[];      // Concept vector
  activation: number;       // 0..1 importance weight
  tag?: CerebroBubbleTag;   // Semantic category
  ts: number;               // Timestamp
  members?: string[];       // Associated tokens/samples
}
```

#### **Model Adapters**

Each neural architecture has an adapter implementing `InjectableLayer`:

- `ProNeuralLMAdapter.ts`: Adapts feedforward hidden layer
- `AdvancedNeuralLMAdapter.ts`: Adapts enhanced feedforward layer
- `TransformerLMAdapter.ts`: Adapts transformer MLP layers

**Usage Example**:

```typescript
import { InjectionEngine } from '../lib/expandable/InjectionEngine';
import { ProNeuralLMAdapter } from '../lib/expandable/ProNeuralLMAdapter';

// Create adapter for model
const adapter = new ProNeuralLMAdapter(model, 'hidden-0');

// Initialize engine
const engine = new InjectionEngine({ epsilon: 0.05, minGain: 0.01 });

// Diagnose current state
const diagnostics = engine.diagnose(bubbles, adapter);
console.log(`Residual energy: ${diagnostics.meanResidual}`);
console.log(`Suggested expansion: ${diagnostics.suggestedK} neurons`);

// Create and execute proposal
const proposal = engine.propose(diagnostics, adapter.getTarget());
const event = engine.execute(proposal, adapter, bubbles);

if (event.accepted) {
  console.log(`Injection accepted. Gain: ${event.delta.estimatedGain}`);
} else {
  console.log('Injection rejected (gain below threshold)');
}
```

**When to Use**:

- Dynamically expanding model capacity
- Injecting domain concepts
- Research on neural network plasticity
- Controllable generation via concept steering

### 14. Advanced Loss Functions (`src/losses/`)

#### **advanced.ts** - Specialized Loss Functions

**Location**: `/home/user/neuro-lingua/src/losses/advanced.ts` (76 lines)

##### **Focal Loss**

**Purpose**: Handle class imbalance by down-weighting easy examples

**Formula**: FL(p_t) = -α(1 - p_t)^γ log(p_t)

```typescript
function focalLoss(
  logits: number[],
  targets: number[],
  options?: { gamma?: number; alpha?: number }
): number;

// Default: gamma=2, alpha=0.25
// Higher gamma = more focus on hard examples
```

**When to Use**:
- Imbalanced datasets
- Hard example mining
- Object detection tasks

**Reference**: Lin et al. (2017) "Focal Loss for Dense Object Detection"

##### **Label Smoothing Cross-Entropy**

**Purpose**: Regularization by softening one-hot targets

**Formula**: y'_i = (1 - ε)·y_i + ε/K

```typescript
function labelSmoothingCrossEntropy(
  logits: number[],
  targetIndex: number,
  classes: number,
  smoothing?: number  // default: 0.1
): number;
```

**When to Use**:
- Preventing overconfident predictions
- Improving generalization
- Calibrating probability estimates

##### **Symmetric Cross-Entropy**

**Purpose**: Noise-robust loss combining forward and reverse KL

**Formula**: L_SCE = α·CE(p,q) + β·CE(q,p)

```typescript
function symmetricCrossEntropy(
  logits: number[],
  targets: number[],
  alpha?: number,  // forward weight (default: 1)
  beta?: number    // reverse weight (default: 1)
): number;
```

**When to Use**:
- Noisy labels
- Self-training
- Semi-supervised learning

##### **Cosine Embedding Loss**

**Purpose**: Similarity learning in embedding space

```typescript
function cosineEmbeddingLoss(
  x: number[],
  y: number[],
  label: 1 | -1,  // 1=similar, -1=dissimilar
  margin?: number // default: 0.0
): number;
```

**When to Use**:
- Siamese networks
- Contrastive learning
- Embedding alignment

#### **information_bottleneck.ts** - Information Bottleneck Loss

**Location**: `/home/user/neuro-lingua/src/losses/information_bottleneck.ts` (316 lines)

**Purpose**: Implement the Information Bottleneck principle for representation learning

**Core Concept**:

Find representation Z that:
- Compresses input X: minimize I(X;Z)
- Preserves relevant information: maximize I(Z;Y)

**Loss**: L_IB = -I(Z;Y) + β·I(X;Z)

where β controls compression-prediction trade-off

**Key Functions**:

```typescript
// Estimate mutual information using histograms
function estimateMutualInformation(
  x: number[],
  y: number[],
  numBins?: number,    // default: 50
  epsilon?: number     // default: 1e-10
): number;

// Compute representation entropy H(Z)
function computeRepresentationEntropy(
  activations: number[][],
  numBins?: number,
  epsilon?: number
): number;

// Full IB loss computation
function informationBottleneckLoss(
  inputs: number[][],
  hiddenActivations: number[][],
  targetLogits: number[][],
  targetIndices: number[],
  config: InformationBottleneckConfig
): InformationMetrics;

// Beta annealing schedules
function getBetaSchedule(
  schedule: 'constant' | 'linear' | 'exponential' | 'cosine',
  epoch: number,
  totalEpochs: number,
  betaStart: number,
  betaEnd: number
): number;

// Hybrid CE + IB loss
function hybridIBLoss(ceLoss: number, ibLoss: number, alpha: number): number;
```

**Information Metrics Interface**:

```typescript
interface InformationMetrics {
  compressionMI: number;        // I(X;Z)
  predictionMI: number;         // I(Z;Y)
  ibLoss: number;               // Combined loss
  beta: number;                 // Current beta
  representationEntropy: number; // H(Z)
  conditionalEntropy: number;   // H(Z|X)
}
```

**Beta Annealing Strategies**:

| Schedule | Formula | Use Case |
|----------|---------|----------|
| constant | β = β_start | Stable training |
| linear | β = β_start + (β_end - β_start)·t | Gradual increase |
| exponential | β = β_start·(β_end/β_start)^t | Aggressive compression |
| cosine | β = β_end + (β_start - β_end)·(1 + cos(πt))/2 | Smooth annealing |

**Usage Example**:

```typescript
import { informationBottleneckLoss, getBetaSchedule } from '../losses/information_bottleneck';

// Get current beta based on epoch
const beta = getBetaSchedule('cosine', epoch, totalEpochs, 0.01, 1.0);

// Compute IB loss
const metrics = informationBottleneckLoss(
  inputs,
  hiddenActivations,
  outputLogits,
  targetIndices,
  { beta, numBins: 50 }
);

console.log(`I(X;Z) = ${metrics.compressionMI.toFixed(4)}`);
console.log(`I(Z;Y) = ${metrics.predictionMI.toFixed(4)}`);
console.log(`IB Loss = ${metrics.ibLoss.toFixed(4)}`);
```

**Reference**: Tishby et al. (1999) "The Information Bottleneck Method"

### 15. Mathematical Analysis (`src/math/`)

#### **analysis.ts** - Spectral and Lyapunov Analysis

**Location**: `/home/user/neuro-lingua/src/math/analysis.ts` (234 lines)

**Purpose**: Stability analysis for linearized training dynamics

##### **Spectral Radius**

Compute dominant eigenvalue magnitude via power iteration:

```typescript
function spectralRadius(
  matrix: Matrix,
  options?: {
    maxIterations?: number;  // default: 1024
    tolerance?: number;      // default: 1e-9
    initialVector?: number[];
  }
): SpectralRadiusResult;

interface SpectralRadiusResult {
  spectralRadius: number;
  iterations: number;
  converged: boolean;
  tolerance: number;
}
```

**Interpretation**:
- ρ < 1: Stable (discrete-time)
- ρ = 1: Marginally stable
- ρ > 1: Unstable (divergent)

##### **Lyapunov Stability Analysis**

Analyze system stability around equilibrium:

```typescript
function analyzeLyapunov(
  matrix: Matrix,
  options?: {
    steps?: number;          // trajectory steps
    perturbation?: number;   // default: 1e-8
    discreteTime?: boolean;  // default: true
  }
): LyapunovAnalysisResult;

interface LyapunovAnalysisResult extends SpectralRadiusResult {
  lyapunovExponent: number;  // Growth rate (base-e)
  stable: boolean;           // Satisfies stability criterion
  stabilityMargin: number;   // Distance to instability
  assumptions: string[];     // Modeling assumptions
}
```

**Stability Criteria**:
- Discrete-time: |λ|_max < 1
- Continuous-time: Re(λ) < 0

**When to Use**:
- Analyzing training convergence
- Detecting unstable learning rates
- Understanding gradient flow dynamics

#### **statistics.ts** - Fisher Information Statistics

**Location**: `/home/user/neuro-lingua/src/math/statistics.ts` (197 lines)

**Purpose**: Information-geometric curvature diagnostics

##### **Empirical Fisher Information**

```typescript
function empiricalFisherFromGradients(
  gradients: Vector[],
  options?: { epsilon?: number }
): Matrix;

function fisherHessianStatistics(
  gradients: Vector[],
  options?: { epsilon?: number }
): FisherStatistics;

interface FisherStatistics {
  fisher: Matrix;           // Empirical Fisher matrix
  frobeniusNorm: number;    // ||F||_F (Hessian proxy)
  trace: number;            // tr(F) = avg squared gradient norm
  spectralNormBound: number; // Gershgorin disc bound
  notes: string[];          // Interpretation notes
}
```

**Mathematical Background**:

The empirical Fisher information approximates the Hessian:

F ≈ E[∇L ∇L^T]

Properties:
- Trace = average squared gradient norm
- Frobenius norm upper-bounds Gauss-Newton Hessian
- Gershgorin discs bound spectral norm

##### **Fisher Quadratic Form**

```typescript
function fisherQuadraticForm(fisher: Matrix, vector: Vector): number;
```

Evaluates v^T F v for trust region analysis.

### 16. Regularizers (`src/models/regularizers.ts`)

**Location**: `/home/user/neuro-lingua/src/models/regularizers.ts` (97 lines)

**Purpose**: Advanced regularization techniques beyond standard dropout

#### **DropConnect**

Randomly masks weight connections (not activations like Dropout):

```typescript
function applyDropConnect(
  matrix: Matrix,
  config: DropConnectConfig
): Matrix;

interface DropConnectConfig {
  rate: number;    // probability of dropping (0-1)
  seed?: number;   // for reproducibility
}
```

**Difference from Dropout**:
- Dropout: Zeros activations at rate p
- DropConnect: Zeros individual weights at rate p

**Benefits**:
- More fine-grained regularization
- Better gradient flow
- Effective for recurrent networks

**Reference**: Wan et al. (2013) "Regularization of Neural Networks using DropConnect"

#### **Batch Renormalization**

Enhanced batch normalization with correction factors:

```typescript
function batchRenormalize(
  inputs: Matrix,
  state: BatchRenormState
): BatchRenormResult;

interface BatchRenormState {
  runningMean: number[];
  runningVar: number[];
  momentum: number;
  epsilon: number;
  rMax: number;  // Clipping for r
  dMax: number;  // Clipping for d
}

interface BatchRenormResult {
  normalized: Matrix;
  r: number[];  // Scale correction
  d: number[];  // Shift correction
}
```

**Correction Factors**:
- r = σ_batch / σ_running (clipped to [1/rMax, rMax])
- d = (μ_batch - μ_running) / σ_running (clipped to [-dMax, dMax])

**Benefits**:
- Works with small batch sizes
- Stable training with non-i.i.d. mini-batches
- Smooth transition from training to inference

**Reference**: Ioffe (2017) "Batch Renormalization"

### 17. Autodiff System (`src/autodiff/graph.ts`)

**Location**: `/home/user/neuro-lingua/src/autodiff/graph.ts` (188 lines)

**Purpose**: Dynamic computation graph with reverse-mode automatic differentiation

**Status**: Experimental - for replacing hand-derived gradients

#### **Variable Class**

```typescript
class Variable {
  value: number;
  grad: number;
  name?: string;

  // Arithmetic operations
  add(other: Variable | number): Variable;
  sub(other: Variable | number): Variable;
  mul(other: Variable | number): Variable;
  div(other: Variable | number): Variable;

  // Unary operations
  neg(): Variable;
  pow(exponent: number): Variable;

  // Activation functions
  tanh(): Variable;
  exp(): Variable;
  log(): Variable;

  // Gradient operations
  backward(gradient?: number): void;
  zeroGrad(): void;
}
```

#### **Usage Example**

```typescript
import { Variable } from '../autodiff/graph';

// Create variables
const x = new Variable(2.0, 'x');
const y = new Variable(3.0, 'y');

// Build computation graph
const z = x.mul(y).add(x.pow(2));  // z = x*y + x^2

// Forward: z.value = 2*3 + 2^2 = 10

// Backward pass
z.backward();

// Gradients: ∂z/∂x = y + 2x = 7, ∂z/∂y = x = 2
console.log(`dz/dx = ${x.grad}`);  // 7
console.log(`dz/dy = ${y.grad}`);  // 2
```

#### **Mean Squared Error Helper**

```typescript
function meanSquaredError(predictions: number[], targets: number[]): Variable;
```

#### **Graph Internals**

- **Topological Sort**: Ensures correct gradient flow order
- **Gradient Accumulation**: Handles shared variables
- **Lazy Evaluation**: Gradients computed only when `backward()` called

**When to Use**:

- Prototyping new loss functions
- Verifying hand-derived gradients
- Educational demonstrations
- Experimental layer implementations

### 18. Edge Learning Diagnostics (`src/backend/edgeLearning.ts`)

**Location**: `/home/user/neuro-lingua/src/backend/edgeLearning.ts` (146 lines)

**Purpose**: Information-theoretic analysis of model learning efficiency

#### **Diagnostics Interface**

```typescript
interface EdgeLearningDiagnostics {
  fisherInformation: number;    // Loss surface curvature
  entropy: number;              // Parameter uncertainty
  estimatorCovariance: number;  // Variance of parameter estimates
  cramerRaoBound: number;       // Theoretical minimum variance
  efficiency: number;           // Achieved vs theoretical (0-1)
  variance: number;             // Actual variance from loss
  timestamp: number;
  status: 'success' | 'error';
  error?: string;
}
```

#### **Main Function**

```typescript
function computeSimulatedEdgeLearningDiagnostics(
  modelSize: number,
  trainingLosses: number[]
): EdgeLearningDiagnostics;
```

#### **Metric Interpretation**

| Metric | Meaning | Ideal Value |
|--------|---------|-------------|
| **Fisher Information** | Sharpness of loss minimum | Higher = distinctive solution |
| **Entropy** | Parameter uncertainty | Moderate (scales with model) |
| **Estimator Covariance** | Variance of estimates (1/Fisher) | Lower = precise estimates |
| **Cramér-Rao Bound** | Theoretical minimum variance | Lower = better possible |
| **Efficiency** | CRB / Achieved variance | Closer to 1 = optimal |
| **Variance** | Actual variance from loss | Lower = better fit |

#### **Usage Example**

```typescript
import { computeSimulatedEdgeLearningDiagnostics } from '../backend/edgeLearning';

const diagnostics = computeSimulatedEdgeLearningDiagnostics(
  model.getParameterCount(),
  trainingHistory.losses
);

if (diagnostics.status === 'success') {
  console.log(`Fisher Information: ${diagnostics.fisherInformation}`);
  console.log(`Learning Efficiency: ${(diagnostics.efficiency * 100).toFixed(1)}%`);

  if (diagnostics.efficiency < 0.5) {
    console.log('Model could potentially learn more from data');
  }
}
```

#### **Theoretical Background**

**Cramér-Rao Bound**: For any unbiased estimator θ̂:

Var(θ̂) ≥ 1 / I(θ)

where I(θ) is the Fisher information.

**Efficiency**: η = CRB / Var(θ̂)

- η = 1: Estimator achieves theoretical limit (optimal)
- η < 1: Room for improvement
- η > 1: Biased estimator (shouldn't happen for unbiased)

**Reference**:
- [Fisher Information](https://en.wikipedia.org/wiki/Fisher_information)
- [Cramér-Rao Bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound)

---

## Development Workflows

### Initial Setup

```bash
# Clone repository
git clone https://github.com/abbrubin150-ui/neuro-lingua.git
cd neuro-lingua

# Install dependencies (use pnpm preferred)
pnpm install
# or: npm install / yarn install

# Start development server
pnpm dev
# Opens browser at http://localhost:5173/neuro-lingua/
```

### Development Commands

```bash
# Development server (hot reload)
pnpm dev

# Type checking
tsc --noEmit

# Linting
pnpm lint

# Format code
pnpm format

# Check formatting without changes
pnpm format:check

# Run tests once
pnpm test

# Run tests in watch mode
pnpm test:watch

# Production build
pnpm build

# Preview production build
pnpm preview

# Train model (Node.js script)
pnpm train

# GPU benchmarks
pnpm benchmark:gpu
```

### Typical Development Workflow

1. **Create feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** with hot reload:

   ```bash
   pnpm dev
   ```

3. **Verify code quality**:

   ```bash
   pnpm lint
   pnpm format
   pnpm test
   tsc --noEmit
   ```

4. **Build and test**:

   ```bash
   pnpm build
   pnpm preview
   ```

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   git push origin feature/your-feature-name
   ```

### File Watching

Vite automatically watches and hot-reloads:

- TypeScript/TSX files in `src/`
- CSS files
- `index.html`

For test watching:

```bash
pnpm test:watch
```

---

## Testing Guidelines

### Test Framework

- **Vitest** with jsdom environment
- **@testing-library/react** for component tests
- **TypeScript** for type-safe tests

### Test Organization

```
tests/
├── ProNeuralLM.test.ts          # Core model tests
├── ProNeuralLM.device.test.ts   # Device-specific model tests
├── AdvancedNeuralLM.test.ts     # Advanced features tests
├── TransformerLM.test.ts        # Transformer architecture tests
├── GovernanceEngine.test.ts     # Autonomous governance tests
├── MathUtils.test.ts            # Math utilities tests
├── tokenizer.test.ts            # Tokenization tests
├── sampler.test.ts              # Generation algorithms tests
├── embedding_extraction.test.ts # Embedding extraction tests
├── App.test.tsx                 # Main React app tests
├── backend/                     # Backend tests (WebGPU, edge learning)
├── components/                  # Component tests (UI panels)
├── compression/                 # Compression system tests
├── contexts/                    # Context provider tests
├── lib/                         # Library module tests
├── losses/                      # Loss function tests
├── math/                        # Mathematical analysis tests
│   └── analysis.test.ts
├── numerics/                    # Numerical correctness tests
│   ├── sampling.test.ts
│   └── bayesian.test.ts
├── test_edge_analyzer.py        # Python edge analyzer tests
├── test_on_the_edge_learning.py # Python edge learning tests
└── test_recursive_optimizer.py  # Python recursive optimizer tests
```

**Total Test Files**: ~38 test files covering frontend, backend, and Python modules

### Writing Tests

**Example test structure**:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { ProNeuralLM } from '../src/lib/ProNeuralLM';

describe('ProNeuralLM', () => {
  let model: ProNeuralLM;

  beforeEach(() => {
    model = new ProNeuralLM(10, 16, 3);
  });

  it('should initialize with correct dimensions', () => {
    expect(model).toBeDefined();
    // assertions...
  });

  it('should train without errors', () => {
    const corpus = 'hello world';
    expect(() => {
      model.train(corpus, 5, 0.1);
    }).not.toThrow();
  });
});
```

### Test Coverage Focus

**Critical areas to test**:

1. **Neural network math**: Forward/backward passes, gradient computation
2. **Numerical stability**: Softmax, log-sum-exp, overflow/underflow
3. **Serialization**: Model export/import, optimizer state preservation
4. **Generation**: Sampling algorithms, beam search correctness
5. **WebGPU**: Fallback behavior, operation equivalence with CPU
6. **Edge cases**: Empty corpus, single token, very long sequences

### Running Tests

```bash
# All tests once
pnpm test

# Watch mode (re-run on changes)
pnpm test:watch

# Specific test file
pnpm test ProNeuralLM

# With coverage (if configured)
pnpm test --coverage
```

### Test Helpers

**Location**: `tests/setup.ts`

Provides:

- Mock localStorage implementation
- Global test utilities
- Matrix/vector comparison helpers

---

## Code Conventions

### TypeScript Style

#### Naming Conventions

- **Classes**: PascalCase (`ProNeuralLM`, `WebGPUBackend`)
- **Interfaces/Types**: PascalCase (`SerializedModel`, `Optimizer`)
- **Functions/Methods**: camelCase (`trainModel`, `computeLoss`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_HYPERPARAMETERS`, `STORAGE_KEYS`)
- **Files**: camelCase for utilities, PascalCase for components (`utils.ts`, `TrainingPanel.tsx`)

#### Type Annotations

- Always use explicit return types for functions
- Prefer `interface` over `type` for object shapes
- Use `const` assertions for readonly data: `as const`
- Enable strict mode (already configured)

**Example**:

```typescript
// Good
function computeLoss(predictions: number[], targets: number[]): number {
  // implementation
}

interface ModelConfig {
  hiddenSize: number;
  learningRate: number;
}

const CONSTANTS = {
  MAX_EPOCHS: 200,
  MIN_LR: 0.001
} as const;

// Avoid
function computeLoss(predictions, targets) {
  // Missing types
  // implementation
}
```

### React Conventions

#### Component Structure

```typescript
import React, { useState, useEffect } from 'react';

interface ComponentProps {
  // Props definition
  title: string;
  onSave: () => void;
}

export function MyComponent({ title, onSave }: ComponentProps) {
  // Hooks first
  const [state, setState] = useState<string>('');

  useEffect(() => {
    // Side effects
  }, []);

  // Event handlers
  const handleClick = () => {
    // handler logic
  };

  // Render
  return (
    <div>
      {/* JSX */}
    </div>
  );
}
```

#### Hooks Usage

- Always use functional components
- Define custom hooks in separate files if reusable
- Follow Rules of Hooks (no conditionals, consistent order)
- Use `useCallback` for event handlers passed to child components
- Use `useMemo` for expensive computations

### File Organization

#### Import Order

1. External libraries (React, third-party)
2. Internal modules (lib/, components/)
3. Types (if separate from implementation)
4. CSS/styles (if applicable)

**Example**:

```typescript
import React, { useState } from 'react';
import { ProNeuralLM } from '../lib/ProNeuralLM';
import { TrainingPanel } from '../components/TrainingPanel';
import type { ModelConfig } from '../types';
```

#### Export Patterns

- Named exports for utilities and components
- Default exports avoided (except for main App component)
- Re-export from index files when creating module boundaries

### Code Formatting

**Prettier Configuration** (`.prettierrc`):

```json
{
  "singleQuote": true,
  "trailingComma": "none",
  "printWidth": 100,
  "semi": true
}
```

**Rules**:

- Single quotes for strings
- No trailing commas
- 100 character line width
- Semicolons required
- 2-space indentation

**Apply formatting**:

```bash
pnpm format
```

### Comments and Documentation

#### When to Comment

- Complex algorithms (neural network forward/backward passes)
- Non-obvious optimizations
- Mathematical formulas (include references)
- Public API methods (JSDoc)
- TODOs with context

**Example**:

```typescript
/**
 * Compute softmax with numerical stability using log-sum-exp trick.
 * Reference: https://en.wikipedia.org/wiki/LogSumExp
 *
 * @param logits - Raw model outputs
 * @returns Normalized probabilities summing to 1
 */
function stableSoftmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}
```

#### JSDoc for Public APIs

```typescript
/**
 * Train the neural language model on provided corpus.
 *
 * @param corpus - Training text (will be tokenized)
 * @param epochs - Number of training iterations (1-200)
 * @param learningRate - Step size for gradient descent (0.001-1.0)
 * @param options - Optional training configuration
 * @throws {Error} If corpus is empty or vocabulary too small
 */
public train(
  corpus: string,
  epochs: number,
  learningRate: number,
  options?: TrainingOptions
): void {
  // implementation
}
```

### Error Handling

#### Principles

- Validate inputs early
- Provide descriptive error messages
- Fail fast on unrecoverable errors
- Log warnings for non-critical issues
- Graceful degradation where possible (e.g., GPU fallback)

**Example**:

```typescript
if (vocabSize < MIN_VOCAB_SIZE) {
  throw new Error(
    `Vocabulary size ${vocabSize} is too small. ` +
      `Minimum required: ${MIN_VOCAB_SIZE}. ` +
      `Provide a larger or more diverse corpus.`
  );
}

// Graceful degradation
try {
  await webgpuBackend.initialize();
  console.log('WebGPU acceleration enabled');
} catch (error) {
  console.warn('WebGPU initialization failed, falling back to CPU:', error);
  this.gpuOps = null; // Will use CPU implementations
}
```

### Performance Considerations

#### Neural Network Code

- Minimize array allocations in hot loops
- Reuse buffers where possible
- Use typed arrays for large numerical data
- Batch operations on GPU when available

**Example**:

```typescript
// Good: Reuse buffer
const activations = new Float32Array(this.hiddenSize);
for (let i = 0; i < this.hiddenSize; i++) {
  activations[i] = Math.max(0, preActivations[i]); // ReLU
}

// Avoid: Creating new arrays in loop
for (let epoch = 0; epoch < epochs; epoch++) {
  const losses = []; // Don't create new array each epoch
  // Instead, clear and reuse
}
```

#### React Performance

- Use `React.memo` for expensive components
- Avoid inline function definitions in render
- Use `useCallback`/`useMemo` appropriately
- Minimize state updates during training loops

---

## Neural Network Architectures

### Architecture Comparison

| Feature               | ProNeuralLM         | AdvancedNeuralLM              | TransformerLM           |
| --------------------- | ------------------- | ----------------------------- | ----------------------- |
| **Architecture**      | Feedforward         | Enhanced Feedforward          | Multi-head Attention    |
| **Layers**            | 2 (hidden + output) | 2 (hidden + output)           | Configurable (1-8)      |
| **Activation**        | ReLU                | ReLU/LeakyReLU/ELU/GELU/Swish | GELU                    |
| **Attention**         | None                | None                          | Multi-head (1-16 heads) |
| **Position Encoding** | None                | None                          | Learned embeddings      |
| **Normalization**     | Optional dropout    | Dropout + Layer norm          | Batch renorm + dropout  |
| **Initialization**    | Random              | He/Xavier                     | Xavier                  |
| **LR Scheduling**     | Constant            | Cosine/Exponential/Warmup     | Constant                |
| **Regularization**    | Dropout             | Dropout + Weight decay        | Dropout                 |
| **Generation**        | Greedy/Sampling     | Greedy/Sampling/Beam          | Greedy/Sampling/Beam    |
| **Complexity**        | Low                 | Medium                        | High                    |
| **Speed**             | Fast                | Medium                        | Slower                  |

### Architecture Selection Guide

**Use ProNeuralLM when**:

- Learning neural network basics
- Need fast training/inference
- Working with small datasets
- Simplicity is preferred
- Baseline comparison needed

**Use AdvancedNeuralLM when**:

- Production use case
- Need better convergence
- Experimenting with training techniques
- Comparing advanced features (activation, scheduling)
- Moderate dataset size

**Use TransformerLM when**:

- Attention mechanisms needed
- Long-range dependencies important
- Sequence modeling tasks
- Research experimentation
- Sufficient computational resources

### Hyperparameter Recommendations

#### ProNeuralLM Baseline

```typescript
{
  hiddenSize: 64,
  epochs: 20,
  learningRate: 0.08,
  optimizer: 'momentum',
  momentum: 0.9,
  dropout: 0.1,
  contextSize: 3
}
```

#### AdvancedNeuralLM Optimized

```typescript
{
  hiddenSize: 128,
  epochs: 40,
  learningRate: 0.1,
  optimizer: 'adam',
  dropout: 0.15,
  contextSize: 4,

  // Advanced config
  activation: 'gelu',
  initialization: 'he',
  lrSchedule: 'cosine',
  lrMin: 1e-6,
  weightDecay: 1e-4,
  useLayerNorm: true
}
```

#### TransformerLM Configuration

```typescript
{
  embeddingDim: 128,
  numHeads: 4,
  numLayers: 2,
  ffDim: 256,
  dropout: 0.1,
  contextSize: 6,
  epochs: 50,
  learningRate: 0.001,
  optimizer: 'adam'
}
```

### Mathematical Details

#### Forward Pass (ProNeuralLM)

```
Input: context tokens [t₁, t₂, ..., tₙ]
1. Embedding lookup: E ∈ ℝ^(V×d)
2. Context concatenation: x = [E[t₁]; E[t₂]; ...; E[tₙ]] ∈ ℝ^(n×d)
3. Hidden layer: h = ReLU(W₁x + b₁) ∈ ℝ^h
4. Output layer: logits = W₂h + b₂ ∈ ℝ^V
5. Softmax: p = exp(logits) / Σexp(logits)
```

#### Backward Pass (Gradient Computation)

```
1. Loss: L = -log(p[target])
2. Output gradient: ∂L/∂logits = p - onehot(target)
3. Hidden gradient: ∂L/∂h = W₂ᵀ × ∂L/∂logits
4. ReLU gradient: ∂L/∂preActivation = ∂L/∂h ⊙ (h > 0)
5. Weight updates: W ← W - lr × ∂L/∂W
```

#### Attention Mechanism (TransformerLM)

```
Q = XWq  (queries)
K = XWk  (keys)
V = XWv  (values)

Attention(Q, K, V) = softmax(QKᵀ/√d)V

Multi-head: concat(head₁, ..., headₕ)Wo
where headᵢ = Attention(QWqᵢ, KWkᵢ, VWvᵢ)
```

---

## Common Tasks

### Task 1: Add New Hyperparameter

**Goal**: Add a new configurable hyperparameter to the training process

**Steps**:

1. **Update type definition** in `src/lib/ProNeuralLM.ts`:

   ```typescript
   export interface TrainingOptions {
     // ... existing options
     myNewParam?: number;
   }
   ```

2. **Add to constants** in `src/config/constants.ts`:

   ```typescript
   export const DEFAULT_HYPERPARAMETERS = {
     // ... existing defaults
     myNewParam: 0.5
   };

   export const HYPERPARAMETER_CONSTRAINTS = {
     // ... existing constraints
     myNewParam: { min: 0, max: 1 }
   };
   ```

3. **Update UI** in `src/components/TrainingPanel.tsx`:

   ```typescript
   const [myNewParam, setMyNewParam] = useState(DEFAULT_HYPERPARAMETERS.myNewParam);

   // Add input control in JSX
   <label>
     My New Param:
     <input
       type="number"
       value={myNewParam}
       onChange={(e) => setMyNewParam(Number(e.target.value))}
       min={HYPERPARAMETER_CONSTRAINTS.myNewParam.min}
       max={HYPERPARAMETER_CONSTRAINTS.myNewParam.max}
       step="0.01"
     />
   </label>
   ```

4. **Use in training logic** in `src/lib/ProNeuralLM.ts`:

   ```typescript
   public train(corpus: string, epochs: number, lr: number, options?: TrainingOptions) {
     const myNewParam = options?.myNewParam ?? DEFAULT_HYPERPARAMETERS.myNewParam;
     // Use myNewParam in training...
   }
   ```

5. **Add tests** in `tests/ProNeuralLM.test.ts`:
   ```typescript
   it('should respect myNewParam configuration', () => {
     const model = new ProNeuralLM(10, 16, 3);
     model.train('test corpus', 5, 0.1, { myNewParam: 0.8 });
     // Verify behavior...
   });
   ```

### Task 2: Implement New Text Generation Method

**Goal**: Add a new generation algorithm (e.g., typical sampling)

**Steps**:

1. **Create implementation** in `src/generation/typical_sampling.ts`:

   ```typescript
   /**
    * Typical sampling (entropy-based truncation)
    */
   export function typicalSampling(logits: number[], tau: number = 0.9): number {
     const probs = softmax(logits);
     const entropy = -probs.reduce((sum, p) => sum + p * Math.log(p), 0);

     // Implementation...
     return selectedIndex;
   }
   ```

2. **Add to generation config** in `src/config/constants.ts`:

   ```typescript
   export const DEFAULT_GENERATION = {
     // ... existing options
     typicalTau: 0.9,
     samplingMode: 'topp' as 'off' | 'topk' | 'topp' | 'typical'
   };
   ```

3. **Integrate into model** in `src/lib/ProNeuralLM.ts`:

   ```typescript
   import { typicalSampling } from '../generation/typical_sampling';

   public generate(prompt: string, maxTokens: number, options?: GenerationOptions): string {
     // ... existing logic

     if (options?.samplingMode === 'typical') {
       nextToken = typicalSampling(logits, options.typicalTau);
     }
   }
   ```

4. **Add UI controls** in `src/components/ChatInterface.tsx`:

   ```typescript
   <select value={samplingMode} onChange={(e) => setSamplingMode(e.target.value)}>
     <option value="off">Greedy</option>
     <option value="topk">Top-k</option>
     <option value="topp">Top-p</option>
     <option value="typical">Typical</option>
   </select>
   ```

5. **Write tests** in `tests/generation/typical_sampling.test.ts`:

   ```typescript
   import { typicalSampling } from '../../src/generation/typical_sampling';

   describe('typicalSampling', () => {
     it('should select high-entropy tokens', () => {
       const logits = [1.0, 2.0, 1.5, 0.5];
       const result = typicalSampling(logits, 0.9);
       expect(result).toBeGreaterThanOrEqual(0);
       expect(result).toBeLessThan(logits.length);
     });
   });
   ```

### Task 3: Add WebGPU Operation

**Goal**: Implement a new GPU-accelerated operation

**Steps**:

1. **Add shader** in `src/backend/webgpu.ts`:

   ```typescript
   private createMyOpShader(): GPUShaderModule {
     const code = `
       @group(0) @binding(0) var<storage, read> input: array<f32>;
       @group(0) @binding(1) var<storage, read_write> output: array<f32>;

       @compute @workgroup_size(64)
       fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
         let i = global_id.x;
         output[i] = myOperation(input[i]);
       }
     `;
     return this.device!.createShaderModule({ code });
   }
   ```

2. **Add operation method** in `src/backend/gpu_neural_ops.ts`:

   ```typescript
   public async myGPUOperation(input: number[]): Promise<number[]> {
     if (!this.backend.isInitialized()) {
       // CPU fallback
       return input.map(x => myCPUOperation(x));
     }

     try {
       const inputTensor = await this.backend.createTensor(input);
       const outputTensor = await this.backend.myOp(inputTensor);
       return outputTensor.toArray();
     } catch (error) {
       console.warn('GPU operation failed, falling back to CPU:', error);
       return input.map(x => myCPUOperation(x));
     }
   }
   ```

3. **Integrate into model** in `src/lib/ProNeuralLM.ts`:

   ```typescript
   private async forwardPass(context: number[]): Promise<number[]> {
     if (this.gpuOps) {
       return await this.gpuOps.myGPUOperation(context);
     } else {
       return this.cpuForwardPass(context);
     }
   }
   ```

4. **Add benchmarks** in `scripts/benchmark_gpu.ts`:

   ```typescript
   async function benchmarkMyOp() {
     const input = Array.from({ length: 1000 }, () => Math.random());

     const cpuStart = performance.now();
     const cpuResult = input.map((x) => myCPUOperation(x));
     const cpuTime = performance.now() - cpuStart;

     const gpuStart = performance.now();
     const gpuResult = await gpuOps.myGPUOperation(input);
     const gpuTime = performance.now() - gpuStart;

     console.log(`CPU: ${cpuTime.toFixed(2)}ms, GPU: ${gpuTime.toFixed(2)}ms`);
     console.log(`Speedup: ${(cpuTime / gpuTime).toFixed(2)}x`);
   }
   ```

### Task 4: Add New React Component

**Goal**: Create a new UI component for the application

**Steps**:

1. **Create component file** in `src/components/MyNewComponent.tsx`:

   ```typescript
   import React, { useState } from 'react';

   interface MyNewComponentProps {
     title: string;
     onAction: (value: string) => void;
   }

   export function MyNewComponent({ title, onAction }: MyNewComponentProps) {
     const [value, setValue] = useState('');

     const handleSubmit = () => {
       onAction(value);
       setValue('');
     };

     return (
       <div className="my-component">
         <h3>{title}</h3>
         <input
           value={value}
           onChange={(e) => setValue(e.target.value)}
           placeholder="Enter value..."
         />
         <button onClick={handleSubmit}>Submit</button>
       </div>
     );
   }
   ```

2. **Export from index** in `src/components/index.ts`:

   ```typescript
   export { MyNewComponent } from './MyNewComponent';
   ```

3. **Use in App** in `src/App.tsx`:

   ```typescript
   import { MyNewComponent } from './components';

   function App() {
     const handleAction = (value: string) => {
       console.log('Action triggered:', value);
     };

     return (
       <div>
         {/* ... existing components */}
         <MyNewComponent title="My Feature" onAction={handleAction} />
       </div>
     );
   }
   ```

4. **Add tests** in `tests/MyNewComponent.test.tsx`:

   ```typescript
   import { render, screen, fireEvent } from '@testing-library/react';
   import { MyNewComponent } from '../src/components/MyNewComponent';

   describe('MyNewComponent', () => {
     it('should call onAction when submitted', () => {
       const mockAction = vi.fn();
       render(<MyNewComponent title="Test" onAction={mockAction} />);

       const input = screen.getByPlaceholderText('Enter value...');
       fireEvent.change(input, { target: { value: 'test value' } });

       const button = screen.getByText('Submit');
       fireEvent.click(button);

       expect(mockAction).toHaveBeenCalledWith('test value');
     });
   });
   ```

### Task 5: Debug Training Issues

**Goal**: Investigate why model isn't learning or loss isn't decreasing

**Debugging Checklist**:

1. **Verify data**:

   ```typescript
   console.log('Corpus length:', corpus.length);
   console.log('Vocabulary size:', this.vocabSize);
   console.log('Unique tokens:', new Set(corpus).size);
   ```

   - Ensure corpus is non-empty
   - Check vocabulary size >= MIN_VOCAB_SIZE (8)
   - Verify corpus has sufficient diversity

2. **Check learning rate**:

   ```typescript
   console.log('Learning rate:', lr);
   console.log('Effective LR:', effectiveLR);
   ```

   - Too high (>0.5): Loss may diverge or oscillate
   - Too low (<0.001): Learning too slow
   - Try lr=0.08 (default) as baseline

3. **Monitor gradients**:

   ```typescript
   const gradNorm = Math.sqrt(gradients.reduce((sum, g) => sum + g * g, 0));
   console.log('Gradient norm:', gradNorm);
   ```

   - Near 0: Dead ReLU, poor initialization
   - Very large (>100): Exploding gradients, reduce LR
   - Use gradient clipping if needed

4. **Inspect loss curve**:

   ```typescript
   console.log('Loss history:', this.trainingHistory.losses);
   ```

   - Increasing: LR too high or gradient issues
   - Flat: LR too low or saturated
   - Noisy: Normal for small batches
   - Should generally decrease over epochs

5. **Test generation**:

   ```typescript
   const sample = model.generate('test', 10, 1.0);
   console.log('Generated:', sample);
   ```

   - Repetitive: Model not learning diversity
   - Random: Model hasn't converged
   - Check if better with lower temperature

6. **Verify architecture**:

   ```typescript
   console.log('Hidden size:', this.hiddenSize);
   console.log('Context size:', this.contextSize);
   ```

   - Too small: Insufficient capacity
   - Too large: Overfitting on small corpus

### Task 6: Compress a Trained Model

**Goal**: Reduce model size using quantization, distillation, or low-rank approximation

**Steps**:

1. **Quantization** (fastest, 4x reduction):

   ```typescript
   import { compressWithQuantization } from '../compression/compress';

   // Compress model weights to int8
   const result = await compressWithQuantization(model);
   console.log(`Original: ${result.originalSize} bytes`);
   console.log(`Compressed: ${result.compressedSize} bytes`);
   console.log(`Ratio: ${result.compressionRatio.toFixed(2)}x`);

   // Export compressed model
   exportCompressedModel(result.compressedModel, 'model-int8.json');
   ```

2. **Knowledge Distillation** (best quality, slower):

   ```typescript
   import { compressWithDistillation } from '../compression/compress';

   // Train smaller student model
   const result = await compressWithDistillation(teacherModel, corpus, {
     temperature: 3.0,
     alpha: 0.7,
     studentHiddenSize: 32, // Much smaller than teacher
     epochs: 30,
     learningRate: 0.1,
     onProgress: (epoch, loss) => {
       console.log(`Distillation epoch ${epoch}: loss ${loss.toFixed(4)}`);
     }
   });

   // Student model is in result.compressedModel
   ```

3. **Low-Rank Approximation** (configurable):

   ```typescript
   import { compressWithLowRank } from '../compression/compress';

   // Compress with target ratio
   const result = await compressWithLowRank(model, {
     targetCompressionRatio: 2.0 // 2x compression
   });

   console.log(`Approximation error: ${result.approximationError.toFixed(4)}`);
   ```

4. **Compare compression methods**:

   ```typescript
   const methods = ['quantization', 'distillation', 'lowrank'];
   const results = [];

   for (const method of methods) {
     const result = await compress(model, method);
     results.push({
       method,
       ratio: result.compressionRatio,
       error: result.approximationError,
       time: result.metadata.compressionTime
     });
   }

   // Analyze trade-offs
   results.sort((a, b) => b.ratio - a.ratio);
   console.table(results);
   ```

5. **Test compressed model**:

   ```typescript
   // Generate text with compressed model
   const prompt = 'The quick brown';
   const original = originalModel.generate(prompt, 20, 0.8);
   const compressed = compressedModel.generate(prompt, 20, 0.8);

   console.log('Original:', original);
   console.log('Compressed:', compressed);

   // Compare perplexity
   const origPPL = originalModel.calculatePerplexity(testCorpus);
   const compPPL = compressedModel.calculatePerplexity(testCorpus);
   console.log(`Perplexity change: ${((compPPL - origPPL) / origPPL * 100).toFixed(2)}%`);
   ```

**When to Use Each Method**:

- **Quantization**: When you need fast compression and can tolerate ~2% accuracy loss
- **Distillation**: When you want best quality and have time for training
- **Low-rank**: When you want fine-grained control over compression ratio

---

## Important Files Reference

### Configuration Files

| File                      | Purpose               | When to Modify            |
| ------------------------- | --------------------- | ------------------------- |
| `package.json`            | Dependencies, scripts | Adding packages, scripts  |
| `tsconfig.json`           | TypeScript compiler   | Type checking rules       |
| `vite.config.ts`          | Build configuration   | Build settings, base path |
| `.eslintrc.cjs`           | Linting rules         | Code quality standards    |
| `.prettierrc`             | Formatting rules      | Code style preferences    |
| `src/config/constants.ts` | App defaults          | Default hyperparameters   |

### Core Implementation Files

| File                            | Lines | Purpose        | Modify For           |
| ------------------------------- | ----- | -------------- | -------------------- |
| `src/lib/ProNeuralLM.ts`        | ~900  | Base neural LM | Core model changes   |
| `src/lib/AdvancedNeuralLM.ts`   | ~750  | Enhanced LM    | Advanced features    |
| `src/lib/TransformerLM.ts`      | ~550  | Transformer    | Attention mechanisms |
| `src/lib/MathUtils.ts`          | ~600  | Math utilities | Numerical operations |
| `src/backend/webgpu.ts`         | ~400  | GPU backend    | GPU operations       |
| `src/backend/gpu_neural_ops.ts` | ~300  | Neural GPU ops | GPU integration      |

### UI Component Files

| File                                       | Lines | Purpose                        | Modify For                   |
| ------------------------------------------ | ----- | ------------------------------ | ---------------------------- |
| `src/App.tsx`                              | ~500  | Main application               | App structure, i18n          |
| `src/components/TrainingPanel.tsx`         | 1671  | Training UI                    | Training controls            |
| `src/components/DecisionEntry2Panel.tsx`   | 841   | Enhanced decision entry        | Governance decisions         |
| `src/components/BrainPanel.tsx`            | 731   | Brain vitals and mood          | Brain system UI              |
| `src/components/ModelMetrics.tsx`          | 712   | Metrics display                | Visualization                |
| `src/components/EmbeddingVisualizationPanel.tsx` | 660   | t-SNE/UMAP visualization      | Embedding plots              |
| `src/components/ChatInterface.tsx`         | 644   | Generation UI                  | Chat features                |
| `src/components/GovernanceBoard.tsx`       | 590   | Governance monitoring          | Governance UI                |
| `src/components/RunComparisonPanel.tsx`    | 558   | Compare training runs          | Experiment comparison        |
| `src/components/CompressionPanel.tsx`      | 513   | Compression UI                 | Model compression            |
| `src/components/ExportPanel.tsx`           | 505   | Model/data export              | Export functionality         |
| `src/components/ProjectManager.tsx`        | 485   | Project management             | Experiment tracking          |
| `src/components/ExplainabilityPanel.tsx`   | 412   | SHAP/gradients/attention       | Model interpretability       |
| `src/components/BrainTelemetryPanel.tsx`   | 373   | Brain telemetry                | Brain stats display          |
| `src/components/InformationTheoryPanel.tsx`| 365   | Information plane analysis     | Info theory visualization    |
| `src/components/ScenarioManager.tsx`       | 285   | Test scenario editor           | Scenario testing             |
| `src/components/DecisionLedgerEditor.tsx`  | 254   | Governance/decision tracking   | Decision ledger              |
| `src/components/TokenizerConfig.tsx`       | 196   | Tokenizer settings             | Tokenization config          |
| `src/components/CerebroPanel.tsx`          | 194   | Cerebro concept injection      | Concept injection system     |
| `src/components/ErrorBoundary.tsx`         | 107   | React error boundary           | Error handling               |
| `src/components/OnboardingCard.tsx`        | 106   | First-time user guide          | Onboarding UX                |
| `src/components/CerebroBubbleGraph.tsx`    | 86    | Cerebro bubble visualization   | Bubble graph rendering       |
| `src/components/OnboardingTooltip.tsx`     | 78    | Interactive onboarding tips    | Tooltips and hints           |
| `src/components/ModelSnapshot.tsx`         | 77    | Model metadata snapshot        | Snapshot display             |

### Compression Module Files

| File                           | Lines | Purpose              | Modify For           |
| ------------------------------ | ----- | -------------------- | -------------------- |
| `src/compression/compress.ts`  | ~324  | Compression API      | High-level interface |
| `src/compression/quantization.ts` | ~181  | Weight quantization  | Int8/16 compression  |
| `src/compression/distillation.ts` | ~272  | Knowledge distillation | Student training     |
| `src/compression/lowrank.ts`   | ~330  | SVD approximation    | Low-rank compression |

### Workflow Files

| File                                 | Purpose        | Triggers               |
| ------------------------------------ | -------------- | ---------------------- |
| `.github/workflows/ci.yml`           | CI pipeline    | Push, PR               |
| `.github/workflows/train-model.yml`  | Model training | Manual, corpus changes |
| `.github/workflows/deploy-pages.yml` | GitHub Pages   | Push to main           |

### Data Files

| File                            | Purpose            | Format     |
| ------------------------------- | ------------------ | ---------- |
| `data/corpus.txt`               | Training data      | Plain text |
| `models/neuro-lingua-v324.json` | Trained model      | JSON (3MB) |
| `docs/experiments/runs/*.json`  | Experiment results | JSON       |

---

## Integration Points

### 1. localStorage Persistence

**Purpose**: Save UI state, model metadata, projects across sessions

**Implementation**: `src/lib/storage.ts`

**Stored Data**:

- UI settings (theme, layout)
- Tokenizer configuration
- Project and run metadata
- Onboarding dismissal state

**Storage Keys** (from `src/config/constants.ts`):

```typescript
const STORAGE_KEYS = {
  UI_SETTINGS: 'neuro-lingua-ui-settings-v1',
  MODEL_META: 'neuro-lingua-model-meta-v1',
  TOKENIZER_CONFIG: 'neuro-lingua-tokenizer-config-v1',
  ONBOARDING_DISMISSED: 'neuro-lingua-onboarding-dismissed'
};
```

**Usage**:

```typescript
import { loadFromStorage, saveToStorage } from './lib/storage';

// Save
saveToStorage(STORAGE_KEYS.UI_SETTINGS, { theme: 'dark' });

// Load
const settings = loadFromStorage<UISettings>(STORAGE_KEYS.UI_SETTINGS);
```

**Important Notes**:

- Not encrypted - do not store sensitive data
- Subject to browser quotas (~5-10MB)
- Cleared when user clears browser data
- Synchronous API (blocks main thread)

### 2. WebGPU Acceleration

**Purpose**: Hardware-accelerated matrix operations for 2-5x faster training

**Architecture**:

```
ProNeuralLM
    ↓ (optional)
GPUNeuralOps
    ↓
WebGPUBackend
    ↓
GPU Hardware
```

**Initialization**:

```typescript
// In App.tsx or training code
const backend = new WebGPUBackend();
try {
  await backend.initialize();
  const gpuOps = new GPUNeuralOps(backend);
  model.setGPUOps(gpuOps);
  console.log('GPU acceleration enabled');
} catch (error) {
  console.warn('GPU not available, using CPU');
}
```

**Automatic Fallback**:

- GPU unavailable: Falls back to CPU
- Operation fails: Retries on CPU
- Transparent to model code

**Performance Monitoring**:

```typescript
const metrics = gpuOps.getMetrics();
console.log('GPU operations:', metrics.gpuOpsCount);
console.log('CPU fallbacks:', metrics.cpuFallbacks);
console.log('Average time:', metrics.averageTimeMs);
```

**Browser Support**:

- Chrome 113+ (enabled by default)
- Edge 113+
- Firefox (experimental, behind flag)
- Safari (not yet supported as of 2025-01)

### 3. Project Context (Σ-SIG Compliance)

**Purpose**: Centralized experiment tracking with governance

**Provider**: `src/contexts/ProjectContext.tsx`

**Structure**:

```typescript
Project
  ├── id: string
  ├── name: string
  ├── description: string
  ├── runs: Run[]
  └── scenarios: Scenario[]

Run
  ├── id: string
  ├── config: FrozenConfig (immutable)
  ├── results: TrainingResults
  └── timestamp: number

Scenario
  ├── id: string
  ├── prompt: string
  ├── expectedOutput?: string
  └── evaluations: RunEvaluation[]
```

**Usage**:

```typescript
import { useProjects } from '../contexts/ProjectContext';

function MyComponent() {
  const { projects, createProject, updateProject, deleteProject, addRun, addScenario } =
    useProjects();

  const handleCreateProject = () => {
    const project = createProject('My Experiment', 'Testing dropout');
    console.log('Created:', project.id);
  };
}
```

**Decision Ledger**:

- Records governance decisions
- Links decisions to experiments
- Ensures reproducibility

### 4. GitHub Actions Integration

**Purpose**: Automated model retraining on corpus changes or manual trigger

**Workflow**: `.github/workflows/train-model.yml`

**Triggers**:

**1. Manual Dispatch**:

```bash
gh workflow run train-model.yml \
  -f epochs=40 \
  -f hidden_size=128 \
  -f learning_rate=0.1 \
  -f optimizer=adam \
  -f dropout=0.15 \
  -f context_size=4
```

**2. Automatic on Push**:

- Changes to `data/corpus.txt`
- Changes to `scripts/train.ts`

**Process**:

1. Checkout repository
2. Install dependencies (pnpm)
3. Validate hyperparameters
4. Run `pnpm train`
5. Check for model changes
6. Commit updated `models/neuro-lingua-v324.json`
7. Upload artifacts (30-day retention)

**Environment Variables** (used by `scripts/train.ts`):

```bash
EPOCHS=30
HIDDEN_SIZE=64
LEARNING_RATE=0.08
OPTIMIZER=momentum
DROPOUT=0.1
CONTEXT_SIZE=3
TOKENIZER_MODE=unicode  # or ascii, custom
```

**Permissions Required**:

- Repository → Settings → Actions → Workflow permissions
- Enable: **Read and write permissions**

### 5. Tokenizer System

**Purpose**: Configurable text tokenization (Unicode, ASCII, custom regex)

**Configuration**: `src/config/constants.ts`

**Modes**:

**1. Unicode** (default):

```typescript
mode: 'unicode';
// Splits on: \p{L}\d\s'-
// Preserves: Letters, digits, whitespace, apostrophes, hyphens
// Use for: Multi-language, proper names, contractions
```

**2. ASCII**:

```typescript
mode: 'ascii';
// Splits on: [^a-zA-Z0-9\s'-]
// Preserves: Only ASCII letters and digits
// Use for: English-only, simple tokenization
```

**3. Custom**:

```typescript
mode: 'custom';
pattern: "[^\\p{L}\\d\\s'-]"; // Custom regex
// User-defined splitting logic
// Use for: Domain-specific tokenization
```

**Export/Import**:

```typescript
// Export tokenizer config
const config = model.getTokenizerConfig();
const json = JSON.stringify(config);
downloadFile('tokenizer.json', json);

// Import tokenizer config
const imported = JSON.parse(fileContent);
model.setTokenizerConfig(imported);
```

**Special Tokens**:

```typescript
const SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
```

- `<PAD>`: Padding for batching
- `<BOS>`: Beginning of sequence
- `<EOS>`: End of sequence
- `<UNK>`: Unknown token

---

## Project Management

### Σ-SIG Compliance

**Σ-SIG** (Sigma-SIG: Scientific Infrastructure for Governance) is a framework for reproducible experiment tracking.

**Core Principles**:

1. **Immutability**: Run configs are frozen after creation
2. **Traceability**: All decisions and changes are logged
3. **Reproducibility**: Complete experiment metadata preserved
4. **Governance**: Decision Ledger tracks rationale

**Implementation**: `src/contexts/ProjectContext.tsx`

### Project Lifecycle

**1. Create Project**:

```typescript
const project = createProject(
  'Dropout Experiment',
  'Investigating optimal dropout rates for small corpora'
);
```

**2. Add Run**:

```typescript
const run = addRun(project.id, {
  architecture: 'AdvancedNeuralLM',
  hyperparameters: {
    hiddenSize: 64,
    epochs: 30,
    learningRate: 0.1,
    dropout: 0.15
  },
  corpus: 'training data...',
  timestamp: Date.now()
});
// Config is now frozen - cannot be modified
```

**3. Record Results**:

```typescript
updateRun(project.id, run.id, {
  results: {
    finalLoss: 1.234,
    perplexity: 3.456,
    trainingTime: 45.2,
    history: losses
  }
});
```

**4. Add Test Scenarios**:

```typescript
const scenario = addScenario(project.id, {
  prompt: 'The quick brown',
  expectedBehavior: 'Should complete common phrase',
  evaluationCriteria: 'Coherence and fluency'
});

// Evaluate scenario across runs
evaluateScenario(project.id, scenario.id, run.id, {
  output: 'The quick brown fox jumps',
  score: 0.85,
  notes: 'Good completion, slightly repetitive'
});
```

**5. Decision Ledger**:

```typescript
addDecision(project.id, {
  title: 'Increase dropout to 0.2',
  rationale: 'Model overfitting on training set, validation loss increasing',
  alternatives: ['Reduce model size', 'Add more data', 'Early stopping'],
  decision: 'Increase dropout',
  implementedInRun: run.id
});
```

### Experiment Comparison

**Compare Runs**:

```typescript
const runs = project.runs.filter((r) => r.config.dropout > 0.1);
const comparison = runs.map((r) => ({
  id: r.id,
  dropout: r.config.dropout,
  finalLoss: r.results.finalLoss,
  perplexity: r.results.perplexity
}));

// Find best performing
const best = comparison.sort((a, b) => a.perplexity - b.perplexity)[0];
```

**Export for Analysis**:

```typescript
const exportData = {
  project: project.name,
  runs: project.runs.map((r) => ({
    config: r.config,
    results: r.results,
    timestamp: r.timestamp
  })),
  scenarios: project.scenarios
};

downloadJSON(`${project.name}-export.json`, exportData);
```

---

## Watch Out For

### Common Pitfalls

#### 1. Vocabulary Too Small

**Problem**: Training fails with "Vocabulary size too small" error

**Cause**: Corpus doesn't have enough unique tokens (min: 8)

**Solution**:

```typescript
// Check vocabulary size before training
const tokens = new Set(corpus.split(/\s+/));
if (tokens.size < MIN_VOCAB_SIZE) {
  alert(`Corpus has ${tokens.size} unique tokens. Minimum: ${MIN_VOCAB_SIZE}`);
  return;
}
```

#### 2. Learning Rate Too High

**Problem**: Loss oscillates or diverges (NaN)

**Cause**: LR > 0.5 causes unstable training

**Solution**:

```typescript
// Clamp learning rate
const safeLR = Math.min(Math.max(lr, 0.001), 0.5);

// Or use LR scheduling (AdvancedNeuralLM)
lrSchedule: 'cosine',  // Gradually reduces LR
```

#### 3. Memory Issues with Large Models

**Problem**: Browser crashes or becomes unresponsive

**Cause**: Model too large for browser memory (hidden size > 256)

**Solution**:

```typescript
// Enforce constraints
if (hiddenSize > HYPERPARAMETER_CONSTRAINTS.hiddenSize.max) {
  throw new Error(`Hidden size ${hiddenSize} exceeds maximum 256`);
}

// Or use incremental training
const batchSize = Math.floor(corpusLength / 10);
trainInBatches(corpus, batchSize);
```

#### 4. WebGPU Not Available

**Problem**: GPU acceleration doesn't work

**Cause**: Browser doesn't support WebGPU

**Solution**:

```typescript
// Always check availability
if (!navigator.gpu) {
  console.warn('WebGPU not supported in this browser');
  // Fall back to CPU automatically
}

// Don't rely on GPU-only features
if (this.gpuOps) {
  result = await this.gpuOps.matmul(a, b);
} else {
  result = this.cpuMatmul(a, b);
}
```

#### 5. localStorage Quota Exceeded

**Problem**: Can't save large models to localStorage

**Cause**: Browser quota ~5-10MB, models can exceed this

**Solution**:

```typescript
try {
  saveToStorage(STORAGE_KEYS.MODEL_META, modelData);
} catch (error) {
  if (error.name === 'QuotaExceededError') {
    alert('Storage quota exceeded. Export model to file instead.');
    // Offer download instead
    downloadModelJSON(modelData);
  }
}
```

#### 6. Base Path Issues in Deployment

**Problem**: Routes don't work on GitHub Pages

**Cause**: Base path `/neuro-lingua/` not configured

**Solution**:

```typescript
// vite.config.ts already has:
base: '/neuro-lingua/',

// Ensure all routes use base path
const imagePath = `${import.meta.env.BASE_URL}assets/logo.png`;

// Not:
const imagePath = '/assets/logo.png';  // Wrong on GitHub Pages
```

#### 7. Training Freezes UI

**Problem**: Browser becomes unresponsive during training

**Cause**: Long synchronous training loop blocks main thread

**Solution**:

```typescript
// Use setTimeout to yield to browser
private async trainWithYield(corpus: string, epochs: number) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    this.trainEpoch(corpus);

    // Yield every epoch
    await new Promise(resolve => setTimeout(resolve, 0));

    // Update UI
    updateProgressBar(epoch / epochs);
  }
}
```

#### 8. Type Errors with Model State

**Problem**: TypeScript errors when updating model config

**Cause**: Type mismatch between UI state and model config

**Solution**:

```typescript
// Use proper typing for state
const [config, setConfig] = useState<ModelConfig>({
  hiddenSize: 64,
  learningRate: 0.08
  // ... all required fields
});

// Don't use partial updates without spreading
setConfig({ hiddenSize: 128 }); // Wrong: loses other fields

setConfig((prev) => ({ ...prev, hiddenSize: 128 })); // Correct
```

### Security Considerations

#### Do NOT Use With Sensitive Data

- App stores data in unencrypted localStorage
- No server-side security
- Browser extensions can access localStorage
- Data may sync across devices

#### Safe Data Types

- Public text (books, articles)
- Generated/synthetic text
- Educational examples
- Non-confidential research data

#### Unsafe Data Types

- Personally Identifiable Information (PII)
- Medical records
- Financial information
- Authentication credentials
- Business secrets
- Private communications

### Performance Tips

#### Optimize Training Speed

```typescript
// 1. Use WebGPU when available (2-5x faster)
await enableGPU(model);

// 2. Reduce context size for faster forward passes
contextSize: 3  // Instead of 6

// 3. Use momentum instead of Adam for speed
optimizer: 'momentum'  // Faster than 'adam'

// 4. Train with smaller epochs, iterate
epochs: 10  // Quick iterations to find good hyperparameters

// 5. Disable expensive features during experimentation
dropout: 0,
useLayerNorm: false,
lrSchedule: 'constant'
```

#### Optimize Generation Speed

```typescript
// 1. Use greedy decoding (fastest)
samplingMode: 'off';

// 2. Limit max tokens
maxTokens: 25;

// 3. Disable beam search
useBeamSearch: false;

// 4. Use higher temperature for less computation
temperature: 1.5; // More randomness, less argmax computation
```

---

## Deployment & CI/CD

### GitHub Pages Deployment

**Workflow**: `.github/workflows/deploy-pages.yml`

**Triggered By**:

- Push to `main` or `master` branch
- Manual workflow dispatch

**Process**:

1. Install dependencies
2. Run tests (`pnpm test`)
3. Build production bundle (`pnpm build`)
4. Upload build artifact to GitHub Pages
5. Deploy to `https://abbrubin150-ui.github.io/neuro-lingua/`

**Configuration**:

```typescript
// vite.config.ts
base: '/neuro-lingua/',  // Must match repo name
```

**Setup Requirements**:

1. Repository → Settings → Pages
2. Source: **GitHub Actions**
3. Workflow: **deploy-pages.yml**

### Continuous Integration

**Workflow**: `.github/workflows/ci.yml`

**Runs On**:

- Every push to any branch
- Every pull request

**Checks**:

1. **Type checking**: `tsc --noEmit`
2. **Linting**: `pnpm lint`
3. **Formatting**: `pnpm format:check`
4. **Tests**: `pnpm test`
5. **Build**: `pnpm build`

**All must pass** for PR to be merged

**Local Check** (run before pushing):

```bash
# Run all CI checks locally
tsc --noEmit && \
pnpm lint && \
pnpm format:check && \
pnpm test && \
pnpm build

# Or create a script in package.json
"ci:local": "tsc --noEmit && pnpm lint && pnpm format:check && pnpm test && pnpm build"
```

### Model Training Automation

**Workflow**: `.github/workflows/train-model.yml`

**Purpose**: Automatically retrain model when corpus changes or manually triggered

**Manual Trigger**:

```bash
# Using GitHub CLI
gh workflow run train-model.yml \
  -f epochs=40 \
  -f hidden_size=128 \
  -f learning_rate=0.1 \
  -f optimizer=adam \
  -f dropout=0.15 \
  -f context_size=4

# Or via GitHub web UI
# Actions → Train Neuro-Lingua Model → Run workflow
```

**Automatic Trigger**:

- Edit `data/corpus.txt`
- Commit and push
- Workflow runs automatically
- Commits updated `models/neuro-lingua-v324.json` if model changes

**Artifacts**:

- Training results uploaded for 30 days
- Download via: Actions → Workflow run → Artifacts

**Permissions**:

- Workflow needs `contents: write` to commit model
- Enable in: Settings → Actions → Workflow permissions

### Release Process

**Versioning**: Semantic versioning (vX.Y.Z)

- **Major (X)**: Breaking changes to model architecture or API
- **Minor (Y)**: New features (new optimizer, generation method, new UI panels)
- **Patch (Z)**: Bug fixes, performance improvements

**Current Version**: v3.2.4 (runtime), v0.0.0 (package.json - not published to npm)

**Creating a Release**:

1. **Update version** in `package.json`:

   ```json
   "version": "3.3.0"
   ```

2. **Update README** with new features

3. **Update CHANGELOG** (if exists):

   ```markdown
   ## [3.3.0] - 2025-11-18

   ### Added

   - New typical sampling generation method
   - GPU performance dashboard

   ### Fixed

   - Learning rate scheduling bug in TransformerLM
   ```

4. **Commit changes**:

   ```bash
   git add package.json README.md CHANGELOG.md
   git commit -m "chore: bump version to 3.3.0"
   ```

5. **Tag release**:

   ```bash
   git tag -a v3.3.0 -m "Release v3.3.0: Typical sampling and GPU dashboard"
   git push origin main --tags
   ```

6. **Create GitHub release**:
   ```bash
   gh release create v3.3.0 \
     --title "v3.3.0 - Typical Sampling & GPU Dashboard" \
     --notes "See CHANGELOG.md for details" \
     models/neuro-lingua-v324.json
   ```

---

## Additional Resources

### Documentation Files

#### Core Documentation
- **README.md**: Project overview and quickstart
- **CLAUDE.md**: This file - comprehensive AI assistant guide
- **CHANGELOG_v3.3.md**: Version history and changes

#### Architecture & Implementation
- **MATHEMATICAL_ENHANCEMENTS.md**: Detailed math formulations
- **NEURO_LINGUA_V4_UPGRADES.md**: v4.0 roadmap with RoPE, SwiGLU, RMSNorm, Mirostat v2
- **TRANSFORMER_GUIDE.md**: Transformer architecture explanation
- **TRANSFORMER_IMPLEMENTATION.md**: Implementation details
- **GPU_ACCELERATION_GUIDE.md**: WebGPU setup and usage

#### Governance & Systems
- **GOVERNANCE_INTEGRATION_GUIDE.md**: Autonomous governance system integration
- **GOVERNANCE_ARCHITECTURE.md**: GovernanceEngine architecture and design
- **INTEGRATION_COMPLETE.md**: System integration status and checklist
- **SYSTEM_CALIBRATION_AGENT_SPEC.md**: Agent calibration and instruction specification

#### Development
- **DEVELOPMENT_SETUP_GUIDE.md**: Development environment setup
- **DEVELOPMENT_ROADMAP.md**: Future plans and features
- **IMMEDIATE_ACTIONS.md**: Priority tasks and action items

#### Theory & Research
- **docs/generalization-theory.md**: Generalization theory documentation
- **docs/on-the-edge-formalism.md**: Edge learning formalism
- **docs/theory/information.md**: Information theory foundations
- **docs/losses.md**: Loss functions documentation
- **docs/experiments/logbook.md**: Experiment logbook

### External References

**Neural Networks**:

- [Deep Learning Book](https://www.deeplearningbook.org/) - Comprehensive theory
- [Neural Network from Scratch](http://neuralnetworksanddeeplearning.com/) - Beginner-friendly
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper

**WebGPU**:

- [WebGPU Spec](https://www.w3.org/TR/webgpu/) - Official specification
- [WebGPU Fundamentals](https://webgpufundamentals.org/) - Tutorials
- [Chrome WebGPU Samples](https://austin-eng.com/webgpu-samples/) - Code examples

**Testing**:

- [Vitest Docs](https://vitest.dev/) - Test framework
- [Testing Library](https://testing-library.com/) - React testing

**TypeScript**:

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)

---

## Contribution Guidelines

### For AI Assistants

**When working on this codebase**:

1. **Always read existing code first** before making changes
2. **Follow established patterns** (don't reinvent the wheel)
3. **Test your changes** (`pnpm test` must pass)
4. **Run linter** (`pnpm lint` must pass)
5. **Format code** (`pnpm format` before committing)
6. **Write tests** for new features
7. **Update documentation** if changing public APIs
8. **Use TypeScript strictly** (no `any` types without good reason)
9. **Consider performance** (browser constraints, memory usage)
10. **Graceful degradation** (CPU fallback, error handling)

### Code Review Checklist

Before submitting changes:

- [ ] Type checking passes: `tsc --noEmit`
- [ ] Linting passes: `pnpm lint`
- [ ] Tests pass: `pnpm test`
- [ ] Code formatted: `pnpm format`
- [ ] Build succeeds: `pnpm build`
- [ ] Manual testing in browser
- [ ] No console errors or warnings
- [ ] Responsive UI (if applicable)
- [ ] WebGPU fallback works (test with GPU disabled)
- [ ] Documentation updated (if API changed)

---

**End of CLAUDE.md**

_This document is maintained for AI assistants working on Neuro-Lingua DOMESTICA. Keep it updated as the codebase evolves._
