# CLAUDE.md - AI Assistant Guide for Neuro-Lingua

> **Last Updated**: 2025-11-18
> **Version**: 3.2.4
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
- **WebGPU Acceleration**: Automatic GPU detection with graceful CPU fallback
- **4 Optimizers**: SGD with momentum, Adam, Damped Newton, L-BFGS
- **5+ Generation Methods**: Greedy, Top-k, Top-p (nucleus), Beam Search, Contrastive Search
- **Advanced Features**: Dropout, layer normalization, learning rate scheduling, weight decay
- **Σ-SIG Compliance**: Experiment tracking with Decision Ledger governance
- **Session Persistence**: localStorage-based state management
- **GitHub Actions**: Automated model retraining workflow

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
│   │   └── gpu_neural_ops.ts        # High-level neural operations on GPU
│   ├── components/                   # React UI components
│   │   ├── TrainingPanel.tsx         # Main training configuration panel
│   │   ├── ModelMetrics.tsx          # Performance metrics dashboard
│   │   ├── ProjectManager.tsx        # Project/run management (Σ-SIG)
│   │   ├── ScenarioManager.tsx       # Test scenario editor
│   │   ├── DecisionLedgerEditor.tsx  # Governance/decision tracking
│   │   ├── TokenizerConfig.tsx       # Tokenizer settings
│   │   ├── ChatInterface.tsx         # Chat-style generation UI
│   │   ├── OnboardingCard.tsx        # First-time user guide
│   │   └── ErrorBoundary.tsx         # React error boundary
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
│   │   ├── storage.ts                # localStorage abstraction
│   │   ├── utils.ts                  # Tokenizer and CSV utilities
│   │   └── traceExport.ts            # Σ-SIG compliant experiment tracing
│   ├── losses/                       # Advanced loss functions
│   ├── math/                         # Mathematical utilities
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
│   ├── AdvancedNeuralLM.test.ts      # Advanced features tests
│   ├── TransformerLM.test.ts         # Transformer tests
│   ├── MathUtils.test.ts             # Math utilities tests
│   ├── tokenizer.test.ts             # Tokenizer tests
│   ├── sampler.test.ts               # Generation tests
│   ├── App.test.tsx                  # Main app component tests
│   ├── numerics/                     # Numerical correctness tests
│   ├── math/                         # Mathematical analysis tests
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
- 4 optimizers: Momentum SGD, Adam, Damped Newton, L-BFGS
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

### 4. Context Providers (`src/contexts/`)

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

### 5. Configuration (`src/config/`)

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
├── AdvancedNeuralLM.test.ts     # Advanced features
├── TransformerLM.test.ts        # Transformer architecture
├── MathUtils.test.ts            # Math utilities
├── tokenizer.test.ts            # Tokenization
├── sampler.test.ts              # Generation algorithms
├── App.test.tsx                 # React component
├── numerics/                    # Numerical correctness
│   ├── sampling.test.ts
│   └── bayesian.test.ts
└── math/                        # Mathematical analysis
    └── analysis.test.ts
```

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

| File                                | Lines | Purpose            | Modify For          |
| ----------------------------------- | ----- | ------------------ | ------------------- |
| `src/App.tsx`                       | ~500  | Main application   | App structure       |
| `src/components/TrainingPanel.tsx`  | ~1400 | Training UI        | Training controls   |
| `src/components/ModelMetrics.tsx`   | ~450  | Metrics display    | Visualization       |
| `src/components/ProjectManager.tsx` | ~550  | Project management | Experiment tracking |
| `src/components/ChatInterface.tsx`  | ~200  | Generation UI      | Chat features       |

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
- **Minor (Y)**: New features (new optimizer, generation method)
- **Patch (Z)**: Bug fixes, performance improvements

**Current Version**: v3.2.4

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

- **README.md**: Project overview and quickstart
- **MATHEMATICAL_ENHANCEMENTS.md**: Detailed math formulations
- **TRANSFORMER_GUIDE.md**: Transformer architecture explanation
- **TRANSFORMER_IMPLEMENTATION.md**: Implementation details
- **GPU_ACCELERATION_GUIDE.md**: WebGPU setup and usage
- **DEVELOPMENT_SETUP_GUIDE.md**: Development environment setup
- **DEVELOPMENT_ROADMAP.md**: Future plans and features

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
