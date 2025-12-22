# Neuro-Lingua DOMESTICA v3.2.4 — Development Setup Guide

## Overview

This guide provides comprehensive instructions for developing, testing, deploying, and extending the Neuro-Lingua DOMESTICA project. It covers environment setup, running the application, executing tests, CI/CD workflows, and a detailed roadmap for future development.

**Current Version:** 3.2.4
**Last Updated:** November 2024

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Development Environment](#development-environment)
4. [Running Tests & Linting](#running-tests--linting)
5. [Building & Deployment](#building--deployment)
6. [Training Models](#training-models)
7. [GitHub Actions Workflows](#github-actions-workflows)
8. [Project Architecture](#project-architecture)
9. [Immediate Development Tasks](#immediate-development-tasks)
10. [Advanced Integration Features](#advanced-integration-features)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Get the project running in 5 minutes

```bash
# Clone the repository
git clone https://github.com/abbrubin150-ui/neuro-lingua.git
cd neuro-lingua

# Install dependencies
pnpm install

# Start development server
pnpm dev

# In another terminal, run tests
pnpm test

# Build for production
pnpm build
```

The application will be available at `http://localhost:5173/` (or another port if 5173 is busy).

---

## Prerequisites & Installation

### System Requirements

- **Node.js:** Version 18 or higher (20+ recommended)
- **pnpm:** Version 8 or higher (package manager)
- **Python:** Version 3.9+ (optional, for data pipeline scripts)
- **Browser:** Chrome/Edge 113+ or Firefox for WebGPU support (optional GPU acceleration)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/abbrubin150-ui/neuro-lingua.git
cd neuro-lingua

# 2. Install Node dependencies
pnpm install

# 3. (Optional) Install Python dependencies for data processing
pip install datasets requests numpy scipy scikit-learn

# 4. Verify installation
pnpm test     # Should show 144+ tests passing
pnpm build    # Should complete without errors
```

### Verifying Setup

```bash
# Check Node version
node --version  # Should be v18.0.0 or higher

# Check pnpm version
pnpm --version  # Should be 8.0.0 or higher

# Run self-tests
pnpm test
# Expected output: Test Files 10 passed (10), Tests 144 passed (144)

# Verify linting
pnpm lint
# Expected output: No errors or warnings

# Check code formatting
pnpm format:check
# Expected output: All matched files use Prettier code style!
```

---

## Development Environment

### Starting the Dev Server

```bash
pnpm dev
```

This command:

- Starts Vite development server on `http://localhost:5173/`
- Enables hot module reloading (HMR)
- Opens browser automatically (on supported platforms)
- Monitors source files for changes

### IDE Recommendations

**VS Code Setup:**

```json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

**Recommended Extensions:**

- ESLint (dbaeumer.vscode-eslint)
- Prettier (esbenp.prettier-vscode)
- TypeScript Vue Plugin (Vue.vscode-typescript-vue-plugin)

### Environment Variables

Create a `.env.local` file (not committed to repo) for local configuration:

```bash
# Example .env.local
VITE_DEBUG=false
```

---

## Running Tests & Linting

### Test Suite

```bash
# Run all tests once
pnpm test

# Run tests in watch mode (reruns on file changes)
pnpm test:watch

# Run specific test file
pnpm test ProNeuralLM

# Run with coverage
pnpm test -- --coverage
```

**Test Structure:**

- `tests/ProNeuralLM.test.ts` — Core language model tests
- `tests/AdvancedNeuralLM.test.ts` — Advanced features tests
- `tests/TransformerLM.test.ts` — Transformer architecture tests
- `tests/tokenizer.test.ts` — Tokenization logic tests
- `tests/sampler.test.ts` — Text generation sampling tests
- `tests/MathUtils.test.ts` — Math utility functions tests
- `tests/App.test.tsx` — React component integration tests
- `tests/numerics/` — Numerical analysis tests
- `tests/math/` — Mathematical operation tests

### Code Quality Tools

```bash
# Run ESLint
pnpm lint

# Fix ESLint issues automatically
pnpm lint -- --fix

# Check code formatting
pnpm format:check

# Auto-format all files
pnpm format

# Type check without building
pnpm exec tsc --noEmit
```

### Running All Checks (as in CI)

```bash
# This replicates the CI pipeline locally
pnpm exec tsc --noEmit && \
pnpm lint && \
pnpm format:check && \
pnpm test && \
pnpm build
```

---

## Building & Deployment

### Production Build

```bash
# Full build with type checking
pnpm build

# Preview production build locally
pnpm preview

# Build with source maps for debugging
pnpm build -- --sourcemap
```

**Output:**

- Built files: `dist/`
- HTML: `dist/index.html`
- JavaScript: `dist/assets/*.js`
- CSS: `dist/assets/*.css`
- All gzipped and optimized for production

### GitHub Pages Deployment

The project auto-deploys to GitHub Pages on every push to `main`:

1. CI workflow (`.github/workflows/ci.yml`) runs all tests
2. If tests pass, `deploy-pages.yml` builds the project
3. Built artifacts deployed to `https://abbrubin150-ui.github.io/neuro-lingua/`

**Manual deployment:**

```bash
# Build and test locally
pnpm build

# Deploy to GitHub Pages (if you have permissions)
# Usually handled automatically by GitHub Actions
```

### Deployment Checklist

- [ ] All tests passing locally (`pnpm test`)
- [ ] No lint errors (`pnpm lint`)
- [ ] Code formatted correctly (`pnpm format:check`)
- [ ] Build completes successfully (`pnpm build`)
- [ ] Model artifact created/updated
- [ ] Commit message is clear and descriptive

---

## Training Models

### Browser-Based Training

1. Open application in web browser
2. Enter training text in the "Training" section (200+ words recommended)
3. Configure hyperparameters:
   - **Hidden Size:** 32-128 (affects model capacity)
   - **Epochs:** 10-50 (training iterations)
   - **Learning Rate:** 0.01-0.2 (step size)
   - **Optimizer:** Momentum (0.9) or Adam
   - **Dropout:** 0.05-0.2 (regularization)
   - **Context Size:** 2-5 (input window)

4. Click "Train" button or press Ctrl/Cmd+Enter
5. Monitor loss and accuracy in the metrics panel
6. Save model with Ctrl/Cmd+S or click "Save" button

### Headless Node.js Training

```bash
# Basic training
pnpm train

# Training with custom hyperparameters
EPOCHS=50 OPTIMIZER=adam LEARNING_RATE=0.1 pnpm train

# With custom corpus
CORPUS_PATH=/path/to/custom_corpus.txt pnpm train

# With custom output path
MODEL_EXPORT_PATH=/path/to/output.json pnpm train

# Supported environment variables
EPOCHS=30                           # Number of training epochs
HIDDEN_SIZE=64                      # Hidden layer dimension
LEARNING_RATE=0.08                  # Learning rate
OPTIMIZER=momentum|adam             # Optimization algorithm
MOMENTUM=0.9                        # Momentum coefficient
DROPOUT=0.1                         # Dropout rate
CONTEXT_SIZE=3                      # Context window size
SEED=1337                           # Random seed for reproducibility
TOKENIZER_MODE=unicode|ascii|custom # Tokenization mode
TOKENIZER_PATTERN="[^...]"          # Custom tokenizer regex (if custom mode)
CORPUS_PATH=/path/to/corpus.txt     # Custom corpus file
MODEL_EXPORT_PATH=/path/to/model    # Custom output path
EXPERIMENT_NAME=my_experiment       # Tag for experiment tracking
```

### Data Preparation

Corpora should be:

- **Plain text** (`.txt` files)
- **200+ words minimum** for meaningful training
- **1000+ words recommended** for good quality models
- **Diverse vocabulary** (varied word types and structures)
- **Representative** of target domain

**Example corpus structure:**

```
Paragraph 1: Background and motivation
Paragraph 2: Technical details
Paragraph 3: Implementation approach
...
```

---

## GitHub Actions Workflows

### CI/CD Pipeline (`.github/workflows/ci.yml`)

**Triggers:** Every push to main and all pull requests

**Steps:**

1. Type check with TypeScript
2. ESLint validation
3. Prettier formatting check
4. 144+ unit tests
5. Production build

**Status badge:**

```markdown
![CI](https://github.com/abbrubin150-ui/neuro-lingua/actions/workflows/ci.yml/badge.svg)
```

### Automated Model Training (`.github/workflows/train-model.yml`)

**Triggers:**

1. Manual workflow dispatch (UI button in GitHub)
2. Automatic push to `data/corpus.txt`

**Manual Trigger Inputs:**

- `epochs` — Number of training epochs (1-200, default: 30)
- `hidden_size` — Hidden layer size (16-256, default: 64)
- `learning_rate` — Learning rate (0.001-1.0, default: 0.08)
- `optimizer` — Choice: `momentum` or `adam` (default: momentum)
- `dropout` — Dropout rate (0-0.5, default: 0.1)
- `context_size` — Context window (2-6, default: 3)

**Usage from GitHub CLI:**

```bash
# Manual trigger with custom hyperparameters
gh workflow run train-model.yml \
  -f epochs=50 \
  -f optimizer=adam \
  -f learning_rate=0.1 \
  -f dropout=0.15

# Check workflow status
gh run list --workflow=train-model.yml

# View workflow run details
gh run view <RUN_ID>
```

**Workflow outputs:**

- Updated model artifact in `models/neuro-lingua-v324.json`
- Git commit with hyperparameter details
- Training artifacts uploaded (30-day retention)

### Deployment Pipeline (`.github/workflows/deploy-pages.yml`)

**Triggers:** Successful CI on main branch

**Steps:**

1. Build production bundle
2. Upload to GitHub Pages
3. Deploy to public URL

---

## Project Architecture

### Directory Structure

```
neuro-lingua/
├── src/                          # Application source code
│   ├── App.tsx                   # Main React component
│   ├── main.tsx                  # React entry point
│   ├── components/               # React UI components
│   │   ├── TrainingPanel.tsx     # Training controls
│   │   ├── ModelMetrics.tsx      # Statistics & charts
│   │   ├── ChatInterface.tsx     # Chat UI
│   │   ├── OnboardingCard.tsx    # Welcome screen
│   │   └── TokenizerConfig.tsx   # Tokenizer settings
│   │
│   ├── lib/                      # Core ML implementations
│   │   ├── ProNeuralLM.ts        # Feedforward language model
│   │   ├── AdvancedNeuralLM.ts   # Advanced features
│   │   ├── TransformerLM.ts      # Transformer architecture
│   │   ├── MathUtils.ts          # Math operations
│   │   ├── utils.ts              # Utility functions
│   │   └── storage.ts            # localStorage management
│   │
│   ├── backend/                  # GPU acceleration
│   │   ├── gpu_neural_ops.ts     # WebGPU kernels
│   │   └── webgpu.ts             # WebGPU utilities
│   │
│   ├── models/                   # Neural network architectures
│   │   ├── mini_transformer.ts   # Transformer implementation
│   │   ├── attention.ts          # Attention mechanisms
│   │   └── regularizers.ts       # L2, dropout, etc.
│   │
│   ├── config/                   # Configuration files
│   │   └── constants.ts          # Constants & defaults
│   │
│   └── types/                    # TypeScript definitions
│
├── tests/                        # Test suite (144+ tests)
│   ├── **/*.test.ts              # Unit & integration tests
│   └── setup.ts                  # Test configuration
│
├── scripts/                      # Build & training scripts
│   ├── train.ts                  # Headless training script
│   └── data/                     # Data processing scripts
│
├── data/                         # Training data
│   ├── corpus.txt                # Example training corpus
│   └── processed/                # Processed datasets
│
├── models/                       # Model artifacts
│   └── neuro-lingua-v324.json    # Latest trained model
│
├── .github/workflows/            # GitHub Actions
│   ├── ci.yml                    # CI pipeline
│   ├── train-model.yml           # Training automation
│   └── deploy-pages.yml          # GitHub Pages deployment
│
├── .eslintrc.cjs                 # ESLint configuration
├── .prettierrc                   # Prettier configuration
├── tsconfig.json                 # TypeScript configuration
├── vite.config.ts                # Vite build configuration
├── package.json                  # Dependencies & scripts
└── README.md                     # Project documentation
```

### Architecture Layers

**Presentation Layer (`src/components/`):**

- React components for UI
- TrainingPanel for hyperparameter controls
- ModelMetrics for visualization
- ChatInterface for text generation

**Core ML Layer (`src/lib/` + `src/models/`):**

- ProNeuralLM: Feedforward neural network
- AdvancedNeuralLM: Extended features (activations, schedules)
- TransformerLM: Transformer architecture with attention
- Shared utilities: Math, tokenization, sampling

**GPU Acceleration (`src/backend/`):**

- WebGPU implementation for matrix operations
- Optional GPU tensor operations
- Automatic CPU fallback

**Data & Storage (`src/lib/storage.ts`):**

- localStorage persistence
- Model import/export
- Tokenizer configuration

**Testing (`tests/`):**

- 144+ tests covering all major components
- Unit tests for individual functions
- Integration tests for complex features

---

## Immediate Development Tasks

### Task 1: UI Language Consistency ✓ COMPLETED

- Changed toggle button from "עברית" to "Hebrew"
- Maintained full bilingual support (English + Hebrew with RTL)
- All UI labels consistent

### Task 2: Improved Persistence & Help ✓ COMPLETED

- All hyperparameters persist in localStorage
- Model metadata (timestamp, vocab size) displayed
- Enhanced info cards with pause/resume documentation
- Privacy warnings in onboarding

### Task 3: GitHub Actions Automation ✓ COMPLETED

- Enhanced train-model.yml with:
  - Expanded hyperparameter inputs
  - pnpm support
  - Better error handling
  - Artifact uploads
  - Conditional commits
- Improved ci.yml with:
  - TypeScript type checking
  - pnpm caching
  - Build verification

### Task 4: Developer Experience ✓ COMPLETED

- ESLint configured and passing
- Prettier formatting enforced
- 144+ tests (10 test files)
- Expanded corpus (2800+ words)
- CI pipeline includes all quality checks

---

## Advanced Integration Features

### Future: WebGPU Acceleration

**Status:** Backend code exists, needs UI integration

**To implement:**

1. Wire GPU toggle in TrainingPanel to actual training
2. Add GPU metrics visualization
3. Benchmark CPU vs GPU performance
4. Implement automatic fallback for unsupported browsers

**Files to modify:**

- `src/components/TrainingPanel.tsx` — Add GPU toggle logic
- `src/App.tsx` — Use GPU acceleration in training loop
- `src/components/ModelMetrics.tsx` — Display GPU metrics

**Expected speedup:** 2-5x training acceleration on supported hardware

### Future: Enhanced Transformer Integration

**Status:** TransformerLM implemented, needs better UI support

**To implement:**

1. Add architecture selection UI
2. Transformer-specific hyperparameters (num_heads, num_layers)
3. Attention visualization
4. Performance comparison charts

**Files to modify:**

- `src/components/TrainingPanel.tsx` — Architecture selector
- `src/components/ModelMetrics.tsx` — Architecture-specific charts

### Future: Edge Learning Diagnostics

**Status:** Python script exists, not connected to browser

**To implement:**

1. Wrap Python script in Node.js subprocess
2. Call after training completion
3. Display Fisher Information metrics
4. Show efficiency bounds

**Files to create/modify:**

- `src/backend/edgeLearning.ts` — Node.js wrapper
- `scripts/edge_learning_wrapper.ts` — Training integration

---

## Troubleshooting

### Common Issues

**Q: Build fails with "Cannot find type definition file for '@webgpu/types'"**

```bash
# Solution: Reinstall dependencies
pnpm install
pnpm build
```

**Q: Tests timeout or fail intermittently**

```bash
# Solution: Clear cache and retry
pnpm test -- --clearCache
```

**Q: Dev server won't start**

```bash
# Solution: Check if port 5173 is in use
lsof -i :5173  # macOS/Linux
netstat -ano | findstr :5173  # Windows

# Use different port
pnpm dev -- --port 3000
```

**Q: Training is slow**

```bash
# Solution: Try reducing hyperparameters
# - Decrease epochs (10-20 instead of 30-50)
# - Reduce hidden_size (32-64 instead of 128)
# - Reduce context_size (2-3 instead of 5-6)

# Or try Adam optimizer (often faster)
OPTIMIZER=adam pnpm train
```

**Q: Model doesn't converge (loss stays high)**

```bash
# Solution: Adjust learning rate
# Try lower rate: 0.01-0.05
# Try higher rate: 0.1-0.2
# Or enable learning rate schedule (in AdvancedNeuralLM)
```

**Q: Corpus validation error**

```bash
# Solution: Ensure corpus.txt meets minimum requirements
# Minimum: 8 unique characters/tokens
# Recommended: 200+ words, 500+ characters
# Check for encoding: should be UTF-8
```

### Debug Mode

**Enable verbose logging:**

```bash
# In browser console
localStorage.setItem('DEBUG', 'true');
location.reload();

# Then check browser console for detailed logs
```

**Inspect model state:**

```javascript
// In browser console
console.log(window.__MODEL__); // Access model if exported
```

### Performance Profiling

**CPU profiling (Chrome DevTools):**

1. Open DevTools (F12)
2. Go to Performance tab
3. Click Record
4. Perform training
5. Click Stop
6. Analyze flame chart

**Memory profiling:**

1. Open DevTools → Memory tab
2. Take heap snapshot before training
3. Train model
4. Take heap snapshot after
5. Compare snapshots

---

## Contributing Guidelines

### Before Starting Work

1. Check IMMEDIATE_ACTIONS.md for current priorities
2. Create a new branch: `git checkout -b feature/description`
3. Link issues in commit messages: `fixes #123`
4. Ensure branch name starts with `claude/` for Claude Code

### Commit Message Format

```
<type>: <description>

<optional body explaining why and how>

Fixes #<issue-number> (if applicable)
```

**Types:**

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style (Prettier, ESLint)
- `refactor:` Code restructuring
- `test:` Test additions/changes
- `chore:` Dependencies, configs

**Example:**

```
feat: add WebGPU acceleration toggle to training panel

Implements GPU tensor operations for matrix multiplication,
improving training speed on supported browsers. Includes
fallback for unsupported hardware.

Fixes #42
```

### Pull Request Checklist

- [ ] Branch created from `main`
- [ ] Tests passing locally
- [ ] Linting passes
- [ ] Code formatted with Prettier
- [ ] New tests added (if applicable)
- [ ] Documentation updated
- [ ] Commit messages clear and descriptive
- [ ] No dependencies added without justification

---

## Resources

### Documentation

- [README.md](./README.md) — Project overview
- [TRANSFORMER_IMPLEMENTATION.md](./TRANSFORMER_IMPLEMENTATION.md) — Transformer details
- [IMMEDIATE_ACTIONS.md](./IMMEDIATE_ACTIONS.md) — Priority tasks

### External References

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)

### Related Research

- [Transformers: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [On-Device ML Best Practices](https://github.com/abbrubin150-ui/neuro-lingua/docs/theory/)

---

## Support & Feedback

Found a bug? Have a feature request?

- [Open an issue](https://github.com/abbrubin150-ui/neuro-lingua/issues)
- Include reproduction steps
- Specify browser/OS/Node version
- Attach console logs if applicable

---

**Last Updated:** November 2024
**Maintainer:** [@abbrubin150-ui](https://github.com/abbrubin150-ui)
