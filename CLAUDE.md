# CLAUDE.md - AI Assistant Guide for Neuro-Lingua

This document provides essential information for AI assistants working with the Neuro-Lingua codebase.

## Project Overview

**Neuro-Lingua DOMESTICA** (v4.4.0) is a browser-native neural language model built with React and TypeScript. It implements multiple neural network architectures (ProNeuralLM, AdvancedNeuralLM, TransformerLM) with WebGPU acceleration, advanced training features, and comprehensive governance tracking.

**Live Demo**: https://abbrubin150-ui.github.io/neuro-lingua/

## Quick Reference

```bash
# Install dependencies
pnpm install

# Development server (http://localhost:5173)
pnpm dev

# Run all tests
pnpm test

# Lint and format check
pnpm check                    # lint + format:check + test

# Build for production
pnpm build

# Desktop app (Electron)
pnpm electron:dev             # Development
pnpm electron:build           # Build Windows installer
```

## Technology Stack

| Category | Technology |
|----------|------------|
| Language | TypeScript 5.2.2 (strict mode) |
| UI Framework | React 18.2.0 |
| Build Tool | Vite 4.4.5 |
| Package Manager | pnpm |
| Test Framework | Vitest 1.5.0 + jsdom |
| Linting | ESLint 8.57.0 + Prettier 3.2.5 |
| Desktop | Electron 39.2.7 |
| GPU Acceleration | WebGPU |

## Directory Structure

```
src/
├── App.tsx                    # Main application component (90KB)
├── main.tsx                   # Entry point with error boundary
├── lib/                       # Core neural network implementations
│   ├── ProNeuralLM.ts         # Baseline feedforward language model
│   ├── TransformerLM.ts       # Transformer with multi-head attention
│   ├── AdvancedNeuralLM.ts    # Advanced feedforward variant
│   ├── GovernanceEngine.ts    # Decision tracking (Σ-SIG compliance)
│   ├── BrainEngine.ts         # Autonomous agent state management
│   └── MathUtils.ts           # Numerical utilities
├── models/                    # Attention mechanisms & sparse patterns
│   ├── attention.ts           # Multi-head attention
│   ├── sparse_attention.ts    # Longformer, BigBird, block sparse
│   └── mini_transformer.ts    # Transformer block
├── training/                  # Optimizers (SGD, Adam, Lion, Sophia, etc.)
├── backend/                   # WebGPU & mixed precision
├── components/                # React UI panels (26+ components)
├── contexts/                  # React Context (BrainContext, ProjectContext)
├── types/                     # TypeScript interfaces
├── data/                      # Dataset management, corpusPrep.ts (offline corpus cleaning/split)
├── factory/                   # Offline model factory pipeline (+ node/ files & local server)
├── pwa/                       # Offline service worker template + registration
├── generation/                # Text generation (sampling.ts)
├── losses/                    # Loss functions
├── math/                      # Mathematical analysis (258KB)
├── compression/               # Model compression (quantization, distillation)
├── explainability/            # Interpretability tools
├── integrations/              # External services (mock implementations)
├── autodiff/                  # Automatic differentiation
├── tokenizer/                 # subword.ts (BPE used by the models), BPETokenizer.ts (library)
└── visualization/             # Embedding visualization

tests/                         # 135+ test files mirroring src/ structure
docs/                          # Comprehensive documentation
├── INDEX.md                   # Documentation hub
├── architecture/              # System design documents
├── guides/                    # Development & integration guides
├── theory/                    # Mathematical foundations
└── roadmap/                   # Changelog and planning
scripts/                       # Training & utility scripts
electron/                      # Desktop app (main.ts)
configs/                       # Baseline configurations (JSON)
```

## Code Style & Conventions

### TypeScript Configuration

- **Strict mode**: Enabled (strict null checks, no implicit any)
- **Target**: ES2020
- **Module**: ESNext with Node resolution
- **JSX**: react-jsx (no React import needed)

### ESLint Rules

- **Max warnings**: 0 (CI will fail on any warnings)
- **Unused variables**: Allowed with `_` prefix
- **No explicit any**: Disabled (but avoid using `any` without justification)

### Prettier Settings

- Single quotes
- No trailing commas
- 100 character line width (inferred)
- Semicolons required

### Naming Conventions

- **Components**: PascalCase (e.g., `TrainingPanel.tsx`)
- **Utilities**: camelCase (e.g., `mathUtils.ts`)
- **Types**: PascalCase (e.g., `TrainingConfig`)
- **Constants**: SCREAMING_SNAKE_CASE

## Testing

```bash
pnpm test                      # Run all tests
pnpm test:watch                # Watch mode
pnpm test:parity               # CPU/GPU parity tests
pnpm test:parity:webgpu        # WebGPU-specific parity
pnpm test:parity:wasm          # WASM parity tests
```

### Test Organization

- Tests mirror source structure in `/tests`
- Component tests use `@testing-library/react`
- Model tests include parity tests (`*.parity.test.ts`) and device tests (`*.device.test.ts`)

### Key Test Files

- `ProNeuralLM.parity.test.ts` - CPU/GPU numerical parity
- `TransformerLM.parity.test.ts` - Transformer parity tests
- `lib/ProNeuralLM.gradcheck.test.ts`, `lib/AdvancedNeuralLM.wiring.test.ts`,
  `lib/TransformerLM.gradcheck.test.ts`, `models/transformer_backprop.test.ts` - finite-difference
  gradient checks of every architecture (keep them passing when touching forward/backward code)
- `lib/injection.integrity.test.ts` - Cerebro injection + rollback for all architectures
- `GovernanceEngine.test.ts` - Decision logic verification
- `MathUtils.test.ts` - Numerical stability tests
- `App.test.tsx` - UI integration tests

## Common Development Tasks

### Adding a New Feature

1. Create type definitions in `/src/types/`
2. Implement core logic in `/src/lib/` or `/src/models/`
3. Create UI component in `/src/components/`
4. Add tests in `/tests/` (mirror the source structure)
5. Run `pnpm lint:fix` and `pnpm test`
6. Update relevant documentation in `/docs/`

### Modifying Neural Network Code

- Core models are in `/src/lib/` (ProNeuralLM.ts, TransformerLM.ts, AdvancedNeuralLM.ts)
- Transformer forward/backward: `/src/models/transformer_backprop.ts` (exact gradients, dropout
  masks, per-head RoPE). `MiniTransformerBlock` / `attention.ts` are the reference forward only.
- AdvancedNeuralLM features act through ProNeuralLM's protected hooks (`activate`,
  `normalizeHidden`, `getGradientClipNorm`, `onEpochStart`, ...); add new MLP features the same way
  so that the UI's `train()` path uses them.
- Optimizers in `/src/training/` (`ParamOptimizer.ts` serves the transformer stack and LayerNorm)
- GPU operations in `/src/backend/` (only the MLP's matrix-vector products use WebGPU)

### Working with Components

- Main UI panels are in `/src/components/`
- Use `useBrain()` hook for brain/autonomous state
- Use `useProject()` hook for project/run management
- Follow existing panel patterns (props-based configuration)

### Running Training Scripts

```bash
pnpm train                     # Run training script
pnpm benchmark:gpu             # GPU performance benchmark
pnpm benchmark:models          # Held-out comparison of the three architectures
pnpm lm train --data <dir>     # Offline model factory → models/<name>/model.nlm.json
pnpm lm train --resume <dir>   # continue an interrupted run exactly
pnpm lm eval|generate|serve|info --model <package>
```

`pnpm lm train|eval` use all CPU cores (`--workers`, `src/factory/node/parallel.ts`): replicas in
worker threads load the TypeScript sources through tsx's CommonJS hook registered inside the
thread (tsx 3 does not load in worker threads otherwise, and Node 20 cannot strip types).

The factory (`scripts/lm.ts`, `src/factory/`) is documented in
`docs/guides/OFFLINE_MODEL_FACTORY.md`. Nothing in the app, CLI or desktop code may call an
external AI service or make network requests; `tests/offline/noExternalServices.test.ts` enforces
it (the service worker's same-origin cache handler is the only allowed `fetch`).

## Architecture Patterns

### State Management

- **React Context**: BrainContext (brain state), ProjectContext (project/run management)
- **localStorage**: Persistent model snapshots, hyperparameters
- **Storage keys**:
  - `neuro-lingua-pro-v324` (ProNeuralLM)
  - `neuro-lingua-transformer-v324` (TransformerLM)

### Model Serialization

```typescript
// Export model
const json = model.toJSON();
// Import model (each class restores its own architecture)
const mlp = ProNeuralLM.fromJSON(json);
const advanced = AdvancedNeuralLM.fromJSON(json); // json.architecture === 'advanced'
const transformer = TransformerLM.fromJSON(json); // json.architecture === 'transformer'
// Files chosen in the UI go through parseModelImport() (src/lib/modelImport.ts)
// Compact packages (.nlm.json, base64 tensors + checksum + model card):
const pkg = packModel(json, { precision: 'float32', card }); // src/lib/modelPackageCodec.ts
const model = restoreModel(unpackModel(pkg));                // src/lib/modelPackage.ts
// saveToLocalStorage() falls back to float32 / float16 packages when the JSON exceeds the quota
```

Transformer models saved without `transformer.config.objective` are legacy models and load with
`objective: 'pooled'` (bidirectional, mean-pooled, full-row RoPE) so their predictions do not
change; new models use `objective: 'causal'`.

### Error Handling

- ErrorBoundary component wraps main sections
- Try-catch blocks for async operations
- Graceful fallback when WebGPU unavailable

## Important Files to Know

| File | Purpose |
|------|---------|
| `src/App.tsx` | Main application (90KB, central logic) |
| `src/lib/ProNeuralLM.ts` | Baseline neural network |
| `src/lib/TransformerLM.ts` | Transformer implementation |
| `src/components/TrainingPanel.tsx` | Main training interface (73KB) |
| `src/backend/webgpu.ts` | WebGPU binding layer |
| `src/generation/sampling.ts` | Text generation methods |
| `src/types/project.ts` | Core type definitions |

## CI/CD Pipeline

GitHub Actions workflows in `.github/workflows/`:

- **ci.yml**: Type check, lint, test, build (on push/PR)
- **deploy-pages.yml**: GitHub Pages deployment
- **train-model.yml**: Automated model retraining
- **build-desktop.yml**: Electron Windows build

### CI Requirements

```bash
# All of these must pass:
tsc --noEmit                   # Type checking
pnpm lint                      # No warnings allowed
pnpm format:check              # Formatting check
pnpm test                      # All tests pass
pnpm build                     # Production build
```

## Performance Considerations

### WebGPU Acceleration

- Offloads only the MLP's matrix-vector products; each product flattens and uploads the weight
  matrix and waits for a readback, so small models are often as fast on the CPU. No speedup has
  been measured in this repository.
- The Transformer always trains on the CPU (≈ 26 ms per context window with the UI defaults)
- Automatic CPU fallback if unavailable
- Browser support: Chrome/Edge 113+, Firefox 127+ (with flag)

### Optimization Techniques

- Transformer mini-batches (`batchSize`, default 8) with one fused optimizer pass per batch and
  DropConnect masks shared within a batch
- Grouped-Query Attention (GQA) shrinks the K/V projections
- Library-only (not used by training): mixed precision, sparse/linearized attention, K-FAC

## Gotchas & Tips

1. **localStorage limits**: ~5-10MB typically, be mindful of model sizes
2. **WebGPU availability**: Always test CPU fallback path
3. **Large files**: `App.tsx` (90KB) and `TrainingPanel.tsx` (73KB) are substantial
4. **Strict TypeScript**: All types must be explicit, use discriminated unions
5. **No React import needed**: JSX transform handles it (react-jsx)
6. **Bilingual UI**: Components support English and Hebrew (en/he toggle)
7. **The MLP is order-blind**: ProNeuralLM averages the context embeddings (bag of words).
   Order-dependent behaviour needs the Transformer (see `docs/benchmarks/MODEL_BENCHMARK.md`).
8. **Learning rates**: Adam ~0.003 (UI default); momentum SGD 0.005–0.01. Adam at 0.05+ does not
   learn on the MLP.
9. **Character tokenizer** keeps one `' '` token per whitespace run; generation joins characters
   without separators (`joinTokens`). Use `tokenizeWithOffsets` when mapping text spans to tokens.
10. **Cerebro injection on transformers** must add a multiple of `numHeads` neurons (each head
    grows equally); `InjectionEngine.propose` rounds k accordingly.
11. **No fabricated metrics**: every number shown in the UI must come from the model or be
    labelled as simulated (e.g. the causal analysis panel).
12. **Model packages**: `src/lib/modelPackageCodec.ts` (format, checksum, `storeModelCompactly`)
    must not import the model classes — ProNeuralLM imports it, so the loaders that need
    TransformerLM / AdvancedNeuralLM live in `src/lib/modelPackage.ts` (`restoreModel`).
13. **Subword tokenizer** (`mode: 'bpe'`): the merges live in the tokenizer config
    (`config.bpe`) and are serialized with the model; the vocabulary is `buildVocabulary(text,
    config)` (specials + every subword). Decoding a continuation keeps the leading space.
14. **Interactive training**: `setInteractiveTraining(ms, signal)` makes `train()` yield and stop
    on abort (the App uses it); it is off by default so the CLI and tests are unaffected. An
    interrupted epoch is not added to the history.
15. **Provenance**: exported model cards claim `offline: true` only for models initialised in the
    app or whose imported card says so (`src/lib/modelExport.ts`); unknown history is `null`.
16. **Transformer output layer**: `projectOutput` / `outputSoftmax` / `outputBackward` in
    TransformerLM.ts are the hot path (keep them module-level and monomorphic); their accumulation
    order is part of the bit-identical-results guarantee.
17. **Training state**: `toJSON()` holds weights, the random stream and the history, but not the
    transformer's optimizer moments or schedule position — use `getTrainingState()` /
    `setTrainingState()` (after `setLearningRateSchedule`, which resets the step). Exact resume
    is tested in `tests/factory/resume.test.ts`; keep it passing when adding training state.
18. **Single-thread vs data-parallel training share code**: `TransformerLM.train` and the
    replicas both go through `prepareTrainingWindows` / `planEpoch` / `trainWindow`; change the
    training step there, not in one path only (`tests/factory/parallel.test.ts` compares them).
19. **Generation loops**: `generate()` takes `RepetitionControls` (`src/generation/sampling.ts`),
    applied on every decoding path and over the prompt too. The defaults are a no-repeat n-gram of
    16 for character models and 3 otherwise, with the CTRL repetition penalty off (it lowered the
    real-word rate). See `docs/benchmarks/GENERATION_REPETITION.md`.

## Documentation

- **docs/INDEX.md** - Documentation hub
- **docs/guides/CLAUDE_ASSISTANT_GUIDE.md** - Detailed AI assistant guide (120KB)
- **docs/guides/DEVELOPMENT_SETUP_GUIDE.md** - Setup instructions
- **docs/guides/OFFLINE_MODEL_FACTORY.md** - Offline model production (CLI, packages, server, PWA)
- **docs/architecture/** - System design documents
- **docs/theory/** - Mathematical foundations

## Git Workflow

- Feature branches: `claude/*`, `feature/*`
- Run `pnpm check` before committing
- CI runs on all PRs to main branch
- Desktop builds triggered separately

## Browser Compatibility

| Browser | Version | WebGPU Support |
|---------|---------|----------------|
| Chrome | 113+ | Full support |
| Edge | 113+ | Full support |
| Opera | 99+ | Full support |
| Firefox | 127+ | Behind flag |
| Safari | — | Not supported |
