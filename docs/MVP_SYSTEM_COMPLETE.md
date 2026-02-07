# System-Complete MVP (Browser-Focused)

This document outlines the minimal browser-ready runtime delivering the green-tier capabilities identified earlier. It keeps scope tightly focused on what is realistic for Web / WebGPU without speculative features.

## Scope (Included vs. Deferred)
- **Included (🟢)**: Tokenizer subsystem, determinism & reproducibility layer, dataset abstraction, numeric precision control (partial), headless programmatic API.
- **Deferred (🟡/🔴)**: KV-cache optimization, advanced memory management, unified checkpoint streaming, compiler-grade graph IR, formal constraint solving.

## Browser Execution Assumptions
- Primary target: modern browsers with WebGPU available; must gracefully degrade to CPU if WebGPU missing.
- All artifacts (tokenizer JSON, config, weights) load from static hosting; no filesystem access assumed.
- No reliance on bit-exact GPU results; determinism is "deterministic-enough" under seeded sampling.

## Module Breakdown

### 1. Tokenizer Subsystem
- **Artifacts**: BPE/Unigram JSON + hash for versioning; optional WASM tokenizer fallback.
- **Runtime**: Stateless encoder/decoder with UTF-8 normalization; batch encode/decode for throughput.
- **Interfaces**: `loadTokenizer(url|blob)`, `encode(text, opts)`, `decode(tokens)`.
- **Metrics**: Coverage, entropy, fertility computed in JS for debug dashboards.

### 2. Determinism & Reproducibility Layer
- **PRNG**: Single seedable generator feeding sampling and dataset shuffling.
- **Hashing**: Config + weights hashed client-side for run IDs; include tokenizer hash in run metadata.
- **Replay**: Capture seeds and sampling parameters per generation; tolerate numeric drift across devices.

### 3. Dataset Abstraction
- **Schema**: Typed dataset descriptor (fields, modalities, split ratios).
- **Ingestion**: Fetch + parse JSON/CSV/parquet-lite via streaming reader; validate against schema.
- **Splits**: Deterministic train/val/test splits derived from seed; shuffle reproducibly.
- **Transformations**: Map/filter/batch utilities that honor deterministic ordering.

### 4. Numeric Precision Control (Partial)
- **Policies**: Per-op precision policy wrapper; FP32 default with FP16 emulation where safe.
- **Loss Scaling**: Manual loss scaling hooks for training loops; debug toggles for overflow checks.
- **Logging**: Precision policy recorded in run metadata; assert mismatches during debug builds.

### 5. Programmatic API (Headless Core)
- **Packaging**: Core TS module with no UI dependencies; UI acts as consumer only.
- **Surface Area**: `createModel(config, weights)`, `trainStep(batch, opts)`, `generate(prompt, opts)`; all return promises.
- **CLI/Scriptability**: Node-compatible entrypoint for offline scripts; same APIs exposed to browser bundles.
- **Telemetry**: Optional hooks for timing/memory metrics; no GPU-specific assumptions.

## Cross-Cutting Concerns
- **Versioning**: Embed version + hash for tokenizer, model config, and weights in a manifest JSON.
- **Error Handling**: Graceful degradation when WebGPU unavailable; emit explicit capability summary at startup.
- **Configuration**: Single JSON config describing model dims, precision policy, dataset schema, and seed; hashed for reproducibility.

## MVP Acceptance Criteria
- End-to-end text generation demo runs in browser with deterministic sampling (given seed) using headless core APIs.
- Tokenizer load + encode/decode succeeds from static artifact; metrics available in console logs.
- Dataset ingest + deterministic split works for sample dataset (few hundred rows) without server code.
- Precision policies logged and enforced during generation/training loops; loss scaling flag toggles behavior.
- API usable headlessly from TS/Node without importing UI modules.

## Build Sequencing (Suggested)
1. Standalone tokenizer loader + metrics.
2. Seeded PRNG utilities shared by sampling and dataset modules.
3. Dataset abstraction with deterministic split + shuffling.
4. Precision policy wrappers and loss-scaling hooks.
5. Headless core API wiring the above into generation/training routines.

## Out-of-Scope for MVP
- GPU kernel fusion, static graph compilation, mmap/streaming checkpoints, and formal constraint solvers remain explicitly excluded.
