# WebGPU Acceleration Guide — Neuro-Lingua DOMESTICA

## Overview

Neuro-Lingua DOMESTICA includes **optional WebGPU offload** of the feed-forward models' matrix-vector products (ProNeuralLM / AdvancedNeuralLM forward and backward).

> **Measured reality.** Each GPU product flattens the weight matrix on the CPU, uploads it, runs one
> kernel and waits for the result to be read back. Flattening alone is as much work as the CPU
> multiplication, so for the model sizes used in this app the GPU path is usually **not faster**
> than the CPU; no speedup has been measured in this repository. The TransformerLM always trains
> on the CPU with exact backpropagation. A real speedup would require keeping weights resident on
> the GPU and running whole mini-batches there.

**Current Status:** ✅ Integrated for the MLP models (matrix-vector products only)
**Supported Browsers:** Chrome/Edge 113+, Firefox (experimental)
**Requirements:** WebGPU-capable GPU (most modern GPUs)

---

## Quick Start

### 1. Check GPU Availability

When you load Neuro-Lingua, the application automatically detects WebGPU support:

- ✅ **Available**: Your browser and GPU support WebGPU
- ❌ **Not Available**: Your browser or GPU doesn't support WebGPU (falls back to CPU)

### 2. Enable GPU Acceleration

1. Open the **Training** panel
2. Look for **⚡ GPU Acceleration (WebGPU)** toggle
3. If available, toggle it to enable
4. You'll see a message confirming that WebGPU is enabled for the MLP matrix-vector products

### 3. Train with GPU

1. Enter training text (200+ words recommended)
2. Configure hyperparameters as usual
3. Click **Train** or press Ctrl/Cmd+Enter
4. MLP training sends its matrix-vector products to the GPU (compare the timings in the GPU metrics panel)

### 4. View GPU Metrics

After training completes:

- **System message** shows: "⚡ GPU Acceleration: X ops, Yms total, Zms/op"
- **Advanced Statistics** panel displays GPU metrics:
  - Total Operations performed on GPU
  - Total Time spent on GPU
  - Average Time per operation

---

## Technical Details

### How GPU Acceleration Works

When GPU acceleration is enabled, Neuro-Lingua performs these operations on the GPU:

1. **Matrix-Vector Multiplication** (y = A @ x)
   - Primary bottleneck in neural network forward/backward passes
   - Most performance-critical operation

2. **Matrix-Vector Transpose Multiplication** (y = A^T @ x)
   - Used in gradient computations
   - Essential for training efficiency

3. **Element-wise Operations** (vector addition, multiplication)
   - Supporting operations for optimization

### GPU Execution Pipeline

```
Forward Pass:
  Input → [GPU] Matrix-Vector Mul → Hidden → [GPU] Matrix-Vector Mul → Output

Backward Pass:
  Loss Gradient → [GPU] Transpose Mul → Hidden Grad → [GPU] Transpose Mul → Input Grad

Weight Updates:
  Gradients → [GPU] Vector Operations → New Weights
```

### CPU Fallback

If GPU operations fail (rare):

- Automatic fallback to CPU computation
- GPU disabled for remainder of session
- Training continues normally (slower)
- Console logs warning message

### Device selection flags

- **Automatic detection:** `GPUNeuralOps.initialize()` enables GPU execution only when `navigator.gpu` is available; otherwise it runs purely on CPU without throwing.
- **Force CPU:** Skip calling `setGPUOps` on `ProNeuralLM` (as in `scripts/train.ts`, the entry point for CLI training) or call `gpuOps.setEnabled(false)` after initialization.
- **Tests:** GPU-specific assertions are gated the same way, so automated runs without WebGPU simply exercise the CPU path.

---

## Performance Benchmarking

### Measuring GPU Benefits

**Setup:** Use identical training parameters, compare CPU vs GPU

```bash
# Training Configuration
- Hidden Size: 64
- Epochs: 20
- Context Size: 3
- Corpus: 500+ words
- Optimizer: Momentum
```

### Expected Performance

No speedup has been measured for the training path (see the note in the overview). Measure it on
your hardware with the instructions below before relying on the GPU, and compare against the CPU.

### Benchmark Instructions

1. **CPU Benchmark:**
   - Disable GPU toggle
   - Train with 20 epochs
   - Note training time from console

2. **GPU Benchmark:**
   - Enable GPU toggle
   - Train with same parameters
   - Compare times
   - View GPU metrics

3. **Calculate Speedup:**
   ```
   Speedup = CPU Time / GPU Time
   ```

---

## Browser Compatibility

### Chrome/Edge (Recommended)

**Status:** ✅ Full WebGPU support

```
Requirements:
- Chrome 113+ or Edge 113+
- Integrated or dedicated GPU
- Enable "Experimental Web Platform features" flag (if needed)
```

**Enable flag (if needed):**

1. Navigate to `chrome://flags`
2. Search for "WebGPU"
3. Enable the flag
4. Restart browser

### Firefox

**Status:** 🟡 Experimental support

```
Requirements:
- Firefox 120+ (experimental)
- Dedicated NVIDIA or Intel GPU
- Enable `dom.webgpu.enabled` in about:config
```

### Safari

**Status:** 🔄 Coming soon (Metal support in progress)

Currently not supported, but Apple is actively implementing WebGPU support.

### Mobile Browsers

**Status:** ❌ Not recommended

- Mobile GPU integration still experimental
- Benefits unclear on mobile hardware
- May consume additional battery/heat

---

## Optimization Tips

### 1. Choose Appropriate Batch Size

GPU acceleration benefits most with larger operations:

- **Smaller corpus (200-300 words):** GPU overhead may exceed benefit
- **Larger corpus (1000+ words):** GPU acceleration shines

### 2. Use Efficient Architectures

GPU acceleration is most effective with:

- **ProNeuralLM** (simple feedforward)
- **AdvancedNeuralLM** (more matrix ops)
- **TransformerLM** (heavy matrix operations - best speedup)

### 3. Monitor GPU Metrics

Check if GPU is actually being used:

- Look for non-zero `totalOperations` in metrics
- If zero, GPU operations failed silently
- Check browser console for errors

### 4. Temperature Management

GPU training generates heat:

- Monitor GPU temperature (use HWInfo64 on Windows)
- Typical safe range: 60-80°C
- If exceeding 85°C, reduce epochs or enable GPU power limits
- Use browser's developer tools to throttle if needed

---

## Troubleshooting

### GPU Toggle Disabled

**Problem:** The GPU toggle is grayed out / "Not Available"

**Causes & Solutions:**

1. **Browser doesn't support WebGPU**
   - Solution: Update to Chrome 113+ or Edge 113+
   - Check: `chrome://version` shows Chrome 113+

2. **No compatible GPU**
   - Solution: Integrated GPU (Intel UHD) usually works
   - Try: Dedicated GPU if available (NVIDIA, AMD)
   - Check: `chrome://gpu` for GPU info

3. **WebGPU disabled in browser**
   - Chrome: Navigate to `chrome://flags`, search "WebGPU", enable
   - Edge: Navigate to `edge://flags`, search "WebGPU", enable
   - Restart browser after enabling

### GPU Toggle Enabled But Not Accelerating

**Problem:** GPU toggle is on but training isn't faster

**Solutions:**

1. **Check GPU metrics after training:**
   - If `totalOperations` is 0, GPU not being used
   - If non-zero but small, overhead exceeds benefit

2. **Increase corpus size:**
   - Larger corpora show bigger benefits
   - Try 1000+ word corpus instead of 200 words

3. **Check console for errors:**
   - Open DevTools (F12)
   - Console tab → Look for warnings
   - Common issue: WASM not available

4. **Try different architecture:**
   - Transformer uses more matrix ops (best for GPU)
   - AdvancedNeuralLM is better than ProNeuralLM
   - Try switching architecture

### Training Slower with GPU

**Problem:** Training is actually slower with GPU enabled

**Causes:**

1. **Corpus too small:** GPU overhead dominates
   - Solution: Use larger corpus (1000+ words)

2. **Browser overhead:** Firefox/Safari may have more overhead
   - Solution: Try Chrome/Edge for best performance

3. **GPU memory bandwidth bottleneck**
   - Solution: Reduce hidden size (smaller model)
   - Or: Train with larger corpus to amortize overhead

### GPU Disabled During Training

**Problem:** Training starts with GPU but switches to CPU

**Causes:**

1. **WebGPU operation failed**
   - Check console for error message
   - Try reloading page

2. **Out of GPU memory**
   - Solution: Reduce `hiddenSize` parameter
   - Solution: Reduce `contextSize` parameter
   - Solution: Use smaller corpus

3. **GPU crashed**
   - Solution: Restart browser
   - Solution: Update GPU drivers
   - Solution: Disable and re-enable GPU toggle

---

## Advanced: GPU Metrics Explained

### Metrics Displayed

After training with GPU enabled:

```
⚡ GPU Acceleration
Status: ACTIVE (or DISABLED)
Total Operations: 1,234
Total Time: 4,567.89 ms
Avg Time/Op: 3.70 ms
Device: WebGPU Device
```

### Interpreting Metrics

| Metric               | Meaning                        | Good Value         |
| -------------------- | ------------------------------ | ------------------ |
| **Total Operations** | Number of GPU kernels launched | Hundreds-Thousands |
| **Total Time**       | Cumulative GPU execution time  | Varies by corpus   |
| **Avg Time/Op**      | Average time per operation     | 1-5 ms             |
| **Device**           | GPU backend being used         | "WebGPU Device"    |

### Example Analysis

**Scenario:** 20-epoch training with 500-word corpus

```
Total Operations: 2,840
Total Time: 8,935 ms
Avg Time/Op: 3.15 ms

Interpretation:
- 2,840 operations = reasonable (142 ops/epoch)
- 8,935 ms ≈ 9 seconds GPU time
- 3.15 ms/op is typical for matrix operations
- Speedup: CPU took ~25 seconds → 9 sec GPU = 2.8x speedup
```

---

## Limitations & Known Issues

### Current Limitations

1. **No multi-GPU support:** Uses single GPU only
2. **No GPU memory optimization:** Can't train extremely large models
3. **Limited shader optimization:** Not as tuned as native CUDA
4. **Browser constraints:** Runs in browser sandbox

### Known Issues

1. **Chrome on Mac with M1/M2:**
   - WebGPU may not use GPU efficiently
   - Workaround: Use Safari (when available) or Sonoma+

2. **Firefox with AMD GPU:**
   - WebGPU support is experimental
   - May not work or be slower than CPU

3. **Integrated GPU on laptop:**
   - Shares system RAM with CPU
   - Overhead can exceed benefit on small operations

---

## Future Improvements

**Planned enhancements:**

1. **Multi-GPU support** — Use multiple GPUs in parallel
2. **Custom WASM kernels** — Specialized GPU kernels for LM operations
3. **Mixed precision** — fp16 for faster, lower-memory training
4. **Persistent GPU state** — Keep model on GPU between batches
5. **Attention optimization** — GPU-accelerated Transformer attention

---

## References

### Documentation

- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [WebGPU Best Practices](https://gpuweb.github.io/gpuweb/#gpubuffer)
- [Chrome WebGPU Support](https://developer.chrome.com/docs/web-platform/webgpu/)

### Related Articles

- "Accelerating Neural Networks in the Browser" — WebGPU research
- "WebGPU Performance Characteristics" — GPU acceleration patterns
- "Browser-based ML: Privacy and Performance" — Neuro-Lingua design principles

### Testing

- Run benchmarks with different corpus sizes
- Compare CPU vs GPU times
- Monitor GPU temperature and memory
- File issues if experiencing problems

---

## Support

If you encounter GPU-related issues:

1. **Check browser compatibility:** Is your browser WebGPU-enabled?
2. **Check GPU availability:** Does `chrome://gpu` show WebGPU support?
3. **Check console errors:** Are there WebGPU error messages?
4. **Try disabling GPU:** Does training work without GPU?
5. **Report issue:** Include browser version, GPU model, and error messages

---

**Last Updated:** November 2024
**GPU Support:** WebGPU (active development)
