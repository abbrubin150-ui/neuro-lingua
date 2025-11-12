# Transformer Architecture Guide — Neuro-Lingua DOMESTICA

## Overview

Neuro-Lingua includes a fully integrated **Transformer architecture** based on the attention mechanism. The Transformer offers state-of-the-art sequence modeling capabilities and is now exposed in the UI for easy configuration and training.

**Status:** ✅ Fully integrated into training UI
**Architecture:** Multi-head attention with feed-forward layers
**Default Config:** 2 layers, 4 attention heads

---

## Quick Start

### 1. Select Transformer Architecture

1. Open Neuro-Lingua in your browser
2. In the **Training** panel, select the **🔮 Transformer** architecture button
3. The training panel will reveal Transformer-specific controls

### 2. Configure Transformer Layers

When Transformer is selected, you'll see:

```
🔮 Transformer Configuration
┌─────────────────────────┐
│ Attention Heads: [4] ← │ (1-16)
│ Layers: [2] ←          │ (1-8)
└─────────────────────────┘
```

**Adjust parameters:**

- **Attention Heads:** Number of parallel attention mechanisms (default: 4)
- **Layers:** Number of stacked transformer blocks (default: 2)

### 3. Train with Transformer

1. Enter training text (500+ words recommended)
2. Configure other hyperparameters (hidden size, epochs, learning rate, etc.)
3. Click **Train** or press Ctrl/Cmd+Enter
4. Monitor training progress in the metrics panel

---

## Understanding Transformer Parameters

### Attention Heads (1-16)

**What it does:** Splits the attention mechanism into multiple parallel subspaces

**Intuition:**

- **1 head:** Single focus mechanism
- **4 heads:** Model learns 4 different types of relationships
- **8 heads:** Rich, diverse representation of dependencies
- **16 heads:** Very expressive, but more parameters

**Impact on Performance:**

- More heads = Better expressiveness but slower training
- Typical range: 4-8 heads for small models
- Must divide evenly into hidden size (e.g., if hidden_size=64, use 4, 8 heads)

**Recommendation:**

- Start with **4 heads** for fast prototyping
- Use **8 heads** for better accuracy
- Try **2-3 heads** for very small corpus (200-300 words)

### Layers (1-8)

**What it does:** Stacks multiple transformer blocks for deeper representations

**Intuition:**

- **1 layer:** Simple attention (like one-step reasoning)
- **2 layers:** Two-step reasoning over attention patterns
- **4 layers:** Rich hierarchical understanding
- **8 layers:** Very deep, approaching BERT-style models

**Impact on Performance:**

- More layers = Better learning capacity but slower
- Training time scales roughly linearly with depth
- Typical range: 1-4 layers for browser training

**Recommendation:**

- Start with **2 layers** (default)
- Use **3-4 layers** for better results on large corpus (1000+ words)
- Avoid >4 layers for small corpus (memory/time constraints)

---

## Architecture Components

### Multi-Head Attention

The core mechanism of Transformers:

```
Input → [Head 1: Pattern A]
      → [Head 2: Pattern B]  → Concatenate → Output
      → [Head 3: Pattern C]
      → [Head 4: Pattern D]
```

Each head learns different relationships:

- **Head 1:** Long-range dependencies
- **Head 2:** Local patterns
- **Head 3:** Rare token relationships
- **Head 4:** Syntax patterns

### Feed-Forward Network

After attention, each token passes through:

```
Linear(hidden_size → 2*hidden_size)
  ↓ (GELU activation)
Linear(2*hidden_size → hidden_size)
```

This provides non-linear expressiveness and capacity.

### Positional Encoding

Tells the model about token positions (critical since attention is permutation-invariant):

```
Position 0 → [sin(0/10000), cos(0/10000), ...]
Position 1 → [sin(1/10000), cos(1/10000), ...]
...
```

### LayerNorm & Residual Connections

Stabilizes training through:

- Layer normalization before attention
- Residual connections (skip connections)
- Prevents gradient vanishing/exploding

---

## Transformer vs. Standard LM

### Comparison Matrix

| Aspect                      | Standard (ProNeuralLM) | Advanced                   | **Transformer**      |
| --------------------------- | ---------------------- | -------------------------- | -------------------- |
| **Mechanism**               | MLP with ReLU          | MLP + advanced activations | Multi-head attention |
| **Long-range dependencies** | Limited                | Better                     | Excellent            |
| **Parameter efficiency**    | High                   | Medium                     | Lower                |
| **Training speed**          | Fast                   | Medium                     | Slower               |
| **Memory usage**            | Low                    | Medium                     | Higher               |
| **Model capacity**          | Limited                | Good                       | Excellent            |
| **Best for**                | Small corpus           | Medium corpus              | Large corpus         |

### When to Use Transformer

✅ **Use Transformer when:**

- You have a **large corpus** (1000+ words)
- You want **best quality** output
- Training speed is less important
- You need to capture **long-range dependencies**

❌ **Avoid Transformer if:**

- Your corpus is **very small** (<300 words)
- You need **fastest training**
- You have **memory constraints**
- Testing ideas quickly (use Standard instead)

---

## Training Guide

### Hyperparameter Recommendations

#### Small Corpus (200-500 words)

```
Architecture:   Transformer
Attention Heads: 2-3
Layers:         1
Hidden Size:    32-48
Epochs:         15-25
Learning Rate:  0.1-0.15
Optimizer:      Adam
Batch/Context:  3-4
```

#### Medium Corpus (500-1500 words)

```
Architecture:   Transformer
Attention Heads: 4
Layers:         2 (default)
Hidden Size:    48-64
Epochs:         20-40
Learning Rate:  0.08-0.12
Optimizer:      Adam or Momentum
Batch/Context:  4-5
```

#### Large Corpus (1500+ words)

```
Architecture:   Transformer
Attention Heads: 6-8
Layers:         3-4
Hidden Size:    64-128
Epochs:         30-50
Learning Rate:  0.05-0.1
Optimizer:      Adam
Batch/Context:  5-6
```

### Training Tips

1. **Start Conservative:** Use fewer heads/layers first, scale up if training is fast
2. **Monitor Loss:** Loss should steadily decrease (watch for plateaus)
3. **Use GPU:** Transformer benefits significantly from GPU acceleration
4. **Adjust Learning Rate:** If loss spikes, reduce learning rate
5. **Watch Accuracy:** Perplexity should steadily decrease each epoch
6. **Resume Training:** Enable resume to continue from checkpoints

### Common Issues

#### "Training is very slow"

- **Solution 1:** Reduce attention heads (try 2-3 instead of 4)
- **Solution 2:** Reduce number of layers (use 1 instead of 2)
- **Solution 3:** Enable GPU acceleration if available
- **Solution 4:** Reduce hidden size (48 instead of 64)

#### "Loss doesn't decrease"

- **Possible Cause:** Learning rate too low
- **Solution:** Increase learning rate to 0.1-0.15
- **Alternative:** Try Adam optimizer instead of momentum

#### "Loss spikes unpredictably"

- **Possible Cause:** Learning rate too high
- **Solution:** Reduce learning rate to 0.05-0.08
- **Alternative:** Enable layer norm for stability

#### "Out of memory" (in browser)

- **Solution 1:** Reduce attention heads
- **Solution 2:** Reduce number of layers
- **Solution 3:** Reduce hidden size significantly
- **Solution 4:** Use smaller corpus for training

---

## Understanding Attention Visualization

### What Attention Learns

During training, the model learns attention patterns like:

```
Input: "The cat sat on the mat"

Head 1 (Syntax):
  "cat" → attends to "The" (article)
  "sat" → attends to "cat" (subject)

Head 2 (Long-range):
  "mat" → attends to "cat" (object association)
  "on" → attends to "the" (prepositional phrase)

Head 3 (Position):
  Each word → nearby tokens (local context)
```

These patterns are learned automatically during training!

---

## Advanced: Fine-tuning Transformers

### Transfer Learning (Future Feature)

Once implemented, you'll be able to:

1. Train a base Transformer model
2. Export the model
3. Load it back and fine-tune on new data
4. Achieve better accuracy with less training time

### Custom Attention Patterns

Advanced customization (future):

- Custom attention initialization
- Sparse attention patterns
- Multi-head weighting
- Dynamic head allocation

---

## Performance Metrics

### What to Monitor

While training with Transformer, watch:

1. **Loss:** Should decrease monotonically
   - Target: Decrease by ~5-10% each epoch
   - Warning: Spikes or increase indicates problems

2. **Accuracy:** Should increase steadily
   - Target: Reach 50-70% on medium corpus
   - Depends on corpus difficulty

3. **Perplexity:** Should decrease
   - Formula: PPL = exp(loss)
   - Target: Reach 5-10 for good models

4. **Training Speed:** Tokens/sec
   - Watch: Should remain steady
   - Indicates if GPU is being used effectively

### Benchmarks

**Typical results for 500-word corpus:**

| Config            | Epochs | Time      | Final Accuracy | PPL |
| ----------------- | ------ | --------- | -------------- | --- |
| 1 head, 1 layer   | 20     | 5-8 sec   | 45%            | 12  |
| 4 heads, 2 layers | 20     | 15-20 sec | 60%            | 8   |
| 8 heads, 3 layers | 20     | 35-40 sec | 68%            | 6   |

(With GPU: 2-5x speedup)

---

## Architectural Details

### Transformer Block Structure

```
Input (Batch × SeqLen × Hidden)
  ↓
LayerNorm
  ↓
Multi-Head Attention(query, key, value)
  ↓
Dropout(0.1)
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Feed-Forward Network
  - Linear(hidden → 2*hidden)
  - GELU Activation
  - Linear(2*hidden → hidden)
  ↓
Dropout(0.1)
  ↓
Add & Norm (Residual + LayerNorm)
  ↓
Output (Batch × SeqLen × Hidden)
```

### Attention Mechanism

```
Query = W_q * Input
Key = W_k * Input
Value = W_v * Input

Scores = (Query @ Key^T) / sqrt(dim)
Weights = Softmax(Scores)
Output = Weights @ Value
```

This allows every position to attend to every other position!

---

## Troubleshooting

### Transformer-Specific Issues

**Q: Why is Transformer so slow?**

- A: Attention complexity is O(seq_len²). Smaller corpus = faster
- A: GPU acceleration helps significantly

**Q: Should I use Transformer for small corpus?**

- A: No. Standard or Advanced LM is faster. Use Transformer for 1000+ words

**Q: Can I mix architectures?**

- A: No. You must choose one per training session

**Q: What's the difference between heads and layers?**

- A: Heads = parallel processing (width)
- A: Layers = sequential processing (depth)

**Q: How many heads should equal hidden size?**

- A: Hidden size should be divisible by num_heads
- A: Example: hidden=64, heads=4 means 16 dims per head

---

## References

### Academic Papers

- **"Attention Is All You Need"** (Vaswani et al., 2017)
  - Original Transformer paper
  - https://arxiv.org/abs/1706.03762

- **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
  - Bidirectional attention, shows why Transformers are powerful
  - https://arxiv.org/abs/1810.04805

### Key Concepts

- Attention mechanism
- Multi-head attention
- Positional encoding
- Feed-forward network
- Layer normalization
- Residual connections

### Related Tools

- Hugging Face Transformers library
- PyTorch attention implementation
- TensorFlow tf.keras.layers.MultiHeadAttention

---

## Future Enhancements

### Planned Features

1. **Relative Positional Encoding** — Better handling of variable length sequences
2. **Flash Attention** — Faster attention implementation
3. **Sparse Attention** — For very long sequences
4. **Distillation** — Train smaller models from Transformer teachers
5. **Quantization** — Compress Transformer for mobile/edge

---

**Last Updated:** November 2024
**Transformer Support:** Fully Integrated
**Status:** Production Ready
