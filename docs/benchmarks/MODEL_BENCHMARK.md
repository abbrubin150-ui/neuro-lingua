# Model benchmark (held-out data)

Reproducible comparison of the three architectures on text they were **not** trained on.

```bash
pnpm benchmark:models                                   # ~10 min on a 4-core Xeon
pnpm benchmark:models -- --quick                        # smoke test
pnpm benchmark:models -- --out docs/benchmarks/MODEL_BENCHMARK_RESULTS.md
```

Source: [`scripts/benchmark_models.ts`](../../scripts/benchmark_models.ts). Seeds are fixed, so the
quality numbers are reproducible; timings depend on the machine.

## Tasks

**Order task (synthetic).** Sentences `<a> <verb> <b> first|second <answer> end` with 8 names and
6 verbs; the answer is `a` after `first` and `b` after `second`. Both orders of every pair are
equally frequent, so the answer depends on **word order**, not on which words occur. 300
training / 100 held-out sentences, context window 6. Metrics: next-token accuracy at the answer
position, and validation perplexity over all tokens. The best possible perplexity is ≈ 2.96: the
first name, the verb, the second name and the question word are random by construction.

**Character-level corpus.txt.** `data/corpus.txt` (≈ 10.7k characters of English prose), split
85/15 by sentence (9,300 training / 1,390 validation characters), character tokenizer, context
window 8. Metric: validation bits per character (bpc = loss / ln 2). For reference, a unigram
character model of English is ≈ 4.1 bpc.

## Models

| Model | Configuration |
| --- | --- |
| MLP (ProNeuralLM) | hidden 64, Adam lr 0.003, dropout 0.05 |
| Advanced | as MLP + GELU + LayerNorm |
| Transformer causal | D 48, 2 layers, 4 query / 2 KV heads, FFN 96, Adam lr 0.003, batch 8 |
| Transformer pooled | same weights layout, legacy objective (bidirectional, mean-pooled, full-row RoPE) |

## Results (2026-11-08, Node 22, Intel Xeon 2.3 GHz)

### Order task (12 epochs)

| Model | Answer accuracy | Validation perplexity | s / epoch |
| --- | --- | --- | --- |
| MLP (ProNeuralLM, H=64) | 4.0% | 16.32 | 0.4 |
| Advanced (GELU + LayerNorm, H=64) | 2.0% | 15.25 | 0.3 |
| Transformer causal (D=48, 2 layers, GQA 4/2) | **100.0%** | **3.57** | 3.1 |
| Transformer pooled (legacy objective) | 81.0% | 3.91 | 3.0 |

### Character-level corpus.txt (6 epochs)

| Model | Validation bpc | Training bpc | s / epoch |
| --- | --- | --- | --- |
| MLP (ProNeuralLM, H=64) | 4.17 | 3.95 | 3.8 |
| Advanced (GELU + LayerNorm, H=64) | 4.02 | 3.84 | 2.0 |
| Transformer causal (D=48, 2 layers, GQA 4/2) | **2.88** | 1.86 | 23.3 |
| Transformer pooled (legacy objective) | 3.26 | 2.93 | 18.2 |

## Reading the results

- **The feed-forward models are order-blind.** ProNeuralLM represents the context as the *mean*
  of its token embeddings (a bag of words, like CBOW). It learns deterministic sequences
  perfectly, but it cannot tell `alice likes bob` from `bob likes alice`, and at the character
  level it stays close to a unigram model. Concatenating position-specific embeddings
  (Bengio et al., 2003) would remove this limitation; it is not implemented.
- **The causal transformer generalises.** It solves the order task on unseen sentences and
  reaches 2.88 bpc on held-out prose after 6 epochs. Its gap between training and validation
  bpc (1.86 vs 2.88) shows it is starting to memorise this small corpus.
- **The causal objective beats the legacy pooled one** on both tasks with the same weights
  layout: per-position next-token training gives far more signal per window, and causal
  attention with per-head RoPE encodes order directly.
- **Cost.** The transformer is about 6–12× slower per epoch than the feed-forward models on the CPU. With the UI
  defaults (D 128, 4 layers, FFN 512, ≈ 1M parameters) one training window costs ≈ 26 ms.

## Earlier word-level attempt

At the word level, corpus.txt has 1,214 training tokens and a 704-word vocabulary: every model
memorises the training text (training perplexity ≈ 1) and validation perplexity explodes, mostly
because held-out words are out-of-vocabulary (`evaluate()` reports `oovRate`). Word-level
generalisation needs a much larger corpus than the one shipped with the repository.
