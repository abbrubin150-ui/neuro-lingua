# UI training defaults: context 32 and the training stride

Measured on 2026-11-15 (Node 22 on a 4-core CPU, single thread) to choose how the browser app
trains the Transformer after its default context grew from 4 to 32 tokens. Seeds are fixed;
timings depend on the machine.

## What the stride is

With the causal objective every position of a context window is a training target. Training on
every overlapping window (`trainingStride` 1) therefore visits each token `contextSize` times per
epoch. A stride of `s` keeps one window in `s`, so each token is a target `contextSize / s` times
per epoch. `pnpm lm train` uses `s = contextSize` (one pass per epoch) with validation and early
stopping.

## Cost per epoch

UI default model (D 128, 4 layers, 8 / 4 heads, FFN 512, 1,173,857 parameters with a 993-subword
vocabulary), BPE tokens, 428 training tokens:

| Context | Stride | ms per token per epoch |
| --- | --- | --- |
| 4 | 1 (former default) | 31.8 |
| 32 | 1 | 187.8 |
| 32 | 8 | 20.0 |
| 32 | 32 | 6.5 |

## Held-out quality

`data/corpus.txt`, split 85 / 15 at a sentence boundary (9,066 training / 1,644 held-out
characters), BPE learned from the training part, a smaller model (D 64, 2 layers, 4 / 2 heads,
FFN 128), context 32. Bits per character on the held-out part:

| Seconds of training | Stride 1 | Stride 8 | Stride 32 |
| --- | --- | --- | --- |
| ≈ 12 | | 3.256 (1 epoch) | 3.248 (4 epochs) |
| ≈ 23 | | 3.263 (2) | 3.289 (8) |
| ≈ 44 | | 3.504 (4) | 3.596 (16) |
| ≈ 63 | 4.291 (1 epoch) | 3.974 (6) | 4.075 (24) |
| ≈ 87 | | 4.375 (8) | 4.452 (32) |
| ≈ 122 | 5.131 (2) | 5.181 (16) | 5.324 (64) |
| ≈ 245 | 5.694 (4) | 6.002 (32) | 5.923 (128) |

- For the same compute the three strides reach about the same held-out loss. The stride does not
  make training more efficient; it sets how many passes over the text one epoch is.
- On this corpus the best held-out loss comes after about 4 passes (≈ 12 s). After that the model
  memorises the training text and the held-out loss rises.
- With stride 1 the first epoch is already 32 passes: 4.29 bpc, against 3.25 reached with an
  eighth of the compute. The app has no held-out validation, so it cannot stop earlier than one
  epoch.

## The app's two built-in texts (UI default model, 30 epochs)

| Text | Stride | Seconds | Training loss | Sample (T 0.8, top-p 0.9) |
| --- | --- | --- | --- | --- |
| default text (110 chars) | 8 | 68 | 0.368 | "neural language model trains in the browser. Pas" |
| default text | 32 | 23 | 0.554 | same |
| example corpus (506 chars) | 8 | 177 | 0.370 | "and artificial intelligence are reshaping the technology landscape. Advanced algorithms l" |
| example corpus | 32 | 50 | 0.826 | "…the technology landscape. Advanced algormhmscap" |

With stride 1 the example corpus needs about 25 minutes for 30 epochs (266 tokens × 188 ms × 30);
that run was not completed.

## Choice

`defaultTrainingStride(contextSize) = max(1, floor(contextSize / 4))`, i.e. 8 at context 32.

- Each token is a target 4 times per epoch, the same as the former default (context 4, every
  window), so "30 epochs" keeps its meaning.
- An epoch at context 32 costs 20 ms per token, less than the former default (31.8 ms).
- On the built-in texts 30 epochs still fit the text (stride 32 leaves the example corpus
  under-trained at 30 epochs).

## Limits

- One run per configuration; the quality table uses a smaller model than the UI default to keep
  the runs short.
- The app's default of 30 epochs is 120 passes at stride 8. On the 9 K-character corpus above
  that is far past the best held-out loss (≈ 4 passes). The app does not hold text out or stop
  early; the CLI does. Fewer epochs, or held-out validation in the app, is the remaining gap.
- Models stored before this change keep the stride saved with them.
