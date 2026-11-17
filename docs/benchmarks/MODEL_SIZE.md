# Model size: `tiny` vs. `small` presets

Measured 2026-11-09 to 2026-11-17 to see whether the main limit on text quality is model size.
Every number below comes from `pnpm lm train` logs (`training-log.jsonl`) and
`scripts/experiments`-style generation runs on the trained packages. One run per preset.

## Setup

Same data, tokenizer and split for both models: Tiny Shakespeare (1.1 MB) with BPE (2,004
tokens), 6,261 training and 693 validation paragraphs, seed 1337, Adam with the factory's
cosine schedule (ending at 0.0003), 4 worker threads.

| Preset | Layers | Heads (KV) | FF width | Context | Parameters | Epochs |
|---|---|---|---|---|---|---|
| `tiny` | 2 | 4 (2) | 128 | 32 | 324,372 | 4 |
| `small` | 4 | 8 (4) | 512 | 64 | 1,433,684 | 6 |

## Held-out quality (bits per character)

| Epoch | `tiny` | `small` |
|---|---|---|
| 1 | 2.150 | 2.123 |
| 2 | 2.042 | 2.034 |
| 3 | 1.983 | 1.969 |
| 4 | **1.959** (final) | 1.918 |
| 5 | | 1.883 |
| 6 | | **1.873** (final) |

The schedules differ: `tiny`'s learning rate reached its floor at epoch 4, `small`'s at epoch 6.
So the epoch-4 rows are not a like-for-like comparison. Compare the final rows instead:
**1.959 vs 1.873 bpc, 4.4% better**.

| | `tiny` | `small` |
|---|---|---|
| Final train loss / validation loss (nats per token) | 4.266 / 4.372 (gap 0.11) | 3.795 / 4.181 (gap 0.39) |
| Training time | 1,285 s (≈ 21 min, 313–329 s per epoch) | 17,523 s (≈ 4.9 h, 1,722–3,973 s per epoch) |

`small` epoch times varied 2.3× because other measurements shared the 4 cores during some
epochs. The run was resumed twice with `--resume` after the 2-hour background limit.

## Generation

Six prompts (`ROMEO:`, `First Citizen:`, …), 50 tokens. Metrics are as in
[GENERATION_REPETITION.md](./GENERATION_REPETITION.md). For the n-gram rows the default block
of 3 is on.

| Decoding | `tiny` loops · known words | `small` loops · known words |
|---|---|---|
| greedy, no control | 6/6 · 100% | 2/6 · 95.7% |
| greedy, n-gram 3 | 0/6 · 100% | 0/6 · 96.3% |
| top-p 0.9, T=0.5, n-gram 3 | 0/24 · 97.4% | 0/24 · 96.9% |
| top-p 0.9, T=0.8, n-gram 3 | 0/24 · 92.5% | 0/24 · 92.7% |

Samples at T=0.5 (seed 1000):

- `tiny`, `First Citizen:`: "So shall you go, my lord, my gracious lord, And in the malice of thy father's love, and the
  rest? This is this. ESCALUS: I'll not be, I have been as weed with"
- `small`, `First Citizen:`: "How? Citizen: I will, I know not. First Citizens: I have not yet been, and so, I'll tell you
  to be done. COMINIUS: I'ld have you, I am not to do, That you have"

## Findings

1. Size helps, but only a little here. A model 4.4 times larger trained for 1.5 times as many
   epochs is 4.4% better in bits per character, at about 14 times the training time.
2. The larger model loops less without any control (greedy: 2/6 prompts vs 6/6). With the default
   n-gram block both are loop-free, and the share of real words is the same (T=0.8: 92.7% vs 92.5%).
   Reading the samples, neither is clearly better. Both produce a speaker line, then phrases that
   are locally plausible and globally meaningless.
3. The data is now a limit as well. `small`'s gap between train and validation loss is 3.5 times
   `tiny`'s and grew every epoch (validation gained only 0.010 bpc in the last epoch while train
   loss fell 0.21 nats). With 1.1 MB of text, more parameters mostly buy memorisation.
4. Keep `tiny` (context 32) as the UI default. `small` is worth it for the CLI factory when there
   is several times more text than Tiny Shakespeare and hours of CPU time are available.

## Limits

- One run per preset and one dataset (English). Hebrew was not measured at `small` size.
- No dropout or weight-decay sweep for `small`; stronger regularisation might narrow the gap.
- 50-token samples; "known words" measures spelling, not meaning.

## Reproduce

```bash
pnpm lm train --data tiny.txt --tokenizer bpe --preset tiny  --epochs 4 --out models/en-bpe
pnpm lm train --data tiny.txt --tokenizer bpe --preset small --epochs 6 --out models/en-bpe-small
# if the run is interrupted:
pnpm lm train --resume models/en-bpe-small
```
