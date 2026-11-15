# Generation loops: repetition penalty vs. no-repeat n-grams

Measured on 2026-11-09 to choose the decoding defaults. Every number below comes from
`scripts/experiments/generation_repetition.ts` run on models trained with `pnpm lm train`.
Seeds are fixed; samples are short (see "Limits").

## Setup

| Model | Data | Tokenizer | Parameters | Held-out (10%) |
|---|---|---|---|---|
| en-char | Tiny Shakespeare (1.1 MB) | character, 68 symbols | 74,628 | PPL 4.51 / char, **2.159 bpc** |
| en-bpe | Tiny Shakespeare | BPE, 2,004 tokens | 324,372 | PPL 79.2 / token, **1.959 bpc** |
| he-char | Torah without niqqud (Sefaria "Tanach with Text Only", 380 K chars) | character, 32 symbols | 69,984 | PPL 5.67 / char, **2.495 bpc** |

All three: `--preset tiny` (D=64, context 32 tokens, 2 layers), 4 epochs, seed 1337. Perplexities
of different tokenizers are not comparable (a BPE token is several characters); bits per character
on the same split are.

Prompts: six per language (`ROMEO:`, `First Citizen:`, … / `שמע ישראל`, `ויאמר אלהים`,
`וידבר יהוה אל משה`, …). Greedy = top-k 1 (one run per prompt); sampling = top-p 0.9 at T=0.5
and T=0.8 (four seeds per prompt). 150 characters (character models) or 50 tokens (BPE).

Metrics, over whitespace-separated words of the generated text:

- **loops**: samples that repeat any word 4-gram
- **rep4**: share of word 4-grams that already occurred earlier in the same sample
- **known**: share of generated words that occur in the training text (catches a control that
  breaks spelling)

## Results

### Greedy decoding

| Control | en-char loops / rep4 / known | he-char loops / rep4 / known | en-bpe loops / rep4 / known |
|---|---|---|---|
| none | 6/6 · 82.5% · 97.8% | 6/6 · 80.9% · 98.5% | 6/6 · 30.4% · 100% |
| repetition penalty θ=1.2 | 5/6 · 22.2% · 96.8% | 6/6 · 64.0% · 99.0% | 0/6 · 0% · 99.5% |
| repetition penalty θ=1.3 | 2/6 · 3.1% · 97.8% | 6/6 · 63.3% · 98.0% | 0/6 · 0% · 99.1% |
| no-repeat n-gram 3 | | | **0/6 · 0% · 100%** |
| no-repeat n-gram 4 | | | 0/6 · 0% · 94.1% |
| no-repeat n-gram 6 | | | 3/6 · 2.9% · 98.4% |
| no-repeat n-gram 12 | 6/6 · 3.6% · 98.9% | 0/6 · 0% · 99.5% | |
| no-repeat n-gram 16 | **0/6 · 0% · 98.4%** | **0/6 · 0% · 96.9%** | |
| no-repeat n-gram 24 | 6/6 · 15.6% · 97.9% | 6/6 · 14.9% · 99.0% | |

### Sampling, top-p 0.9: share of known words (loops)

| Control | en-char T0.5 | en-char T0.8 | he-char T0.5 | he-char T0.8 | en-bpe T0.5 | en-bpe T0.8 |
|---|---|---|---|---|---|---|
| none | 97.4% (0/24) | 92.7% | 96.3% (**2/24**) | 93.6% | 97.1% (**1/24**) | 92.3% |
| θ=1.2 | 96.6% | **89.0%** | 96.6% | **87.7%** | 95.8% | **90.7%** |
| θ=1.3 | 94.4% | **86.6%** | 95.5% | **84.4%** | 95.9% | **90.3%** |
| n-gram 16 (char) / 3 (BPE) | 97.4% | 92.7% | 96.6% (0/24) | 93.6% | 97.4% (0/24) | 92.5% |

At T=0.8 no configuration looped.

## Findings

1. Greedy decoding loops on every prompt for all three models. This is the "the complain the
   complain" / "ויאמר אלהים את האדם ויאמר אלהים" failure.
2. A no-repeat n-gram block removes the loops without hurting spelling. However, its length has
   to fit the tokenizer: about two words. That is 16 for characters (12 and 24 leave loops with
   another period) and 3 for BPE (4 blocks pieces inside words: "wedom", "baws").
3. The CTRL repetition penalty does not fit small vocabularies. With characters every common
   letter and the space are always in the window, so the penalty lowers them too. It did not stop
   the Hebrew loops even at θ=1.3, and under sampling it lowered the share of real words in all
   three models (T=0.8: 92.7 → 89.0%, 93.6 → 87.7%, 92.3 → 90.7% at θ=1.2).
4. The block breaks exact loops; it does not make the text good. Greedy output still circles the
   model's most likely phrase ("the stand the country the stand the man …"). Sampling at T≈0.5–0.8
   gives better text than greedy with any control.
5. The prompt has little influence with the `tiny` preset. Its 32-character context is about six
   words, and with the same seed different Hebrew prompts produced nearly the same continuation.
   A longer context (`--preset small`, 64) is the lever here.
6. BPE vs. characters, same data and split: 1.959 vs 2.159 bits per character (9% better). The
   BPE model has 4.3× the parameters, almost all of them in the embedding and output layers.

## Defaults chosen

| Setting | Default | Where |
|---|---|---|
| no-repeat n-gram | 16 for `character`, 3 for every other tokenizer | `recommendedNoRepeatNgram()` in `src/generation/sampling.ts`; UI, `pnpm lm generate`, `pnpm lm serve` |
| repetition penalty | 1 (off) | `DEFAULT_GENERATION.repetitionPenalty` |

The value 3 for word-level tokenizers (`unicode`, `ascii`, `custom`) is the usual word-trigram
block. It was not measured here.

## Limits

- One run per configuration and seed set; small models (70–320 K parameters) trained for 4 epochs.
- Samples are short (150 characters ≈ 30 words; 50 BPE tokens), so loops with a period longer than
  the sample are not counted.
- "known" measures spelling, not meaning. A sample can be all real words and still be nonsense.

## Reproduce

```bash
pnpm lm train --data tiny.txt --tokenizer character --preset tiny --epochs 4 --out models/en-char
THETAS=1.2,1.3 NGRAMS=12,16,24 pnpm exec tsx scripts/experiments/generation_repetition.ts \
  models/en-char/model.nlm.json tiny.txt "ROMEO:|First Citizen:|KING HENRY:|What say you|I will not|Now is the" 4 150
```
