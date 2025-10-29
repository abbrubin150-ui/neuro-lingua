# Experiment Logbook

All experiments were executed via `python scripts/train_experiment.py` after generating the sample corpora with `scripts/data/build_corpus.py`. Metrics are stored under `docs/experiments/runs/` together with per-run model exports in `models/experiments/`.

| Experiment | Dataset | Epochs | Hidden | Optimizer | Dropout | Seed | Final Loss | Accuracy |
|------------|---------|--------|--------|-----------|---------|------|------------|----------|
| `hebrew_news_baseline` | Hebrew news sample | 15 | 56 | momentum | 0.15 | 2024 | 1.40 | 0.62 |
| `wikitext_baseline` | WikiText sample | 12 | 48 | momentum | 0.10 | 3407 | 4.04 | 0.03 |
| `wikitext_dropout` | WikiText sample | 12 | 64 | adam | 0.25 | 3407 | 4.01 | 0.06 |

## Observations

- The custom Hebrew regex tokenizer paired with a lower context window yielded a steady accuracy improvement across epochs, with momentum providing stable convergence despite the tiny corpus.
- Increasing dropout on the WikiText sample with Adam recovered from early volatility and produced a modest accuracy lift compared to the baseline, suggesting regularisation helps even on small vocabularies.
- Momentum-based training on WikiText plateaued quickly; future runs should increase corpus size or context length to capture richer structure.

## Reproduction Checklist

1. Build the corpora (sample mode):
   ```bash
   python scripts/data/build_corpus.py --dataset wikitext --mode sample
   python scripts/data/build_corpus.py --dataset hebrew_news --mode sample
   ```
2. Launch the experiment suite:
   ```bash
   python scripts/train_experiment.py
   ```
3. Inspect metrics and artifacts:
   - Metrics JSON: `docs/experiments/runs/<experiment>.json`
   - Models: `models/experiments/<experiment>.json`
   - Summary: `docs/experiments/runs/summary.json`
