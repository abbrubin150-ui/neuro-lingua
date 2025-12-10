# Dataset Pipeline Overview

This directory hosts the data lifecycle for the Neuro-Lingua experiments. Raw artifacts live in `data/raw/`, processed corpora in `data/processed/`, and experiment-specific metadata in `data/experiments/` (created on demand).

Three corpora are supported out-of-the-box:

- **Wikitext** – encyclopaedic English paragraphs derived from the WikiText project.
- **Hebrew News** – short Hebrew news briefs spanning technology, weather, sports, health, and transport beats.
- **Hebrew Opinion** – argumentative op-eds that include thesis, primary argumentation, counterpoints, and calls to action.

Use the helpers under `scripts/data/` to download, clean, split, and summarise the corpora. Each script writes both the text files required by the training stack and a small JSON metadata report with token statistics, document counts, and the random seed used for reproducibility.

```
python scripts/data/build_corpus.py --dataset wikitext --mode sample
python scripts/data/build_corpus.py --dataset hebrew_news --mode sample
python scripts/data/build_corpus.py --dataset hebrew_opinion --mode sample
```

Passing `--mode full` will attempt to download the publicly available sources listed inside the script (requires the `datasets` and `requests` Python packages). Processed files are written beneath `data/processed/<dataset>/` and named `train`, `validation`, and `test`.
