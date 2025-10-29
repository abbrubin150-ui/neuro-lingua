# Advanced Loss Catalogue

This document summarises the loss functions implemented in `src/losses/advanced.ts`
and provides a brief discussion of when each loss is advantageous.

## Focal Loss

Designed for dense detection tasks with extreme class imbalance. The factor
$(1 - p_t)^\gamma$ down-weights confident predictions, letting the model focus on
hard negatives without requiring complex sampling heuristics. The implementation
supports per-class weighting via the parameter $\alpha$.

## Label Smoothing Cross-Entropy

Replaces one-hot targets with a convex combination of the gold label and a
uniform prior. The smoothed targets prevent the network from becoming over-
confident, improve calibration, and act as a regulariser for over-parameterised
models.

## Symmetric Cross-Entropy

Combines the standard forward cross-entropy with its reverse counterpart to
handle label noise. By symmetrising the divergence we avoid degenerate cases
where corrupted labels dominate training.

## Cosine Embedding Loss

Measures similarity between paired embeddings. Positive pairs are driven toward
cosine similarity one, while negative pairs are only penalised if their cosine
similarity exceeds a margin. This formulation is particularly effective when the
feature extractor is shared across modalities.
