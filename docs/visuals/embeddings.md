# Embedding Projections

Embedding projections are generated via `npx tsx scripts/visualize_embeddings.ts <model>`, which calls the helpers in `src/visualization/embeddings.ts` to produce paired t-SNE and UMAP layouts. Each JSON artifact contains the 2D coordinates, projection metadata, and a small vocabulary sample for quick inspection.

| Model | Method | Summary | Artifact |
|-------|--------|---------|----------|
| `hebrew_news_baseline` | t-SNE | Tight clustering of topical words with mean coordinate near the origin. | `docs/visuals/hebrew_news_baseline-tsne.json` |
| `hebrew_news_baseline` | UMAP | Reveals a wider spread while maintaining separation between technology and weather terms. | `docs/visuals/hebrew_news_baseline-umap.json` |
| `wikitext_dropout` | t-SNE | Points remain dispersed, highlighting the sparse wiki vocabulary learned from the toy corpus. | `docs/visuals/wikitext_dropout-tsne.json` |
| `wikitext_dropout` | UMAP | Compresses the vocabulary cloud and hints at thematic subclusters despite limited data. | `docs/visuals/wikitext_dropout-umap.json` |

To regenerate the projections, run:

```bash
npx tsx scripts/visualize_embeddings.ts models/experiments/hebrew_news_baseline.json
npx tsx scripts/visualize_embeddings.ts models/experiments/wikitext_dropout.json
```
