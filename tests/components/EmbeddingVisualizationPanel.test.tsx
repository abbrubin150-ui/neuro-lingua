import { describe, expect, it, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

const projections = vi.hoisted(() => ({
  umap: vi.fn((points: number[][]) => ({
    coordinates: points.map((_, i) => [i, -i]),
    summary: { min: [0, -points.length], max: [points.length, 0], mean: [0, 0] },
    metadata: { method: 'umap' }
  })),
  tsne: vi.fn((points: number[][]) => ({
    coordinates: points.map((_, i) => [i, i]),
    summary: { min: [0, -points.length], max: [points.length, 0], mean: [0, 0] },
    metadata: { method: 'tsne' }
  }))
}));

vi.mock('../../src/visualization/embeddings', () => ({
  projectWithUMAP: projections.umap,
  projectWithTSNE: projections.tsne
}));

import {
  EmbeddingVisualizationPanel,
  selectProjectionTokens
} from '../../src/components/EmbeddingVisualizationPanel';
import type { ProNeuralLM } from '../../src/lib/ProNeuralLM';

function fakeModel(vocab: string[]): ProNeuralLM {
  return {
    getVocab: () => vocab,
    getEmbeddings: () => vocab.map((_, i) => [i, i * 2, i * 3])
  } as unknown as ProNeuralLM;
}

beforeEach(() => {
  projections.umap.mockClear();
  projections.tsne.mockClear();
  // jsdom has no canvas implementation
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as never;
});

describe('selectProjectionTokens', () => {
  it('prefers words and subwords (in vocabulary order), then characters, never specials', () => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', '▁', '▁a', '▁the', 'ing', 'שָׁ'];
    expect(selectProjectionTokens(vocab, 4).map((i) => vocab[i])).toEqual([
      '▁the',
      'ing',
      'שָׁ',
      'a'
    ]);
    expect(selectProjectionTokens(vocab, 100)).toHaveLength(7);
  });
});

describe('EmbeddingVisualizationPanel', () => {
  it('does not project automatically: a large vocabulary must not freeze the page', async () => {
    vi.useFakeTimers();
    render(<EmbeddingVisualizationPanel model={fakeModel(Array.from({ length: 2004 }, (_, i) => `tok${i}`))} />);
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    vi.useRealTimers();
    expect(projections.umap).not.toHaveBeenCalled();
    expect(projections.tsne).not.toHaveBeenCalled();
    expect(screen.getByText(/to project 300 of\s+2004 tokens/)).toBeInTheDocument();
  });

  it('projects a bounded selection on request', async () => {
    const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ...Array.from({ length: 600 }, (_, i) => `w${i}`)];
    render(<EmbeddingVisualizationPanel model={fakeModel(vocab)} />);
    fireEvent.change(screen.getByLabelText('Tokens to project'), { target: { value: '120' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Compute projection' }));
      await new Promise((r) => setTimeout(r, 30));
    });
    expect(projections.umap).toHaveBeenCalledTimes(1);
    const points = projections.umap.mock.calls[0][0] as number[][];
    expect(points).toHaveLength(120);
    // The first selected token is "w0" (index 4): specials are skipped.
    expect(points[0]).toEqual([4, 8, 12]);
  });
});
