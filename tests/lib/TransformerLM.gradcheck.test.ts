import { describe, expect, it } from 'vitest';
import { TransformerLM, type TransformerConfig } from '../../src/lib/TransformerLM';

/**
 * Model-level finite-difference check of TransformerLM: token embeddings → transformer blocks →
 * final RMSNorm → per-position output projection → mean next-token cross-entropy over the scored
 * positions of one window. Gradients are read from the internal accumulators.
 */

type Internals = {
  embedding: number[][];
  wOutput: number[][];
  bOutput: number[];
  attentionWeights: { query: number[][]; key: number[][]; value: number[][] }[];
  ffWeights1: number[][][];
  ffWeights2: number[][][];
  renormStates: { attention: { gamma: number[] }; ffn: { gamma: number[] } }[];
  finalNorm: { gamma: number[] };
  gradBuffers: {
    wOutput: number[][];
    bOutput: number[];
    finalGamma: number[] | null;
    layers: {
      query: number[][];
      key: number[][];
      value: number[][];
      ff1: number[][];
      ff2: number[][];
      attnGamma: number[];
      ffnGamma: number[];
    }[];
    embedding: Map<number, number[]>;
  } | null;
  batchDropConnect: unknown;
  stackForward(inputs: number[], train: boolean, scored?: number[], useDC?: boolean): Pass;
  accumulateGradients(inputs: number[], pass: Pass, targets: number[]): number;
  sampleBatchDropConnect(): unknown;
  finalizeDropConnectGradients(g: unknown): void;
};
type Pass = { probs: number[][] };

const VOCAB = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'a', 'b', 'c', 'd', 'e'];

function build(config: TransformerConfig) {
  const model = new TransformerLM(VOCAB, 8, 0.01, 4, 'adam', 0.9, 0, 5, undefined, {
    numLayers: 2,
    numHeads: 4,
    numKVHeads: 2,
    ffHiddenDim: 6,
    attentionDropout: 0,
    dropConnectRate: 0,
    ...config
  });
  const m = model as unknown as Internals;
  // Non-trivial gains so that their gradients are exercised.
  let s = 3;
  const rnd = () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648 - 0.5;
  };
  m.finalNorm.gamma = m.finalNorm.gamma.map(() => 1 + rnd() * 0.4);
  m.renormStates.forEach((r) => {
    r.attention.gamma = r.attention.gamma.map(() => 1 + rnd() * 0.4);
    r.ffn.gamma = r.ffn.gamma.map(() => 1 + rnd() * 0.4);
  });
  m.wOutput = m.wOutput.map((row) => row.map(() => rnd()));
  return m;
}

function check(m: Internals, useDropConnect: boolean) {
  const ctx = [4, 5, 4, 6]; // token 'a' repeats
  const scored = [0, 1, 2, 3];
  const targets = [5, 4, 6, 7];

  if (useDropConnect) m.batchDropConnect = m.sampleBatchDropConnect();
  const dc = m.batchDropConnect;
  const loss = () => {
    // Rebuild W̃ = W ⊙ M from the current (possibly perturbed) weights with the same masks.
    if (useDropConnect) {
      const layers = dc as { masks: Record<string, number[][] | null>; masked: unknown }[];
      layers.forEach((layer, l) => {
        const att = m.attentionWeights[l];
        const mask = (W: number[][], M: number[][] | null) =>
          M ? W.map((row, i) => row.map((v, j) => v * M[i][j])) : W;
        layer.masked = {
          query: mask(att.query, layer.masks.query),
          key: mask(att.key, layer.masks.key),
          value: mask(att.value, layer.masks.value)
        };
      });
      m.batchDropConnect = layers;
    }
    const pass = m.stackForward(ctx, useDropConnect, scored, useDropConnect);
    return (
      -targets.reduce((sum, t, k) => sum + Math.log(pass.probs[k][t]), 0) / targets.length
    );
  };

  loss(); // build masked weights from the initial weights
  const pass = m.stackForward(ctx, useDropConnect, scored, useDropConnect);
  m.accumulateGradients(ctx, pass, targets);
  const g = m.gradBuffers!;
  m.batchDropConnect = dc;
  m.finalizeDropConnectGradients(g);

  const h = 1e-6;
  const numeric = (get: () => number, set: (v: number) => void) => {
    const o = get();
    set(o + h);
    const lp = loss();
    set(o - h);
    const lm = loss();
    set(o);
    return (lp - lm) / (2 * h);
  };
  const close = (a: number, n: number, label: string) =>
    expect(Math.abs(a - n), label).toBeLessThan(2e-6 + 1e-4 * Math.abs(n));

  const matrices: [string, number[][], number[][]][] = [
    ['wOutput', m.wOutput, g.wOutput],
    ['L0.query', m.attentionWeights[0].query, g.layers[0].query],
    ['L1.key', m.attentionWeights[1].key, g.layers[1].key],
    ['L0.value', m.attentionWeights[0].value, g.layers[0].value],
    ['L1.ff1', m.ffWeights1[1], g.layers[1].ff1],
    ['L0.ff2', m.ffWeights2[0], g.layers[0].ff2]
  ];
  for (const [label, P, G] of matrices) {
    for (let i = 0; i < P.length; i++) {
      for (let j = 0; j < P[i].length; j++) {
        close(
          G[i][j],
          numeric(
            () => P[i][j],
            (v) => (P[i][j] = v)
          ),
          `${label}[${i}][${j}]`
        );
      }
    }
  }
  const vectors: [string, number[], number[]][] = [
    ['bOutput', m.bOutput, g.bOutput],
    ['finalGamma', m.finalNorm.gamma, g.finalGamma!],
    ['L1.attnGamma', m.renormStates[1].attention.gamma, g.layers[1].attnGamma]
  ];
  for (const [label, P, G] of vectors) {
    for (let i = 0; i < P.length; i++) {
      close(
        G[i],
        numeric(
          () => P[i],
          (v) => (P[i] = v)
        ),
        `${label}[${i}]`
      );
    }
  }
  for (const tok of [4, 5, 6]) {
    const row = m.embedding[tok];
    const grad = g.embedding.get(tok)!;
    for (let i = 0; i < row.length; i++) {
      close(
        grad[i],
        numeric(
          () => row[i],
          (v) => (row[i] = v)
        ),
        `embedding[${tok}][${i}]`
      );
    }
  }
}

describe('TransformerLM backpropagation (model level)', () => {
  it('matches finite differences for the causal objective (all positions, GQA)', () => {
    check(build({}), false);
  });

  it('matches finite differences with mini-batch DropConnect masks', () => {
    check(build({ dropConnectRate: 0.3 }), true);
  });
});
