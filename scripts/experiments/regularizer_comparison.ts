#!/usr/bin/env tsx
import { MiniTransformerBlock } from '../../src/models/mini_transformer';
import { batchRenormalize, BatchRenormState, applyDropConnect } from '../../src/models/regularizers';

function randomMatrix(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * 0.2)
  );
}

function createRenormState(features: number): BatchRenormState {
  return {
    runningMean: new Array(features).fill(0),
    runningVar: new Array(features).fill(1),
    momentum: 0.9,
    epsilon: 1e-5,
    rMax: 3,
    dMax: 5
  };
}

function runDropConnectDemo(rate: number) {
  const weights = randomMatrix(4, 4);
  const dropped = applyDropConnect(weights, { rate, seed: 42 });
  const sparsity =
    dropped.flat().filter((value) => value === 0).length / (dropped.length * dropped[0].length);
  console.log(`DropConnect rate=${rate.toFixed(2)} -> empirical sparsity=${sparsity.toFixed(2)}`);
}

function runBatchRenormDemo() {
  const inputs = randomMatrix(8, 4);
  const state = createRenormState(4);
  const { normalized, r, d } = batchRenormalize(inputs, state);
  console.log('Batch ReNorm stats:', {
    r: r.map((v) => Number(v.toFixed(3))),
    d: d.map((v) => Number(v.toFixed(3)))
  });
  console.log('First row before/after:', inputs[0], normalized[0].map((v) => Number(v.toFixed(3))));
}

function runTransformerDemo() {
  const renormState = createRenormState(8);
  const block = new MiniTransformerBlock({
    modelDim: 8,
    heads: 2,
    ff: { hiddenDim: 16 },
    attentionDropout: 0.1,
    dropConnectRate: 0.2,
    renormState
  });

  const inputs = randomMatrix(6, 8);
  const attentionWeights = {
    query: randomMatrix(8, 8),
    key: randomMatrix(8, 8),
    value: randomMatrix(8, 8)
  };
  const ff1 = randomMatrix(8, 16);
  const ff2 = randomMatrix(16, 8);
  const outputs = block.forward(inputs, attentionWeights, ff1, ff2);
  console.log('MiniTransformer output sample:', outputs[0].map((v) => Number(v.toFixed(3))));
}

runDropConnectDemo(0.1);
runDropConnectDemo(0.5);
runBatchRenormDemo();
runTransformerDemo();
