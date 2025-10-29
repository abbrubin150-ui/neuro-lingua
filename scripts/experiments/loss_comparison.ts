#!/usr/bin/env tsx
import {
  focalLoss,
  labelSmoothingCrossEntropy,
  symmetricCrossEntropy,
  cosineEmbeddingLoss
} from '../../src/losses/advanced';

function demoFocalLoss() {
  const logits = [2.5, -1.2, 0.3];
  const targets = [1, 0, 0];
  console.log('Focal loss:', focalLoss(logits, targets, { gamma: 2, alpha: 0.5 }).toFixed(4));
}

function demoLabelSmoothing() {
  const logits = [0.1, 1.4, -0.2, 0.7];
  console.log(
    'Label smoothing cross-entropy:',
    labelSmoothingCrossEntropy(logits, 1, 4, 0.1).toFixed(4)
  );
}

function demoSymmetricCE() {
  const logits = [0.9, -0.7, 0.2];
  const targets = [0.7, 0.2, 0.1];
  console.log('Symmetric cross-entropy:', symmetricCrossEntropy(logits, targets).toFixed(4));
}

function demoCosineEmbedding() {
  const x = [0.1, 0.8, -0.3];
  const y = [0.0, 0.7, -0.4];
  const z = [-0.5, 0.2, 0.1];
  console.log('Cosine embedding (positive pair):', cosineEmbeddingLoss(x, y, 1).toFixed(4));
  console.log('Cosine embedding (negative pair):', cosineEmbeddingLoss(x, z, -1, 0.1).toFixed(4));
}

demoFocalLoss();
demoLabelSmoothing();
demoSymmetricCE();
demoCosineEmbedding();
