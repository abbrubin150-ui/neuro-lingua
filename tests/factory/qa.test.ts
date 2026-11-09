// @vitest-environment node
import { describe, expect, it } from 'vitest';
import { runModelFactory, type FactoryConfig } from '../../src/factory/pipeline';
import { dataParallelBackend } from '../../src/factory/node/parallel';
import { DEFAULT_QA_TEMPLATE, qaDocument, qaPrompt } from '../../src/data/qa';
import { renderModelCard } from '../../src/lib/modelPackage';
import { complete } from '../../src/factory/node/server';

const PAIRS = [
  ['what color is the sky', 'the sky is blue'],
  ['what color is grass', 'grass is green'],
  ['what color is snow', 'snow is white'],
  ['what color is coal', 'coal is black'],
  ['what color is blood', 'blood is red'],
  ['what color is the sun', 'the sun is yellow']
];
// Repeated pairs are removed by de-duplication, so vary the wording slightly per copy.
const JSONL = Array.from({ length: 4 }, (_, k) =>
  PAIRS.map(([question, answer]) =>
    JSON.stringify({ question: `${question}${'?'.repeat(k + 1)}`, answer: `${answer}.` })
  ).join('\n')
).join('\n');

const CONFIG: FactoryConfig = {
  name: 'qa-test',
  architecture: 'transformer',
  tokenizer: { mode: 'unicode' },
  model: {
    hiddenSize: 32,
    contextSize: 8,
    dropout: 0,
    transformer: {
      numLayers: 2,
      numHeads: 4,
      numKVHeads: 2,
      ffHiddenDim: 64,
      attentionDropout: 0,
      dropConnectRate: 0,
      batchSize: 4,
      trainingStride: 1
    }
  },
  training: { epochs: 25, learningRate: 0.01, warmupSteps: 0, patience: 0, seed: 3 },
  data: { validationFraction: 0.2 },
  qa: DEFAULT_QA_TEMPLATE
};

const docs = () => [qaDocument('colors.jsonl', JSONL)];

describe('question–answer training', () => {
  it('learns the answers, scores only them and records the chat template', async () => {
    const result = await runModelFactory(docs(), CONFIG);
    const card = result.card;
    expect(card.chat).toEqual({ format: 'qa', questionLabel: 'Question:', answerLabel: 'Answer:' });
    expect(renderModelCard(card)).toContain('## Chat');

    const first = result.history[0];
    const last = result.history[result.history.length - 1];
    expect(last.validationLoss).toBeLessThan(first.validationLoss);

    // A trained question gets its answer (greedy decoding).
    const answer = await result.model.generate(qaPrompt('what color is grass?'), 4, 0.05, 1);
    expect(answer).toContain('green');

    // The local server's chat endpoint asks with the model's labels and stops at the next question.
    const reply = await complete(
      result.model,
      {
        messages: [
          { role: 'system', content: 'ignored' },
          { role: 'user', content: 'what color is snow?' }
        ],
        max_tokens: 12,
        temperature: 0.05,
        top_k: 1
      },
      512,
      card.chat
    );
    expect(reply.text).toContain('white');
    expect(reply.text).not.toContain('Question');
    expect(reply.text).toBe(reply.text.trim());
  }, 120000);

  it('in-process and data-parallel Q&A training agree (no dropout)', async () => {
    const config = { ...CONFIG, training: { ...CONFIG.training, epochs: 2 } };
    const single = await runModelFactory(docs(), config);
    const parallel = await runModelFactory(docs(), config, {}, undefined, {
      backend: dataParallelBackend({ workers: 2 })
    });
    single.history.forEach((h, i) => {
      expect(parallel.history[i].trainLoss).toBeCloseTo(h.trainLoss, 9);
      expect(parallel.history[i].validationLoss).toBeCloseTo(h.validationLoss, 9);
    });
  }, 120000);
});
