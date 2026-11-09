/**
 * Held-out benchmark of the three Neuro-Lingua architectures.
 *
 *   pnpm benchmark:models                 # both tasks, default epochs
 *   pnpm benchmark:models -- --quick      # fewer epochs (smoke test)
 *   pnpm benchmark:models -- --out docs/benchmarks/MODEL_BENCHMARK.md
 *
 * Task "order": a synthetic language "<a> <verb> <b> first|second <answer> end" in which the
 *   answer depends on word ORDER, not on which words occur (both orders of the same pair are
 *   equally frequent). Metric: next-token accuracy at the answer position on held-out
 *   sentences, plus validation perplexity.
 * Task "chars": character-level language modelling of data/corpus.txt (85/15 split by
 *   sentence). Metric: validation bits per character (bpc).
 *
 * Every number comes from model.evaluate() / getLogitsForContext() on text the model never
 * trained on. Seeds are fixed; timings depend on the machine.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { ProNeuralLM, type TokenizerConfig } from '../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../src/lib/AdvancedNeuralLM';
import { TransformerLM, type TransformerConfig } from '../src/lib/TransformerLM';

const args = process.argv.slice(2);
const quick = args.includes('--quick');
const outIdx = args.indexOf('--out');
const outPath = outIdx >= 0 ? args[outIdx + 1] : null;

type Builder = (vocab: string[], context: number, tokenizer?: TokenizerConfig) => ProNeuralLM;

const TRANSFORMER: TransformerConfig = {
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  ffHiddenDim: 96,
  attentionDropout: 0.05,
  dropConnectRate: 0,
  batchSize: 8
};

const MODELS: { name: string; build: Builder }[] = [
  {
    name: 'MLP (ProNeuralLM, H=64)',
    build: (v, c, t) => new ProNeuralLM(v, 64, 0.003, c, 'adam', 0.9, 0.05, 1, t)
  },
  {
    name: 'Advanced (GELU + LayerNorm, H=64)',
    build: (v, c, t) =>
      new AdvancedNeuralLM(v, 64, 0.003, c, 'adam', 0.9, 0.05, 1, t, {
        activation: 'gelu',
        useLayerNorm: true,
        lrSchedule: 'constant'
      })
  },
  {
    name: 'Transformer causal (D=48, 2 layers, GQA 4/2)',
    build: (v, c, t) => new TransformerLM(v, 48, 0.003, c, 'adam', 0.9, 0.05, 1, t, TRANSFORMER)
  },
  {
    name: 'Transformer pooled (legacy objective)',
    build: (v, c, t) =>
      new TransformerLM(v, 48, 0.003, c, 'adam', 0.9, 0.05, 1, t, {
        ...TRANSFORMER,
        objective: 'pooled'
      })
  }
];

// ---------------------------------------------------------------------------
// Task "order"
// ---------------------------------------------------------------------------

const NAMES = ['alice', 'bob', 'carol', 'dave', 'erin', 'frank', 'grace', 'heidi'];
const VERBS = ['likes', 'sees', 'helps', 'calls', 'meets', 'thanks'];

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
}

function orderExamples(n: number, seed: number): string[][] {
  const r = lcg(seed);
  const out: string[][] = [];
  for (let i = 0; i < n; i++) {
    const a = NAMES[Math.floor(r() * NAMES.length)];
    let b = a;
    while (b === a) b = NAMES[Math.floor(r() * NAMES.length)];
    const verb = VERBS[Math.floor(r() * VERBS.length)];
    const q = r() < 0.5 ? 'first' : 'second';
    out.push([a, verb, b, q, q === 'first' ? a : b, 'end']);
  }
  return out;
}

async function answerAccuracy(model: ProNeuralLM, examples: string[][], context: number) {
  const stream: string[] = Array(context).fill('<BOS>');
  let correct = 0;
  for (const ex of examples) {
    for (let k = 0; k < ex.length; k++) {
      if (k === 4) {
        const ctx = stream.slice(-context).map((t) => Math.max(0, model.getTokenIndex(t)));
        const logits = await model.getLogitsForContext(ctx);
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        if (model.getVocab()[best] === ex[4]) correct++;
      }
      stream.push(ex[k]);
    }
  }
  return correct / examples.length;
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

type Row = { model: string; metric: string; secondary: string; secondsPerEpoch: string };

async function trainTimed(model: ProNeuralLM, text: string, epochs: number) {
  const t0 = performance.now();
  for (let e = 0; e < epochs; e++) await model.train(text, 1);
  return (performance.now() - t0) / 1000 / epochs;
}

async function runOrderTask(epochs: number): Promise<Row[]> {
  const train = orderExamples(300, 1);
  const val = orderExamples(100, 2);
  const trainText = train.map((e) => e.join(' ')).join(' ');
  const valText = val.map((e) => e.join(' ')).join(' ');
  const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', ...NAMES, ...VERBS, 'first', 'second', 'end'];
  const context = 6;
  const rows: Row[] = [];
  for (const spec of MODELS) {
    const model = spec.build(vocab, context);
    const seconds = await trainTimed(model, trainText, epochs);
    const accuracy = await answerAccuracy(model, val, context);
    const evalResult = await model.evaluate(valText);
    rows.push({
      model: spec.name,
      metric: `${(accuracy * 100).toFixed(1)}%`,
      secondary: evalResult.perplexity.toFixed(2),
      secondsPerEpoch: seconds.toFixed(1)
    });
    console.log(`[order] ${spec.name}: answer acc ${rows[rows.length - 1].metric}`);
  }
  return rows;
}

async function runCharTask(epochs: number): Promise<Row[]> {
  const text = readFileSync(new URL('../data/corpus.txt', import.meta.url), 'utf8');
  const sentences = text.split(/(?<=[.!?])\s+/);
  const cut = Math.floor(sentences.length * 0.85);
  const train = sentences.slice(0, cut).join(' ');
  const val = sentences.slice(cut).join(' ');
  const tokenizer: TokenizerConfig = { mode: 'character' };
  const vocab = [
    '<PAD>',
    '<BOS>',
    '<EOS>',
    '<UNK>',
    ...Array.from(new Set(ProNeuralLM.tokenizeText(train, tokenizer)))
  ];
  const context = 8;
  const rows: Row[] = [];
  for (const spec of MODELS) {
    const model = spec.build(vocab, context, tokenizer);
    const seconds = await trainTimed(model, train, epochs);
    const evalVal = await model.evaluate(val);
    const evalTrain = await model.evaluate(train);
    rows.push({
      model: spec.name,
      metric: (evalVal.loss / Math.LN2).toFixed(2),
      secondary: (evalTrain.loss / Math.LN2).toFixed(2),
      secondsPerEpoch: seconds.toFixed(1)
    });
    console.log(`[chars] ${spec.name}: val bpc ${rows[rows.length - 1].metric}`);
  }
  return rows;
}

function table(headers: string[], rows: string[][]): string {
  return [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map((r) => `| ${r.join(' | ')} |`)
  ].join('\n');
}

async function main() {
  const orderEpochs = quick ? 3 : 12;
  const charEpochs = quick ? 1 : 6;
  const order = await runOrderTask(orderEpochs);
  const chars = await runCharTask(charEpochs);

  const report = [
    `## Order task (${orderEpochs} epochs, held-out sentences)`,
    '',
    table(
      ['Model', 'Answer accuracy', 'Validation perplexity', 's / epoch'],
      order.map((r) => [r.model, r.metric, r.secondary, r.secondsPerEpoch])
    ),
    '',
    `## Character-level corpus.txt (${charEpochs} epochs, 85/15 split)`,
    '',
    table(
      ['Model', 'Validation bpc', 'Training bpc', 's / epoch'],
      chars.map((r) => [r.model, r.metric, r.secondary, r.secondsPerEpoch])
    ),
    ''
  ].join('\n');

  console.log(`\n${report}`);
  if (outPath) {
    writeFileSync(outPath, report);
    console.log(`Written to ${outPath}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
