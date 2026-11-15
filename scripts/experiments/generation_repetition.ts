/**
 * Loops vs. damage of decoding settings on a trained package
 * (results: docs/benchmarks/GENERATION_REPETITION.md).
 *
 *   THETAS=1.2,1.3 NGRAMS=12,16,24 pnpm exec tsx scripts/experiments/generation_repetition.ts \
 *     <model.nlm.json> <training corpus> "<prompt1>|<prompt2>|..." [samples=6] [maxTokens=120]
 *
 * Env: THETAS (repetition penalties), NGRAMS (no-repeat n-gram sizes), COMBOS ("θ:n,..."),
 * SHOW (configurations whose first sample per prompt is printed).
 * rep4   = share of word 4-grams in a sample that already occurred earlier in that sample
 * dist2  = distinct word bigrams / bigrams
 * known  = share of generated words that occur in the training corpus (spelling damage guard)
 * loopy  = share of samples with any word 4-gram repeated
 */
import { readFileSync } from 'node:fs';
import { loadModelFile } from '../../src/factory/node/files';
import type { RepetitionControls } from '../../src/generation/sampling';

const [modelPath, corpusPath, promptArg, samplesArg, maxArg] = process.argv.slice(2);
const samples = Number(samplesArg ?? 6);
const maxTokens = Number(maxArg ?? 120);
const prompts = promptArg.split('|');
const vocab = new Set(
  readFileSync(corpusPath, 'utf8')
    .toLowerCase()
    .split(/[^\p{L}']+/u)
    .filter(Boolean)
);
const words = (t: string) =>
  t
    .toLowerCase()
    .split(/[^\p{L}']+/u)
    .filter(Boolean);

interface Cfg {
  name: string;
  temperature: number;
  topK: number;
  topP: number;
  rep: RepetitionControls;
}
const modes: [string, number, number, number][] = [
  ['greedy', 1, 1, 0],
  ['T0.5', 0.5, 0, 0.9],
  ['T0.8', 0.8, 0, 0.9]
];
const reps: [string, RepetitionControls][] = [['', {}]];
for (const t of (process.env.THETAS ?? '').split(',').filter(Boolean).map(Number))
  reps.push([`θ=${t}`, { repetitionPenalty: t }]);
for (const g of (process.env.NGRAMS ?? '').split(',').filter(Boolean).map(Number))
  reps.push([`ngram=${g}`, { noRepeatNgramSize: g }]);
for (const c of (process.env.COMBOS ?? '').split(',').filter(Boolean)) {
  const [t, g] = c.split(':').map(Number);
  reps.push([`θ=${t}+ngram=${g}`, { repetitionPenalty: t, noRepeatNgramSize: g }]);
}
const cfgs: Cfg[] = [];
for (const [m, temperature, topK, topP] of modes)
  for (const [r, rep] of reps)
    cfgs.push({ name: `${m} ${r}`.trim(), temperature, topK, topP, rep });

const { model } = await loadModelFile(modelPath);
const showFor = new Set((process.env.SHOW ?? 'greedy,greedy θ=1.2').split(','));
for (const c of cfgs) {
  let rep = 0,
    tot4 = 0,
    d2 = 0,
    tot2 = 0,
    known = 0,
    totw = 0,
    loopy = 0,
    runs = 0;
  const shown: string[] = [];
  for (const p of prompts) {
    const reps = c.topK === 1 ? 1 : samples;
    for (let s = 0; s < reps; s++) {
      model.reseed(1000 + s);
      const text = await model.generate(
        p,
        maxTokens,
        c.temperature,
        c.topK,
        c.topP,
        0,
        0,
        0,
        0,
        0.1,
        c.rep
      );
      if (s === 0 && showFor.has(c.name))
        shown.push(`  [${p}] → ${text.replace(/\s+/g, ' ').slice(0, 160)}`);
      const w = words(text);
      const seen = new Set<string>();
      let r = 0;
      for (let i = 0; i + 4 <= w.length; i++) {
        const g = w.slice(i, i + 4).join(' ');
        if (seen.has(g)) r++;
        seen.add(g);
        tot4++;
      }
      rep += r;
      if (r > 0) loopy++;
      const bi = new Set<string>();
      for (let i = 0; i + 2 <= w.length; i++) bi.add(w[i] + ' ' + w[i + 1]);
      d2 += bi.size;
      tot2 += Math.max(0, w.length - 1);
      for (const x of w) if (vocab.has(x)) known++;
      totw += w.length;
      runs++;
    }
  }
  const f = (x: number) => (x * 100).toFixed(1).padStart(5) + '%';
  console.log(
    `${c.name.padEnd(26)} rep4 ${f(rep / Math.max(1, tot4))}  dist2 ${f(d2 / Math.max(1, tot2))}  known ${f(known / Math.max(1, totw))}  loopy ${loopy}/${runs}`
  );
  for (const s of shown) console.log(s);
}
