/**
 * Neuro-Lingua offline model factory CLI.
 *
 *   pnpm lm train    --data <file|dir> [--data …] [--out models/my-model] [options]
 *   pnpm lm eval     --model <package> --data <file|dir>
 *   pnpm lm generate --model <package> --prompt "text" [--max-tokens 60] [--temperature 0.8]
 *   pnpm lm serve    --model <package> [--port 8787] [--host 127.0.0.1] [--cors]
 *   pnpm lm info     --model <package>
 *
 * Everything runs locally on the CPU from local files: no network access, no external AI API.
 * Run `pnpm lm help` for every option.
 */

import { mkdir, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import {
  runModelFactory,
  evaluateText,
  withBitsPerCharacter,
  type FactoryConfig
} from '../src/factory/pipeline';
import { readCorpusDocuments, loadModelFile } from '../src/factory/node/files';
import { createInferenceServer } from '../src/factory/node/server';
import {
  DataParallelTrainer,
  dataParallelBackend,
  parseWorkerCount
} from '../src/factory/node/parallel';
import { TransformerLM } from '../src/lib/TransformerLM';
import { packModel, renderModelCard } from '../src/lib/modelPackage';
import { prepareCorpus } from '../src/data/corpusPrep';
import type { Optimizer } from '../src/lib/ProNeuralLM';

type Args = { _: string[]; options: Map<string, string[]> };

function parseArgs(argv: string[]): Args {
  const args: Args = { _: [], options: new Map() };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const [key, inline] = a.slice(2).split('=', 2);
      const value = inline ?? (argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true');
      args.options.set(key, [...(args.options.get(key) ?? []), value]);
    } else {
      args._.push(a);
    }
  }
  return args;
}

const str = (a: Args, key: string, fallback?: string) => a.options.get(key)?.at(-1) ?? fallback;
const num = (a: Args, key: string, fallback: number) => {
  const v = str(a, key);
  const n = v === undefined ? NaN : Number(v);
  return Number.isFinite(n) ? n : fallback;
};
const flag = (a: Args, key: string) => a.options.has(key) && str(a, key) !== 'false';
const list = (a: Args, key: string) =>
  (a.options.get(key) ?? []).flatMap((v) => v.split(',')).filter((v) => v.length > 0);

const PRESETS = {
  tiny: { hidden: 64, context: 32, layers: 2, heads: 4, kvHeads: 2, ff: 128 },
  small: { hidden: 128, context: 64, layers: 4, heads: 8, kvHeads: 4, ff: 512 },
  base: { hidden: 256, context: 128, layers: 6, heads: 8, kvHeads: 4, ff: 1024 }
} as const;

const HELP = `Neuro-Lingua offline model factory

Commands
  train     Prepare local documents, train a tokenizer and a model, validate every epoch and
            write the best model as a package with a model card.
  eval      Loss, perplexity and bits per character of a model on local text.
  generate  Generate text from a prompt.
  serve     Serve a model on a local HTTP API (OpenAI-compatible completions / chat).
  info      Print a package's model card.

train options
  --data <path>          file or directory (repeatable / comma-separated): .txt .md .jsonl .json .csv
  --out <dir>            output directory (default models/<name>)
  --name <name>          model name (default local-model)
  --arch <a>             transformer | advanced | feedforward (default transformer)
  --preset <p>           tiny | small | base (default tiny); --hidden --context --layers --heads
                         --kv-heads --ff override it
  --tokenizer <t>        bpe | character | unicode (default bpe); --vocab <n> (default 2000);
                         --lowercase
  --epochs <n>           maximum epochs (default 10); --patience <n> early stopping (default 3)
  --lr <x>               learning rate (default 0.003); --warmup <steps> (default 100)
  --optimizer <o>        adam | momentum | lion (default adam)
  --batch <n>            windows per optimizer step (default 8)
  --stride <n>           training stride (default = context: non-overlapping chunks)
  --weight-decay <x>     AdamW decay (default 0.01); --dropout <x> (default 0.05);
                         --attention-dropout <x> (default 0.05); --dropconnect <x> (default 0)
  --validation <f>       held-out fraction (default 0.1); --min-paragraph <chars> (default 20);
                         --split-seed <n> (a different, still deterministic split)
  --precision <p>        float32 | float16 package (default float32)
  --init <package>       continue training an existing model on the new data
  --seed <n>             random seed (default 1337)
  --workers <n|auto>     processes for transformer training and evaluation (default auto: one
                         per CPU core, at most 8; 1 trains in-process). Results are reproducible
                         for a given seed and worker count.

eval options
  --workers <n|auto>     as for train

generate / serve options
  --model <package>  --prompt <text>  --max-tokens <n>  --temperature <x>  --top-p <x>  --top-k <n>
  --seed <n>  --port <n> (default 8787)  --host <h> (default 127.0.0.1)  --cors
`;

function progressBar(done: number, total: number, epoch: number, started: number) {
  const fraction = total > 0 ? done / total : 0;
  const elapsed = (Date.now() - started) / 1000;
  const eta = fraction > 0 ? elapsed / fraction - elapsed : 0;
  const width = 30;
  const filled = Math.round(fraction * width);
  process.stdout.write(
    `\r  epoch ${epoch + 1} [${'#'.repeat(filled)}${'.'.repeat(width - filled)}] ${(fraction * 100).toFixed(0)}%  ETA ${eta.toFixed(0)}s   `
  );
  if (done === total) process.stdout.write('\r' + ' '.repeat(70) + '\r');
}

async function train(a: Args) {
  const data = list(a, 'data');
  if (data.length === 0) throw new Error('train needs at least one --data <file|dir>');
  const name = str(a, 'name', 'local-model')!;
  const out = path.resolve(str(a, 'out', path.join('models', name))!);
  const preset =
    PRESETS[(str(a, 'preset', 'tiny') as keyof typeof PRESETS) ?? 'tiny'] ?? PRESETS.tiny;
  const context = num(a, 'context', preset.context);
  const config: FactoryConfig = {
    name,
    description: str(a, 'description'),
    architecture: (str(a, 'arch', 'transformer') as FactoryConfig['architecture']) ?? 'transformer',
    tokenizer: {
      mode: (str(a, 'tokenizer', 'bpe') as FactoryConfig['tokenizer']['mode']) ?? 'bpe',
      vocabSize: num(a, 'vocab', 2000),
      lowercase: flag(a, 'lowercase')
    },
    model: {
      hiddenSize: num(a, 'hidden', preset.hidden),
      contextSize: context,
      dropout: num(a, 'dropout', 0.05),
      transformer: {
        numLayers: num(a, 'layers', preset.layers),
        numHeads: num(a, 'heads', preset.heads),
        numKVHeads: num(a, 'kv-heads', preset.kvHeads),
        ffHiddenDim: num(a, 'ff', preset.ff),
        attentionDropout: num(a, 'attention-dropout', 0.05),
        dropConnectRate: num(a, 'dropconnect', 0),
        batchSize: num(a, 'batch', 8),
        trainingStride: num(a, 'stride', context),
        weightDecay: num(a, 'weight-decay', 0.01)
      }
    },
    training: {
      epochs: num(a, 'epochs', 10),
      learningRate: num(a, 'lr', 0.003),
      optimizer: (str(a, 'optimizer', 'adam') as Optimizer) ?? 'adam',
      warmupSteps: num(a, 'warmup', 100),
      patience: num(a, 'patience', 3),
      seed: num(a, 'seed', 1337)
    },
    data: {
      validationFraction: num(a, 'validation', 0.1),
      minParagraphChars: num(a, 'min-paragraph', 20),
      splitSeed: num(a, 'split-seed', 0)
    },
    precision: str(a, 'precision', 'float32') === 'float16' ? 'float16' : 'float32'
  };

  const { documents, files } = await readCorpusDocuments(data);
  console.log(`Neuro-Lingua model factory — offline, no external AI services`);
  console.log(`Read ${files.length} files (${documents.length} documents)`);
  await mkdir(out, { recursive: true });
  const logFile = path.join(out, 'training-log.jsonl');
  await writeFile(logFile, '');

  const init = str(a, 'init');
  const initialModel = init ? (await loadModelFile(path.resolve(init))).model : undefined;
  const workers = parseWorkerCount(str(a, 'workers'));
  let epochStarted = Date.now();
  const result = await runModelFactory(
    documents,
    config,
    {
      onLog: (m) => console.log(m),
      onProgress: (done, total, epoch) => {
        if (done === 1) epochStarted = Date.now();
        if (process.stdout.isTTY) progressBar(done, total, epoch, epochStarted);
      },
      onEpoch: async (report, model) => {
        await appendFile(logFile, JSON.stringify(report) + '\n');
        // Latest state (with optimizer buffers) for inspection or manual recovery.
        const checkpoint = packModel(model.toJSON() as unknown as Record<string, unknown>, {
          includeOptimizerState: true
        });
        await writeFile(path.join(out, 'checkpoint.nlm.json'), JSON.stringify(checkpoint));
      }
    },
    initialModel,
    { backend: dataParallelBackend({ workers }) }
  );
  result.card.environment = {
    runtime: `Node ${process.version}`,
    platform: `${os.platform()} ${os.arch()}`,
    cpu: os.cpus()[0]?.model ?? 'unknown',
    files: files.map((f) => path.relative(process.cwd(), f))
  };
  const pkg = { ...result.package, card: result.card };
  await writeFile(path.join(out, 'model.nlm.json'), JSON.stringify(pkg));
  await writeFile(path.join(out, 'model-card.md'), renderModelCard(result.card));
  const best = result.history[result.bestEpoch - 1];
  console.log(
    `\nBest epoch ${result.bestEpoch}: validation ${best.validationBitsPerCharacter.toFixed(3)} bpc, perplexity ${best.validationPerplexity.toFixed(2)}`
  );
  console.log(`Wrote ${path.join(out, 'model.nlm.json')} and model-card.md`);
}

async function evaluate(a: Args) {
  const { model } = await loadModelFile(path.resolve(str(a, 'model', '')!));
  const { documents } = await readCorpusDocuments(list(a, 'data'));
  const text = prepareCorpus(documents, { validationFraction: 0, minParagraphChars: 1 }).train;
  const workers = parseWorkerCount(str(a, 'workers'));
  let r: Awaited<ReturnType<typeof evaluateText>>;
  if (model instanceof TransformerLM && workers > 1) {
    const pool = await DataParallelTrainer.start(
      { model, trainText: '', validationText: text, seed: 0 },
      { workers }
    );
    try {
      r = withBitsPerCharacter(await pool.evaluate(), text);
    } finally {
      await pool.dispose();
    }
  } else {
    r = await evaluateText(model, text);
  }
  console.log(
    JSON.stringify(
      {
        tokens: r.tokens,
        loss: r.loss,
        perplexity: r.perplexity,
        bitsPerCharacter: r.bitsPerCharacter
      },
      null,
      2
    )
  );
}

async function generate(a: Args) {
  const { model } = await loadModelFile(path.resolve(str(a, 'model', '')!));
  if (a.options.has('seed')) model.reseed(num(a, 'seed', 1337));
  const text = await model.generate(
    str(a, 'prompt', '')!,
    num(a, 'max-tokens', 60),
    num(a, 'temperature', 0.8),
    num(a, 'top-k', 0),
    num(a, 'top-p', 0.9)
  );
  console.log(text);
}

async function serve(a: Args) {
  const file = path.resolve(str(a, 'model', '')!);
  const { model, pkg } = await loadModelFile(file);
  const port = num(a, 'port', 8787);
  const host = str(a, 'host', '127.0.0.1')!;
  const name = pkg?.card?.name ?? path.basename(file);
  const server = createInferenceServer(model, { modelName: name, cors: flag(a, 'cors') });
  server.listen(port, host, () => {
    console.log(
      `Serving "${name}" on http://${host}:${port} (local model only, no external calls)`
    );
    console.log(`  POST /v1/completions      {"prompt": "...", "max_tokens": 60}`);
    console.log(`  POST /v1/chat/completions {"messages": [{"role": "user", "content": "..."}]}`);
  });
}

async function info(a: Args) {
  const { model, pkg } = await loadModelFile(path.resolve(str(a, 'model', '')!));
  if (pkg?.card) console.log(renderModelCard(pkg.card));
  else
    console.log(
      `Parameters: ${model.getParametersCount()}, vocabulary: ${model.getVocabSize()}, context: ${model.getContextSize()} (no model card)`
    );
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  const command = a._[0] ?? 'help';
  const commands: Record<string, (a: Args) => Promise<void>> = {
    train,
    eval: evaluate,
    generate,
    serve,
    info
  };
  if (!commands[command]) {
    console.log(HELP);
    return;
  }
  await commands[command](a);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
