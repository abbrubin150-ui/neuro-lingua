# Offline Model Factory · מפעל מודלים אופליין

Train, evaluate, package and serve your own language model **entirely on your machine** — from
your own text files, with no external AI model, API key or network service.

---

## התחלה מהירה (עברית)

**מה זה?** כלי שורת-פקודה ואפליקציית דפדפן שמייצרים מודל שפה קטן מהטקסטים שלכם, על המחשב
שלכם בלבד. אין קריאה לשום שירות בינה מלאכותית חיצוני, אין מפתח API, ואין צורך ברשת אחרי
ההתקנה הראשונית של החבילות.

**1. התקנה (פעם אחת, דורשת רשת רק להורדת חבילות npm):**

```bash
pnpm install
```

**2. אימון מודל מתיקייה של קבצי טקסט** (נתמכים: ‎.txt ‏‎.md ‏‎.jsonl ‏‎.json ‏‎.csv):

```bash
pnpm lm train --data ./my-texts --name my-model
```

התוצאה נכתבת לתיקייה `models/my-model`:
קובץ המודל (`model.nlm.json`), כרטיס מודל (`model-card.md`) ויומן אימון (`training-log.jsonl`).

**3. יצירת טקסט:**

```bash
pnpm lm generate --model models/my-model/model.nlm.json --prompt "שלום"
```

**מודל שעונה על שאלות:** אפשר לאמן על זוגות של שאלה ותשובה (קובצי jsonl/csv, או טקסט עם שורות
"ש:" / "ת:"). האימון לומד רק את התשובות, ואז שואלים בשיחה במסוף:

```bash
pnpm lm train --format qa --data data/qa --name faq --question-label "שאלה:" --answer-label "תשובה:"
pnpm lm chat --model models/faq/model.nlm.json
```

**ליבות ועצירות:** האימון מתחלק אוטומטית בין כל ליבות המעבד. אם ריצה נעצרה באמצע (כיבוי, Ctrl-C),
ממשיכים בדיוק מאותה נקודה — התוצאה זהה לריצה שלא נעצרה:

```bash
pnpm lm train --resume models/my-model
```

**4. הערכה על טקסט שהמודל לא ראה:**

```bash
pnpm lm eval --model models/my-model/model.nlm.json --data ./held-out-texts
```

**5. הפעלת שרת מקומי** (תואם לממשק completions/chat הנפוץ, מאזין רק ל-127.0.0.1):

```bash
pnpm lm serve --model models/my-model/model.nlm.json --port 8787
```

**באפליקציית הדפדפן:** בחרו בלוח ה-Tokenizer את האפשרות Subword BPE, אמנו, ולחצו על
‎📦 Package‎ כדי לשמור קובץ ‎.nlm.json‎ — אותו פורמט שהכלים בשורת הפקודה טוענים. האפליקציה
נשמרת במטמון הדפדפן אחרי ביקור אחד, ולכן נפתחת ועובדת גם בלי חיבור לרשת.

**מה לצפות:** אלה מודלים קטנים (מאות אלפים עד מיליוני פרמטרים) שרצים על המעבד. הם לומדים את
הסגנון, אוצר המילים והתבניות של הטקסט שלכם; הם אינם תחליף למודלי שפה גדולים. המדד העיקרי הוא
bits per character על טקסט מוחזק בצד — ככל שהוא נמוך יותר, המודל טוב יותר.

---

## Reference (English)

### What "offline" means here

- **Data, tokenizer, training, evaluation, packaging and serving** all run in-process on the CPU,
  from local files. Nothing is downloaded and no request leaves the machine.
- **No external AI**: no hosted model, no API key, no pretrained weights. Every weight is learned
  from your text.
- **Enforced by a test**: `tests/offline/noExternalServices.test.ts` fails if the app, CLI,
  desktop or public files gain an AI-service client dependency or import, a hosted AI API
  hostname, or any network call other than the offline service worker's same-origin cache.
- The only network step is the one-time `pnpm install` of the npm dependencies. (The optional
  Python helper `scripts/data/build_corpus.py --mode full` can download *public text datasets*;
  it is never needed.)

### Commands

```text
pnpm lm train    --data <file|dir> [--data …] [--out models/<name>] [options]
pnpm lm train    --resume <dir>
pnpm lm eval     --model <package> --data <file|dir>
pnpm lm generate --model <package> --prompt "text" [--max-tokens 60] [--temperature 0.8]
pnpm lm serve    --model <package> [--port 8787] [--host 127.0.0.1] [--cors]
pnpm lm chat     --model <package>          (models trained with --format qa)
pnpm lm info     --model <package>
pnpm lm help
```

`--model` accepts a package (`.nlm.json`), a plain `model.toJSON()` export or a trace export from
the browser app.

### Data

| Format | How it is read |
| --- | --- |
| `.txt` / `.text` | one document |
| `.md` / `.markdown` | one document; code blocks, tables, links, emphasis and HTML are stripped |
| `.jsonl` / `.ndjson` | one document per line: a string or an object with `text`, `content`, `body` or `document` |
| `.json` | an array (or one object) of the same |
| `.csv` | the `text`/`content`/`body`/`document` column, else the first column (RFC 4180 quoting) |

Directories are read recursively in sorted order (hidden folders, `node_modules`, `.git`, `dist`
and `build` are skipped). Preparation (`src/data/corpusPrep.ts`): Unicode NFC, Unix newlines,
control characters removed, whitespace collapsed, paragraphs shorter than `--min-paragraph`
(default 20 characters) dropped, exact duplicate paragraphs removed. The **train / validation
split is decided per paragraph by a content hash** (`--validation`, default 0.1; `--split-seed`
for a different split), so adding or reordering files never moves an existing paragraph to the
other side.

### Tokenizer

`--tokenizer bpe` (default) learns a byte-pair-encoding subword vocabulary from the training
split (`--vocab`, default 2000; `src/tokenizer/subword.ts`):

- pieces are runs of letters (with their combining marks — Hebrew niqqud never becomes a separate
  token), digits or other symbols; a piece that follows whitespace starts with the word marker
  `▁`, so decoding restores spaces exactly;
- merges are learned with incremental pair counts and a lazy max-heap (≈ 0.2 s for 200 K
  characters), ties broken by string order — training is deterministic;
- every character seen in training is in the vocabulary, so text made of known characters is
  never out-of-vocabulary.

`--tokenizer character` and `--tokenizer unicode` (word-level) are also available.

### Model and training

| Preset | hidden | context | layers | heads / KV heads | FFN |
| --- | --- | --- | --- | --- | --- |
| `tiny` (default) | 64 | 32 | 2 | 4 / 2 | 128 |
| `small` | 128 | 64 | 4 | 8 / 4 | 512 |
| `base` | 256 | 128 | 6 | 8 / 4 | 1024 |

The default architecture is the decoder-only causal transformer (`--arch transformer`; RoPE,
grouped-query attention, SwiGLU, RMSNorm, exact backpropagation). `--arch advanced|feedforward`
train the MLP models, which see the context as a bag of words.

- **Chunked training**: one window every `--stride` targets (default = context length), with the
  loss on every position of the window and the chunk alignment re-drawn every epoch.
- **Optimizer**: Adam (`--optimizer`), mini-batches of `--batch` windows (default 8), warm-up
  (`--warmup`, default 100 steps) then cosine decay of the learning rate to 10 %, decoupled weight
  decay (`--weight-decay`, default 0.01; norm gains and output bias are not decayed).
- **Validation every epoch** on the held-out paragraphs; **early stopping** after `--patience`
  epochs without improvement (default 3). The written model is the **best epoch**, not the last.
- `--init <package>` continues training an existing model on new data (its tokenizer is kept).
- `--seed` makes the whole run reproducible.

**Multi-core training** (`--workers <n|auto>`, default one per CPU core, at most 8). Every
mini-batch is split over worker threads that hold copies of the model; they write their gradients
into shared memory, the main thread adds them up and takes the optimizer step (schedule, clipping
and optimizer state stay in one place), and the new parameters go back through shared memory.
Windows, batches and learning-rate steps are exactly those of single-thread training. Dropout noise
comes from one seeded stream per worker, so a run is reproducible for a given `--seed` and
`--workers` — pass `--workers` explicitly to reproduce a run on a machine with another number of
cores. Without dropout the result equals single-thread training up to floating-point summation
order. Validation and `pnpm lm eval` are split over the workers as well. Larger `--batch` values
parallelise better (more windows per synchronisation).

**Resuming** (`--resume <dir>`). After every epoch the run writes a lossless checkpoint (see
Output). `pnpm lm train --resume models/my-model` continues from it with the original data and
settings, and the final model is bit-identical to an uninterrupted run with the same `--workers`.
The checkpoint is written to a temporary file and renamed, so an interruption in the middle of a
write leaves the previous checkpoint intact. A resumed run refuses training data that differs from
the original (fingerprint of the prepared text). `--epochs`, `--patience` and `--workers` may be
changed on resume (another worker count changes the remaining dropout noise).

### Question–answer models (`--format qa`)

A model trained on plain text continues text. Trained on question–answer pairs, it answers:

```bash
pnpm lm train --format qa --data data/qa --name faq            # pairs only
pnpm lm train --format qa --data data/qa --init models/docs/model.nlm.json --name faq
pnpm lm chat  --model models/faq/model.nlm.json                 # ask on the console
echo "How do I resume training?" | pnpm lm chat --model models/faq/model.nlm.json
```

- **Input**: `.jsonl` / `.json` objects with `question`/`answer` (also `prompt`/`completion`,
  `instruction`/`output`, or chat `messages` with user → assistant turns), `.csv` with
  question/answer columns (else the first two), or text / Markdown with `Q:` / `A:` blocks (also
  `Question:` / `Answer:` and `ש:` / `ת:`, `שאלה:` / `תשובה:`). `data/qa/neuro-lingua-faq.jsonl` is
  a hand-written example.
- **Template**: every pair becomes one paragraph `Question: … Answer: …`; the labels are set with
  `--question-label` / `--answer-label` (e.g. `"שאלה:"` / `"תשובה:"`) and stored in the model card,
  so `chat`, `serve` and `eval` use the same prompt. Pairs are cleaned, de-duplicated and split into
  training / validation like any paragraph (short pairs are kept).
- **Loss on the answers only**: the questions and labels are context. The token that follows each
  answer — the first token of the next question — is also learned, so the model knows where an
  answer ends; generation stops at that token.
- **Validation** reports the loss per answer character, and at the end the share of validation
  questions answered exactly right (case, spacing and final punctuation ignored) and the
  word-overlap F1 of the generated answers, from free (greedy) generation. The loss alone is
  misleading here: once an answer has begun, a memorised continuation is cheap, while choosing
  the right answer is the hard part.
- `--init <package>` starts from a model trained on plain text (its tokenizer is kept).
- The local server's `/v1/chat/completions` asks the last user message with the model's labels and
  stops at the end of the answer; `pnpm lm eval` on a Q&A model scores the answers only.
- These are small models trained on your pairs: they reproduce answers they were trained on,
  answer some rephrasings, and give a (wrong) trained answer to questions outside their data —
  they cannot say "I don't know". Several phrasings per question help most.

Measured on `data/qa/neuro-lingua-faq.jsonl` (72 pairs: 24 facts × 3 phrasings; 58 training /
14 validation pairs, `tiny` preset, 4 workers). Every validation question rephrases a fact whose
answer is in the training pairs, so the scores measure whether a reworded question finds its
answer; with 14 questions, differences of one or two answers are noise.

| Run | Validation bits per answer character | Exact answers | Word-overlap F1 |
| --- | --- | --- | --- |
| fine-tuned from the docs model (`--init`) | 0.088 | 2 / 14 | 0.27 |
| from scratch on the pairs | 0.097 | 4 / 14 | 0.35 |
| from scratch, `--dropout 0.2 --attention-dropout 0.1` | 0.107 | 4 / 14 | 0.42 |

The lowest loss did not give the best answers: pretraining on the docs made the loss converge
much faster (0.63 vs 1.93 bits per answer character after 5 epochs) but did not help it pick
answers, and stronger dropout gave the best answers with the highest loss. Greedy answers of the
from-scratch model:

| Question | Answer |
| --- | --- |
| What tokenizer does the factory use? *(trained)* | A subword BPE tokenizer learned from your own corpus, with 2000 subwords by default. ✓ |
| איך ממשיכים אימון שנעצר? *(trained)* | מריצים pnpm lm train --resume עם תיקיית המודל, והאימון ממשיך בדיוק מאותה נקודה. ✓ |
| How can I resume a stopped training run? *(new wording)* | Run pnpm lm train --resume with the model folder; it continues exactly where it stopped. ✓ |
| Which tokenizer splits my text? *(new wording)* | Plain text, Markdown, JSONL, JSON and CSV files. ✗ |
| What is the capital of France? *(outside the data)* | Run pnpm lm train --resume with the model folder; it continues exactly where it stopped. ✗ |

### Output

| File | Content |
| --- | --- |
| `model.nlm.json` | the best model as a package, with its model card |
| `model-card.md` | the card as Markdown: sizes, tokenizer, data statistics, training settings, per-epoch metrics, environment, provenance |
| `training-log.jsonl` | one JSON line per epoch (train/validation loss, perplexity, bits per character, learning rate, seconds) |
| `checkpoint.nlm.json` | lossless state after the latest epoch for `--resume`: float64 weights, optimizer moments and schedule position, random streams, the best model so far and the run settings (≈ 70 bytes per parameter) |

**Bits per character** (validation loss × tokens ÷ characters ÷ ln 2) is the headline metric: it
does not depend on the tokenizer, so models with different vocabularies can be compared.

### Package format (`neuro-lingua-model`, version 1)

```json
{
  "format": "neuro-lingua-model",
  "formatVersion": 1,
  "createdAt": "…",
  "architecture": "transformer",
  "precision": "float32",
  "includesOptimizerState": false,
  "checksum": "fnv1a64:…",
  "card": { "name": "…", "provenance": { "offline": true, "externalServices": [], "statement": "…" } },
  "model": { "…": "the model's toJSON() with every numeric array stored as base64 float32/float16" }
}
```

- `--precision float16` halves the size again (weights rounded to ~3 significant digits).
- The optimizer state (Adam moments) is dropped unless requested, which is most of the size of a
  plain export: the browser app's default transformer is 23 MB as plain JSON, 5.7 MB as a float32
  package and 2.9 MB as float16.
- The checksum is verified on every load; a modified or truncated file is rejected.
- `provenance.offline` is `true` only when the producer knows that no external service was
  involved, `null` when the history is unknown (e.g. a model imported from a file without a card).

### Local server

```bash
pnpm lm serve --model models/my-model/model.nlm.json --port 8787
```

| Endpoint | Body |
| --- | --- |
| `GET /health` | → `{"status":"ok"}` |
| `GET /v1/models` | → the loaded model |
| `POST /v1/completions` | `{"prompt": "…", "max_tokens": 60, "temperature": 0.8, "top_p": 0.9, "top_k": 0, "stop": ["\n\n"], "seed": 1}` |
| `POST /v1/chat/completions` | `{"messages": [{"role": "user", "content": "…"}], …same options}` |

```bash
curl -s http://127.0.0.1:8787/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The model", "max_tokens": 40, "seed": 7}'
```

- Responses follow the common completions / chat-completions shapes, so local tools that speak
  that format can use the model. The server never forwards a request anywhere.
- It listens on `127.0.0.1` unless `--host` says otherwise, and sends CORS headers only with
  `--cors`. `max_tokens` is capped at 512 and request bodies at 1 MB.
- Chat requests are turned into a plain prompt (system messages are ignored, the other contents
  are joined with newlines): these models are not instruction-tuned.

### Browser app

- **Tokenizer panel → "Subword BPE"**: the merges are learned from the training text when a new
  model is created (a resumed model keeps its merges).
- **📦 Package** downloads a `.nlm.json` package with a model card; **⬆️ Import** loads packages,
  plain exports and trace exports. The card's provenance is tracked: a model initialised in the
  app is `offline: true`; a model that started from an imported file keeps that file's provenance.
- **Saving in the browser** falls back to a float32 and then a float16 package when the plain JSON
  exceeds the localStorage quota, and the app says which one was stored.
- **Responsive training**: training yields to the page every 50 ms, so the progress bar moves
  within an epoch and **Stop** acts within one training window.
- **Works offline**: production builds register a service worker that pre-caches every file of
  the build. After one online visit the app starts and trains without a network connection, and
  it can be installed as an app (web app manifest). The desktop app (Electron) loads from disk and
  is offline by construction.

### Measured example

A model produced end to end in this repository with nothing but the CLI and local files (one
2.3 GHz Xeon core, Node 22):

```bash
pnpm lm train --data data/corpus.txt \
  --data data/processed/wikitext/train.txt,data/processed/hebrew_news/train.txt,data/processed/hebrew_opinion/train.txt \
  --data docs --name neuro-lingua-docs --preset tiny --epochs 14 --patience 3
```

| | |
| --- | --- |
| Data | 43 files → 35 documents (JSON files without a text field contribute nothing); 935 training / 105 validation paragraphs; 186,265 / 21,592 characters; 6 duplicate paragraphs removed |
| Tokenizer | BPE, 2,000 subwords (231 base symbols including Hebrew, 1,769 merges); ≈ 3.0 characters per token |
| Model | transformer, `tiny` preset: 324,372 parameters, context of 32 tokens |
| Training | early stop after epoch 12 (3 epochs without improvement), best epoch 9 kept; ≈ 95 s per epoch (≈ 68 s of training on ≈ 61 K tokens + ≈ 27 s of validation); 19 min in total |
| Validation (held-out paragraphs) | **2.307 bits per character** (perplexity 94.8 per subword) |
| Files never used in training: README.md + CLAUDE.md (25,339 characters) | **2.241 bits per character**. For orientation only, on the same text: gzip -9 3.384, brotli -11 2.813 (compressors start without training data; these files share topics and phrasing with the docs) |
| Package | 1.86 MB (float32) |

Validation bits per character per epoch: 3.07 → 2.62 → 2.46 → 2.38 → 2.35 → 2.32 → 2.31 → 2.32 →
**2.31** → 2.33 → 2.34 → 2.36, while the training loss kept falling (6.82 → 2.95 nats per token):
with ~190 K characters the model starts to memorise after ~9 epochs, so more text helps more than
more epochs. Early stopping kept epoch 9.

Sample (`pnpm lm generate --prompt "The transformer" --temperature 0.7 --seed 7`):

> The transformer: ights When to Use: Related: Self-lay in the Well-ste. Sumstical External model
> toward passes: Learning RThic

The model has picked up the vocabulary and layout of the documentation (headings, "External model",
"Learning") but not coherent sentences. That is the honest picture of 0.3 M parameters trained on
0.2 MB of text: it predicts related text well, it does not write. Larger presets and much more
text are the levers.

The same package then served completions and chat completions on `127.0.0.1` (`pnpm lm serve`)
and loaded in the browser app through ⬆️ Import.

The run above used one core. Multi-core training was added afterwards; on the same 4-core machine,
one epoch of the same model (training + validation, on the same documents plus this guide: 955
training paragraphs) took:

| `--workers` | 1 | 2 | 4 |
| --- | --- | --- | --- |
| Seconds per epoch | 99.4 | 55.8 (1.8×) | 38.6 (2.6×) |

`pnpm lm eval` on the held-out files took 31.4 s with one worker and 12.5 s with four, with the
same result to 15 significant digits.

### Limitations

- CPU-only JavaScript: on one core the `tiny` preset trains at ≈ 900 tokens per second with a
  2,000-subword vocabulary, and scoring (validation, `eval`) costs ≈ 4 ms per token because every
  token is predicted from a full context window. Worker threads multiply this by about 2.6 on
  4 cores (measured below). `small` and `base` are several times slower per token. Corpora of a
  few megabytes are practical; hundreds of megabytes are not.
- Small models learn the style, vocabulary and local structure of the training text. They do not
  have world knowledge, follow instructions or reason; judge them by held-out bits per character
  and by reading their samples.
- You are responsible for having the right to use the training text.

### Code map

| Path | Role |
| --- | --- |
| `scripts/lm.ts` | CLI |
| `src/factory/pipeline.ts` | documents → corpus → tokenizer → model → validated package |
| `src/factory/node/files.ts`, `src/factory/node/server.ts` | Node-only file reading and the local server |
| `src/factory/node/parallel.ts`, `src/factory/node/trainingWorker.ts` | multi-core training and evaluation (coordinator and replica threads) |
| `src/data/qa.ts`, `src/factory/answer.ts` | question–answer data, answer-only loss masks, end-of-answer token, scoring |
| `src/data/corpusPrep.ts` | parsing, cleaning, de-duplication, hash split |
| `src/tokenizer/subword.ts` | BPE training, encoding with exact offsets, decoding |
| `src/lib/modelPackageCodec.ts`, `src/lib/modelPackage.ts` | package format, checksums, loaders |
| `src/lib/modelExport.ts` | model cards and provenance for models made in the browser |
| `src/pwa/` | offline service worker (generated at build time) and its registration |
| `tests/offline/noExternalServices.test.ts` | the no-external-AI guard |
