# Neuro-Lingua DOMESTICA: תיאור ומפרט

גרסה 4.4.0 · מצב הריפו ב-15 בנובמבר 2026 (אחרי PR #164) · רישיון MIT

---

## 1. תיאור

**בשורה אחת (לתיאור הריפו ב-GitHub):**

> Browser-native neural language models in React + TypeScript (MLP and causal Transformer). Train,
> evaluate, package and serve them offline from your own text, with no external AI service.

**בפסקה:**

Neuro-Lingua היא מעבדה למודלי שפה קטנים שרצה כולה על המחשב של המשתמש. היא כוללת שלושה חלקים:

- **אפליקציית דפדפן דו-לשונית (עברית/אנגלית).** מאמנים בה מודל על טקסט שמדביקים, מייצרים טקסט,
  ובוחנים את המודל דרך לוחות של הסברתיות, תורת האינפורמציה וממשל.
- **כלי שורת פקודה (`pnpm lm`).** הכלי מאמן מודל מתיקיית קבצים על כל ליבות המעבד, מעריך אותו על
  טקסט מוחזק בצד, אורז אותו לקובץ עם checksum וכרטיס מודל, ומגיש אותו בשרת מקומי.
- **אפליקציית שולחן עבודה (Electron, Windows).**

כל המשקלות נלמדים מהטקסט של המשתמש. אין מודל מאומן מראש, אין מפתח API ואין קריאה לשירות חיצוני.
בדיקה אוטומטית אוכפת זאת.

**מה זה לא:** זה לא תחליף למודל שפה גדול. המודלים כאן קטנים, ממאות אלפים ועד כמיליון פרמטרים,
ורצים על המעבד ב-JavaScript. הם לומדים סגנון, אוצר מילים ומבנה מקומי של טקסט. אין להם ידע על
העולם, הם לא ממלאים הוראות והם לא מסיקים מסקנות (ראו סעיף 3).

---

## 2. עקרונות מחייבים

| עיקרון | מה הוא אומר | איך נאכף |
|---|---|---|
| אופליין | הקוד של האפליקציה, ה-CLI ושולחן העבודה לא פונה לרשת ולא לשירות AI | `tests/offline/noExternalServices.test.ts`. החריג היחיד הוא מטמון ה-service worker מאותו מקור |
| בלי מספרים מומצאים | כל מספר בממשק מחושב מהמודל, או שמסומן במפורש כסימולציה | סקירת קוד. לוח הניתוח הסיבתי מסומן כסימולציה |
| שחזוריות | זרעים קבועים. אימון מרובה ליבות משוחזר עבור אותו `--seed` ואותו `--workers`. המשך ריצה שנקטעה זהה לריצה שלא נקטעה | `tests/factory/resume.test.ts`, `tests/factory/parallel.test.ts` |
| גרדיאנטים מדויקים | ה-backprop של כל ארכיטקטורה נבדק מול הפרשים סופיים | `tests/lib/*gradcheck*`, `tests/models/transformer_backprop.test.ts` |
| מקוריות מוצהרת | כרטיס מודל מצהיר `offline: true` רק כשהמקור ידוע. כשההיסטוריה לא ידועה, הערך הוא `null` | `src/lib/modelExport.ts` |

---

## 3. מצב איכות נוכחי (מדוד)

כל המספרים מחושבים על טקסט שהמודל לא ראה באימון.

**השוואת ארכיטקטורות** (`docs/benchmarks/MODEL_BENCHMARK.md`, `data/corpus.txt` לפי תווים):

| מודל | ביטים לתו על טקסט מוחזק | משימת סדר מילים (דיוק) |
|---|---|---|
| ProNeuralLM (MLP) | 4.17 | 4% |
| AdvancedNeuralLM | 4.02 | 2% |
| Transformer סיבתי | **2.88** | **100%** |

**מודלים מאומנים ב-CLI** (preset `tiny`, 4 אפוקים, 10% מוחזקים; `docs/benchmarks/GENERATION_REPETITION.md`):

| נתונים | טוקנייזר | פרמטרים | ביטים לתו |
|---|---|---|---|
| Tiny Shakespeare | תווים | 74,628 | 2.159 |
| Tiny Shakespeare | BPE, 2,000 | 324,372 | **1.959** |
| התורה בלי ניקוד | תווים | 69,984 | 2.495 |

**מסקנות:**

- רק ה-Transformer מייצר טקסט שנראה כמו שפה. ה-MLP-ים "עיוורים לסדר": הם ממצעים את ההטמעות
  של ההקשר (bag of words), ולכן לא יכולים ללמוד תחביר.
- ב-greedy כל המודלים נכנסים ללולאות. חסימת n-gram (ברירת המחדל מאז 4.4.0) עוצרת את הלולאות,
  אבל לא יוצרת משמעות.
- הרמה היא של הדגמה ומחקר, עדיין לא של מודל שפה שימושי.

---

## 4. רכיבי המערכת

```
┌──────────────────────── src/ (TypeScript, ~70K שורות, 156 קבצים) ────────────────────────┐
│  App.tsx + components/    ← אפליקציית הדפדפן (React 18, Vite)                            │
│  lib/  models/  training/  generation/  tokenizer/  losses/   ← ליבת המודלים (משותפת)   │
│  factory/pipeline.ts  +  factory/node/ (קבצים, workers, שרת)  ← מפעל המודלים האופליין    │
│  pwa/                     ← service worker להתקנה ועבודה בלי רשת                           │
│  math/ explainability/ compression/ backend/ ...  ← ספריות ניתוח ותשתית                   │
└───────────────────────────────────────────────────────────────────────────────────────────┘
scripts/lm.ts          ← CLI: train / eval / generate / serve / info
electron/main.ts       ← אפליקציית שולחן עבודה (Windows NSIS)
edge_formalism/  symmetry_coupling/  neurosync/   ← מודולי Python מחקריים, לא חלק מזרימת המודל
```

ליבת המודלים אחת, והיא משמשת את כל החזיתות. אותה מחלקת `TransformerLM` מאומנת בדפדפן, ב-CLI
וב-worker threads. קובץ `.nlm.json` שנוצר באחת מהן נטען בכל האחרות.

---

## 5. מודלים

### 5.1 ProNeuralLM (`src/lib/ProNeuralLM.ts`)

מודל feedforward בסיסי. ההקשר מיוצג כממוצע ההטמעות של `contextSize` הטוקנים האחרונים, ואחריו
שכבה נסתרת ו-softmax על אוצר המילים.

- backprop מדויק, ו-dropout רק באימון.
- תומך בכל האופטימייזרים (סעיף 7).
- WebGPU אופציונלי למכפלות מטריצה-וקטור, עם נפילה אוטומטית למעבד.
- זו מחלקת הבסיס: `generate`, `evaluate`, הטוקניזציה והסריאליזציה מוגדרים כאן, ושתי המחלקות האחרות
  יורשות אותם.

### 5.2 AdvancedNeuralLM (`src/lib/AdvancedNeuralLM.ts`)

ProNeuralLM בתוספת:

- פונקציות אקטיבציה: ReLU, LeakyReLU, ELU או GELU.
- אתחול He או Xavier, ו-LayerNorm.
- תזמוני קצב למידה: cosine, exponential ו-warmup.
- weight decay ו-gradient clipping.
- שיטות יצירה נוספות: beam search, contrastive, nucleus ו-greedy.

התכונות פועלות דרך hooks מוגנים של מחלקת הבסיס (`activate`, `normalizeHidden`,
`getGradientClipNorm`, `onEpochStart`), ולכן מסלול האימון של הממשק משתמש בהן.

### 5.3 TransformerLM (`src/lib/TransformerLM.ts`, `src/models/transformer_backprop.ts`)

Transformer מסוג decoder-only וסיבתי:

- **Attention:** קשב רב-ראשי סיבתי עם GQA/MQA (`numKVHeads`) ו-RoPE לכל ראש.
- **נרמול:** pre-norm RMSNorm לפני הקשב ולפני ה-FFN, ו-RMSNorm סופי.
- **FFN וחיבורים:** SwiGLU, עם residual connections.
- **רגולריזציה:** attention dropout ו-DropConnect (המסכות משותפות בתוך mini-batch).
- **Loss:** next-token loss בכל מיקום.
- **אימון:** mini-batches (ברירת מחדל 8) עם צעד אופטימייזר מאוחד לכל batch. האימון תמיד על המעבד.
- **אופטימייזרים:** Adam, momentum או Lion. בחירה מסדר שני נופלת ל-Adam.
- **מודלים ישנים:** מודל שנשמר בלי `objective` נטען כ-`pooled` (דו-כיווני, ממוצע, RoPE על שורה
  מלאה), כדי שהתחזיות שלו לא ישתנו.

| הגדרה | ברירת מחדל בממשק | `tiny` | `small` | `base` |
|---|---|---|---|---|
| `hiddenSize` (D) | 128 | 64 | 128 | 256 |
| `numLayers` | 4 | 2 | 4 | 6 |
| `numHeads` / `numKVHeads` | 8 / 4 | 4 / 2 | 8 / 4 | 8 / 4 |
| `ffHiddenDim` | 512 | 128 | 512 | 1024 |
| `contextSize` | **32** (4 ל-MLP-ים) | 32 | 64 | 128 |

### 5.4 הזרקת נוירונים (Cerebro, `src/lib/expandable/`)

הזרקה מוסיפה נוירונים לשכבה נסתרת של מודל קיים, ופועלת על כל הארכיטקטורות:

- ב-MLP ההזרקה משמרת את הפונקציה של המודל.
- ב-Transformer מוסיפים כפולה של `numHeads`.
- Rollback משחזר את המודל המקורי במדויק (`tests/lib/injection.integrity.test.ts`).

---

## 6. טוקניזציה

| מצב | תיאור |
|---|---|
| `unicode` | מילים וסימני פיסוק לפי קטגוריות Unicode |
| `ascii` | מילים ב-ASCII |
| `custom` | ביטוי רגולרי של המשתמש |
| `character` | תו לכל טוקן. כל רצף רווחים הופך לטוקן `' '` אחד |
| `bpe` (ברירת מחדל בממשק וב-CLI, 2,000 תת-מילים) | תת-מילים שנלמדות מהקורפוס (`src/tokenizer/subword.ts`): דטרמיניסטי, משאיר ניקוד מחובר לאות, ושומר offsets מדויקים. ה-merges נשמרים בתוך קובץ המודל |

`src/tokenizer/BPETokenizer.ts` הוא מודול BPE ישן יותר, שהמודלים לא משתמשים בו.

---

## 7. אימון

- **אופטימייזרים** (`src/training/`):
  - Adam, momentum SGD, Lion, Sophia, L-BFGS ו-damped Newton על ה-MLP-ים.
  - Adam, momentum או Lion על ה-Transformer.
  - K-FAC ו-EMA קיימים כספרייה בלבד.
- **ברירות מחדל:** Adam בקצב 0.003. Adam בקצב 0.05 ומעלה לא לומד על ה-MLP.
- **Loss:** cross-entropy. אפשר להפעיל loss mask: אחרי `=`, אחרי תג תשובה או לפי regex
  (`src/losses/lossMask.ts`). Information bottleneck ו-VIB קיימים לניטור ולניסוי.
- **אימון אינטראקטיבי:** `setInteractiveTraining` גורם ל-`train()` לפנות זמן לממשק ולעצור באמצע
  אפוק. אפוק שנקטע לא נכנס להיסטוריה.
- **מרובה ליבות (CLI):** כל mini-batch מתחלק בין worker threads דרך זיכרון משותף. האופטימייזר רץ
  בחוט הראשי. נמדד: פי 2.6 על 4 ליבות.
- **המשך ריצה:** אחרי כל אפוק נכתב checkpoint מלא, בדרך אטומית (כתיבה לקובץ זמני ושינוי שם).
  `--resume` מגיע לתוצאה זהה לביט לריצה שלא נקטעה, ומסרב לנתונים שהשתנו.
- **עצירה מוקדמת:** `--patience`, לפי loss על נתוני האימות.

---

## 8. יצירת טקסט (`src/generation/sampling.ts`)

**שיטות פענוח:** greedy, temperature, top-k, top-p, typical, Mirostat v2, beam search
ו-contrastive search.

**בקרת לולאות** (`RepetitionControls`, מגרסה 4.4.0):

- הבקרות חלות על כל שיטות הפענוח, ומסתכלות גם על הפרומפט.
- **no-repeat n-gram:** אוסר טוקן שישלים רצף שכבר הופיע. ברירת המחדל (`recommendedNoRepeatNgram`)
  היא חסימה באורך של כ-16 תווים:
  - 16 לטוקנייזר תווים (נמדד).
  - ב-BPE: ⌊16 חלקי האורך הממוצע של תת-מילה⌋, ולפחות 3. במודל שנמדד (ממוצע 4.4 תווים) זה 3.
    בקורפוס קטן תת-המילים קרובות לתווים בודדים, והחסימה מתארכת בהתאם.
  - 3 לטוקניזציה לפי מילים. זה הערך המקובל, והוא לא נמדד כאן.
- **עונש חזרות CTRL** (θ, מעל חלון של 64 טוקנים): **כבוי כברירת מחדל**. במדידות הוא הוריד את
  שיעור המילים התקינות ולא עצר לולאות בעברית.
- עונשי frequency ו-presence בסגנון OpenAI קיימים, וכבויים כברירת מחדל.

**ברירות מחדל בממשק:** top-p 0.9, טמפרטורה 0.8, 25 טוקנים.

---

## 9. הערכה

- **`model.evaluate(text)`** מחזיר loss, perplexity, דיוק top-1, מספר טוקנים ו-`oovRate`. הוא משתמש
  באותם חלונות של האימון, בלי dropout.
- **מדד הכותרת הוא ביטים לתו** (loss × טוקנים ÷ תווים ÷ ln 2). הוא לא תלוי בטוקנייזר, ולכן משמש
  להשוואה בין אוצרות מילים. PPL של טוקני BPE ו-PPL של תווים לא ברי השוואה ישירה.
- **מבחנים:**
  - `pnpm benchmark:models`: שלוש הארכיטקטורות.
  - `scripts/experiments/generation_repetition.ts`: לולאות ופגיעה בכתיב.
  - `pnpm benchmark:gpu`.

---

## 10. שמירה ופורמט קבצים

**חבילת מודל `.nlm.json`** (`neuro-lingua-model`, `formatVersion: 1`):

- tensors ב-base64, בדיוק float32 או float16. checksum מסוג FNV-1a-64 נבדק בכל טעינה.
- כרטיס מודל עם מקוריות, נתונים, הגדרות ומדדים לכל אפוק.
- מצב האופטימייזר לא נכלל, אלא אם מבקשים אותו.
- דוגמה, ה-Transformer של ברירת המחדל בממשק: 23MB כ-JSON רגיל, 5.7MB כ-float32 ו-2.9MB כ-float16.

**localStorage בדפדפן:**

- מפתחות `neuro-lingua-pro-v324` (MLP-ים) ו-`neuro-lingua-transformer-v324`.
- כשה-JSON חורג מהמכסה, השמירה נופלת לחבילה דחוסה.
- הגדרות, טוקנייזר, מטא-דאטה וטקסט אימון נשמרים במפתחות `neuro-lingua-*-v1`.

**API:**

```ts
const json = model.toJSON();                  // משקלות, זרם אקראיות, היסטוריה
TransformerLM.fromJSON(json);                  // כל מחלקה משחזרת את עצמה
const pkg = packModel(json, { precision: 'float32', card });   // modelPackageCodec.ts
const model = restoreModel(unpackModel(pkg));                   // modelPackage.ts
model.getTrainingState(); model.setTrainingState(s);           // מומנטים ומיקום בתזמון
```

`modelPackageCodec.ts` לא מייבא את מחלקות המודל, כי `ProNeuralLM` מייבא אותו. לכן הטוען שצריך את
כל המחלקות נמצא בקובץ נפרד, `modelPackage.ts`.

---

## 11. שורת פקודה ושרת

```bash
pnpm lm train    --data <file|dir> [--arch transformer|advanced|feedforward]
                 [--preset tiny|small|base] [--tokenizer bpe|character|unicode] [--vocab 2000]
                 [--epochs 10] [--patience 3] [--lr 0.003] [--batch 8] [--workers auto] ...
pnpm lm train    --resume <dir>
pnpm lm eval     --model <package> --data <file|dir>
pnpm lm generate --model <package> --prompt "..." [--temperature 0.8] [--top-p 0.9]
                 [--no-repeat-ngram n] [--repetition-penalty x] [--seed n]
pnpm lm serve    --model <package> [--port 8787] [--host 127.0.0.1] [--cors]
pnpm lm info     --model <package>
```

- **קלט:** קבצי `.txt`, `.md`, `.jsonl`, `.json` ו-`.csv`. הכנת הקורפוס
  (`src/data/corpusPrep.ts`) מנרמלת את הטקסט, מסירה פסקאות כפולות ומחלקת בין אימון לאימות לפי
  פסקה, בדרך דטרמיניסטית.
- **פלט:** `model.nlm.json`, `model-card.md`, `training-log.jsonl` ו-`checkpoint.nlm.json`.

**שרת** (`src/factory/node/server.ts`):

- **נקודות קצה:** `GET /health`, `GET /v1/models`, `POST /v1/completions` ו-`POST /v1/chat/completions`.
- **פרמטרים:** `prompt`/`messages`, `max_tokens` (עד 512), `temperature`, `top_p`, `top_k`, `stop`,
  `seed`, `repetition_penalty` ו-`no_repeat_ngram_size`.
- **הגבלות:** מאזין ל-127.0.0.1 כברירת מחדל. גוף הבקשה מוגבל ל-1MB. CORS רק עם `--cors`.

---

## 12. אפליקציית הדפדפן

- **מבנה:** `App.tsx` (2,843 שורות) ו-26 רכיבים ב-`src/components/`.
- **עברית ואנגלית:** החלפת שפה מלאה, כולל כיוון RTL.
- **ארכיטקטורת ברירת המחדל:** Transformer (מגרסה 4.4.0). הגדרות ממשק שמורות גוברות עליה.
- **לוחות עיקריים:**

| לוח | מה הוא עושה | מקור הנתונים |
|---|---|---|
| Training | נתונים, היפר-פרמטרים, ארכיטקטורה, אופטימייזר, טוקנייזר, עצירה באמצע אפוק | המודל |
| Chat | יצירה עם כל שיטות הפענוח, סליידרים לבקרת לולאות, Monte-Carlo לאי-ודאות | המודל |
| Model Metrics | עקומות loss, ייצוא CSV | המודל |
| Explainability | ערכי Shapley (מדויקים עד 10 טוקנים), integrated gradients, attention rollout | המודל |
| Information Theory | הערכות I(X;Z) ו-I(Z;Y) בשיטת binning (ניטור בלבד) | המודל |
| Edge learning | Fisher אמפירי של שכבת הפלט, אנטרופיה, חסם Cramér-Rao | המודל |
| Embeddings | t-SNE / UMAP | המודל |
| Cerebro | הזרקת נוירונים ו-rollback | המודל |
| Compression | quantization, distillation, low-rank | המודל |
| Projects / Runs / Scenarios | ניהול פרויקטים, השוואת ריצות, יומן החלטות (Decision Ledger) | נתוני המשתמש |
| Brain | מצב רוח וצרכים של "המוח" לפי אירועי אימון ויצירה. רק מציע פעולות | אירועים |
| Governance | כיול אוטומטי מבוקר: שינוי קטן ויחיד, מתועד ביומן (Σ-SIG) | מדדי האימון |
| Causal Analysis | מנוע הסקה סיבתית | **סימולטור, ומסומן ככזה** |
| Export | חבילה, JSON, טוקנייזר, היסטוריה | המודל |

- **PWA:** service worker שומר במטמון כל קובץ של ה-build. האפליקציה ניתנת להתקנה ועובדת בלי רשת
  אחרי ביקור אחד.
- **שולחן עבודה:** Electron 39 עם `contextIsolation`, `sandbox` ובלי `nodeIntegration`. WebGPU
  מופעל, ואם תהליך ה-GPU קורס האפליקציה עולה מחדש בלי האצת חומרה. ה-build הוא Windows NSIS.

---

## 13. מודולים שקיימים כספרייה בלבד

הקוד קיים ובדוק, אבל לולאת האימון והממשק לא משתמשים בו:

- `BPETokenizer.ts`
- שכבת dataset ושחזוריות (`src/data/Dataset.ts`, `src/lib/Reproducibility.ts`)
- attention דליל ולינארי (BigBird, Longformer, block, strided, axial)
- K-FAC ו-mixed precision
- `src/integrations`: שירותי HuggingFace, Slack ו-Prometheus מדומים
- `src/autodiff`
- הרכיב `GovernanceBoard`
- `src/math`: ניתוח נומרי, התכנסות, תורת האינפורמציה, גרפים ספקטרליים, NTK ו-DAG

---

## 14. מבנה תיקיות

| תיקייה | קבצים | שורות | תוכן |
|---|---|---|---|
| `src/lib` | 36 | 18,792 | מודלים, אריזה, ממשל, Brain, הזרקה, כלים מתמטיים |
| `src/components` | 26 | 13,175 | לוחות React |
| `src/math` | 13 | 8,795 | ספריית ניתוח מתמטי |
| `src/types` | 14 | 4,670 | טיפוסים |
| `src/integrations` | 8 | 4,009 | שירותים מדומים |
| `src/models` | 7 | 2,508 | attention, backprop של ה-Transformer, attention דליל |
| `src/training` | 8 | 2,445 | אופטימייזרים, clipping, EMA |
| `src/data` | 5 | 1,988 | הכנת קורפוס, Dataset |
| `src/backend` | 4 | 1,827 | WebGPU, edge learning, mixed precision |
| `src/losses` | 4 | 1,721 | loss mask, IB, VIB |
| `src/compression` | 5 | 1,253 | quantization, distillation, low-rank |
| `src/contexts` | 2 | 1,253 | BrainContext, ProjectContext |
| `src/factory` | 5 | 1,154 | pipeline, workers, שרת |
| `src/explainability` | 6 | 1,046 | SHAP, IG, rollout, conformal |
| `src/tokenizer` | 3 | 759 | subword, BPE ישן |
| `src/generation` | 1 | 740 | דגימה ובקרת לולאות |
| `tests/` | 90 קבצי בדיקה | 26,143 | מראה של `src/` |

**תיקיות נוספות:**

- `docs/`: מדריכים, ארכיטקטורה, תיאוריה, מבחנים ו-roadmap. נקודת הכניסה היא `docs/INDEX.md`.
- `scripts/`: CLI, מבחנים, ניסויים ובניית קורפוס.
- `configs/`: קונפיגורציות בסיס.
- `data/`: `corpus.txt` ודוגמאות של חדשות בעברית, דעות בעברית ו-wikitext.
- מודולי Python:
  - `edge_formalism/`: ניתוח למידה בקצה היציבות.
  - `symmetry_coupling/`: מדד צימוד סימטריה ומילוני מילים נרדפות בעברית ובאנגלית.
  - `neurosync/`: ניתוח EEG.

---

## 15. בנייה, בדיקות ו-CI

| תחום | פרטים |
|---|---|
| שפה | TypeScript 5 במצב strict, ES2020 |
| ממשק | React 18 |
| בנייה | Vite 4 |
| מנהל חבילות | pnpm 10 |
| Node | Node 20 ב-CI |
| בדיקות | Vitest 1 עם jsdom, ו-Testing Library |
| איכות קוד | ESLint 8 (אפס אזהרות) ו-Prettier 3 |

- **בדיקות:** 90 קבצים, 1,641 בדיקות עוברות ו-4 מדולגות.
  - gradient checks לכל ארכיטקטורה.
  - parity בין CPU ל-GPU.
  - שחזוריות מדויקת של resume ושל אימון מקבילי.
  - שלמות ההזרקה.
  - אכיפת אופליין.
- **CI** (`.github/workflows/ci.yml`, בכל push ל-main ובכל PR): `tsc --noEmit`, `pnpm check`
  (lint, format, test) ו-`pnpm build`.
- **Workflows נוספים:**
  - `deploy-pages.yml`: פריסה ל-GitHub Pages בכל push ל-main.
  - `train-model.yml`: אימון מחדש, מופעל ידנית, ומבצע commit למודל.
  - `build-desktop.yml`: build של Electron לכל תג `v*`.
- **פקודות:** `pnpm install`, `pnpm dev`, `pnpm test`, `pnpm check`, `pnpm build`,
  `pnpm electron:dev` ו-`pnpm electron:build`.

**דפדפנים:** האפליקציה רצה בכל דפדפן מודרני. WebGPU נתמך ב-Chrome ו-Edge מגרסה 113, ב-Opera
מגרסה 99, וב-Firefox מגרסה 127 רק עם flag. ב-Safari אין תמיכה, והאפליקציה נופלת למעבד.

---

## 16. מגבלות ידועות וכיווני המשך

| # | מגבלה | השפעה | כיוון |
|---|---|---|---|
| 1 | ה-MLP-ים עיוורים לסדר | לא לומדים תחביר. גרועים ממודל n-gram פשוט | שרשור הטמעות לפי מיקום (Bengio 2003), או הורדה ממעמד מודל ראשי |
| 2 | הקשר 32 מייקר את האימון בממשק | עם ברירות המחדל של הממשק (כ-1.17 מיליון פרמטרים) נמדדו 188 אלפיות שנייה לטוקן, לעומת 32 בהקשר 4. הממשק מאמן על כל חלון חופף (`trainingStride` 1) | `trainingStride` 8 (נמדד: 20 אלפיות שנייה לטוקן) או 32 (6.5), כמו ב-CLI |
| 3 | BPE על טקסט קצר | בקורפוס של מאות תווים כמעט לא נלמדים מיזוגים, והטוקנים קרובים לתווים בודדים (1.16 תווים לטוקן בטקסט ברירת המחדל, 3.8 ב-`data/corpus.txt`) | קורפוס של אלפי תווים ומעלה |
| 4 | ההקשר של `tiny` הוא 32 תווים | הפרומפט כמעט לא משפיע על ההמשך | preset `small` ומעלה |
| 5 | חישוב ב-JavaScript על המעבד | `tiny` מאמן כ-900 טוקנים בשנייה על ליבה אחת. קורפוס של כמה MB מעשי, של מאות MB לא | — |
| 6 | WebGPU מאיץ רק את ה-MLP | לא נמדדה האצה. ה-Transformer תמיד על המעבד | — |

---

## 17. רישיון

MIT, © 2025 Neuro-Lingua Contributors. המשתמש אחראי לזכויות על טקסט האימון.
