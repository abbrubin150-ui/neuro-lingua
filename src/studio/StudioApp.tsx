/**
 * Neuro-Lingua Studio: a focused workspace (data → model → train → generate → models) built on
 * the same model classes, training path and storage as the classic UI (src/App.tsx).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ProNeuralLM } from '../lib/ProNeuralLM';
import { downloadBlob } from '../lib/utils';
import { splitForValidation } from '../lib/validationSplit';
import { DEFAULT_VALIDATION, EXAMPLE_CORPUS } from '../config/constants';
import type { ModelOrigin } from '../lib/modelExport';
import {
  DEFAULT_STUDIO_SETTINGS,
  applyPreset,
  loadSettings,
  saveSettings,
  structuredCloneSettings,
  validateSettings,
  withArchitecture,
  type PresetId,
  type StudioSettings
} from './settings';
import { TrainError, createModel, trainModel, type EpochReport, type TrainEvent } from './trainer';
import {
  deleteStoredModel,
  describeStoredModels,
  exportJson,
  exportPackage,
  generateText,
  importModelText,
  loadStoredModel,
  noRepeatDefault,
  storeModel,
  summarizeModel,
  type GenerationResult,
  type SaveEncoding,
  type StorageSlot
} from './modelIO';
import { STRINGS, type Locale, type Strings } from './strings';
import {
  Card,
  LineChart,
  NumberField,
  ProgressBar,
  Segmented,
  SelectField,
  Stat,
  Toggle
} from './components';

type Section = 'data' | 'model' | 'train' | 'generate' | 'files';
const SECTIONS: Section[] = ['data', 'model', 'train', 'generate', 'files'];
const LOCALE_KEY = 'neuro-lingua-studio-locale';
const MAX_LOG = 200;

interface LogEntry {
  id: number;
  time: number;
  text: string;
  tone: 'info' | 'ok' | 'warn' | 'error';
}

interface Output {
  id: number;
  prompt: string;
  result: GenerationResult;
}

function loadLocale(): Locale {
  try {
    return localStorage.getItem(LOCALE_KEY) === 'he' ? 'he' : 'en';
  } catch {
    return 'en';
  }
}

function formatDuration(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${s % 60}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

function formatCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  return n.toLocaleString();
}

function eventText(t: Strings, e: TrainEvent): { text: string; tone: LogEntry['tone'] } {
  const unit = (u: 'paragraph' | 'sentence') =>
    u === 'paragraph' ? t.data.unitParagraph : t.data.unitSentence;
  switch (e.kind) {
    case 'split':
      return { text: t.log.split(e.every, unit(e.unit), e.heldOut, e.total), tone: 'info' };
    case 'no-split':
      return { text: t.log.noSplit, tone: 'warn' };
    case 'tokenizer':
      return { text: t.log.tokenizer(e.subwords), tone: 'info' };
    case 'init':
      return { text: t.log.init(e.architecture, e.vocab, e.parameters), tone: 'info' };
    case 'continue':
      return { text: t.log.continue(e.epochsBefore), tone: 'info' };
    case 'early-stop':
      return { text: t.log.earlyStop(e.epoch, e.patience), tone: 'info' };
    case 'kept':
      return { text: t.log.kept(e.epoch, e.of, e.bpc), tone: 'ok' };
    case 'best':
      return { text: t.log.best(e.epoch, e.of, e.bpc), tone: 'ok' };
  }
}

export default function StudioApp() {
  const [locale, setLocale] = useState<Locale>(loadLocale);
  const t = STRINGS[locale];
  const [section, setSection] = useState<Section>('data');
  const [settings, setSettings] = useState<StudioSettings>(loadSettings);
  const [text, setText] = useState('');

  const [model, setModel] = useState<ProNeuralLM | null>(null);
  const [modelRevision, setModelRevision] = useState(0);
  const originRef = useRef<ModelOrigin | undefined>(undefined);
  const [saved, setSaved] = useState(false);
  const [lastStats, setLastStats] = useState<{ loss: number | null; bpc: number | null }>({
    loss: null,
    bpc: null
  });

  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [epochs, setEpochs] = useState<EpochReport[]>([]);
  const [runStarted, setRunStarted] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [outputs, setOutputs] = useState<Output[]>([]);
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const [log, setLog] = useState<LogEntry[]>([]);
  const logId = useRef(0);
  const [stored, setStored] = useState(describeStoredModels);
  const importRef = useRef<HTMLInputElement>(null);
  const textFileRef = useRef<HTMLInputElement>(null);

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => {
    try {
      localStorage.setItem(LOCALE_KEY, locale);
    } catch {
      // not remembered
    }
    document.documentElement.lang = locale;
    document.documentElement.dir = t.dir;
  }, [locale, t.dir]);

  const addLog = useCallback((text: string, tone: LogEntry['tone'] = 'info') => {
    setLog((prev) => {
      const next = [...prev, { id: ++logId.current, time: Date.now(), text, tone }];
      return next.length > MAX_LOG ? next.slice(next.length - MAX_LOG) : next;
    });
  }, []);

  const errors = useMemo(() => validateSettings(settings), [settings]);
  const summary = useMemo(
    () => (model ? summarizeModel(model) : null),
    // modelRevision changes when the same instance is trained further.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model, modelRevision]
  );

  const textStats = useMemo(() => {
    const trimmed = text.trim();
    return {
      characters: text.length,
      words: trimmed ? trimmed.split(/\s+/u).length : 0,
      lines: trimmed ? text.split('\n').length : 0
    };
  }, [text]);
  const split = useMemo(
    () =>
      settings.validation && text.trim()
        ? splitForValidation(text, { fraction: DEFAULT_VALIDATION.fraction })
        : null,
    [text, settings.validation]
  );

  // Parameter estimate for the chosen size (2,000-token vocabulary assumed).
  const estimate = useMemo(() => {
    if (errors.length > 0) return null;
    try {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
      for (let i = vocab.length; i < 2000; i++) vocab.push(`t${i}`);
      return createModel(settings, vocab, { mode: 'unicode' }).getParametersCount();
    } catch {
      return null;
    }
    // Only the size-related fields matter.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    errors.length,
    settings.architecture,
    settings.hiddenSize,
    settings.contextSize,
    settings.transformer.numLayers,
    settings.transformer.numHeads,
    settings.transformer.numKVHeads,
    settings.transformer.ffHiddenDim,
    settings.advanced.useLayerNorm
  ]);

  const patch = (p: Partial<StudioSettings>) => setSettings((s) => ({ ...s, ...p }));
  const patchT = (p: Partial<StudioSettings['transformer']>) =>
    setSettings((s) => ({ ...s, transformer: { ...s.transformer, ...p } }));
  const patchA = (p: Partial<StudioSettings['advanced']>) =>
    setSettings((s) => ({ ...s, advanced: { ...s.advanced, ...p } }));
  const patchG = (p: Partial<StudioSettings['generation']>) =>
    setSettings((s) => ({ ...s, generation: { ...s.generation, ...p } }));

  const adoptModel = useCallback(
    (next: ProNeuralLM | null, origin: ModelOrigin | undefined, isSaved: boolean) => {
      setModel(next);
      setModelRevision((r) => r + 1);
      originRef.current = origin;
      setSaved(isSaved);
      setEpochs([]);
      setLastStats({ loss: null, bpc: null });
    },
    []
  );

  const reportSave = useCallback(
    (encoding: SaveEncoding) => {
      if (encoding === null) addLog(t.log.saved.none, 'warn');
      else addLog(t.log.saved[encoding], encoding === 'float16' ? 'warn' : 'ok');
      setStored(describeStoredModels());
    },
    [addLog, t]
  );

  const startTraining = useCallback(
    async (mode: 'new' | 'continue') => {
      if (training) return;
      if (!text.trim()) {
        addLog(t.train.noText, 'warn');
        setSection('data');
        return;
      }
      if (mode === 'new' && errors.length > 0) {
        addLog(`${t.model.errors} ${errors.join(', ')}`, 'error');
        setSection('model');
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      setTraining(true);
      setProgress(0);
      setEpochs([]);
      setRunStarted(Date.now());
      setSection('train');
      try {
        const outcome = await trainModel(
          {
            text,
            settings: structuredCloneSettings(settings),
            mode,
            current: model,
            origin: originRef.current
          },
          {
            signal: controller.signal,
            onEvent: (e) => {
              const { text: line, tone } = eventText(t, e);
              addLog(line, tone);
            },
            onProgress: setProgress,
            onEpoch: (report, m) => {
              setEpochs((prev) => [...prev, report]);
              addLog(
                t.log.epoch(
                  report.epoch,
                  report.total,
                  report.loss,
                  report.accuracy,
                  report.validationBpc
                )
              );
              if (mode === 'continue') setModelRevision((r) => r + 1);
              else if (report.epoch === 1) setModel(m);
            }
          }
        );
        if (outcome.epochsRun === 0 && mode === 'new') {
          // Stopped before the first epoch finished: keep the previous model.
          setModel(model);
        } else {
          setModel(outcome.model);
          originRef.current = outcome.origin;
          setLastStats({ loss: outcome.loss, bpc: outcome.validationBpc });
        }
        setModelRevision((r) => r + 1);
        if (outcome.reason === 'stopped') addLog(t.log.stopped(outcome.epochsRun), 'warn');
        else addLog(t.log.done(outcome.epochsRun, formatDuration(outcome.durationMs)), 'ok');
        if (outcome.epochsRun > 0) {
          const encoding = storeModel(outcome.model, outcome.origin, {
            loss: outcome.loss,
            accuracy: outcome.accuracy
          });
          setSaved(encoding !== null);
          reportSave(encoding);
        }
      } catch (error) {
        if (error instanceof TrainError && error.code === 'vocab-mismatch') {
          addLog(t.log.vocabMismatch, 'error');
        } else if (error instanceof TrainError && error.code === 'vocab') {
          addLog(t.log.tooFewTokens, 'error');
        } else {
          addLog(
            t.log.errorPrefix + (error instanceof Error ? error.message : String(error)),
            'error'
          );
        }
      } finally {
        abortRef.current = null;
        setTraining(false);
      }
    },
    [training, text, errors, settings, model, t, addLog, reportSave]
  );

  const stopTraining = useCallback(() => abortRef.current?.abort(), []);

  const runGeneration = useCallback(async () => {
    if (!model) {
      addLog(t.gen.noModel, 'warn');
      return;
    }
    if (training || generating) return;
    setGenerating(true);
    try {
      const result = await generateText(model, prompt, settings.generation);
      setOutputs((prev) => [{ id: Date.now(), prompt, result }, ...prev].slice(0, 30));
    } catch (error) {
      addLog(t.log.errorPrefix + (error instanceof Error ? error.message : String(error)), 'error');
    } finally {
      setGenerating(false);
    }
  }, [model, training, generating, prompt, settings.generation, t, addLog]);

  // Keyboard shortcuts: Ctrl/⌘+Enter train or stop, Ctrl/⌘+G generate.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (e.key === 'Enter' && section !== 'generate') {
        e.preventDefault();
        if (training) stopTraining();
        else void startTraining('new');
      } else if (e.key.toLowerCase() === 'g' || (e.key === 'Enter' && section === 'generate')) {
        e.preventDefault();
        setSection('generate');
        void runGeneration();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [section, training, startTraining, stopTraining, runGeneration]);

  async function onTextFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const parts = await Promise.all(Array.from(files).map((f) => f.text()));
    const joined = parts.map((p) => p.trim()).join('\n\n');
    setText((prev) => (prev.trim() ? `${prev.trim()}\n\n${joined}` : joined));
    addLog(t.log.fileRead(files.length, joined.length), 'ok');
  }

  async function onImportFile(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    try {
      const imported = importModelText(await file.text(), file.name);
      adoptModel(imported.model, imported.origin, false);
      addLog(t.log.imported(imported.name ?? file.name), 'ok');
      setSection('files');
    } catch (error) {
      addLog(t.log.importFailed(error instanceof Error ? error.message : String(error)), 'error');
    }
  }

  function slotName(slot: StorageSlot) {
    return slot === 'transformer' ? t.files.slotTransformer : t.files.slotMlp;
  }

  function onLoadSlot(slot: StorageSlot) {
    const loaded = loadStoredModel(slot);
    if (!loaded) {
      addLog(t.log.notFound, 'warn');
      return;
    }
    adoptModel(loaded.model, loaded.origin, true);
    addLog(t.log.loaded(slotName(slot)), 'ok');
  }

  function onDeleteSlot(slot: StorageSlot) {
    if (!window.confirm(t.files.confirmDelete)) return;
    deleteStoredModel(slot);
    setStored(describeStoredModels());
    if (model && summary && (summary.architecture === 'transformer') === (slot === 'transformer')) {
      setSaved(false);
    }
    addLog(t.log.deleted(slotName(slot)), 'info');
  }

  function onSave() {
    if (!model) return;
    const encoding = storeModel(model, originRef.current, { loss: lastStats.loss });
    setSaved(encoding !== null);
    reportSave(encoding);
  }

  function onExport(kind: 'package' | 'json') {
    if (!model) return;
    try {
      const file = kind === 'package' ? exportPackage(model, originRef.current) : exportJson(model);
      downloadBlob(file.blob, file.fileName);
      addLog(t.log.exported(file.fileName, (file.blob.size / 1048576).toFixed(2)), 'ok');
    } catch (error) {
      addLog(t.log.errorPrefix + (error instanceof Error ? error.message : String(error)), 'error');
    }
  }

  async function onCopy(output: Output) {
    try {
      await navigator.clipboard.writeText(output.prompt + output.result.text);
      setCopiedId(output.id);
      setTimeout(() => setCopiedId((id) => (id === output.id ? null : id)), 1500);
    } catch {
      // Clipboard unavailable: nothing to do.
    }
  }

  const lastEpoch = epochs[epochs.length - 1];
  const historyPoints = useMemo(() => {
    if (epochs.length > 0) return epochs.map((e) => ({ x: e.epoch, y: e.loss }));
    if (!model) return [];
    return model
      .getTrainingHistory()
      .map((h, i) => ({ x: i + 1, y: h.loss }))
      .filter((p) => Number.isFinite(p.y));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [epochs, model, modelRevision]);
  const bpcPoints = epochs
    .filter((e) => e.validationBpc !== undefined && Number.isFinite(e.validationBpc))
    .map((e) => ({ x: e.epoch, y: e.validationBpc as number }));

  const origin = originRef.current;
  const provenance = origin?.training
    ? t.files.provTrained(origin.training.characters.toLocaleString(), origin.training.epochs)
    : origin?.base
      ? t.files.provImported(origin.base.name)
      : '—';

  const archLabel = (a: string) =>
    a === 'transformer'
      ? t.model.archTransformer
      : a === 'advanced'
        ? t.model.archAdvanced
        : t.model.archFeedforward;
  const tokLabel = (m: string) =>
    m === 'bpe' ? t.model.tokBpe : m === 'character' ? t.model.tokCharacter : t.model.tokWord;

  const navBadge: Record<Section, string | null> = {
    data: textStats.characters ? formatCount(textStats.characters) : null,
    model: errors.length ? '!' : null,
    train: training ? `${Math.round(progress * 100)}%` : null,
    generate: outputs.length ? String(outputs.length) : null,
    files: model ? '●' : null
  };

  const s = settings;
  const isTransformer = s.architecture === 'transformer';

  return (
    <div className="studio" dir={t.dir}>
      <header className="topbar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            NL
          </span>
          <div>
            <div className="brand__name">{t.appName}</div>
            <div className="brand__tag">{t.tagline}</div>
          </div>
        </div>
        <div className="topbar__status" aria-live="polite">
          {training ? (
            <div className="status status--busy">
              <span className="status__dot" />
              <span>
                {t.status.training} · {t.train.epoch} {lastEpoch ? lastEpoch.epoch : 0}/
                {Math.round(s.epochs)}
              </span>
              <ProgressBar value={progress} label={t.status.training} />
            </div>
          ) : summary ? (
            <div className="status status--ready">
              <span className="status__dot" />
              <span>
                {archLabel(summary.architecture)} · {formatCount(summary.parameters)} ·{' '}
                {saved ? t.status.saved : t.status.unsaved}
              </span>
            </div>
          ) : (
            <div className="status">
              <span className="status__dot" />
              <span>{t.status.noModel}</span>
            </div>
          )}
        </div>
        <div className="topbar__actions">
          <button
            type="button"
            className="btn btn--ghost"
            aria-label={t.languageAria}
            onClick={() => setLocale(locale === 'en' ? 'he' : 'en')}
          >
            {t.language}
          </button>
          <a className="btn btn--ghost" href="./classic.html">
            {t.classicUi}
          </a>
        </div>
      </header>

      <div className="layout">
        <nav className="sidenav" aria-label="Studio">
          {SECTIONS.map((id, i) => (
            <button
              key={id}
              type="button"
              className="sidenav__item"
              aria-current={section === id ? 'page' : undefined}
              onClick={() => setSection(id)}
            >
              <span className="sidenav__num">{i + 1}</span>
              <span className="sidenav__text">
                <span className="sidenav__title">{t.nav[id]}</span>
                <span className="sidenav__hint">{t.navHint[id]}</span>
              </span>
              {navBadge[id] && <span className="sidenav__badge">{navBadge[id]}</span>}
            </button>
          ))}
        </nav>

        <main className="content">
          {section === 'data' && (
            <Card
              title={t.data.title}
              actions={
                <>
                  <button type="button" className="btn" onClick={() => setText(EXAMPLE_CORPUS)}>
                    {t.data.example}
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => textFileRef.current?.click()}
                  >
                    {t.data.openFile}
                  </button>
                  <button
                    type="button"
                    className="btn btn--ghost"
                    disabled={!text}
                    onClick={() => setText('')}
                  >
                    {t.data.clear}
                  </button>
                  <input
                    ref={textFileRef}
                    type="file"
                    accept=".txt,.md,text/plain"
                    multiple
                    hidden
                    onChange={(e) => {
                      void onTextFiles(e.target.files);
                      e.target.value = '';
                    }}
                  />
                </>
              }
            >
              <p className="lead">{t.data.lead}</p>
              <textarea
                className="corpus"
                value={text}
                dir="auto"
                spellCheck={false}
                placeholder={t.data.placeholder}
                aria-label={t.data.title}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="stats-row">
                <Stat label={t.data.characters} value={textStats.characters.toLocaleString()} />
                <Stat label={t.data.words} value={textStats.words.toLocaleString()} />
                <Stat label={t.data.lines} value={textStats.lines.toLocaleString()} />
              </div>
              {text.trim() && (
                <p className="note">
                  {!s.validation
                    ? t.data.splitOff
                    : split
                      ? t.data.splitYes(
                          split.every,
                          split.unit === 'paragraph' ? t.data.unitParagraph : t.data.unitSentence,
                          split.validation.length
                        )
                      : t.data.splitNo}
                </p>
              )}
              <p className="note note--warn">{t.data.privacy}</p>
            </Card>
          )}

          {section === 'model' && (
            <>
              <Card
                title={t.model.title}
                actions={
                  <button
                    type="button"
                    className="btn btn--ghost"
                    disabled={training}
                    onClick={() => setSettings(structuredCloneSettings(DEFAULT_STUDIO_SETTINGS))}
                  >
                    {t.model.reset}
                  </button>
                }
              >
                <div className="preset-row">
                  <span className="field__label">{t.model.presets}</span>
                  {(
                    [
                      ['transformer-tiny', t.model.presetTiny],
                      ['transformer-small', t.model.presetSmall],
                      ['mlp', t.model.presetMlp]
                    ] as [PresetId, string][]
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      className="chip"
                      disabled={training}
                      onClick={() => setSettings((prev) => applyPreset(prev, id))}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <p className="note">{t.model.presetNote}</p>
                <Segmented
                  label={t.model.architecture}
                  value={s.architecture}
                  disabled={training}
                  onChange={(a) => setSettings((prev) => withArchitecture(prev, a))}
                  options={[
                    {
                      value: 'transformer',
                      label: t.model.archTransformer,
                      note: t.model.archTransformerNote
                    },
                    {
                      value: 'feedforward',
                      label: t.model.archFeedforward,
                      note: t.model.archFeedforwardNote
                    },
                    {
                      value: 'advanced',
                      label: t.model.archAdvanced,
                      note: t.model.archAdvancedNote
                    }
                  ]}
                />
                <div className="grid">
                  <SelectField
                    label={t.model.tokenizer}
                    value={s.tokenizer}
                    disabled={training}
                    onChange={(v) => patch({ tokenizer: v })}
                    options={[
                      { value: 'bpe', label: t.model.tokBpe },
                      { value: 'character', label: t.model.tokCharacter },
                      { value: 'unicode', label: t.model.tokWord }
                    ]}
                  />
                  {s.tokenizer === 'bpe' && (
                    <NumberField
                      label={t.model.bpeVocab}
                      value={s.bpeVocabSize}
                      min={100}
                      max={8000}
                      step={100}
                      onChange={(v) => patch({ bpeVocabSize: v })}
                    />
                  )}
                </div>
              </Card>

              <Card title={t.model.size}>
                <div className="grid">
                  <NumberField
                    label={t.model.hiddenSize}
                    value={s.hiddenSize}
                    min={16}
                    max={256}
                    step={8}
                    onChange={(v) => patch({ hiddenSize: v })}
                  />
                  <NumberField
                    label={t.model.contextSize}
                    value={s.contextSize}
                    min={2}
                    max={128}
                    step={1}
                    onChange={(v) => patch({ contextSize: v })}
                  />
                  {isTransformer && (
                    <>
                      <NumberField
                        label={t.model.layers}
                        value={s.transformer.numLayers}
                        min={1}
                        max={8}
                        step={1}
                        onChange={(v) => patchT({ numLayers: v })}
                      />
                      <NumberField
                        label={t.model.heads}
                        value={s.transformer.numHeads}
                        min={1}
                        max={16}
                        step={1}
                        onChange={(v) => patchT({ numHeads: v })}
                      />
                      <NumberField
                        label={t.model.kvHeads}
                        value={s.transformer.numKVHeads}
                        min={1}
                        max={16}
                        step={1}
                        onChange={(v) => patchT({ numKVHeads: v })}
                      />
                      <NumberField
                        label={t.model.ffHidden}
                        value={s.transformer.ffHiddenDim}
                        min={32}
                        max={2048}
                        step={32}
                        onChange={(v) => patchT({ ffHiddenDim: v })}
                      />
                    </>
                  )}
                </div>
                {estimate !== null && (
                  <p className="note">{t.model.estimate(estimate.toLocaleString())}</p>
                )}
              </Card>

              <Card title={t.model.training}>
                <div className="grid">
                  <NumberField
                    label={t.model.epochs}
                    value={s.epochs}
                    min={1}
                    max={200}
                    step={1}
                    onChange={(v) => patch({ epochs: v })}
                  />
                  <NumberField
                    label={t.model.learningRate}
                    value={s.learningRate}
                    min={0.0001}
                    max={0.1}
                    step={0.0005}
                    onChange={(v) => patch({ learningRate: v })}
                  />
                  <SelectField
                    label={t.model.optimizer}
                    value={s.optimizer}
                    onChange={(v) =>
                      patch({ optimizer: v, learningRate: v === 'adam' ? 0.003 : 0.01 })
                    }
                    options={[
                      { value: 'adam', label: 'Adam' },
                      { value: 'momentum', label: 'SGD + momentum' }
                    ]}
                  />
                  {s.optimizer === 'momentum' && (
                    <NumberField
                      label={t.model.momentum}
                      value={s.momentum}
                      min={0}
                      max={0.99}
                      step={0.01}
                      onChange={(v) => patch({ momentum: v })}
                    />
                  )}
                </div>
                <Toggle
                  label={t.model.validation}
                  checked={s.validation}
                  onChange={(v) => patch({ validation: v })}
                />
                <details className="more">
                  <summary>{t.model.more}</summary>
                  <div className="grid">
                    <NumberField
                      label={t.model.dropout}
                      value={s.dropout}
                      min={0}
                      max={0.5}
                      step={0.01}
                      onChange={(v) => patch({ dropout: v })}
                    />
                    <NumberField
                      label={t.model.seed}
                      value={s.seed}
                      step={1}
                      onChange={(v) => patch({ seed: v })}
                    />
                    {isTransformer && (
                      <>
                        <NumberField
                          label={t.model.attnDropout}
                          value={s.transformer.attentionDropout}
                          min={0}
                          max={0.5}
                          step={0.01}
                          onChange={(v) => patchT({ attentionDropout: v })}
                        />
                        <NumberField
                          label={t.model.dropConnect}
                          value={s.transformer.dropConnectRate}
                          min={0}
                          max={0.5}
                          step={0.01}
                          onChange={(v) => patchT({ dropConnectRate: v })}
                        />
                      </>
                    )}
                  </div>
                </details>
              </Card>

              {s.architecture === 'advanced' && (
                <Card title={t.model.advanced}>
                  <div className="grid">
                    <SelectField
                      label={t.model.activation}
                      value={s.advanced.activation}
                      onChange={(v) => patchA({ activation: v })}
                      options={[
                        { value: 'relu', label: 'ReLU' },
                        { value: 'leaky_relu', label: 'Leaky ReLU' },
                        { value: 'elu', label: 'ELU' },
                        { value: 'gelu', label: 'GELU' }
                      ]}
                    />
                    <SelectField
                      label={t.model.lrSchedule}
                      value={s.advanced.lrSchedule}
                      onChange={(v) => patchA({ lrSchedule: v })}
                      options={[
                        { value: 'constant', label: 'constant' },
                        { value: 'cosine', label: 'cosine' },
                        { value: 'exponential', label: 'exponential' },
                        { value: 'warmup_cosine', label: 'warmup + cosine' }
                      ]}
                    />
                    <NumberField
                      label={t.model.weightDecay}
                      value={s.advanced.weightDecay}
                      min={0}
                      max={0.01}
                      step={0.0001}
                      onChange={(v) => patchA({ weightDecay: v })}
                    />
                    <NumberField
                      label={t.model.clipNorm}
                      value={s.advanced.gradientClipNorm}
                      min={1}
                      max={10}
                      step={0.5}
                      onChange={(v) => patchA({ gradientClipNorm: v })}
                    />
                  </div>
                  <Toggle
                    label={t.model.layerNorm}
                    checked={s.advanced.useLayerNorm}
                    onChange={(v) => patchA({ useLayerNorm: v })}
                  />
                </Card>
              )}

              {errors.length > 0 && (
                <p className="note note--error" role="alert">
                  {t.model.errors} {errors.join(' · ')}
                </p>
              )}
            </>
          )}

          {section === 'train' && (
            <>
              <Card title={t.train.title} actions={<span className="kbd">{t.train.shortcut}</span>}>
                <div className="train-actions">
                  {training ? (
                    <button
                      type="button"
                      className="btn btn--danger btn--lg"
                      onClick={stopTraining}
                    >
                      {t.train.stop}
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="btn btn--primary btn--lg"
                        disabled={!text.trim() || errors.length > 0}
                        onClick={() => void startTraining('new')}
                      >
                        {t.train.trainNew}
                      </button>
                      <button
                        type="button"
                        className="btn btn--lg"
                        disabled={!text.trim() || !model}
                        onClick={() => void startTraining('continue')}
                      >
                        {t.train.continue}
                      </button>
                    </>
                  )}
                </div>
                {!text.trim() && <p className="note note--warn">{t.train.noText}</p>}
                {model && !training && <p className="note">{t.train.continueHint}</p>}
                {(training || epochs.length > 0) && (
                  <ProgressBar value={progress} label={t.train.title} />
                )}
                <div className="stats-row">
                  <Stat
                    label={t.train.epoch}
                    value={lastEpoch ? `${lastEpoch.epoch}/${lastEpoch.total}` : '—'}
                  />
                  <Stat
                    label={t.train.loss}
                    value={
                      lastEpoch
                        ? lastEpoch.loss.toFixed(3)
                        : lastStats.loss !== null
                          ? lastStats.loss.toFixed(3)
                          : '—'
                    }
                  />
                  <Stat
                    label={t.train.accuracy}
                    value={lastEpoch ? `${(lastEpoch.accuracy * 100).toFixed(1)}%` : '—'}
                  />
                  <Stat
                    tone="accent"
                    label={t.train.valBpc}
                    value={
                      !training && lastStats.bpc !== null && Number.isFinite(lastStats.bpc)
                        ? lastStats.bpc.toFixed(3)
                        : lastEpoch?.validationBpc !== undefined
                          ? lastEpoch.validationBpc.toFixed(3)
                          : '—'
                    }
                  />
                  <Stat
                    label={t.train.speed}
                    value={lastEpoch ? formatCount(Math.round(lastEpoch.tokensPerSec)) : '—'}
                  />
                  <Stat
                    label={t.train.elapsed}
                    value={
                      runStarted && lastEpoch
                        ? formatDuration(epochs.reduce((a, e) => a + e.ms, 0))
                        : '—'
                    }
                  />
                </div>
              </Card>

              <Card title={t.train.chartTitle}>
                {historyPoints.length === 0 ? (
                  <p className="empty">{t.train.chartEmpty}</p>
                ) : (
                  <div className="charts">
                    <LineChart label={t.train.chartLoss} points={historyPoints} color="a" />
                    {bpcPoints.length > 0 && (
                      <LineChart label={t.train.chartBpc} points={bpcPoints} color="b" />
                    )}
                  </div>
                )}
                <p className="note">{t.train.bpcNote}</p>
              </Card>

              <ActivityLog t={t} log={log} onClear={() => setLog([])} />
            </>
          )}

          {section === 'generate' && (
            <>
              <Card title={t.gen.title} actions={<span className="kbd">{t.gen.shortcut}</span>}>
                {!model && <p className="note note--warn">{t.gen.noModel}</p>}
                <label className="field field--wide">
                  <span className="field__label">{t.gen.prompt}</span>
                  <textarea
                    className="prompt"
                    dir="auto"
                    value={prompt}
                    placeholder={t.gen.promptPlaceholder}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </label>
                <div className="grid">
                  <SelectField
                    label={t.gen.decoding}
                    value={s.generation.sampling}
                    onChange={(v) => patchG({ sampling: v })}
                    options={[
                      { value: 'topp', label: t.gen.topp },
                      { value: 'topk', label: t.gen.topk },
                      { value: 'typical', label: t.gen.typical },
                      { value: 'greedy', label: t.gen.greedy }
                    ]}
                  />
                  {s.generation.sampling !== 'greedy' && (
                    <NumberField
                      label={t.gen.temperature}
                      value={s.generation.temperature}
                      min={0.05}
                      max={2}
                      step={0.05}
                      onChange={(v) => patchG({ temperature: v })}
                    />
                  )}
                  {s.generation.sampling === 'topk' && (
                    <NumberField
                      label={t.gen.topK}
                      value={s.generation.topK}
                      min={1}
                      max={1000}
                      step={1}
                      onChange={(v) => patchG({ topK: v })}
                    />
                  )}
                  {s.generation.sampling === 'topp' && (
                    <NumberField
                      label={t.gen.topP}
                      value={s.generation.topP}
                      min={0.05}
                      max={0.99}
                      step={0.01}
                      onChange={(v) => patchG({ topP: v })}
                    />
                  )}
                  {s.generation.sampling === 'typical' && (
                    <NumberField
                      label={t.gen.tau}
                      value={s.generation.typicalTau}
                      min={0.1}
                      max={0.99}
                      step={0.01}
                      onChange={(v) => patchG({ typicalTau: v })}
                    />
                  )}
                  <NumberField
                    label={t.gen.maxTokens}
                    value={s.generation.maxTokens}
                    min={1}
                    max={400}
                    step={10}
                    onChange={(v) => patchG({ maxTokens: v })}
                  />
                  <NoRepeatField
                    t={t}
                    value={s.generation.noRepeatNgram}
                    auto={model ? noRepeatDefault(model) : null}
                    onChange={(v) => patchG({ noRepeatNgram: v })}
                  />
                </div>
                <div className="train-actions">
                  <button
                    type="button"
                    className="btn btn--primary btn--lg"
                    disabled={!model || training || generating}
                    onClick={() => void runGeneration()}
                  >
                    {generating ? t.gen.running : t.gen.run}
                  </button>
                </div>
              </Card>

              <Card
                title={t.gen.output}
                actions={
                  outputs.length > 0 && (
                    <button type="button" className="btn btn--ghost" onClick={() => setOutputs([])}>
                      {t.gen.clear}
                    </button>
                  )
                }
              >
                {outputs.length === 0 ? (
                  <p className="empty">{t.gen.empty}</p>
                ) : (
                  <ol className="outputs">
                    {outputs.map((o) => (
                      <li key={o.id} className="output">
                        <p className="output__text" dir="auto">
                          <span className="output__prompt">{o.prompt}</span>
                          {o.result.text}
                        </p>
                        <div className="output__meta">
                          <span>
                            {t.gen.meta(o.result.tokens, o.result.ms, o.result.finishReason)}
                          </span>
                          <button
                            type="button"
                            className="btn btn--ghost btn--sm"
                            onClick={() => void onCopy(o)}
                          >
                            {copiedId === o.id ? t.gen.copied : t.gen.copy}
                          </button>
                        </div>
                      </li>
                    ))}
                  </ol>
                )}
              </Card>
            </>
          )}

          {section === 'files' && (
            <>
              <Card
                title={t.files.current}
                actions={
                  model && (
                    <button
                      type="button"
                      className="btn btn--ghost"
                      disabled={training}
                      onClick={() => {
                        adoptModel(null, undefined, false);
                        addLog(t.log.unloaded);
                      }}
                    >
                      {t.files.unload}
                    </button>
                  )
                }
              >
                {!summary ? (
                  <p className="empty">{t.files.none}</p>
                ) : (
                  <>
                    <dl className="facts">
                      <Fact k={t.files.arch} v={archLabel(summary.architecture)} />
                      <Fact k={t.files.params} v={summary.parameters.toLocaleString()} />
                      <Fact k={t.files.vocab} v={summary.vocab.toLocaleString()} />
                      <Fact k={t.files.tokenizer} v={tokLabel(summary.tokenizer)} />
                      <Fact k={t.files.hidden} v={String(summary.hiddenSize)} />
                      <Fact k={t.files.context} v={String(summary.contextSize)} />
                      {summary.layers !== undefined && (
                        <Fact
                          k={t.files.layersHeads}
                          v={`${summary.layers} · ${summary.heads} (${summary.kvHeads})`}
                        />
                      )}
                      <Fact k={t.files.epochs} v={String(summary.epochsTrained)} />
                      <Fact k={t.files.provenance} v={provenance} />
                    </dl>
                    <div className="train-actions">
                      <button
                        type="button"
                        className="btn btn--primary"
                        disabled={training}
                        onClick={onSave}
                      >
                        {t.files.saveBrowser}
                      </button>
                      <button
                        type="button"
                        className="btn"
                        disabled={training}
                        onClick={() => onExport('package')}
                      >
                        {t.files.exportPackage}
                      </button>
                      <button
                        type="button"
                        className="btn btn--ghost"
                        disabled={training}
                        onClick={() => onExport('json')}
                      >
                        {t.files.exportJson}
                      </button>
                    </div>
                    <p className="note">{t.files.packageNote}</p>
                  </>
                )}
              </Card>

              <Card title={t.files.stored}>
                <ul className="slots">
                  {stored.map((slot) => (
                    <li key={slot.slot} className="slot">
                      <div>
                        <div className="slot__name">{slotName(slot.slot)}</div>
                        <div className="slot__meta">
                          {slot.present
                            ? [
                                slot.timestamp ? new Date(slot.timestamp).toLocaleString() : null,
                                slot.vocab ? `${slot.vocab.toLocaleString()} tokens` : null
                              ]
                                .filter(Boolean)
                                .join(' · ') || '✓'
                            : t.files.empty}
                        </div>
                      </div>
                      <div className="slot__actions">
                        <button
                          type="button"
                          className="btn btn--sm"
                          disabled={!slot.present || training}
                          onClick={() => onLoadSlot(slot.slot)}
                        >
                          {t.files.load}
                        </button>
                        <button
                          type="button"
                          className="btn btn--ghost btn--sm"
                          disabled={!slot.present || training}
                          onClick={() => onDeleteSlot(slot.slot)}
                        >
                          {t.files.delete}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
                <p className="note">{t.files.sharedNote}</p>
              </Card>

              <Card title={t.files.import}>
                <button
                  type="button"
                  className="btn"
                  disabled={training}
                  onClick={() => importRef.current?.click()}
                >
                  {t.files.import}
                </button>
                <input
                  ref={importRef}
                  type="file"
                  accept=".json,application/json"
                  hidden
                  onChange={(e) => {
                    void onImportFile(e.target.files);
                    e.target.value = '';
                  }}
                />
                <p className="note">{t.files.importNote}</p>
              </Card>
              <ActivityLog t={t} log={log} onClear={() => setLog([])} />
            </>
          )}
        </main>
      </div>

      {log.length > 0 && section !== 'train' && section !== 'files' && (
        <footer className="ticker" aria-live="polite">
          <span className={`ticker__dot ticker__dot--${log[log.length - 1].tone}`} />
          <span className="ticker__text">{log[log.length - 1].text}</span>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => setSection('train')}
          >
            {t.train.log}
          </button>
        </footer>
      )}
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="fact">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  );
}

function NoRepeatField(props: {
  t: Strings;
  value: number | null;
  auto: number | null;
  onChange: (value: number | null) => void;
}) {
  const { t, value, auto, onChange } = props;
  return (
    <div className="field field--full">
      <Toggle
        label={`${t.gen.noRepeat}: ${t.gen.noRepeatAuto(auto ?? 3)}`}
        checked={value === null}
        onChange={(checked) => onChange(checked ? null : (auto ?? 3))}
      />
      {value !== null && (
        <NumberField
          label={t.gen.noRepeat}
          value={value}
          min={0}
          max={32}
          step={1}
          onChange={onChange}
        />
      )}
      <span className="field__hint">{t.gen.noRepeatNote}</span>
    </div>
  );
}

function ActivityLog(props: { t: Strings; log: LogEntry[]; onClear: () => void }) {
  const { t, log, onClear } = props;
  const listRef = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const list = listRef.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [log.length]);
  return (
    <Card
      title={t.train.log}
      className="card--log"
      actions={
        log.length > 0 && (
          <button type="button" className="btn btn--ghost btn--sm" onClick={onClear}>
            {t.train.clearLog}
          </button>
        )
      }
    >
      {log.length === 0 ? (
        <p className="empty">—</p>
      ) : (
        <ol className="log" ref={listRef}>
          {log.map((entry) => (
            <li key={entry.id} className={`log__entry log__entry--${entry.tone}`} dir="auto">
              <time>{new Date(entry.time).toLocaleTimeString()}</time>
              <span>{entry.text}</span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
