import React, { useCallback, useEffect, useRef, useState } from 'react';

import {
  ProNeuralLM,
  type Optimizer,
  type TokenizerConfig,
  clamp,
  MODEL_VERSION,
  MODEL_STORAGE_KEY
} from './lib/ProNeuralLM';

import {
  AdvancedNeuralLM,
  type ActivationFunction,
  type LRSchedule,
  type InitializationScheme
} from './lib/AdvancedNeuralLM';

import { TransformerLM } from './lib/TransformerLM';

import { StorageManager } from './lib/storage';
import { buildVocab, parseTokenizerConfig, downloadBlob } from './lib/utils';
import type { GPUNeuralOps } from './backend/gpu_neural_ops';
import {
  STORAGE_KEYS,
  DEFAULT_TRAINING_TEXT,
  DEFAULT_HYPERPARAMETERS,
  DEFAULT_GENERATION,
  DEFAULT_ADVANCED_CONFIG,
  DEFAULT_TOKENIZER_CONFIG,
  MIN_VOCAB_SIZE,
  TRAINING_UI_UPDATE_DELAY,
  EXAMPLE_CORPUS
} from './config/constants';

import {
  OnboardingCard,
  type OnboardingCardStrings,
  ModelMetrics,
  ChatInterface,
  type ChatInterfaceStrings,
  TrainingPanel,
  type Architecture,
  ErrorBoundary,
  type Message
} from './components';

type UiSettings = {
  architecture: Architecture;
  hiddenSize: number;
  epochs: number;
  lr: number;
  optimizer: Optimizer;
  momentum: number;
  dropout: number;
  contextSize: number;
  temperature: number;
  topK: number;
  topP: number;
  samplingMode: 'off' | 'topk' | 'topp';
  seed: number;
  resume: boolean;
  tokenizerConfig: TokenizerConfig;
  // Advanced features
  useAdvanced: boolean;
  useGPU: boolean;
  activation: ActivationFunction;
  leakyReluAlpha: number;
  eluAlpha: number;
  initialization: InitializationScheme;
  lrSchedule: LRSchedule;
  lrMin: number;
  lrDecayRate: number;
  warmupEpochs: number;
  weightDecay: number;
  gradientClipNorm: number;
  useLayerNorm: boolean;
  useBeamSearch: boolean;
  beamWidth: number;
};

type ModelMeta = { timestamp: number; vocab: number };

type Locale = 'en' | 'he';

type AppTranslations = {
  toggle: { button: string; aria: string };
  title: string;
  subtitle: string;
  training: {
    heading: string;
    placeholder: string;
    textareaAria: string;
    characters: string;
    words: string;
    tip: string;
    example: string;
    exampleAria: string;
  };
  infoCards: { title: string; body: string }[];
  chat: ChatInterfaceStrings & { trainFirst: string };
  onboarding: OnboardingCardStrings;
};

const TRANSLATIONS: Record<Locale, AppTranslations> = {
  en: {
    toggle: { button: 'עברית / RTL', aria: 'Switch interface language to Hebrew (RTL)' },
    title: '🧠 Neuro‑Lingua DOMESTICA — v{version}',
    subtitle:
      'Advanced neural language model with Momentum/Adam, training-only dropout, real-time charts, and flexible context windows.',
    training: {
      heading: '🎓 Training',
      placeholder: 'Enter training text (ideally 200+ words in English or Hebrew)...',
      textareaAria: 'Training corpus',
      characters: 'Characters',
      words: 'Words',
      tip: '💡 Tip: Start with the example corpus, then paste your own dataset to compare results.',
      example: '📚 Example',
      exampleAria: 'Load example corpus'
    },
    infoCards: [
      {
        title: '🎯 Training Tips',
        body: '• 200–500 words • 20–50 epochs • LR: 0.05–0.1 • Context: 3–5'
      },
      {
        title: '🎲 Text Generation',
        body: '• Temperature: 0.7–1.0 • Choose top‑k or top‑p (top‑p ≈ 0.85–0.95)'
      },
      {
        title: '⚡ Performance',
        body: '• Momentum: 0.9 or Adam • Save tokenizer presets • Export CSV to compare runs'
      },
      {
        title: '⌨️ Shortcuts',
        body: '• Ctrl/Cmd+Enter: Train/Stop • Ctrl/Cmd+S: Save • Ctrl/Cmd+G: Generate'
      }
    ],
    chat: {
      title: '💬 Chat Console',
      replies: (count) => `${count} replies`,
      regionAria: 'Chat interface',
      logAria: 'Chat messages',
      inputAria: 'Chat prompt',
      generateAria: 'Generate text from model',
      placeholderReady: 'Type a message for the model…',
      placeholderEmpty: 'Train the model first…',
      generate: '✨ Generate',
      tip: '💡 Press Shift+Enter to add a new line. Responses reflect the active sampling mode and temperature.',
      userLabel: '👤 You',
      assistantLabel: '🤖 Model',
      systemLabel: '⚙️ System',
      userRole: 'User',
      assistantRole: 'Model',
      systemRole: 'System',
      messageSuffix: 'message',
      trainFirst: '❌ Please train the model first.'
    },
    onboarding: {
      welcomeTitle: '👋 Welcome! Here is how Neuro-Lingua keeps your session in sync',
      privacyWarningTitle: 'Privacy Warning',
      privacyWarningLead: 'DO NOT train with sensitive data.',
      privacyWarningBody:
        'Settings live in browser storage without encryption. Avoid PII, passwords, financial, medical, or confidential data.',
      bulletPauseResume:
        'Pause / Resume: use the Stop button to pause training. With Resume training enabled we pick up from the latest checkpoint when you train again.',
      bulletImportExport:
        'Import / Export: save models and tokenizer presets to JSON for safekeeping or sharing. Importing immediately refreshes the charts and metadata.',
      bulletPersistence:
        'Session Persistence: hyperparameters and tokenizer preferences live in localStorage; training text stays in memory only.',
      gotIt: 'Got it',
      reopenInfo: 'You can reopen this info from localStorage by clearing the flag.'
    }
  },
  he: {
    toggle: { button: 'English / LTR', aria: 'החלף את שפת הממשק לאנגלית (כיוון LTR)' },
    title: '🧠 ניורו-לינגואה DOMESTICA — גרסה {version}',
    subtitle:
      'מודל שפה מתקדם עם Momentum/Adam, דרופאאוט באימון בלבד, תרשימים בזמן אמת וחלונות הקשר גמישים.',
    training: {
      heading: '🎓 אימון',
      placeholder: 'הזינו טקסט אימון (200+ מילים בעברית או באנגלית)...',
      textareaAria: 'קורפוס אימון',
      characters: 'תווים',
      words: 'מילים',
      tip: '💡 טיפ: התחילו בקורפוס הדוגמה ואז הדביקו את הדאטהסט שלכם להשוואת תוצאות.',
      example: '📚 דוגמה',
      exampleAria: 'טען קורפוס דוגמה'
    },
    infoCards: [
      {
        title: '🎯 טיפים לאימון',
        body: '• 200–500 מילים • 20–50 אפוקים • קצב למידה: 0.05–0.1 • הקשר: 3–5'
      },
      {
        title: '🎲 יצירת טקסט',
        body: '• טמפ׳: 0.7–1.0 • בחרו top‑k או top‑p (top‑p ≈ 0.85–0.95)'
      },
      {
        title: '⚡ ביצועים',
        body: '• מומנטום 0.9 או Adam • שמרו פרופילי טוקנייזר • ייצאו CSV להשוואת ריצות'
      },
      {
        title: '⌨️ קיצורי דרך',
        body: '• Ctrl/Cmd+Enter: התחלה/עצירה • Ctrl/Cmd+S: שמירה • Ctrl/Cmd+G: דגימה'
      }
    ],
    chat: {
      title: '💬 קונסולת שיחה',
      replies: (count) => `${count} תגובות`,
      regionAria: 'ממשק שיחה',
      logAria: 'הודעות שיחה',
      inputAria: 'הזנת שיחה',
      generateAria: 'הפיקו טקסט מהמודל',
      placeholderReady: 'כתבו הודעה למודל…',
      placeholderEmpty: 'אימנו את המודל תחילה…',
      generate: '✨ הפקה',
      tip: '💡 Shift+Enter מוסיף שורה חדשה. התגובות תלויות במצב הדגימה ובטמפרטורה הנבחרת.',
      userLabel: '👤 אתם',
      assistantLabel: '🤖 המודל',
      systemLabel: '⚙️ מערכת',
      userRole: 'משתמש',
      assistantRole: 'מודל',
      systemRole: 'מערכת',
      messageSuffix: 'הודעה',
      trainFirst: '❌ נא לאמן את המודל תחילה.'
    },
    onboarding: {
      welcomeTitle: '👋 ברוכים הבאים! כך Neuro-Lingua מסנכרנת את הסשן שלכם',
      privacyWarningTitle: 'אזהרת פרטיות',
      privacyWarningLead: 'אל תאמן עם נתונים רגישים.',
      privacyWarningBody:
        'ההגדרות נשמרות באחסון הדפדפן ללא הצפנה. הימנעו מ-PII, סיסמאות, מידע פיננסי, רפואי או חסוי.',
      bulletPauseResume:
        'עצירה / המשך: השתמשו בכפתור העצירה כדי להקפיא את האימון. עם “המשך אימון” פעיל נמשיך מהצ׳קפוינט האחרון.',
      bulletImportExport:
        'ייבוא / ייצוא: שמרו מודלים ופרופילי טוקנייזר ל-JSON לגיבוי או שיתוף. ייבוא מרענן מיידית תרשימים ומטא־דאטה.',
      bulletPersistence:
        'התמדה בסשן: היפר־פרמטרים ופרופילי טוקנייזר נשמרים ב-localStorage; טקסט האימון נשאר בזיכרון בלבד.',
      gotIt: 'הבנתי',
      reopenInfo: 'אפשר לפתוח שוב את הכרטיס על ידי ניקוי הדגל ב-localStorage.'
    }
  }
} as const;

function loadLatestModel(): ProNeuralLM | null {
  const primary = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);
  if (primary) return primary;

  for (const legacyKey of STORAGE_KEYS.LEGACY_MODELS) {
    const legacy = ProNeuralLM.loadFromLocalStorage(legacyKey);
    if (legacy) {
      try {
        legacy.saveToLocalStorage(MODEL_STORAGE_KEY);
        StorageManager.remove(legacyKey);
      } catch (err) {
        console.warn('Failed to migrate legacy model', err);
      }
      return legacy;
    }
  }

  return null;
}

export default function NeuroLinguaDomesticaV324() {
  // Training corpus
  const [trainingText, setTrainingText] = useState(DEFAULT_TRAINING_TEXT);

  // Chat interface
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [locale, setLocale] = useState<Locale>('en');
  const direction: 'ltr' | 'rtl' = locale === 'he' ? 'rtl' : 'ltr';
  const t = TRANSLATIONS[locale];
  const toggleLocale = useCallback(() => {
    setLocale((prev) => (prev === 'en' ? 'he' : 'en'));
  }, []);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({ loss: 0, acc: 0, ppl: 0, lossEMA: 0, tokensPerSec: 0 });
  const [info, setInfo] = useState({ V: 0, P: 0 });
  const [trainingHistory, setTrainingHistory] = useState<
    { loss: number; accuracy: number; timestamp: number }[]
  >([]);

  // Hyperparameters
  const [hiddenSize, setHiddenSize] = useState(DEFAULT_HYPERPARAMETERS.hiddenSize);
  const [epochs, setEpochs] = useState(DEFAULT_HYPERPARAMETERS.epochs);
  const [lr, setLr] = useState(DEFAULT_HYPERPARAMETERS.learningRate);
  const [optimizer, setOptimizer] = useState<Optimizer>(DEFAULT_HYPERPARAMETERS.optimizer);
  const [momentum, setMomentum] = useState(DEFAULT_HYPERPARAMETERS.momentum);
  const [dropout, setDropout] = useState(DEFAULT_HYPERPARAMETERS.dropout);
  const [contextSize, setContextSize] = useState(DEFAULT_HYPERPARAMETERS.contextSize);
  const [temperature, setTemperature] = useState(DEFAULT_GENERATION.temperature);
  const [topK, setTopK] = useState(DEFAULT_GENERATION.topK);
  const [topP, setTopP] = useState(DEFAULT_GENERATION.topP);
  const [samplingMode, setSamplingMode] = useState<'off' | 'topk' | 'topp'>(
    DEFAULT_GENERATION.samplingMode
  );
  const [seed, setSeed] = useState(DEFAULT_HYPERPARAMETERS.seed);
  const [resume, setResume] = useState(DEFAULT_HYPERPARAMETERS.resume);

  // Architecture selection
  const [architecture, setArchitecture] = useState<Architecture>('feedforward');

  // Advanced features
  const [useAdvanced, setUseAdvanced] = useState(DEFAULT_ADVANCED_CONFIG.useAdvanced);
  const [activation, setActivation] = useState<ActivationFunction>(
    DEFAULT_ADVANCED_CONFIG.activation
  );
  const [leakyReluAlpha, setLeakyReluAlpha] = useState(DEFAULT_ADVANCED_CONFIG.leakyReluAlpha);
  const [eluAlpha, setEluAlpha] = useState(DEFAULT_ADVANCED_CONFIG.eluAlpha);
  const [initialization, setInitialization] = useState<InitializationScheme>(
    DEFAULT_ADVANCED_CONFIG.initialization
  );
  const [lrSchedule, setLrSchedule] = useState<LRSchedule>(DEFAULT_ADVANCED_CONFIG.lrSchedule);
  const [lrMin, setLrMin] = useState(DEFAULT_ADVANCED_CONFIG.lrMin);
  const [lrDecayRate, setLrDecayRate] = useState(DEFAULT_ADVANCED_CONFIG.lrDecayRate);
  const [warmupEpochs, setWarmupEpochs] = useState(DEFAULT_ADVANCED_CONFIG.warmupEpochs);
  const [weightDecay, setWeightDecay] = useState(DEFAULT_ADVANCED_CONFIG.weightDecay);
  const [gradientClipNorm, setGradientClipNorm] = useState(
    DEFAULT_ADVANCED_CONFIG.gradientClipNorm
  );
  const [useLayerNorm, setUseLayerNorm] = useState(DEFAULT_ADVANCED_CONFIG.useLayerNorm);
  const [useBeamSearch, setUseBeamSearch] = useState(DEFAULT_GENERATION.useBeamSearch);
  const [beamWidth, setBeamWidth] = useState(DEFAULT_GENERATION.beamWidth);

  // GPU acceleration
  const [useGPU, setUseGPU] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [gpuMetrics, setGpuMetrics] = useState<{
    enabled: boolean;
    available: boolean;
    totalOperations: number;
    totalTimeMs: number;
    averageTimeMs: number;
    deviceInfo?: string;
  } | null>(null);

  // Tokenizer
  const [tokenizerConfig, setTokenizerConfig] = useState<TokenizerConfig>(DEFAULT_TOKENIZER_CONFIG);
  const [customTokenizerPattern, setCustomTokenizerPattern] = useState('');
  const [tokenizerError, setTokenizerError] = useState<string | null>(null);

  // Model metadata and UI
  const [lastModelUpdate, setLastModelUpdate] = useState<ModelMeta | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(true);

  const modelRef = useRef<ProNeuralLM | null>(null);
  const trainingRef = useRef({ running: false, currentEpoch: 0 });
  const abortControllerRef = useRef<AbortController | null>(null);
  const gpuOpsRef = useRef<GPUNeuralOps | null>(null);

  // Helper to add system messages
  const addSystemMessage = useCallback((content: string) => {
    setMessages((m) => [...m, { type: 'system' as const, content, timestamp: Date.now() }]);
  }, []);

  const persistModelMeta = useCallback((meta: ModelMeta | null) => {
    if (!meta) {
      StorageManager.remove(STORAGE_KEYS.MODEL_META);
      setLastModelUpdate(null);
      return;
    }
    StorageManager.set(STORAGE_KEYS.MODEL_META, meta);
    setLastModelUpdate(meta);
  }, []);

  const applyModelMeta = useCallback(
    (model: ProNeuralLM) => {
      const updatedAt = model.getLastUpdatedAt();
      if (updatedAt) {
        persistModelMeta({ timestamp: updatedAt, vocab: model.getVocabSize() });
      }
    },
    [persistModelMeta]
  );

  const syncTokenizerFromModel = useCallback((model: ProNeuralLM) => {
    const config = model.getTokenizerConfig();
    setTokenizerConfig(config);
    if (config.mode === 'custom') {
      setCustomTokenizerPattern(config.pattern ?? '');
    }
  }, []);

  async function runSelfTests() {
    try {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
      const m = new ProNeuralLM(vocab, 16, 0.05, 3, 'momentum', 0.9, 0, 1234);
      const res = await m.train('hello world hello', 1);
      console.assert(
        res.loss > 0 && res.accuracy >= 0 && res.accuracy <= 1,
        '[SelfTest] train metrics valid'
      );
      const out = await m.generate('hello', 5, 0.9, 0, 0.9);
      console.assert(
        typeof out === 'string' && out.length <= 5 * 10,
        '[SelfTest] generate output (bounded)'
      );

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const m2: any = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0.3, 42);
      const fTrain = await m2.forward([1, 1, 1], true);
      const fEval = await m2.forward([1, 1, 1], false);
      console.assert(!!fTrain.dropMask && !fEval.dropMask, '[SelfTest] dropout train vs eval');

      const m3 = new ProNeuralLM(vocab, 8, 0.05, 3, 'adam', 0.9, 0.0, 7);
      const hLen0 = m3.getTrainingHistory().length;
      await m3.train('hello world hello', 2);
      const hLen1 = m3.getTrainingHistory().length;
      console.assert(hLen1 === hLen0 + 2, '[SelfTest] history grows per epoch (adam)');

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const m4: any = new ProNeuralLM(vocab, 8, 0.05, 5, 'momentum', 0.9, 0.0, 9);
      const seqs = m4.createTrainingSequences('hello');
      console.assert(seqs.length > 0, '[SelfTest] sequences exist');
      console.assert(seqs[0][0].length === 5, '[SelfTest] context window = 5');

      const sK = await m.generate('hello', 5, 0.9, 2, 0);
      const sP = await m.generate('hello', 5, 0.9, 0, 0.8);
      console.assert(typeof sK === 'string' && typeof sP === 'string', '[SelfTest] sampling modes');

      const mA = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0.0, 111);
      const mB = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0.0, 111);
      const gA = await mA.generate('hello', 6, 0.7, 0, 0.9);
      const gB = await mB.generate('hello', 6, 0.7, 0, 0.9);
      console.assert(gA === gB, '[SelfTest] deterministic generation with same seed');

      const vocab2 = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'friends'];
      const s1 = new ProNeuralLM(vocab).getVocabSignature();
      const s2 = new ProNeuralLM(vocab2).getVocabSignature();
      console.assert(s1 !== s2, '[SelfTest] vocab signature reflects vocab');

      const m5 = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0.0, 3);
      const r5 = await m5.train('hello world', 1);
      const ppl = Math.exp(Math.max(1e-8, r5.loss));
      console.assert(!Number.isNaN(ppl), '[SelfTest] perplexity finite');

      console.log('[SelfTest] OK');
    } catch (e) {
      console.warn('[SelfTest] failed', e);
    }
  }

  useEffect(() => {
    document.documentElement.setAttribute('dir', direction);
    document.documentElement.setAttribute('lang', locale === 'he' ? 'he' : 'en');
  }, [direction, locale]);

  // Load settings on mount
  useEffect(() => {
    const saved = StorageManager.get<Partial<UiSettings>>(STORAGE_KEYS.UI_SETTINGS, {});
    if (
      saved.architecture === 'feedforward' ||
      saved.architecture === 'advanced' ||
      saved.architecture === 'transformer'
    ) {
      setArchitecture(saved.architecture);
    }
    if (typeof saved.hiddenSize === 'number') setHiddenSize(saved.hiddenSize);
    if (typeof saved.epochs === 'number') setEpochs(saved.epochs);
    if (typeof saved.lr === 'number') setLr(saved.lr);
    if (
      saved.optimizer === 'adam' ||
      saved.optimizer === 'momentum' ||
      saved.optimizer === 'newton' ||
      saved.optimizer === 'bfgs'
    ) {
      setOptimizer(saved.optimizer);
    }
    if (typeof saved.momentum === 'number') setMomentum(saved.momentum);
    if (typeof saved.dropout === 'number') setDropout(saved.dropout);
    if (typeof saved.contextSize === 'number') setContextSize(saved.contextSize);
    if (typeof saved.temperature === 'number') setTemperature(saved.temperature);
    if (typeof saved.topK === 'number') setTopK(saved.topK);
    if (typeof saved.topP === 'number') setTopP(saved.topP);
    if (
      saved.samplingMode === 'off' ||
      saved.samplingMode === 'topk' ||
      saved.samplingMode === 'topp'
    ) {
      setSamplingMode(saved.samplingMode);
    }
    if (typeof saved.seed === 'number') setSeed(saved.seed);
    if (typeof saved.resume === 'boolean') setResume(saved.resume);
    if (saved.tokenizerConfig) {
      const parsed = parseTokenizerConfig(saved.tokenizerConfig);
      setTokenizerConfig(parsed);
      if (parsed.mode === 'custom') setCustomTokenizerPattern(parsed.pattern ?? '');
    }

    // Load advanced features
    if (typeof saved.useAdvanced === 'boolean') setUseAdvanced(saved.useAdvanced);
    if (typeof saved.useGPU === 'boolean') setUseGPU(saved.useGPU);
    if (saved.activation) setActivation(saved.activation);
    if (typeof saved.leakyReluAlpha === 'number') setLeakyReluAlpha(saved.leakyReluAlpha);
    if (typeof saved.eluAlpha === 'number') setEluAlpha(saved.eluAlpha);
    if (saved.initialization) setInitialization(saved.initialization);
    if (saved.lrSchedule) setLrSchedule(saved.lrSchedule);
    if (typeof saved.lrMin === 'number') setLrMin(saved.lrMin);
    if (typeof saved.lrDecayRate === 'number') setLrDecayRate(saved.lrDecayRate);
    if (typeof saved.warmupEpochs === 'number') setWarmupEpochs(saved.warmupEpochs);
    if (typeof saved.weightDecay === 'number') setWeightDecay(saved.weightDecay);
    if (typeof saved.gradientClipNorm === 'number') setGradientClipNorm(saved.gradientClipNorm);
    if (typeof saved.useLayerNorm === 'boolean') setUseLayerNorm(saved.useLayerNorm);
    if (typeof saved.useBeamSearch === 'boolean') setUseBeamSearch(saved.useBeamSearch);
    if (typeof saved.beamWidth === 'number') setBeamWidth(saved.beamWidth);

    const tokenizerRaw = StorageManager.get<unknown>(STORAGE_KEYS.TOKENIZER_CONFIG, null);
    if (tokenizerRaw) {
      const parsed = parseTokenizerConfig(tokenizerRaw);
      setTokenizerConfig(parsed);
      if (parsed.mode === 'custom') setCustomTokenizerPattern(parsed.pattern ?? '');
    }

    const meta = StorageManager.get<ModelMeta | null>(STORAGE_KEYS.MODEL_META, null);
    if (meta && typeof meta.timestamp === 'number' && typeof meta.vocab === 'number') {
      setLastModelUpdate(meta);
    }

    if (localStorage.getItem(STORAGE_KEYS.ONBOARDING_DISMISSED) === 'true') {
      setShowOnboarding(false);
    }
  }, []);

  // Load model on mount
  useEffect(() => {
    runSelfTests();
    const saved = loadLatestModel();
    if (saved) {
      modelRef.current = saved;
      setInfo({ V: saved.getVocabSize(), P: saved.getParametersCount() });
      setTrainingHistory(saved.getTrainingHistory());
      syncTokenizerFromModel(saved);
      applyModelMeta(saved);
      addSystemMessage(`📀 Model v${MODEL_VERSION} loaded from local storage`);
    }
  }, [applyModelMeta, syncTokenizerFromModel, addSystemMessage]);

  // Check WebGPU availability and initialize GPU metrics
  useEffect(() => {
    const checkGPU = async () => {
      try {
        // Check if WebGPU is available
        if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
          const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter();
          if (adapter) {
            // Initialize GPUNeuralOps
            const { GPUNeuralOps } = await import('./backend/gpu_neural_ops');
            const ops = new GPUNeuralOps();
            const initialized = await ops.initialize();
            if (initialized) {
              gpuOpsRef.current = ops;
              setGpuAvailable(true);
              setGpuMetrics({
                enabled: false,
                available: true,
                totalOperations: 0,
                totalTimeMs: 0,
                averageTimeMs: 0,
                deviceInfo: 'WebGPU Device'
              });
              console.log('✅ WebGPU is available and GPUNeuralOps initialized');
            } else {
              setGpuAvailable(false);
              setGpuMetrics({
                enabled: false,
                available: false,
                totalOperations: 0,
                totalTimeMs: 0,
                averageTimeMs: 0
              });
              console.log('⚠️ GPUNeuralOps initialization failed');
            }
          } else {
            setGpuAvailable(false);
            setGpuMetrics({
              enabled: false,
              available: false,
              totalOperations: 0,
              totalTimeMs: 0,
              averageTimeMs: 0
            });
            console.log('⚠️ WebGPU adapter not available');
          }
        } else {
          setGpuAvailable(false);
          setGpuMetrics({
            enabled: false,
            available: false,
            totalOperations: 0,
            totalTimeMs: 0,
            averageTimeMs: 0
          });
          console.log('⚠️ WebGPU is not supported in this browser');
        }
      } catch (error) {
        setGpuAvailable(false);
        setGpuMetrics({
          enabled: false,
          available: false,
          totalOperations: 0,
          totalTimeMs: 0,
          averageTimeMs: 0
        });
        console.warn('⚠️ Error checking WebGPU availability:', error);
      }
    };
    checkGPU();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const modKey = isMac ? e.metaKey : e.ctrlKey;

      // Ctrl/Cmd + Enter: Start/Stop training
      if (modKey && e.key === 'Enter') {
        e.preventDefault();
        if (isTraining) {
          onStopTraining();
        } else {
          onTrain();
        }
      }

      // Ctrl/Cmd + S: Save model
      if (modKey && e.key === 's') {
        e.preventDefault();
        onSave();
      }

      // Ctrl/Cmd + G: Generate (if input is focused)
      if (modKey && e.key === 'g') {
        e.preventDefault();
        if (modelRef.current && input.trim()) {
          onGenerate();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isTraining, input]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist settings when they change
  useEffect(() => {
    const settings: UiSettings = {
      architecture,
      hiddenSize,
      epochs,
      lr,
      optimizer,
      momentum,
      dropout,
      contextSize,
      temperature,
      topK,
      topP,
      samplingMode,
      seed,
      resume,
      tokenizerConfig,
      useAdvanced,
      useGPU,
      activation,
      leakyReluAlpha,
      eluAlpha,
      initialization,
      lrSchedule,
      lrMin,
      lrDecayRate,
      warmupEpochs,
      weightDecay,
      gradientClipNorm,
      useLayerNorm,
      useBeamSearch,
      beamWidth
    };
    StorageManager.set(STORAGE_KEYS.UI_SETTINGS, settings);
  }, [
    architecture,
    hiddenSize,
    epochs,
    lr,
    optimizer,
    momentum,
    dropout,
    contextSize,
    temperature,
    topK,
    topP,
    samplingMode,
    seed,
    resume,
    tokenizerConfig,
    useAdvanced,
    useGPU,
    activation,
    leakyReluAlpha,
    eluAlpha,
    initialization,
    lrSchedule,
    lrMin,
    lrDecayRate,
    warmupEpochs,
    weightDecay,
    gradientClipNorm,
    useLayerNorm,
    useBeamSearch,
    beamWidth
  ]);

  // Persist tokenizer config separately
  useEffect(() => {
    StorageManager.set(STORAGE_KEYS.TOKENIZER_CONFIG, tokenizerConfig);
    if (modelRef.current) {
      modelRef.current.importTokenizerConfig(tokenizerConfig);
    }
  }, [tokenizerConfig]);

  async function onTrain() {
    if (!trainingText.trim() || trainingRef.current.running) return;
    if (tokenizerError) {
      addSystemMessage('❌ Resolve tokenizer configuration errors before training.');
      return;
    }

    // Create new AbortController for this training session
    abortControllerRef.current = new AbortController();
    trainingRef.current = { running: true, currentEpoch: 0 };
    setIsTraining(true);
    setProgress(0);

    const vocab = buildVocab(trainingText, tokenizerConfig);
    if (vocab.length < MIN_VOCAB_SIZE) {
      addSystemMessage(`❌ Need more training text (at least ${MIN_VOCAB_SIZE} unique tokens).`);
      setIsTraining(false);
      trainingRef.current.running = false;
      return;
    }

    const sigNew = vocab.join('\u241F');
    const sigOld = modelRef.current?.getVocabSignature();
    const shouldReinit = !resume || !modelRef.current || sigNew !== sigOld;

    if (shouldReinit) {
      if (architecture === 'transformer') {
        // Use TransformerLM
        modelRef.current = new TransformerLM(
          vocab,
          hiddenSize,
          lr,
          clamp(contextSize, 2, 6),
          optimizer,
          momentum,
          clamp(dropout, 0, 0.5),
          seed,
          tokenizerConfig,
          {
            numLayers: 2,
            numHeads: 4,
            ffHiddenDim: hiddenSize * 2,
            attentionDropout: dropout,
            dropConnectRate: 0.1
          }
        );
        addSystemMessage(
          `🔮 Starting fresh training with TransformerLM (${vocab.length} vocabulary tokens, 2 layers, 4 heads)…`
        );
      } else if (architecture === 'advanced' || useAdvanced) {
        // Use AdvancedNeuralLM with advanced configuration
        modelRef.current = new AdvancedNeuralLM(
          vocab,
          hiddenSize,
          lr,
          clamp(contextSize, 2, 6),
          optimizer,
          momentum,
          clamp(dropout, 0, 0.5),
          seed,
          tokenizerConfig,
          {
            activation,
            leakyReluAlpha,
            eluAlpha,
            initialization,
            lrSchedule,
            lrMin,
            lrDecayRate,
            warmupEpochs,
            weightDecay,
            gradientClipNorm,
            useLayerNorm,
            beamWidth
          }
        );
        addSystemMessage(
          `🚀 Starting fresh training with AdvancedNeuralLM (${vocab.length} vocabulary tokens)…`
        );
      } else {
        // Use standard ProNeuralLM
        modelRef.current = new ProNeuralLM(
          vocab,
          hiddenSize,
          lr,
          clamp(contextSize, 2, 6),
          optimizer,
          momentum,
          clamp(dropout, 0, 0.5),
          seed,
          tokenizerConfig
        );
        addSystemMessage(
          `📊 Starting fresh training with ProNeuralLM (${vocab.length} vocabulary tokens)…`
        );
      }
    } else {
      modelRef.current!.importTokenizerConfig(tokenizerConfig);
      addSystemMessage('🔁 Continuing training on the current model…');
    }

    // Set GPU operations if enabled
    if (useGPU && gpuOpsRef.current) {
      modelRef.current!.setGPUOps(gpuOpsRef.current);
      addSystemMessage('⚡ GPU acceleration enabled');
    } else {
      modelRef.current!.setGPUOps(null);
    }

    const total = Math.max(1, epochs);
    let aggLoss = 0;
    let aggAcc = 0;
    let lossEMA = 0;
    const emaAlpha = 0.1; // EMA smoothing factor
    const tokens = ProNeuralLM.tokenizeText(trainingText, tokenizerConfig);
    const totalTokens = tokens.length;

    for (let e = 0; e < total; e++) {
      // Check both AbortController and running flag
      if (abortControllerRef.current?.signal.aborted || !trainingRef.current.running) break;
      trainingRef.current.currentEpoch = e;

      const epochStartTime = Date.now();
      const res = await modelRef.current!.train(trainingText, 1);
      const epochEndTime = Date.now();

      aggLoss += res.loss;
      aggAcc += res.accuracy;

      // Calculate EMA of loss
      lossEMA = e === 0 ? res.loss : emaAlpha * res.loss + (1 - emaAlpha) * lossEMA;

      // Calculate tokens/sec for this epoch
      const epochDuration = (epochEndTime - epochStartTime) / 1000; // in seconds
      const tokensPerSec = epochDuration > 0 ? totalTokens / epochDuration : 0;

      const meanLoss = aggLoss / (e + 1);
      setStats({
        loss: meanLoss,
        acc: aggAcc / (e + 1),
        ppl: Math.exp(Math.max(1e-8, meanLoss)),
        lossEMA: lossEMA,
        tokensPerSec: tokensPerSec
      });
      setTrainingHistory(modelRef.current!.getTrainingHistory());
      setProgress(((e + 1) / total) * 100);
      await new Promise((r) => setTimeout(r, TRAINING_UI_UPDATE_DELAY));
    }

    if (trainingRef.current.running) {
      setInfo({ V: modelRef.current!.getVocabSize(), P: modelRef.current!.getParametersCount() });
      applyModelMeta(modelRef.current!);
      addSystemMessage(
        `✅ Training complete! Average accuracy: ${((aggAcc / total) * 100).toFixed(1)}%`
      );
      modelRef.current!.saveToLocalStorage(MODEL_STORAGE_KEY);
    }

    setIsTraining(false);
    trainingRef.current.running = false;
  }

  function onStopTraining() {
    abortControllerRef.current?.abort();
    trainingRef.current.running = false;
    setIsTraining(false);
    addSystemMessage('⏹️ Training stopped');
  }

  function onSave() {
    modelRef.current?.saveToLocalStorage(MODEL_STORAGE_KEY);
    addSystemMessage('💾 Saved locally');
  }

  function onLoad() {
    const m = loadLatestModel();
    if (m) {
      modelRef.current = m;
      setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
      setTrainingHistory(m.getTrainingHistory());
      syncTokenizerFromModel(m);
      applyModelMeta(m);
      addSystemMessage('📀 Loaded from local storage');
    }
  }

  function onExport() {
    if (!modelRef.current) return;
    const modelData = modelRef.current.toJSON();
    const blob = new Blob([JSON.stringify(modelData, null, 2)], {
      type: 'application/json'
    });

    // Create filename with timestamp and hash
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const jsonStr = JSON.stringify(modelData);
    const hash = Math.abs(
      jsonStr.split('').reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0)
    )
      .toString(16)
      .slice(0, 8);
    const filename = `neuro-lingua-v${MODEL_VERSION.replace(/\./g, '')}-${timestamp}-${hash}.json`;

    downloadBlob(blob, filename);
    addSystemMessage(`📦 Exported: ${filename}`);
  }

  function onImport(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const d = JSON.parse(String(reader.result));
        StorageManager.set(MODEL_STORAGE_KEY, d);
        const m = loadLatestModel();
        if (m) {
          modelRef.current = m;
          setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
          setTrainingHistory(m.getTrainingHistory());
          syncTokenizerFromModel(m);
          applyModelMeta(m);
          addSystemMessage('📥 Imported model file');
        }
      } catch {
        addSystemMessage('❌ Failed to import model file');
      }
    };
    reader.readAsText(file);
  }

  function onReset() {
    modelRef.current = null;
    StorageManager.remove(MODEL_STORAGE_KEY);
    StorageManager.removeMultiple([...STORAGE_KEYS.LEGACY_MODELS]);
    persistModelMeta(null);
    setInfo({ V: 0, P: 0 });
    setStats({ loss: 0, acc: 0, ppl: 0, lossEMA: 0, tokensPerSec: 0 });
    setTrainingHistory([]);
    addSystemMessage('🔄 Model reset. Ready to train again.');
  }

  async function onGenerate() {
    if (!modelRef.current || !input.trim()) {
      addSystemMessage(t.chat.trainFirst);
      return;
    }
    setMessages((m) => [...m, { type: 'user', content: input, timestamp: Date.now() }]);

    let txt: string;
    if (useBeamSearch && modelRef.current instanceof AdvancedNeuralLM) {
      // Use beam search generation
      const result = await modelRef.current.generateBeamSearch(
        input,
        DEFAULT_GENERATION.maxTokens,
        beamWidth,
        temperature
      );
      txt = result.text;
    } else {
      // Use standard generation
      const k = samplingMode === 'topk' ? topK : 0;
      const p = samplingMode === 'topp' ? topP : 0;
      txt = await modelRef.current.generate(input, DEFAULT_GENERATION.maxTokens, temperature, k, p);
    }

    setMessages((m) => [...m, { type: 'assistant', content: txt, timestamp: Date.now() }]);
    setInput('');
  }

  function onExample() {
    setTrainingText(EXAMPLE_CORPUS);
  }

  return (
    <ErrorBoundary>
      <div
        dir={direction}
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
          color: '#e2e8f0',
          padding: 20,
          fontFamily: "'Segoe UI', system-ui, sans-serif",
          direction
        }}
      >
        <div style={{ maxWidth: 1400, margin: '0 auto' }}>
          <OnboardingCard
            show={showOnboarding}
            onDismiss={() => setShowOnboarding(false)}
            strings={t.onboarding}
            direction={direction}
          />

          <header style={{ marginBottom: 32 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: direction === 'rtl' ? 'flex-start' : 'flex-end',
                marginBottom: 16
              }}
            >
              <button
                type="button"
                onClick={toggleLocale}
                aria-label={t.toggle.aria}
                style={{
                  padding: '8px 16px',
                  background: 'linear-gradient(90deg, #22d3ee, #818cf8)',
                  border: 'none',
                  borderRadius: 999,
                  color: '#0f172a',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                {t.toggle.button}
              </button>
            </div>
            <div style={{ textAlign: 'center' }}>
              <h1
                style={{
                  fontSize: '2.8rem',
                  fontWeight: 800,
                  background: 'linear-gradient(90deg, #a78bfa 0%, #34d399 50%, #60a5fa 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  marginBottom: 8
                }}
              >
                {t.title.replace('{version}', MODEL_VERSION)}
              </h1>
              <p style={{ color: '#94a3b8', fontSize: '1.05rem', margin: 0 }}>{t.subtitle}</p>
            </div>
          </header>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 20,
              marginBottom: 20
            }}
          >
            <TrainingPanel
              architecture={architecture}
              hiddenSize={hiddenSize}
              epochs={epochs}
              lr={lr}
              optimizer={optimizer}
              momentum={momentum}
              dropout={dropout}
              contextSize={contextSize}
              temperature={temperature}
              topK={topK}
              topP={topP}
              samplingMode={samplingMode}
              seed={seed}
              resume={resume}
              tokenizerConfig={tokenizerConfig}
              customTokenizerPattern={customTokenizerPattern}
              tokenizerError={tokenizerError}
              isTraining={isTraining}
              progress={progress}
              currentEpoch={trainingRef.current.currentEpoch}
              // Advanced features
              useAdvanced={useAdvanced}
              useGPU={useGPU}
              gpuAvailable={gpuAvailable}
              activation={activation}
              leakyReluAlpha={leakyReluAlpha}
              eluAlpha={eluAlpha}
              initialization={initialization}
              lrSchedule={lrSchedule}
              lrMin={lrMin}
              lrDecayRate={lrDecayRate}
              warmupEpochs={warmupEpochs}
              weightDecay={weightDecay}
              gradientClipNorm={gradientClipNorm}
              useLayerNorm={useLayerNorm}
              useBeamSearch={useBeamSearch}
              beamWidth={beamWidth}
              // Callbacks
              onArchitectureChange={setArchitecture}
              onHiddenSizeChange={setHiddenSize}
              onEpochsChange={setEpochs}
              onLrChange={setLr}
              onOptimizerChange={setOptimizer}
              onMomentumChange={setMomentum}
              onDropoutChange={setDropout}
              onContextSizeChange={setContextSize}
              onTemperatureChange={setTemperature}
              onTopKChange={setTopK}
              onTopPChange={setTopP}
              onSamplingModeChange={setSamplingMode}
              onSeedChange={setSeed}
              onResumeChange={setResume}
              // Advanced callbacks
              onUseAdvancedChange={setUseAdvanced}
              onUseGPUChange={setUseGPU}
              onActivationChange={setActivation}
              onLeakyReluAlphaChange={setLeakyReluAlpha}
              onEluAlphaChange={setEluAlpha}
              onInitializationChange={setInitialization}
              onLrScheduleChange={setLrSchedule}
              onLrMinChange={setLrMin}
              onLrDecayRateChange={setLrDecayRate}
              onWarmupEpochsChange={setWarmupEpochs}
              onWeightDecayChange={setWeightDecay}
              onGradientClipNormChange={setGradientClipNorm}
              onUseLayerNormChange={setUseLayerNorm}
              onUseBeamSearchChange={setUseBeamSearch}
              onBeamWidthChange={setBeamWidth}
              onTokenizerConfigChange={setTokenizerConfig}
              onCustomPatternChange={setCustomTokenizerPattern}
              onTokenizerError={setTokenizerError}
              onTrain={onTrain}
              onStop={onStopTraining}
              onReset={onReset}
              onSave={onSave}
              onLoad={onLoad}
              onExport={onExport}
              onImport={onImport}
              onMessage={addSystemMessage}
            />

            <ModelMetrics
              stats={stats}
              info={info}
              lastModelUpdate={lastModelUpdate}
              trainingHistory={trainingHistory}
              gpuMetrics={gpuMetrics}
              onMessage={addSystemMessage}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div
              style={{
                background: 'rgba(30,41,59,0.9)',
                border: '1px solid #334155',
                borderRadius: 16,
                padding: 20,
                display: 'flex',
                flexDirection: 'column',
                direction
              }}
            >
              <h3 style={{ color: '#a78bfa', marginTop: 0, marginBottom: 16 }}>
                {t.training.heading}
              </h3>
              <textarea
                value={trainingText}
                onChange={(e) => setTrainingText(e.target.value)}
                placeholder={t.training.placeholder}
                aria-label={t.training.textareaAria}
                style={{
                  width: '100%',
                  minHeight: 200,
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 12,
                  padding: 16,
                  color: '#e2e8f0',
                  fontSize: 14,
                  resize: 'vertical',
                  flex: 1,
                  fontFamily: 'inherit'
                }}
              />
              <div
                style={{
                  fontSize: 12,
                  color: '#94a3b8',
                  marginTop: 8,
                  display: 'flex',
                  justifyContent: 'space-between'
                }}
              >
                <span>
                  {t.training.characters}: {trainingText.length}
                </span>
                <span>
                  {t.training.words}: {trainingText.split(/\s+/).filter((w) => w.length > 0).length}
                </span>
              </div>
              <div style={{ fontSize: 12, color: '#cbd5f5', marginTop: 8 }}>{t.training.tip}</div>
              <div style={{ marginTop: 12 }}>
                <button
                  type="button"
                  onClick={onExample}
                  aria-label={t.training.exampleAria}
                  style={{
                    padding: '10px 14px',
                    background: '#6366f1',
                    border: 'none',
                    borderRadius: 10,
                    color: 'white',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  {t.training.example}
                </button>
              </div>
            </div>

            <ChatInterface
              messages={messages}
              input={input}
              modelExists={!!modelRef.current}
              onInputChange={setInput}
              onGenerate={onGenerate}
              strings={t.chat}
              direction={direction}
              locale={locale}
            />
          </div>

          <div
            style={{
              marginTop: 20,
              padding: 20,
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 12,
              fontSize: 13,
              color: '#94a3b8'
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              {t.infoCards.map((card) => (
                <div key={card.title}>
                  <strong>{card.title}</strong>
                  <div style={{ fontSize: 12, marginTop: 4 }}>{card.body}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}
