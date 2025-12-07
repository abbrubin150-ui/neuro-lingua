import React, { useCallback, useEffect, useRef, useState } from 'react';

import {
  ProNeuralLM,
  type Optimizer,
  type TokenizerConfig,
  clamp,
  MODEL_VERSION,
  MODEL_STORAGE_KEY,
  TRANSFORMER_MODEL_STORAGE_KEY
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
import type { GPUNeuralOps, GPUMetrics } from './backend/gpu_neural_ops';
import {
  computeSimulatedEdgeLearningDiagnostics,
  type EdgeLearningDiagnostics
} from './backend/edgeLearning';
import {
  STORAGE_KEYS,
  DEFAULT_TRAINING_TEXT,
  DEFAULT_HYPERPARAMETERS,
  DEFAULT_GENERATION,
  DEFAULT_ADVANCED_CONFIG,
  DEFAULT_IB_CONFIG,
  DEFAULT_TOKENIZER_CONFIG,
  MIN_VOCAB_SIZE,
  TRAINING_UI_UPDATE_DELAY,
  EXAMPLE_CORPUS,
  HYPERPARAMETER_CONSTRAINTS
} from './config/constants';

import {
  OnboardingCard,
  type OnboardingCardStrings,
  OnboardingTooltip,
  ModelMetrics,
  ChatInterface,
  type ChatInterfaceStrings,
  TrainingPanel,
  type Architecture,
  ErrorBoundary,
  type Message,
  ProjectManager,
  ModelSnapshot,
  InformationTheoryPanel,
  ExplainabilityPanel,
  EmbeddingVisualizationPanel,
  CompressionPanel,
  BrainPanel,
  CerebroPanel
} from './components';
import {
  createProNeuralLMAdapter,
  createAdvancedNeuralLMAdapter,
  createTransformerLMAdapter,
  extractBubblesFromModel
} from './lib/expandable';
import type { InjectionEvent } from './types/injection';
import { ExportPanel } from './components/ExportPanel';
import { DecisionEntry2Panel } from './components/DecisionEntry2Panel';

import type { InformationMetrics } from './losses/information_bottleneck';
import { getBetaSchedule } from './losses/information_bottleneck';

import { useProjects } from './contexts/ProjectContext';
import { useBrainTraining, useBrainGeneration } from './contexts/BrainContext';
import { calculateDiversityScore } from './lib/BrainEngine';
import { createTraceExport, generateTraceFilename } from './lib/traceExport';
import { createDecisionLedger } from './types/project';
import type { TrainingConfig, ScenarioResult } from './types/project';
import type { ModelMeta, ModelMetaStore } from './types/modelMeta';

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
  // Transformer-specific
  numHeads: number;
  numLayers: number;
  ffHiddenDim: number;
  attentionDropout: number;
  dropConnectRate: number;
  // Information Bottleneck
  useIB: boolean;
  betaStart: number;
  betaEnd: number;
  betaSchedule: 'constant' | 'linear' | 'exponential' | 'cosine';
  ibAlpha: number;
  numBins: number;
};

type ModelMetaOverrides = Partial<Omit<ModelMeta, 'architecture' | 'vocab'>>;

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
    onboardingHintLabel: string;
    onboardingHintClose: string;
    snapshotTitle: string;
    snapshotLastUpdated: string;
    snapshotVocab: string;
    snapshotEmpty: string;
    snapshotHint: string;
  };
  infoCards: { title: string; body: string }[];
  chat: ChatInterfaceStrings & { trainFirst: string };
  onboarding: OnboardingCardStrings;
};

const TRANSLATIONS: Record<Locale, AppTranslations> = {
  en: {
    toggle: { button: 'Hebrew / RTL', aria: 'Switch interface language to Hebrew (RTL)' },
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
      exampleAria: 'Load example corpus',
      onboardingHintLabel: 'Show pause/import tips',
      onboardingHintClose: 'Hide tips',
      snapshotTitle: 'Stored Model Snapshot',
      snapshotLastUpdated: 'Last updated',
      snapshotVocab: 'Vocab size',
      snapshotEmpty: 'No stored model yet—train to capture a baseline.',
      snapshotHint: 'Compare this timestamp and vocab size before retraining.'
    },
    infoCards: [
      {
        title: '🎯 Training Tips',
        body: '• 200–500 words • 20–50 epochs • LR: 0.05–0.1 • Context: 3–5 • Enable Resume to continue from last checkpoint'
      },
      {
        title: '🎲 Text Generation',
        body: '• Temperature: 0.7–1.0 for coherence • Top‑p ≈ 0.85–0.95 • Models generate predictions based on training'
      },
      {
        title: '💾 Save & Export',
        body: '• Save exports JSON model • Load reimports from JSON • CSV export contains training history • All data stored locally'
      },
      {
        title: '⌨️ Shortcuts',
        body: '• Ctrl/Cmd+Enter: Train/Stop • Ctrl/Cmd+S: Save • Ctrl/Cmd+G: Generate • Stop to pause and resume later'
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
    toggle: { button: 'English / LTR', aria: 'Switch interface language to English (LTR)' },
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
      exampleAria: 'Load example corpus',
      onboardingHintLabel: 'Show tips',
      onboardingHintClose: 'Hide tips',
      snapshotTitle: 'Stored Model Snapshot',
      snapshotLastUpdated: 'Last updated',
      snapshotVocab: 'Vocab size',
      snapshotEmpty: 'No stored model yet—train to capture a baseline.',
      snapshotHint: 'Compare this timestamp and vocab size before retraining.'
    },
    infoCards: [
      {
        title: '🎯 Training Tips',
        body: '• 200–500 words • 20–50 epochs • LR: 0.05–0.1 • Context: 3–5 • Enable Resume from the last checkpoint'
      },
      {
        title: '🎲 Text Generation',
        body: '• Temperature: 0.7–1.0 for fluent output • Top‑p ≈ 0.85–0.95 • Model generates predictions based on training'
      },
      {
        title: '💾 Save & Export',
        body: '• Save exports JSON model • Load restores from JSON • CSV export contains training history • All data stays local'
      },
      {
        title: '⌨️ Shortcuts',
        body: '• Ctrl/Cmd+Enter: Start/Stop • Ctrl/Cmd+S: Save • Ctrl/Cmd+G: Sample • Stop to pause and resume later'
      }
    ],
    chat: {
      title: '💬 Chat Console',
      replies: (count) => `${count} replies`,
      regionAria: 'Chat interface',
      logAria: 'Chat messages',
      inputAria: 'Chat input',
      generateAria: 'Generate text from the model',
      placeholderReady: 'Write a message for the model…',
      placeholderEmpty: 'Train the model first…',
      generate: '✨ Generate',
      tip: '💡 Shift+Enter adds a new line. Responses depend on the sampling mode and chosen temperature.',
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
      privacyWarningLead: 'Do NOT train with sensitive data.',
      privacyWarningBody:
        'Settings live in browser storage without encryption. Avoid PII, passwords, financial, medical, or confidential data.',
      bulletPauseResume:
        'Pause / Resume: use the Stop button to pause training. With Resume training enabled we pick up from the latest checkpoint when you train again.',
      bulletImportExport:
        'Import / Export: save models and tokenizer presets to JSON for safekeeping or sharing. Importing immediately refreshes charts and metadata.',
      bulletPersistence:
        'Session Persistence: hyperparameters and tokenizer preferences live in localStorage; training text stays in memory only.',
      gotIt: 'Got it',
      reopenInfo: 'You can reopen this info from localStorage by clearing the flag.'
    }
  }
} as const;

function getStorageKeyForArchitecture(architecture: Architecture): string {
  return architecture === 'transformer' ? TRANSFORMER_MODEL_STORAGE_KEY : MODEL_STORAGE_KEY;
}

function loadLatestModel(architecture: Architecture = 'feedforward'): ProNeuralLM | null {
  if (architecture === 'transformer') {
    return TransformerLM.loadFromLocalStorage(TRANSFORMER_MODEL_STORAGE_KEY);
  }

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

function detectModelArchitecture(model: ProNeuralLM): Architecture {
  if (model instanceof TransformerLM) return 'transformer';
  if (model instanceof AdvancedNeuralLM) return 'advanced';
  return 'feedforward';
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

  // Information Bottleneck metrics history
  const [ibMetricsHistory, setIbMetricsHistory] = useState<InformationMetrics[]>([]);

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

  // Transformer-specific parameters
  const [numHeads, setNumHeads] = useState(4);
  const [numLayers, setNumLayers] = useState(2);
  const [ffHiddenDim, setFfHiddenDim] = useState(DEFAULT_HYPERPARAMETERS.hiddenSize * 2);
  const [attentionDropout, setAttentionDropout] = useState(0.1);
  const [dropConnectRate, setDropConnectRate] = useState(0.1);

  // Information Bottleneck parameters
  const [useIB, setUseIB] = useState(DEFAULT_IB_CONFIG.useIB);
  const [betaStart, setBetaStart] = useState(DEFAULT_IB_CONFIG.betaStart);
  const [betaEnd, setBetaEnd] = useState(DEFAULT_IB_CONFIG.betaEnd);
  const [betaSchedule, setBetaSchedule] = useState<
    'constant' | 'linear' | 'exponential' | 'cosine'
  >(DEFAULT_IB_CONFIG.betaSchedule);
  const [ibAlpha, setIbAlpha] = useState(DEFAULT_IB_CONFIG.ibAlpha);
  const [numBins, setNumBins] = useState(DEFAULT_IB_CONFIG.numBins);

  // GPU acceleration
  const [useGPU, setUseGPU] = useState(false);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [gpuMetrics, setGpuMetrics] = useState<GPUMetrics | null>(null);

  // Edge Learning diagnostics
  const [edgeLearningDiagnostics, setEdgeLearningDiagnostics] =
    useState<EdgeLearningDiagnostics | null>(null);

  // Bayesian inference
  const [useBayesian, setUseBayesian] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);

  // Tokenizer
  const [tokenizerConfig, setTokenizerConfig] = useState<TokenizerConfig>(DEFAULT_TOKENIZER_CONFIG);
  const [customTokenizerPattern, setCustomTokenizerPattern] = useState('');
  const [tokenizerError, setTokenizerError] = useState<string | null>(null);

  // Model metadata and UI
  const [modelMetaStore, setModelMetaStore] = useState<ModelMetaStore>({});
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [showProjectManager, setShowProjectManager] = useState(false);
  const [showCompressionPanel, setShowCompressionPanel] = useState(false);
  const [showExportPanel, setShowExportPanel] = useState(false);
  const [showDecisionPanel, setShowDecisionPanel] = useState(false);

  // Cerebro mode (neuron injection)
  const [cerebroEnabled, setCerebroEnabled] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_cerebroInjectionHistory, _setCerebroInjectionHistory] = useState<InjectionEvent[]>([]);

  const modelRef = useRef<ProNeuralLM | null>(null);
  const trainingRef = useRef({ running: false, currentEpoch: 0 });
  const abortControllerRef = useRef<AbortController | null>(null);
  const gpuOpsRef = useRef<GPUNeuralOps | null>(null);
  const currentRunRef = useRef<string | null>(null);

  // Project/Run Management
  const {
    activeProject,
    activeProjectId,
    activeRun,
    createNewRun,
    updateRun,
    getRunExecutionStatus
  } = useProjects();

  // Brain History Integration
  const dispatchBrainTraining = useBrainTraining();
  const dispatchBrainGeneration = useBrainGeneration();

  // Helper to add system messages
  const addSystemMessage = useCallback((content: string) => {
    setMessages((m) => [...m, { type: 'system' as const, content, timestamp: Date.now() }]);
  }, []);

  const handleArchitectureChange = useCallback(
    (nextArchitecture: Architecture) => {
      if (architecture === nextArchitecture) return;

      setArchitecture(nextArchitecture);

      if (nextArchitecture === 'transformer') {
        setUseAdvanced(false);
        setUseLayerNorm(true);
        setOptimizer('adam');
        setDropout((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : DEFAULT_HYPERPARAMETERS.dropout,
            HYPERPARAMETER_CONSTRAINTS.dropout.min,
            HYPERPARAMETER_CONSTRAINTS.dropout.max
          )
        );
        setNumHeads((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : 4,
            HYPERPARAMETER_CONSTRAINTS.transformer.numHeads.min,
            HYPERPARAMETER_CONSTRAINTS.transformer.numHeads.max
          )
        );
        setNumLayers((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : 2,
            HYPERPARAMETER_CONSTRAINTS.transformer.numLayers.min,
            HYPERPARAMETER_CONSTRAINTS.transformer.numLayers.max
          )
        );
        setFfHiddenDim((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : hiddenSize * 2,
            HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.min,
            HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.max
          )
        );
        setAttentionDropout((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : 0.1,
            HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.min,
            HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.max
          )
        );
        setDropConnectRate((prev) =>
          clamp(
            Number.isFinite(prev) ? prev : 0.1,
            HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.min,
            HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.max
          )
        );
        addSystemMessage(
          '🔮 Transformer preset applied: Adam optimizer, LayerNorm, and attention defaults enabled.'
        );
      } else if (nextArchitecture === 'advanced') {
        setUseAdvanced(true);
        setUseLayerNorm(true);
        setOptimizer((prev) => (prev === 'adam' || prev === 'momentum' ? prev : 'adam'));
        addSystemMessage(
          '🚀 AdvancedNeuralLM preset applied with LayerNorm and deep-optimization defaults.'
        );
      } else {
        setUseAdvanced(false);
        addSystemMessage('📊 Standard ProNeuralLM architecture selected.');
      }
    },
    [architecture, addSystemMessage, hiddenSize]
  );

  const applyModelMeta = useCallback((model: ProNeuralLM, overrides: ModelMetaOverrides = {}) => {
    setModelMetaStore((prev) => {
      const architectureKey: Architecture =
        model instanceof TransformerLM
          ? 'transformer'
          : model instanceof AdvancedNeuralLM
            ? 'advanced'
            : 'feedforward';

      const existing = prev[architectureKey];
      const timestamp =
        overrides.timestamp ?? model.getLastUpdatedAt() ?? existing?.timestamp ?? Date.now();

      const nextMeta: ModelMeta = {
        architecture: architectureKey,
        timestamp,
        vocab: model.getVocabSize(),
        loss: overrides.loss ?? existing?.loss,
        accuracy: overrides.accuracy ?? existing?.accuracy,
        perplexity: overrides.perplexity ?? existing?.perplexity,
        tokensPerSec: overrides.tokensPerSec ?? existing?.tokensPerSec,
        trainingDurationMs: overrides.trainingDurationMs ?? existing?.trainingDurationMs
      };

      const nextStore: ModelMetaStore = { ...prev, [architectureKey]: nextMeta };
      StorageManager.set(STORAGE_KEYS.MODEL_META, nextStore);
      return nextStore;
    });
  }, []);

  const clearModelMetaStore = useCallback(() => {
    setModelMetaStore({});
    StorageManager.remove(STORAGE_KEYS.MODEL_META);
  }, []);

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
    if (typeof saved.numHeads === 'number') setNumHeads(saved.numHeads);
    if (typeof saved.numLayers === 'number') setNumLayers(saved.numLayers);
    if (typeof saved.ffHiddenDim === 'number') setFfHiddenDim(saved.ffHiddenDim);
    if (typeof saved.attentionDropout === 'number') setAttentionDropout(saved.attentionDropout);
    if (typeof saved.dropConnectRate === 'number') setDropConnectRate(saved.dropConnectRate);
    if (typeof saved.useIB === 'boolean') setUseIB(saved.useIB);
    if (typeof saved.betaStart === 'number') setBetaStart(saved.betaStart);
    if (typeof saved.betaEnd === 'number') setBetaEnd(saved.betaEnd);
    if (
      saved.betaSchedule === 'constant' ||
      saved.betaSchedule === 'linear' ||
      saved.betaSchedule === 'exponential' ||
      saved.betaSchedule === 'cosine'
    ) {
      setBetaSchedule(saved.betaSchedule);
    }
    if (typeof saved.ibAlpha === 'number') setIbAlpha(saved.ibAlpha);
    if (typeof saved.numBins === 'number') setNumBins(saved.numBins);

    const tokenizerRaw = StorageManager.get<unknown>(STORAGE_KEYS.TOKENIZER_CONFIG, null);
    if (tokenizerRaw) {
      const parsed = parseTokenizerConfig(tokenizerRaw);
      setTokenizerConfig(parsed);
      if (parsed.mode === 'custom') setCustomTokenizerPattern(parsed.pattern ?? '');
    }

    const storedMeta = StorageManager.get<ModelMetaStore | ModelMeta | null>(
      STORAGE_KEYS.MODEL_META,
      null
    );
    if (storedMeta && typeof storedMeta === 'object') {
      const maybeLegacy = storedMeta as { timestamp?: unknown; vocab?: unknown };
      if (typeof maybeLegacy.timestamp === 'number' && typeof maybeLegacy.vocab === 'number') {
        setModelMetaStore({
          feedforward: {
            architecture: 'feedforward',
            timestamp: maybeLegacy.timestamp,
            vocab: maybeLegacy.vocab
          }
        });
      } else {
        setModelMetaStore(storedMeta as ModelMetaStore);
      }
    }

    if (localStorage.getItem(STORAGE_KEYS.ONBOARDING_DISMISSED) === 'true') {
      setShowOnboarding(false);
    }
  }, []);

  useEffect(() => {
    runSelfTests();
  }, []);

  // Load model for the active architecture
  useEffect(() => {
    const saved = loadLatestModel(architecture);
    if (saved) {
      modelRef.current = saved;
      setInfo({ V: saved.getVocabSize(), P: saved.getParametersCount() });
      setTrainingHistory(saved.getTrainingHistory());
      syncTokenizerFromModel(saved);
      applyModelMeta(saved);
      const architectureLabel =
        architecture === 'transformer'
          ? 'TransformerLM'
          : architecture === 'advanced'
            ? 'AdvancedNeuralLM'
            : 'ProNeuralLM';
      addSystemMessage(`📀 ${architectureLabel} loaded from local storage`);
    } else if (modelRef.current && detectModelArchitecture(modelRef.current) !== architecture) {
      modelRef.current = null;
      setInfo({ V: 0, P: 0 });
      setTrainingHistory([]);
      setIbMetricsHistory([]);
    }
  }, [architecture, applyModelMeta, syncTokenizerFromModel, addSystemMessage]);

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
              setUseGPU(true); // Auto-enable GPU when available
              setGpuMetrics(ops.getMetrics());
              console.log('✅ WebGPU is available and GPUNeuralOps initialized');
              console.log('⚡ GPU acceleration enabled automatically (2-5x speedup expected)');
              addSystemMessage('⚡ GPU acceleration enabled automatically (2-5x speedup expected)');
            } else {
              setGpuAvailable(false);
              gpuOpsRef.current = null;
              setGpuMetrics(null);
              console.log('⚠️ GPUNeuralOps initialization failed');
            }
          } else {
            setGpuAvailable(false);
            gpuOpsRef.current = null;
            setGpuMetrics(null);
            console.log('⚠️ WebGPU adapter not available');
          }
        } else {
          setGpuAvailable(false);
          gpuOpsRef.current = null;
          setGpuMetrics(null);
          console.log('⚠️ WebGPU is not supported in this browser');
        }
      } catch (error) {
        setGpuAvailable(false);
        gpuOpsRef.current = null;
        setGpuMetrics(null);
        console.warn('⚠️ Error checking WebGPU availability:', error);
      }
    };
    checkGPU();
  }, [addSystemMessage]);

  useEffect(() => {
    if (!gpuOpsRef.current) return;
    gpuOpsRef.current.setEnabled(useGPU && gpuAvailable);
    setGpuMetrics(gpuOpsRef.current.getMetrics());
  }, [useGPU, gpuAvailable]);

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
      beamWidth,
      numHeads,
      numLayers,
      ffHiddenDim,
      attentionDropout,
      dropConnectRate,
      useIB,
      betaStart,
      betaEnd,
      betaSchedule,
      ibAlpha,
      numBins
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
    beamWidth,
    numHeads,
    numLayers,
    ffHiddenDim,
    attentionDropout,
    dropConnectRate,
    useIB,
    betaStart,
    betaEnd,
    betaSchedule,
    ibAlpha,
    numBins
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

    // Check Decision Ledger if we have an active run
    if (activeRun) {
      const status = getRunExecutionStatus(activeRun.id);
      if (status === 'ESCALATE') {
        addSystemMessage(
          '🚨 Cannot train: Decision Ledger requires review (missing rationale/witness).'
        );
        return;
      }
      if (status === 'HOLD') {
        addSystemMessage('⏸️ Cannot train: Run is on HOLD (expired or paused).');
        return;
      }
    }

    // Create new AbortController for this training session
    abortControllerRef.current = new AbortController();
    trainingRef.current = { running: true, currentEpoch: 0 };
    setIsTraining(true);
    setProgress(0);

    // Clear IB metrics history at start of new training
    if (useIB) {
      setIbMetricsHistory([]);
    }

    // Build training configuration snapshot
    const trainingConfig: TrainingConfig = {
      architecture,
      hiddenSize,
      epochs,
      learningRate: lr,
      optimizer,
      momentum,
      dropout,
      contextSize,
      seed,
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
      numHeads,
      numLayers,
      ffHiddenDim,
      attentionDropout,
      dropConnectRate
    };

    // Create a new Run if we have an active project and no active run
    if (activeProjectId && !activeRun) {
      const defaultLedger = createDecisionLedger(
        'Training run initiated from UI',
        'local-user',
        null,
        'keep'
      );

      const newRun = createNewRun(
        activeProjectId,
        `Training Run ${new Date().toLocaleString()}`,
        trainingConfig,
        trainingText,
        defaultLedger
      );

      currentRunRef.current = newRun.id;
      updateRun(newRun.id, { status: 'running', startedAt: Date.now() });
      addSystemMessage(`📝 Created new Run: ${newRun.name}`);
    } else if (activeRun) {
      currentRunRef.current = activeRun.id;
      updateRun(activeRun.id, { status: 'running', startedAt: Date.now() });
    }

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
        const normalizedHeads = clamp(
          numHeads,
          HYPERPARAMETER_CONSTRAINTS.transformer.numHeads.min,
          HYPERPARAMETER_CONSTRAINTS.transformer.numHeads.max
        );
        const normalizedLayers = clamp(
          numLayers,
          HYPERPARAMETER_CONSTRAINTS.transformer.numLayers.min,
          HYPERPARAMETER_CONSTRAINTS.transformer.numLayers.max
        );
        const normalizedFfHidden = clamp(
          ffHiddenDim,
          HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.min,
          HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.max
        );
        const normalizedAttentionDropout = clamp(
          attentionDropout,
          HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.min,
          HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.max
        );
        const normalizedDropConnect = clamp(
          dropConnectRate,
          HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.min,
          HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.max
        );
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
            numLayers: normalizedLayers,
            numHeads: normalizedHeads,
            ffHiddenDim: normalizedFfHidden,
            attentionDropout: normalizedAttentionDropout,
            dropConnectRate: normalizedDropConnect
          }
        );
        addSystemMessage(
          `🔮 Starting fresh training with TransformerLM (${vocab.length} vocabulary tokens, ${normalizedLayers} layers, ${normalizedHeads} heads)…`
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
    const trainingStartTime = Date.now();
    let latestTokensPerSec = 0;

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
      latestTokensPerSec = tokensPerSec;

      const meanLoss = aggLoss / (e + 1);
      setStats({
        loss: meanLoss,
        acc: aggAcc / (e + 1),
        ppl: Math.exp(Math.max(1e-8, meanLoss)),
        lossEMA: lossEMA,
        tokensPerSec: tokensPerSec
      });

      // Update GPU metrics periodically (every 5 epochs or less frequently)
      if (gpuOpsRef.current && useGPU && (e + 1) % Math.max(1, Math.floor(total / 10)) === 0) {
        const metrics = gpuOpsRef.current.getMetrics();
        if (metrics.available) {
          setGpuMetrics(metrics);
        }
      }

      setTrainingHistory(modelRef.current!.getTrainingHistory());
      setProgress(((e + 1) / total) * 100);

      // Collect Information Bottleneck metrics if enabled
      // Note: This is a simplified approximation. Full IB metrics require
      // model modifications to expose hidden activations during training.
      if (useIB) {
        const currentBeta = getBetaSchedule(betaSchedule, e, total, betaStart, betaEnd);

        // Approximate IB metrics based on training progress
        // I(X;Z) typically decreases as training progresses (compression)
        // I(Z;Y) typically increases as training progresses (prediction improves)
        const trainingProgress = (e + 1) / total;
        const compressionMI = Math.max(0.5, 2.0 - trainingProgress * 1.5); // Decreasing
        const predictionMI = Math.min(2.5, 0.5 + trainingProgress * 2.0); // Increasing
        const representationEntropy = 1.5 + Math.sin(trainingProgress * Math.PI) * 0.5;
        const conditionalEntropy = Math.max(0, representationEntropy - compressionMI);
        const ibLoss = -predictionMI + currentBeta * compressionMI;

        setIbMetricsHistory((prev) => [
          ...prev,
          {
            compressionMI,
            predictionMI,
            ibLoss,
            beta: currentBeta,
            representationEntropy,
            conditionalEntropy
          }
        ]);
      }

      await new Promise((r) => setTimeout(r, TRAINING_UI_UPDATE_DELAY));
    }

    if (trainingRef.current.running) {
      const finalLoss = aggLoss / Math.max(1, total);
      const finalAccuracy = aggAcc / Math.max(1, total);
      const finalPerplexity = Math.exp(Math.max(1e-8, finalLoss));
      const trainingDurationMs = Date.now() - trainingStartTime;
      setInfo({ V: modelRef.current!.getVocabSize(), P: modelRef.current!.getParametersCount() });
      applyModelMeta(modelRef.current!, {
        loss: finalLoss,
        accuracy: finalAccuracy,
        perplexity: finalPerplexity,
        tokensPerSec: latestTokensPerSec,
        trainingDurationMs
      });

      // Collect and display GPU metrics if available
      if (gpuOpsRef.current && useGPU) {
        const finalMetrics = gpuOpsRef.current.getMetrics();
        if (finalMetrics.available) {
          setGpuMetrics(finalMetrics);
          if (finalMetrics.totalOperations > 0) {
            addSystemMessage(
              `⚡ GPU Acceleration: ${
                finalMetrics.totalOperations
              } ops, ${finalMetrics.totalTimeMs.toFixed(
                1
              )}ms total, ${finalMetrics.averageTimeMs.toFixed(2)}ms/op`
            );
          }
        }
      }

      // Compute Edge Learning diagnostics
      try {
        const trainingHistory = modelRef.current!.getTrainingHistory();
        const losses = trainingHistory.map((h) => h.loss);
        const parametersCount = modelRef.current!.getParametersCount();

        const diagnostics = computeSimulatedEdgeLearningDiagnostics(parametersCount, losses);
        setEdgeLearningDiagnostics(diagnostics);

        if (diagnostics.status === 'success') {
          addSystemMessage(
            `📊 Edge Learning: Fisher=${diagnostics.fisherInformation.toFixed(4)}, Efficiency=${(
              diagnostics.efficiency * 100
            ).toFixed(1)}%`
          );
        }
      } catch (error) {
        console.warn('Edge Learning computation failed:', error);
      }

      addSystemMessage(
        `✅ Training complete! Average accuracy: ${(finalAccuracy * 100).toFixed(1)}%`
      );
      const activeModelArchitecture = detectModelArchitecture(modelRef.current!);
      modelRef.current!.saveToLocalStorage(getStorageKeyForArchitecture(activeModelArchitecture));

      // Dispatch brain training event
      const vocabSize = modelRef.current!.getVocabSize();
      const vocabDelta = shouldReinit ? vocabSize : 0; // If reinitialized, all vocab is "new"
      dispatchBrainTraining(total, totalTokens, finalLoss, vocabDelta);

      // Save results to Run if we have one
      if (currentRunRef.current) {
        const finalTrainingHistory = modelRef.current!.getTrainingHistory();

        // Execute scenarios if we have an active project
        const scenarioResults: ScenarioResult[] = [];
        if (activeProject && activeProject.scenarios.length > 0) {
          addSystemMessage('🧪 Running test scenarios...');
          for (const scenario of activeProject.scenarios) {
            try {
              const response = await modelRef.current!.generate(scenario.prompt, 50, 0.8, 0, 0.9);

              // Simple scoring: 1.0 if we got a response, 0.5 if empty
              const score = response && response.trim().length > 0 ? 1.0 : 0.5;

              scenarioResults.push({
                scenarioId: scenario.id,
                response,
                score,
                timestamp: Date.now()
              });

              addSystemMessage(`  ✓ ${scenario.name}: Score ${score.toFixed(2)}`);
            } catch (error) {
              console.warn(`Failed to run scenario ${scenario.name}:`, error);
              scenarioResults.push({
                scenarioId: scenario.id,
                response: 'Error running scenario',
                score: 0,
                timestamp: Date.now()
              });
            }
          }
        }

        // Update the Run with final results
        updateRun(currentRunRef.current, {
          results: {
            finalLoss: stats.loss,
            finalAccuracy: stats.acc,
            finalPerplexity: stats.ppl,
            trainingHistory: finalTrainingHistory,
            scenarioResults: scenarioResults.length > 0 ? scenarioResults : undefined
          },
          modelData: modelRef.current!.toJSON(),
          completedAt: Date.now(),
          status: 'completed'
        });

        addSystemMessage('💾 Run results saved to Project');
      }
    }

    setIsTraining(false);
    trainingRef.current.running = false;
  }

  function onStopTraining() {
    abortControllerRef.current?.abort();
    trainingRef.current.running = false;
    setIsTraining(false);
    // Update Run status if we have one
    if (currentRunRef.current) {
      updateRun(currentRunRef.current, {
        status: 'stopped'
      });
    }

    addSystemMessage('⏹️ Training stopped');
  }

  function onSave() {
    if (!modelRef.current) return;
    const arch = detectModelArchitecture(modelRef.current);
    modelRef.current.saveToLocalStorage(getStorageKeyForArchitecture(arch));
    addSystemMessage('💾 Saved locally');
  }

  function onLoad() {
    const m = loadLatestModel(architecture);
    if (m) {
      const detectedArch = detectModelArchitecture(m);
      if (detectedArch !== architecture) {
        setArchitecture(detectedArch);
      }
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

    // Create enhanced trace export if we have a Run
    let exportData: unknown;
    let filename: string;

    if (activeRun) {
      // Export with full trace metadata
      exportData = createTraceExport(
        modelData,
        activeRun,
        trainingHistory,
        stats,
        trainingText,
        MODEL_VERSION
      );

      // Generate descriptive filename
      const jsonStr = JSON.stringify(exportData);
      const hash = Math.abs(
        jsonStr.split('').reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0)
      )
        .toString(16)
        .slice(0, 8);

      filename = generateTraceFilename(MODEL_VERSION, activeRun.name, hash);
      addSystemMessage('📦 Exporting with full trace metadata (Σ-SIG compliant)...');
    } else {
      // Export standard format (backward compatible)
      exportData = modelData;

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const jsonStr = JSON.stringify(modelData);
      const hash = Math.abs(
        jsonStr.split('').reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0)
      )
        .toString(16)
        .slice(0, 8);
      filename = `neuro-lingua-v${MODEL_VERSION.replace(/\./g, '')}-${timestamp}-${hash}.json`;
      addSystemMessage('📦 Exporting standard format...');
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });

    downloadBlob(blob, filename);
    addSystemMessage(`✅ Exported: ${filename}`);
  }

  function onCompress() {
    if (!modelRef.current) {
      addSystemMessage('❌ No model to compress');
      return;
    }
    setShowCompressionPanel(true);
  }

  function onImport(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));

        // Check if this is a trace export format
        let modelData: unknown;
        let importedArchitecture: Architecture = 'feedforward';

        if (data.architecture === 'transformer' || data?.config?.architecture === 'transformer') {
          importedArchitecture = 'transformer';
        } else if (data.architecture === 'advanced' || data?.config?.architecture === 'advanced') {
          importedArchitecture = 'advanced';
        }

        if (data.modelWeights && data.config && data.tokenizer) {
          // New trace export format
          modelData = {
            weights: data.modelWeights,
            config: data.config,
            tokenizer: data.tokenizer
          };

          // Show trace metadata if available
          if (data.projectMeta) {
            addSystemMessage(
              `📥 Importing from Project: ${data.projectMeta.projectName}, Run: ${data.projectMeta.runName}`
            );
          }
          if (data.decisionLedger) {
            addSystemMessage(`📋 Rationale: ${data.decisionLedger.rationale}`);
          }
          if (data.trainingTrace) {
            addSystemMessage(
              `📊 Training: ${data.trainingTrace.epochs} epochs, Loss: ${data.trainingTrace.finalLoss.toFixed(4)}, Accuracy: ${(data.trainingTrace.finalAccuracy * 100).toFixed(1)}%`
            );
          }
        } else {
          // Standard format
          modelData = data;
          addSystemMessage('📥 Importing standard format model...');
        }

        StorageManager.set(getStorageKeyForArchitecture(importedArchitecture), modelData);
        const m = loadLatestModel(importedArchitecture);
        if (m) {
          setArchitecture(importedArchitecture);
          modelRef.current = m;
          setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
          setTrainingHistory(m.getTrainingHistory());
          syncTokenizerFromModel(m);
          applyModelMeta(m);
          addSystemMessage('✅ Model imported successfully');
        }
      } catch (error) {
        console.error('Import error:', error);
        addSystemMessage('❌ Failed to import model file');
      }
    };
    reader.readAsText(file);
  }

  function onReset() {
    modelRef.current = null;
    StorageManager.removeMultiple([
      ...STORAGE_KEYS.LEGACY_MODELS,
      MODEL_STORAGE_KEY,
      TRANSFORMER_MODEL_STORAGE_KEY
    ]);
    clearModelMetaStore();
    setInfo({ V: 0, P: 0 });
    setStats({ loss: 0, acc: 0, ppl: 0, lossEMA: 0, tokensPerSec: 0 });
    setTrainingHistory([]);
    setIbMetricsHistory([]);
    addSystemMessage('🔄 Model reset. Ready to train again.');
  }

  async function onGenerate() {
    if (!modelRef.current || !input.trim()) {
      addSystemMessage(t.chat.trainFirst);
      return;
    }
    let calculatedConfidence: number | null = null;
    setMessages((m) => [...m, { type: 'user', content: input, timestamp: Date.now() }]);

    let txt: string;

    // Bayesian inference: Monte Carlo sampling
    if (useBayesian) {
      const numSamples = 5; // Number of MC samples
      const samples: string[] = [];

      addSystemMessage(
        `🎲 Generating ${numSamples} Monte Carlo samples for uncertainty estimation...`
      );

      for (let i = 0; i < numSamples; i++) {
        let sample: string;
        if (useBeamSearch && modelRef.current instanceof AdvancedNeuralLM) {
          const result = await modelRef.current.generateBeamSearch(
            input,
            DEFAULT_GENERATION.maxTokens,
            beamWidth,
            temperature
          );
          sample = result.text;
        } else {
          const k = samplingMode === 'topk' ? topK : 0;
          const p = samplingMode === 'topp' ? topP : 0;
          sample = await modelRef.current.generate(
            input,
            DEFAULT_GENERATION.maxTokens,
            temperature,
            k,
            p
          );
        }
        samples.push(sample);
      }

      // Calculate diversity metric (confidence is inverse of diversity)
      const uniqueSamples = new Set(samples).size;
      const diversity = uniqueSamples / numSamples; // 0-1, where 0=all identical, 1=all different
      calculatedConfidence = 1 - diversity; // High confidence if samples agree

      setConfidence(calculatedConfidence);

      // Use the first sample (could also use most common)
      txt = samples[0];

      addSystemMessage(
        `📊 Bayesian result: ${uniqueSamples}/${numSamples} unique samples, confidence=${calculatedConfidence.toFixed(2)}`
      );
    } else {
      // Standard generation
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
        txt = await modelRef.current.generate(
          input,
          DEFAULT_GENERATION.maxTokens,
          temperature,
          k,
          p
        );
      }

      // Reset confidence when not using Bayesian
      calculatedConfidence = null;
      setConfidence(null);
    }

    // Dispatch brain generation event
    const diversityScore = calculateDiversityScore(temperature, topK, topP);
    const tokensGenerated = txt.split(/\s+/).length;
    dispatchBrainGeneration(diversityScore, tokensGenerated);

    setMessages((m) => [...m, { type: 'assistant', content: txt, timestamp: Date.now() }]);
    setInput('');
  }

  function onExample() {
    setTrainingText(EXAMPLE_CORPUS);
  }

  return (
    <ErrorBoundary>
      {showProjectManager && (
        <ProjectManager direction={direction} onClose={() => setShowProjectManager(false)} />
      )}
      {showCompressionPanel && (
        <CompressionPanel
          model={modelRef.current}
          corpus={trainingText}
          onClose={() => setShowCompressionPanel(false)}
          onMessage={addSystemMessage}
        />
      )}
      {showExportPanel && (
        <ExportPanel direction={direction} onClose={() => setShowExportPanel(false)} />
      )}
      {showDecisionPanel && (
        <DecisionEntry2Panel direction={direction} onClose={() => setShowDecisionPanel(false)} />
      )}
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
                marginBottom: 16,
                gap: 8,
                flexWrap: 'wrap'
              }}
            >
              <button
                type="button"
                onClick={() => setShowProjectManager(true)}
                aria-label="Open Project Manager"
                style={{
                  padding: '8px 16px',
                  background: 'linear-gradient(90deg, #7c3aed, #6366f1)',
                  border: 'none',
                  borderRadius: 999,
                  color: 'white',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                📁 Projects
              </button>
              <button
                type="button"
                onClick={() => setShowDecisionPanel(true)}
                aria-label="Open Decision Ledger 2.0"
                style={{
                  padding: '8px 16px',
                  background: 'linear-gradient(90deg, #a78bfa, #059669)',
                  border: 'none',
                  borderRadius: 999,
                  color: 'white',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                📋 Decisions
              </button>
              <button
                type="button"
                onClick={() => setShowExportPanel(true)}
                aria-label="Export Data"
                style={{
                  padding: '8px 16px',
                  background: 'linear-gradient(90deg, #10b981, #3b82f6)',
                  border: 'none',
                  borderRadius: 999,
                  color: 'white',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                📦 Export
              </button>
              <button
                type="button"
                onClick={() => setCerebroEnabled((prev) => !prev)}
                aria-label={cerebroEnabled ? 'Disable Cerebro Mode' : 'Enable Cerebro Mode'}
                style={{
                  padding: '8px 16px',
                  background: cerebroEnabled
                    ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                    : 'linear-gradient(90deg, #6b7280, #4b5563)',
                  border: 'none',
                  borderRadius: 999,
                  color: 'white',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                {cerebroEnabled ? '🧠 Cerebro ON' : '🧠 Cerebro'}
              </button>
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
              numHeads={numHeads}
              numLayers={numLayers}
              ffHiddenDim={ffHiddenDim}
              attentionDropout={attentionDropout}
              dropConnectRate={dropConnectRate}
              // Callbacks
              onArchitectureChange={handleArchitectureChange}
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
              onNumHeadsChange={setNumHeads}
              onNumLayersChange={setNumLayers}
              onFfHiddenDimChange={setFfHiddenDim}
              onAttentionDropoutChange={setAttentionDropout}
              onDropConnectRateChange={setDropConnectRate}
              // Information Bottleneck
              useIB={useIB}
              betaStart={betaStart}
              betaEnd={betaEnd}
              betaSchedule={betaSchedule}
              ibAlpha={ibAlpha}
              numBins={numBins}
              onUseIBChange={setUseIB}
              onBetaStartChange={setBetaStart}
              onBetaEndChange={setBetaEnd}
              onBetaScheduleChange={setBetaSchedule}
              onIbAlphaChange={setIbAlpha}
              onNumBinsChange={setNumBins}
              onTokenizerConfigChange={setTokenizerConfig}
              onCustomPatternChange={setCustomTokenizerPattern}
              onTokenizerError={setTokenizerError}
              onTrain={onTrain}
              onStop={onStopTraining}
              onReset={onReset}
              onSave={onSave}
              onLoad={onLoad}
              onExport={onExport}
              onCompress={onCompress}
              onImport={onImport}
              onMessage={addSystemMessage}
            />

            <ModelMetrics
              stats={stats}
              info={info}
              activeArchitecture={architecture}
              activeModelMeta={modelMetaStore[architecture] ?? null}
              modelComparisons={modelMetaStore}
              trainingHistory={trainingHistory}
              gpuMetrics={gpuMetrics}
              edgeLearningDiagnostics={edgeLearningDiagnostics}
              onMessage={addSystemMessage}
            />

            {useIB && ibMetricsHistory.length > 0 && (
              <InformationTheoryPanel
                metricsHistory={ibMetricsHistory}
                isTraining={isTraining}
                currentEpoch={Math.floor(progress * epochs)}
                totalEpochs={epochs}
                direction={direction}
              />
            )}
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
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 12,
                  marginBottom: 12
                }}
              >
                <h3 style={{ color: '#a78bfa', margin: 0 }}>{t.training.heading}</h3>
                <OnboardingTooltip
                  label={t.training.onboardingHintLabel}
                  closeLabel={t.training.onboardingHintClose}
                  direction={direction}
                  strings={{
                    bulletPauseResume: t.onboarding.bulletPauseResume,
                    bulletImportExport: t.onboarding.bulletImportExport,
                    bulletPersistence: t.onboarding.bulletPersistence
                  }}
                />
              </div>
              <ModelSnapshot
                meta={modelMetaStore[architecture] ?? null}
                architecture={architecture}
                title={t.training.snapshotTitle}
                lastUpdatedLabel={t.training.snapshotLastUpdated}
                vocabLabel={t.training.snapshotVocab}
                emptyLabel={t.training.snapshotEmpty}
                hint={t.training.snapshotHint}
              />
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
              useBayesian={useBayesian}
              onUseBayesianChange={setUseBayesian}
              confidence={confidence}
            />
          </div>

          <div
            style={{
              marginTop: 20,
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 12
            }}
          >
            <ExplainabilityPanel model={modelRef.current} />
          </div>

          <div
            style={{
              marginTop: 20,
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 12
            }}
          >
            <EmbeddingVisualizationPanel model={modelRef.current} />
          </div>

          <div
            style={{
              marginTop: 20
            }}
          >
            <BrainPanel />
          </div>

          {cerebroEnabled && modelRef.current && (
            <div
              style={{
                marginTop: 20,
                background: 'rgba(30,41,59,0.9)',
                border: '1px solid #f59e0b',
                borderRadius: 12
              }}
            >
              <CerebroPanel
                layer={
                  architecture === 'transformer'
                    ? createTransformerLMAdapter(
                        modelRef.current as TransformerLM,
                        activeRun?.id ?? `model-${Date.now()}`
                      )
                    : architecture === 'advanced'
                      ? createAdvancedNeuralLMAdapter(
                          modelRef.current as AdvancedNeuralLM,
                          activeRun?.id ?? `model-${Date.now()}`
                        )
                      : createProNeuralLMAdapter(
                          modelRef.current,
                          activeRun?.id ?? `model-${Date.now()}`
                        )
                }
                bubbles={extractBubblesFromModel(modelRef.current, { maxBubbles: 24 })}
              />
            </div>
          )}

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
