import React, { useRef } from 'react';
import type { Optimizer, TokenizerConfig as TokenizerConfigType } from '../lib/ProNeuralLM';
import type { ActivationFunction, LRSchedule, InitializationScheme } from '../lib/AdvancedNeuralLM';
import type { GradientClipMode } from '../training/GradientClipping';
import { clamp } from '../lib/ProNeuralLM';
import { TokenizerConfig } from './TokenizerConfig';
import { DEFAULT_HYPERPARAMETERS, HYPERPARAMETER_CONSTRAINTS } from '../config/constants';

export type Architecture = 'feedforward' | 'advanced' | 'transformer';

interface TrainingPanelProps {
  // Architecture selection
  architecture: Architecture;

  // Hyperparameters
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
  samplingMode: 'off' | 'topk' | 'topp' | 'typical' | 'mirostat' | 'contrastive';
  seed: number;
  resume: boolean;

  // Advanced features
  useAdvanced: boolean;
  useGPU: boolean;
  gpuAvailable: boolean;
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
  gradientClipMode: GradientClipMode;
  useLayerNorm: boolean;
  useEMA: boolean;
  emaDecay: number;
  useBeamSearch: boolean;
  beamWidth: number;

  // Transformer-specific
  numHeads: number;
  numLayers: number;
  ffHiddenDim: number;
  attentionDropout: number;
  dropConnectRate: number;
  numKVHeads: number; // GQA: number of key-value heads

  // Information Bottleneck
  useIB: boolean;
  betaStart: number;
  betaEnd: number;
  betaSchedule: 'constant' | 'linear' | 'exponential' | 'cosine';
  ibAlpha: number;
  numBins: number;

  // Advanced Sampling Parameters
  typicalTau: number;
  mirostatTau: number;
  mirostatEta: number;

  // Advanced Loss Functions
  lossFunction: 'cross_entropy' | 'focal' | 'label_smoothing' | 'symmetric_ce';
  focalGamma: number;
  focalAlpha: number;
  labelSmoothingEpsilon: number;
  sceBeta: number;

  // Loss Mask (Answer-Only Training)
  lossMaskMode: 'none' | 'afterEquals' | 'afterAnswerTag' | 'customRegExp';
  lossMaskAnswerTag: string;
  lossMaskCustomPattern: string;
  lossMaskRegExpPosition: 'after' | 'before' | 'match' | 'exclude';
  lossMaskPatternError: string | null;

  // Optimizer-specific parameters (Lion v4.0)
  lionBeta1: number;
  lionBeta2: number;
  lionWeightDecay: number;

  // Optimizer-specific parameters (Sophia v4.2)
  sophiaBeta1: number;
  sophiaBeta2: number;
  sophiaRho: number;
  sophiaHessianFreq: number;

  // Tokenizer
  tokenizerConfig: TokenizerConfigType;
  customTokenizerPattern: string;
  tokenizerError: string | null;

  // Training state
  isTraining: boolean;
  progress: number;
  currentEpoch: number;

  // Callbacks
  onArchitectureChange: (value: Architecture) => void;
  onHiddenSizeChange: (value: number) => void;
  onEpochsChange: (value: number) => void;
  onLrChange: (value: number) => void;
  onOptimizerChange: (value: Optimizer) => void;
  onMomentumChange: (value: number) => void;
  onDropoutChange: (value: number) => void;
  onContextSizeChange: (value: number) => void;
  onTemperatureChange: (value: number) => void;
  onTopKChange: (value: number) => void;
  onTopPChange: (value: number) => void;
  onSamplingModeChange: (value: 'off' | 'topk' | 'topp' | 'typical' | 'mirostat') => void;
  onSeedChange: (value: number) => void;
  onResumeChange: (value: boolean) => void;

  // Advanced callbacks
  onUseAdvancedChange: (value: boolean) => void;
  onUseGPUChange: (value: boolean) => void;
  onActivationChange: (value: ActivationFunction) => void;
  onLeakyReluAlphaChange: (value: number) => void;
  onEluAlphaChange: (value: number) => void;
  onInitializationChange: (value: InitializationScheme) => void;
  onLrScheduleChange: (value: LRSchedule) => void;
  onLrMinChange: (value: number) => void;
  onLrDecayRateChange: (value: number) => void;
  onWarmupEpochsChange: (value: number) => void;
  onWeightDecayChange: (value: number) => void;
  onGradientClipNormChange: (value: number) => void;
  onGradientClipModeChange: (value: GradientClipMode) => void;
  onUseLayerNormChange: (value: boolean) => void;
  onUseEMAChange: (value: boolean) => void;
  onEmaDecayChange: (value: number) => void;
  onUseBeamSearchChange: (value: boolean) => void;
  onBeamWidthChange: (value: number) => void;

  // Transformer-specific callbacks
  onNumHeadsChange: (value: number) => void;
  onNumLayersChange: (value: number) => void;
  onFfHiddenDimChange: (value: number) => void;
  onAttentionDropoutChange: (value: number) => void;
  onDropConnectRateChange: (value: number) => void;
  onNumKVHeadsChange: (value: number) => void; // GQA callback

  // Information Bottleneck callbacks
  onUseIBChange: (value: boolean) => void;
  onBetaStartChange: (value: number) => void;
  onBetaEndChange: (value: number) => void;
  onBetaScheduleChange: (value: 'constant' | 'linear' | 'exponential' | 'cosine') => void;
  onIbAlphaChange: (value: number) => void;
  onNumBinsChange: (value: number) => void;

  // Advanced Sampling callbacks
  onTypicalTauChange: (value: number) => void;
  onMirostatTauChange: (value: number) => void;
  onMirostatEtaChange: (value: number) => void;

  // Advanced Loss Function callbacks
  onLossFunctionChange: (
    value: 'cross_entropy' | 'focal' | 'label_smoothing' | 'symmetric_ce'
  ) => void;
  onFocalGammaChange: (value: number) => void;
  onFocalAlphaChange: (value: number) => void;
  onLabelSmoothingEpsilonChange: (value: number) => void;
  onSceBetaChange: (value: number) => void;

  // Loss Mask callbacks
  onLossMaskModeChange: (value: 'none' | 'afterEquals' | 'afterAnswerTag' | 'customRegExp') => void;
  onLossMaskAnswerTagChange: (value: string) => void;
  onLossMaskCustomPatternChange: (value: string) => void;
  onLossMaskRegExpPositionChange: (value: 'after' | 'before' | 'match' | 'exclude') => void;
  onLossMaskPatternErrorChange: (value: string | null) => void;

  // Lion optimizer callbacks
  onLionBeta1Change: (value: number) => void;
  onLionBeta2Change: (value: number) => void;
  onLionWeightDecayChange: (value: number) => void;

  // Sophia optimizer callbacks
  onSophiaBeta1Change: (value: number) => void;
  onSophiaBeta2Change: (value: number) => void;
  onSophiaRhoChange: (value: number) => void;
  onSophiaHessianFreqChange: (value: number) => void;

  onTokenizerConfigChange: (config: TokenizerConfigType) => void;
  onCustomPatternChange: (pattern: string) => void;
  onTokenizerError: (error: string | null) => void;
  onTrain: () => void;
  onStop: () => void;
  onReset: () => void;
  onSave: () => void;
  onLoad: () => void;
  onApplyDefaults: () => void;
  onExport: () => void;
  onCompress: () => void;
  onImport: (ev: React.ChangeEvent<HTMLInputElement>) => void;
  onMessage: (message: string) => void;
}

/**
 * TrainingPanel provides controls for all training hyperparameters and model operations
 */
export function TrainingPanel(props: TrainingPanelProps) {
  const importRef = useRef<HTMLInputElement>(null);

  return (
    <div
      role="region"
      aria-label="Training controls and hyperparameters"
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20
      }}
    >
      {/* Architecture Selection */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>🏗️ Architecture</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          <button
            onClick={() => props.onArchitectureChange('feedforward')}
            style={{
              padding: '12px 16px',
              background:
                props.architecture === 'feedforward'
                  ? 'linear-gradient(90deg, #7c3aed, #6366f1)'
                  : '#374151',
              border:
                props.architecture === 'feedforward' ? '2px solid #a78bfa' : '1px solid #4b5563',
              borderRadius: 10,
              color: 'white',
              fontWeight: props.architecture === 'feedforward' ? 700 : 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            📊 Standard
            <div style={{ fontSize: 10, opacity: 0.8, marginTop: 4 }}>ProNeuralLM</div>
          </button>
          <button
            onClick={() => props.onArchitectureChange('advanced')}
            style={{
              padding: '12px 16px',
              background:
                props.architecture === 'advanced'
                  ? 'linear-gradient(90deg, #7c3aed, #6366f1)'
                  : '#374151',
              border: props.architecture === 'advanced' ? '2px solid #a78bfa' : '1px solid #4b5563',
              borderRadius: 10,
              color: 'white',
              fontWeight: props.architecture === 'advanced' ? 700 : 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            🚀 Advanced
            <div style={{ fontSize: 10, opacity: 0.8, marginTop: 4 }}>AdvancedNeuralLM</div>
          </button>
          <button
            onClick={() => props.onArchitectureChange('transformer')}
            style={{
              padding: '12px 16px',
              background:
                props.architecture === 'transformer'
                  ? 'linear-gradient(90deg, #7c3aed, #6366f1)'
                  : '#374151',
              border:
                props.architecture === 'transformer' ? '2px solid #a78bfa' : '1px solid #4b5563',
              borderRadius: 10,
              color: 'white',
              fontWeight: props.architecture === 'transformer' ? 700 : 600,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            🔮 Transformer
            <div style={{ fontSize: 10, opacity: 0.8, marginTop: 4 }}>Multi-Head Attention</div>
          </button>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 8 }}>
          <button
            onClick={props.onApplyDefaults}
            aria-label="Apply recommended hyperparameter preset"
            style={{
              padding: '10px 12px',
              background: 'rgba(99,102,241,0.1)',
              border: '1px solid #6366f1',
              borderRadius: 10,
              color: '#c4d3ff',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span>
                🎯 Apply RMSNorm preset ({DEFAULT_HYPERPARAMETERS.hiddenSize}d /{' '}
                {DEFAULT_HYPERPARAMETERS.transformer.numLayers} layers / Adam{' '}
                {DEFAULT_HYPERPARAMETERS.learningRate})
              </span>
              <span style={{ fontSize: 12, opacity: 0.8 }}>
                Pre-norm residuals with RMSNorm stability and lower dropout defaults
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* Transformer-Specific Configuration */}
      {props.architecture === 'transformer' && (
        <div
          style={{
            background: 'rgba(139, 92, 246, 0.1)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
            borderRadius: 12,
            padding: 12,
            marginBottom: 12
          }}
        >
          <div
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: '#a78bfa',
              marginBottom: 8,
              display: 'flex',
              alignItems: 'center',
              gap: 6
            }}
          >
            🔮 Transformer Configuration
            <span
              title="Transformer uses multi-head self-attention to capture long-range dependencies. More heads = more diverse attention patterns. More layers = deeper representations. Higher FF dim = more model capacity."
              style={{ cursor: 'help', fontSize: 11, opacity: 0.7 }}
            >
              ℹ️
            </span>
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
              gap: 12
            }}
          >
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="Number of parallel attention mechanisms. Each head learns different patterns. More heads = richer representations but slower training. Recommended: 4-8"
              >
                Attention Heads ℹ️
              </div>
              <input
                aria-label="Number of attention heads"
                type="number"
                min="1"
                max="16"
                value={props.numHeads}
                onChange={(e) =>
                  props.onNumHeadsChange(Math.max(1, Math.min(16, parseInt(e.target.value || '4'))))
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>1-16 (default: 4)</div>
            </div>
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="Number of Key-Value heads for Grouped-Query Attention (GQA). Multiple query heads share KV heads to reduce memory. Must divide numHeads evenly. Lower = less memory but may affect quality. numKVHeads=numHeads is standard MHA."
              >
                KV Heads (GQA) ℹ️
              </div>
              <input
                aria-label="Number of key-value heads for GQA"
                type="number"
                min={HYPERPARAMETER_CONSTRAINTS.transformer.numKVHeads.min}
                max={props.numHeads}
                value={props.numKVHeads}
                onChange={(e) => {
                  const val = parseInt(e.target.value || `${props.numHeads}`, 10);
                  // Must be between 1 and numHeads, and must divide numHeads evenly
                  const clamped = Math.max(1, Math.min(props.numHeads, val));
                  props.onNumKVHeadsChange(clamped);
                }}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                1-{props.numHeads} (ratio: {props.numHeads / props.numKVHeads}:1,{' '}
                {props.numKVHeads === props.numHeads ? 'MHA' : 'GQA'})
              </div>
            </div>
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="Number of stacked transformer blocks. Each layer can capture increasingly abstract patterns. More layers = deeper model but slower training. Recommended: 2-4"
              >
                Layers ℹ️
              </div>
              <input
                aria-label="Number of transformer layers"
                type="number"
                min="1"
                max="8"
                value={props.numLayers}
                onChange={(e) =>
                  props.onNumLayersChange(Math.max(1, Math.min(8, parseInt(e.target.value || '2'))))
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>1-8 (default: 2)</div>
            </div>
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="Feed-forward network hidden dimension. Controls model capacity between attention layers. Typically 2-4x the hidden size. Higher = more parameters and capacity."
              >
                FF Hidden Dim ℹ️
              </div>
              <input
                aria-label="Transformer feed-forward hidden dimension"
                type="number"
                min={HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.min}
                max={HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.max}
                value={props.ffHiddenDim}
                onChange={(e) =>
                  props.onFfHiddenDimChange(
                    clamp(
                      parseInt(e.target.value || `${props.ffHiddenDim}`, 10) || props.ffHiddenDim,
                      HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.min,
                      HYPERPARAMETER_CONSTRAINTS.transformer.ffHiddenDim.max
                    )
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                xHidden (default: 2×)
              </div>
            </div>
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="Dropout applied to attention weights. Helps prevent overfitting by randomly dropping attention connections during training. Typical range: 0.1-0.2"
              >
                Attention Dropout ℹ️
              </div>
              <input
                aria-label="Attention dropout"
                type="number"
                step="0.01"
                min={HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.min}
                max={HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.max}
                value={props.attentionDropout}
                onChange={(e) =>
                  props.onAttentionDropoutChange(
                    clamp(
                      parseFloat(e.target.value || `${props.attentionDropout}`),
                      HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.min,
                      HYPERPARAMETER_CONSTRAINTS.transformer.attentionDropout.max
                    )
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                Stabilize attention
              </div>
            </div>
            <div>
              <div
                style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4, cursor: 'help' }}
                title="DropConnect randomly drops connections (weights) in feed-forward layers during training. Similar to dropout but drops weights instead of activations. Helps prevent overfitting."
              >
                DropConnect ℹ️
              </div>
              <input
                aria-label="Transformer DropConnect rate"
                type="number"
                step="0.01"
                min={HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.min}
                max={HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.max}
                value={props.dropConnectRate}
                onChange={(e) =>
                  props.onDropConnectRateChange(
                    clamp(
                      parseFloat(e.target.value || `${props.dropConnectRate}`),
                      HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.min,
                      HYPERPARAMETER_CONSTRAINTS.transformer.dropConnectRate.max
                    )
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              />
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>
                Regularize FF layers
              </div>
            </div>
          </div>
          <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 8 }}>
            💡 Tune heads/layers for capacity, FF dim for expressiveness, dropout/dropconnect for
            stability
          </div>
        </div>
      )}

      {/* Hyperparameters Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(6, 1fr)',
          gap: 12,
          alignItems: 'end',
          marginBottom: 12
        }}
      >
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Hidden</div>
          <input
            aria-label="Hidden size"
            type="number"
            value={props.hiddenSize}
            onChange={(e) =>
              props.onHiddenSizeChange(
                clamp(
                  parseInt(e.target.value || '64'),
                  HYPERPARAMETER_CONSTRAINTS.hiddenSize.min,
                  HYPERPARAMETER_CONSTRAINTS.hiddenSize.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Epochs</div>
          <input
            aria-label="Epochs"
            type="number"
            value={props.epochs}
            onChange={(e) =>
              props.onEpochsChange(
                clamp(
                  parseInt(e.target.value || '20'),
                  HYPERPARAMETER_CONSTRAINTS.epochs.min,
                  HYPERPARAMETER_CONSTRAINTS.epochs.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Learning Rate</div>
          <input
            aria-label="Learning rate"
            type="number"
            step="0.01"
            value={props.lr}
            onChange={(e) =>
              props.onLrChange(
                clamp(
                  parseFloat(e.target.value || '0.08'),
                  HYPERPARAMETER_CONSTRAINTS.learningRate.min,
                  HYPERPARAMETER_CONSTRAINTS.learningRate.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Context</div>
          <input
            aria-label="Context window"
            type="number"
            min={HYPERPARAMETER_CONSTRAINTS.contextSize.min}
            max={HYPERPARAMETER_CONSTRAINTS.contextSize.max}
            value={props.contextSize}
            onChange={(e) =>
              props.onContextSizeChange(
                clamp(
                  parseInt(e.target.value || '3'),
                  HYPERPARAMETER_CONSTRAINTS.contextSize.min,
                  HYPERPARAMETER_CONSTRAINTS.contextSize.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Optimizer</div>
          <select
            aria-label="Optimizer"
            value={props.optimizer}
            onChange={(e) => props.onOptimizerChange(e.target.value as Optimizer)}
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          >
            <option value="momentum">Momentum</option>
            <option value="adam">Adam</option>
            <option value="lion">Lion (v4.0)</option>
            <option value="sophia">Sophia (v4.2)</option>
            <option value="newton">Damped Newton</option>
            <option value="bfgs">Quasi-Newton (L-BFGS)</option>
          </select>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Momentum</div>
          <input
            aria-label="Momentum"
            type="number"
            step="0.05"
            value={props.momentum}
            onChange={(e) =>
              props.onMomentumChange(
                clamp(
                  parseFloat(e.target.value || '0.9'),
                  HYPERPARAMETER_CONSTRAINTS.momentum.min,
                  HYPERPARAMETER_CONSTRAINTS.momentum.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Dropout</div>
          <input
            aria-label="Dropout"
            type="number"
            step="0.01"
            value={props.dropout}
            onChange={(e) =>
              props.onDropoutChange(
                clamp(
                  parseFloat(e.target.value || '0.1'),
                  HYPERPARAMETER_CONSTRAINTS.dropout.min,
                  HYPERPARAMETER_CONSTRAINTS.dropout.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
      </div>

      {/* Sampling Parameters */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(6, 1fr)',
          gap: 12,
          alignItems: 'end',
          marginBottom: 12
        }}
      >
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Temperature</div>
          <input
            aria-label="Temperature"
            type="number"
            step="0.05"
            value={props.temperature}
            onChange={(e) =>
              props.onTemperatureChange(
                clamp(
                  parseFloat(e.target.value || '0.8'),
                  HYPERPARAMETER_CONSTRAINTS.temperature.min,
                  HYPERPARAMETER_CONSTRAINTS.temperature.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div style={{ gridColumn: 'span 2 / span 2' }}>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Sampling</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="radio"
                checked={props.samplingMode === 'off'}
                onChange={() => props.onSamplingModeChange('off')}
              />{' '}
              off
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="radio"
                checked={props.samplingMode === 'topk'}
                onChange={() => props.onSamplingModeChange('topk')}
              />{' '}
              top‑k
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="radio"
                checked={props.samplingMode === 'topp'}
                onChange={() => props.onSamplingModeChange('topp')}
              />{' '}
              top‑p
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="radio"
                checked={props.samplingMode === 'typical'}
                onChange={() => props.onSamplingModeChange('typical')}
              />{' '}
              typical
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="radio"
                checked={props.samplingMode === 'mirostat'}
                onChange={() => props.onSamplingModeChange('mirostat')}
              />{' '}
              mirostat
            </label>
          </div>
        </div>
        <div>
          <div
            style={{
              fontSize: 12,
              color: '#94a3b8',
              opacity: props.samplingMode === 'topk' ? 1 : 0.5
            }}
          >
            Top‑K
          </div>
          <input
            aria-label="Top K"
            type="number"
            value={props.topK}
            disabled={props.samplingMode !== 'topk'}
            onChange={(e) =>
              props.onTopKChange(
                clamp(
                  parseInt(e.target.value || '20'),
                  HYPERPARAMETER_CONSTRAINTS.topK.min,
                  HYPERPARAMETER_CONSTRAINTS.topK.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div
            style={{
              fontSize: 12,
              color: '#94a3b8',
              opacity: props.samplingMode === 'topp' ? 1 : 0.5
            }}
          >
            Top‑P
          </div>
          <input
            aria-label="Top P"
            type="number"
            step="0.01"
            value={props.topP}
            disabled={props.samplingMode !== 'topp'}
            onChange={(e) =>
              props.onTopPChange(
                clamp(
                  parseFloat(e.target.value || '0.9'),
                  HYPERPARAMETER_CONSTRAINTS.topP.min,
                  HYPERPARAMETER_CONSTRAINTS.topP.max
                )
              )
            }
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8' }}>Seed</div>
          <input
            aria-label="Random seed"
            type="number"
            value={props.seed}
            onChange={(e) => props.onSeedChange(parseInt(e.target.value || '1337'))}
            style={{
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              borderRadius: 6,
              padding: 8,
              color: 'white'
            }}
          />
        </div>
      </div>

      {/* Advanced Sampling Parameters */}
      {(props.samplingMode === 'typical' || props.samplingMode === 'mirostat') && (
        <div
          style={{
            background: 'rgba(168, 85, 247, 0.1)',
            border: '1px solid rgba(168, 85, 247, 0.3)',
            borderRadius: 12,
            padding: 12,
            marginBottom: 12
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: '#c4b5fd', marginBottom: 8 }}>
            {props.samplingMode === 'typical' ? '🎯 Typical Sampling' : '🌀 Mirostat v2'} Parameters
          </div>
          {props.samplingMode === 'typical' && (
            <div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                Typical Tau (τ): {props.typicalTau.toFixed(2)}
              </div>
              <input
                aria-label="Typical tau"
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={props.typicalTau}
                onChange={(e) => props.onTypicalTauChange(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                Lower τ = more focused on typical tokens (0.9 recommended)
              </div>
            </div>
          )}
          {props.samplingMode === 'mirostat' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  Target Entropy (τ): {props.mirostatTau.toFixed(1)}
                </div>
                <input
                  aria-label="Mirostat target entropy"
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={props.mirostatTau}
                  onChange={(e) => props.onMirostatTauChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                  Lower = more predictable, Higher = more creative
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  Learning Rate (η): {props.mirostatEta.toFixed(2)}
                </div>
                <input
                  aria-label="Mirostat learning rate"
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={props.mirostatEta}
                  onChange={(e) => props.onMirostatEtaChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                  How fast to adapt (0.1 recommended)
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Optimizer-Specific Parameters */}
      {(props.optimizer === 'lion' || props.optimizer === 'sophia') && (
        <div
          style={{
            background: 'rgba(34, 197, 94, 0.1)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: 12,
            padding: 12,
            marginBottom: 12
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: '#86efac', marginBottom: 8 }}>
            {props.optimizer === 'lion' ? '🦁 Lion (v4.0)' : '🧠 Sophia (v4.2)'} Optimizer Settings
          </div>
          {props.optimizer === 'lion' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  β₁ (Update): {props.lionBeta1.toFixed(2)}
                </div>
                <input
                  aria-label="Lion beta1"
                  type="range"
                  min="0.8"
                  max="0.99"
                  step="0.01"
                  value={props.lionBeta1}
                  onChange={(e) => props.onLionBeta1Change(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  β₂ (State): {props.lionBeta2.toFixed(2)}
                </div>
                <input
                  aria-label="Lion beta2"
                  type="range"
                  min="0.9"
                  max="0.999"
                  step="0.001"
                  value={props.lionBeta2}
                  onChange={(e) => props.onLionBeta2Change(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  Weight Decay: {props.lionWeightDecay.toFixed(3)}
                </div>
                <input
                  aria-label="Lion weight decay"
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
                  value={props.lionWeightDecay}
                  onChange={(e) => props.onLionWeightDecayChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
          {props.optimizer === 'sophia' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  β₁ (Momentum): {props.sophiaBeta1.toFixed(3)}
                </div>
                <input
                  aria-label="Sophia beta1"
                  type="range"
                  min="0.9"
                  max="0.99"
                  step="0.005"
                  value={props.sophiaBeta1}
                  onChange={(e) => props.onSophiaBeta1Change(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  β₂ (Hessian): {props.sophiaBeta2.toFixed(2)}
                </div>
                <input
                  aria-label="Sophia beta2"
                  type="range"
                  min="0.9"
                  max="0.999"
                  step="0.001"
                  value={props.sophiaBeta2}
                  onChange={(e) => props.onSophiaBeta2Change(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  ρ (Clip Bound): {props.sophiaRho.toFixed(1)}
                </div>
                <input
                  aria-label="Sophia rho clipping"
                  type="range"
                  min="0.1"
                  max="5"
                  step="0.1"
                  value={props.sophiaRho}
                  onChange={(e) => props.onSophiaRhoChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                  Hessian Update Freq: {props.sophiaHessianFreq}
                </div>
                <input
                  aria-label="Sophia Hessian update frequency"
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  value={props.sophiaHessianFreq}
                  onChange={(e) => props.onSophiaHessianFreqChange(parseInt(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            </div>
          )}
          <div style={{ fontSize: 9, color: '#64748b', marginTop: 8 }}>
            {props.optimizer === 'lion'
              ? '💡 Lion: 50% less memory than Adam, use lower LR (~3e-4)'
              : '💡 Sophia: 2× faster convergence via curvature-aware updates'}
          </div>
        </div>
      )}

      {/* Advanced Loss Function Selector */}
      {props.useAdvanced && (
        <div
          style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 12,
            padding: 12,
            marginBottom: 12
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: '#fca5a5', marginBottom: 8 }}>
            🎯 Loss Function
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <select
                aria-label="Loss function"
                value={props.lossFunction}
                onChange={(e) =>
                  props.onLossFunctionChange(
                    e.target.value as 'cross_entropy' | 'focal' | 'label_smoothing' | 'symmetric_ce'
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              >
                <option value="cross_entropy">Cross-Entropy (Standard)</option>
                <option value="focal">Focal Loss (Class Imbalance)</option>
                <option value="label_smoothing">Label Smoothing CE</option>
                <option value="symmetric_ce">Symmetric CE (Noise Robust)</option>
              </select>
            </div>
            {props.lossFunction === 'focal' && (
              <>
                <div>
                  <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>
                    γ (Focus): {props.focalGamma.toFixed(1)}
                  </div>
                  <input
                    aria-label="Focal gamma"
                    type="range"
                    min="0"
                    max="5"
                    step="0.5"
                    value={props.focalGamma}
                    onChange={(e) => props.onFocalGammaChange(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>
                    α (Balance): {props.focalAlpha.toFixed(2)}
                  </div>
                  <input
                    aria-label="Focal alpha"
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={props.focalAlpha}
                    onChange={(e) => props.onFocalAlphaChange(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </div>
              </>
            )}
            {props.lossFunction === 'label_smoothing' && (
              <div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>
                  ε (Smoothing): {props.labelSmoothingEpsilon.toFixed(2)}
                </div>
                <input
                  aria-label="Label smoothing epsilon"
                  type="range"
                  min="0"
                  max="0.3"
                  step="0.01"
                  value={props.labelSmoothingEpsilon}
                  onChange={(e) => props.onLabelSmoothingEpsilonChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            )}
            {props.lossFunction === 'symmetric_ce' && (
              <div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>
                  β (Reverse Weight): {props.sceBeta.toFixed(1)}
                </div>
                <input
                  aria-label="SCE beta"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={props.sceBeta}
                  onChange={(e) => props.onSceBetaChange(parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            )}
          </div>
          <div style={{ fontSize: 9, color: '#64748b', marginTop: 8 }}>
            {props.lossFunction === 'cross_entropy' &&
              'Standard cross-entropy loss for classification'}
            {props.lossFunction === 'focal' &&
              'Focal loss down-weights easy examples (γ=2, α=0.25 typical)'}
            {props.lossFunction === 'label_smoothing' &&
              'Prevents overconfident predictions (ε=0.1 typical)'}
            {props.lossFunction === 'symmetric_ce' && 'Robust to noisy labels via bidirectional KL'}
          </div>
        </div>
      )}

      {/* Loss Mask (Answer-Only Training) */}
      {props.useAdvanced && (
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: 12,
            padding: 12,
            marginBottom: 12
          }}
        >
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6ee7b7', marginBottom: 8 }}>
            🎭 Loss Mask (Answer-Only Training)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>Mask Mode</div>
              <select
                aria-label="Loss mask mode"
                value={props.lossMaskMode}
                onChange={(e) =>
                  props.onLossMaskModeChange(
                    e.target.value as 'none' | 'afterEquals' | 'afterAnswerTag' | 'customRegExp'
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12
                }}
              >
                <option value="none">None (Full Sequence)</option>
                <option value="afterEquals">After &quot;=&quot; (Math Tasks)</option>
                <option value="afterAnswerTag">After Answer Tag (Q&A)</option>
                <option value="customRegExp">Custom RegExp Pattern</option>
              </select>
            </div>
            {props.lossMaskMode === 'afterAnswerTag' && (
              <div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>Answer Tag</div>
                <input
                  aria-label="Answer tag delimiter"
                  type="text"
                  value={props.lossMaskAnswerTag}
                  onChange={(e) => props.onLossMaskAnswerTagChange(e.target.value)}
                  placeholder="A:"
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white',
                    fontSize: 12,
                    boxSizing: 'border-box'
                  }}
                />
              </div>
            )}
            {props.lossMaskMode === 'customRegExp' && (
              <div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>Position Mode</div>
                <select
                  aria-label="RegExp position mode"
                  value={props.lossMaskRegExpPosition}
                  onChange={(e) =>
                    props.onLossMaskRegExpPositionChange(
                      e.target.value as 'after' | 'before' | 'match' | 'exclude'
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white',
                    fontSize: 12
                  }}
                >
                  <option value="after">After Match (loss on content after pattern)</option>
                  <option value="before">Before Match (loss on content before pattern)</option>
                  <option value="match">On Match Only (loss only on matched text)</option>
                  <option value="exclude">Exclude Match (loss on everything except matched)</option>
                </select>
              </div>
            )}
          </div>
          {/* Custom RegExp Pattern Input */}
          {props.lossMaskMode === 'customRegExp' && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
                Custom Pattern (JS RegExp)
              </div>
              <input
                aria-label="Custom RegExp pattern"
                type="text"
                value={props.lossMaskCustomPattern}
                onChange={(e) => props.onLossMaskCustomPatternChange(e.target.value)}
                placeholder="Answer:\\s*|^A:|\\[ANSWER\\]"
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: `1px solid ${props.lossMaskPatternError ? '#ef4444' : '#475569'}`,
                  borderRadius: 6,
                  padding: 8,
                  color: 'white',
                  fontSize: 12,
                  fontFamily: 'monospace',
                  boxSizing: 'border-box'
                }}
              />
              {props.lossMaskPatternError && (
                <div
                  style={{
                    marginTop: 4,
                    fontSize: 10,
                    color: '#ef4444',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4
                  }}
                >
                  <span>⚠</span> {props.lossMaskPatternError}
                </div>
              )}
            </div>
          )}
          <div style={{ fontSize: 9, color: '#64748b', marginTop: 8 }}>
            {props.lossMaskMode === 'none' && 'Compute loss on all tokens (standard training)'}
            {props.lossMaskMode === 'afterEquals' &&
              'Only compute loss after "=" - ideal for "20+7=27" format'}
            {props.lossMaskMode === 'afterAnswerTag' &&
              `Only compute loss after "${props.lossMaskAnswerTag}" - ideal for "Q:...\\nA:..." format`}
            {props.lossMaskMode === 'customRegExp' && (
              <>
                Use RegExp to define mask boundaries.{' '}
                {props.lossMaskRegExpPosition === 'after' && 'Loss computed AFTER first match.'}
                {props.lossMaskRegExpPosition === 'before' && 'Loss computed BEFORE first match.'}
                {props.lossMaskRegExpPosition === 'match' && 'Loss computed ONLY on matched text.'}
                {props.lossMaskRegExpPosition === 'exclude' &&
                  'Loss computed on all text EXCEPT matches.'}
              </>
            )}
          </div>
          {props.lossMaskMode === 'afterEquals' && props.tokenizerConfig.mode !== 'character' && (
            <div
              style={{
                marginTop: 8,
                padding: 8,
                background: 'rgba(234, 179, 8, 0.15)',
                border: '1px solid rgba(234, 179, 8, 0.3)',
                borderRadius: 6,
                fontSize: 10,
                color: '#fde68a'
              }}
            >
              <strong>Warning:</strong> For math tasks, use the <strong>Character-level</strong>{' '}
              tokenizer. The current tokenizer strips symbols like + and = from the corpus, so the
              model cannot learn arithmetic operators.
            </div>
          )}
          {props.lossMaskMode !== 'none' && (
            <div
              style={{
                marginTop: 8,
                padding: 8,
                background: 'rgba(16, 185, 129, 0.15)',
                borderRadius: 6,
                fontSize: 10,
                color: '#a7f3d0'
              }}
            >
              <strong>Tip:</strong> Format your corpus as:
              {props.lossMaskMode === 'afterEquals' && (
                <code style={{ display: 'block', marginTop: 4, fontFamily: 'monospace' }}>
                  20+7=27
                  <br />
                  15-3=12
                </code>
              )}
              {props.lossMaskMode === 'afterAnswerTag' && (
                <code style={{ display: 'block', marginTop: 4, fontFamily: 'monospace' }}>
                  Q: What is 20+7?
                  <br />
                  {props.lossMaskAnswerTag} 27
                </code>
              )}
              {props.lossMaskMode === 'customRegExp' && (
                <div style={{ marginTop: 4 }}>
                  <div style={{ marginBottom: 4 }}>
                    <strong>Example patterns:</strong>
                  </div>
                  <code style={{ display: 'block', fontFamily: 'monospace', fontSize: 9 }}>
                    Answer:\\s* - Match &quot;Answer:&quot; followed by whitespace
                    <br />
                    ^A: - Match &quot;A:&quot; at start of line
                    <br />
                    \\[RESPONSE\\] - Match literal &quot;[RESPONSE]&quot; tag
                    <br />
                    (?:Output|Result):\\s* - Match &quot;Output:&quot; or &quot;Result:&quot;
                  </code>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Advanced Features Toggle */}
      <div style={{ marginTop: 12, marginBottom: 12 }}>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: 14,
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          <input
            type="checkbox"
            checked={props.useAdvanced}
            onChange={(e) => props.onUseAdvancedChange(e.target.checked)}
          />
          <span>🚀 Enable Advanced Features (AdvancedNeuralLM)</span>
        </label>
      </div>

      {/* GPU Acceleration Toggle */}
      <div
        style={{
          marginBottom: 12,
          padding: 12,
          borderRadius: 12,
          border: '1px solid rgba(99,102,241,0.3)',
          background: 'rgba(99,102,241,0.08)'
        }}
      >
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            fontSize: 14,
            fontWeight: 600,
            cursor: props.gpuAvailable ? 'pointer' : 'not-allowed',
            opacity: props.gpuAvailable ? 1 : 0.6
          }}
        >
          <input
            type="checkbox"
            checked={props.useGPU}
            onChange={(e) => props.onUseGPUChange(e.target.checked)}
            disabled={!props.gpuAvailable}
            aria-label={
              props.gpuAvailable
                ? 'Toggle WebGPU acceleration'
                : 'WebGPU not available on this device'
            }
          />
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            ⚡ GPU Acceleration (WebGPU)
            <span
              style={{
                fontSize: 11,
                padding: '2px 8px',
                borderRadius: 999,
                background: props.gpuAvailable ? 'rgba(34,197,94,0.2)' : 'rgba(248,113,113,0.2)',
                color: props.gpuAvailable ? '#34d399' : '#f87171'
              }}
            >
              {props.gpuAvailable ? 'Available' : 'Unavailable'}
            </span>
          </span>
        </label>
        <div
          style={{
            fontSize: 11,
            color: props.gpuAvailable ? '#cbd5f5' : '#fca5a5',
            marginInlineStart: 24,
            marginTop: 6
          }}
        >
          {props.gpuAvailable
            ? props.useGPU
              ? 'WebGPU acceleration enabled. Expect 2-5x faster epochs on compatible hardware.'
              : 'WebGPU detected. Enable the toggle to accelerate matrix operations during training.'
            : 'WebGPU is not available in this browser. Training will safely fall back to CPU.'}
        </div>
      </div>

      {/* Advanced Features Panel */}
      {props.useAdvanced && (
        <div
          style={{
            background: 'rgba(99, 102, 241, 0.1)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 12
          }}
        >
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, color: '#a78bfa' }}>
            Advanced Neural Network Configuration
          </div>

          {/* Activation & Initialization */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 12,
              marginBottom: 12
            }}
          >
            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Activation Function</div>
              <select
                aria-label="Activation function"
                value={props.activation}
                onChange={(e) => props.onActivationChange(e.target.value as ActivationFunction)}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              >
                <option value="relu">ReLU</option>
                <option value="leaky_relu">Leaky ReLU</option>
                <option value="elu">ELU</option>
                <option value="gelu">GELU</option>
              </select>
            </div>

            {props.activation === 'leaky_relu' && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Leaky ReLU Alpha</div>
                <input
                  aria-label="Leaky ReLU alpha"
                  type="number"
                  step="0.01"
                  value={props.leakyReluAlpha}
                  onChange={(e) =>
                    props.onLeakyReluAlphaChange(
                      clamp(
                        parseFloat(e.target.value || '0.01'),
                        HYPERPARAMETER_CONSTRAINTS.leakyReluAlpha.min,
                        HYPERPARAMETER_CONSTRAINTS.leakyReluAlpha.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}

            {props.activation === 'elu' && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>ELU Alpha</div>
                <input
                  aria-label="ELU alpha"
                  type="number"
                  step="0.1"
                  value={props.eluAlpha}
                  onChange={(e) =>
                    props.onEluAlphaChange(
                      clamp(
                        parseFloat(e.target.value || '1.0'),
                        HYPERPARAMETER_CONSTRAINTS.eluAlpha.min,
                        HYPERPARAMETER_CONSTRAINTS.eluAlpha.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}

            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Weight Initialization</div>
              <select
                aria-label="Weight initialization"
                value={props.initialization}
                onChange={(e) =>
                  props.onInitializationChange(e.target.value as InitializationScheme)
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              >
                <option value="default">Default</option>
                <option value="xavier">Xavier</option>
                <option value="he">He</option>
              </select>
            </div>
          </div>

          {/* Learning Rate Schedule */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12,
              marginBottom: 12
            }}
          >
            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>LR Schedule</div>
              <select
                aria-label="Learning rate schedule"
                value={props.lrSchedule}
                onChange={(e) => props.onLrScheduleChange(e.target.value as LRSchedule)}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              >
                <option value="constant">Constant</option>
                <option value="cosine">Cosine Annealing</option>
                <option value="exponential">Exponential Decay</option>
                <option value="warmup_cosine">Warmup + Cosine</option>
              </select>
            </div>

            {(props.lrSchedule === 'cosine' || props.lrSchedule === 'warmup_cosine') && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Min LR</div>
                <input
                  aria-label="Minimum learning rate"
                  type="number"
                  step="0.000001"
                  value={props.lrMin}
                  onChange={(e) =>
                    props.onLrMinChange(
                      clamp(
                        parseFloat(e.target.value || '0.000001'),
                        HYPERPARAMETER_CONSTRAINTS.lrMin.min,
                        HYPERPARAMETER_CONSTRAINTS.lrMin.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}

            {props.lrSchedule === 'exponential' && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Decay Rate</div>
                <input
                  aria-label="Learning rate decay rate"
                  type="number"
                  step="0.01"
                  value={props.lrDecayRate}
                  onChange={(e) =>
                    props.onLrDecayRateChange(
                      clamp(
                        parseFloat(e.target.value || '0.95'),
                        HYPERPARAMETER_CONSTRAINTS.lrDecayRate.min,
                        HYPERPARAMETER_CONSTRAINTS.lrDecayRate.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}

            {props.lrSchedule === 'warmup_cosine' && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Warmup Epochs</div>
                <input
                  aria-label="Warmup epochs"
                  type="number"
                  value={props.warmupEpochs}
                  onChange={(e) =>
                    props.onWarmupEpochsChange(
                      clamp(
                        parseInt(e.target.value || '0'),
                        HYPERPARAMETER_CONSTRAINTS.warmupEpochs.min,
                        HYPERPARAMETER_CONSTRAINTS.warmupEpochs.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}
          </div>

          {/* Regularization */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 12,
              marginBottom: 12
            }}
          >
            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Weight Decay (L2)</div>
              <input
                aria-label="Weight decay"
                type="number"
                step="0.0001"
                value={props.weightDecay}
                onChange={(e) =>
                  props.onWeightDecayChange(
                    clamp(
                      parseFloat(e.target.value || '0.0001'),
                      HYPERPARAMETER_CONSTRAINTS.weightDecay.min,
                      HYPERPARAMETER_CONSTRAINTS.weightDecay.max
                    )
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              />
            </div>

            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Gradient Clip Norm</div>
              <input
                aria-label="Gradient clip norm"
                type="number"
                step="0.5"
                value={props.gradientClipNorm}
                onChange={(e) =>
                  props.onGradientClipNormChange(
                    clamp(
                      parseFloat(e.target.value || '5.0'),
                      HYPERPARAMETER_CONSTRAINTS.gradientClipNorm.min,
                      HYPERPARAMETER_CONSTRAINTS.gradientClipNorm.max
                    )
                  )
                }
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              />
            </div>

            <div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Clip Mode</div>
              <select
                aria-label="Gradient clip mode"
                value={props.gradientClipMode}
                onChange={(e) => props.onGradientClipModeChange(e.target.value as GradientClipMode)}
                style={{
                  width: '100%',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 6,
                  padding: 8,
                  color: 'white'
                }}
              >
                <option value="global_norm">Global Norm</option>
                <option value="per_parameter">Per-Parameter</option>
                <option value="both">Both</option>
              </select>
            </div>
          </div>

          {/* Layer Norm & EMA */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 12,
              marginBottom: 12
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={props.useLayerNorm}
                  onChange={(e) => props.onUseLayerNormChange(e.target.checked)}
                />
                Layer Normalization
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={props.useEMA}
                  onChange={(e) => props.onUseEMAChange(e.target.checked)}
                />
                EMA Weight Averaging
              </label>
            </div>

            {props.useEMA && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>EMA Decay</div>
                <input
                  aria-label="EMA decay"
                  type="number"
                  step="0.001"
                  value={props.emaDecay}
                  onChange={(e) =>
                    props.onEmaDecayChange(
                      clamp(
                        parseFloat(e.target.value || '0.999'),
                        HYPERPARAMETER_CONSTRAINTS.emaDecay.min,
                        HYPERPARAMETER_CONSTRAINTS.emaDecay.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}
          </div>

          {/* Beam Search */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 12,
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={props.useBeamSearch}
                  onChange={(e) => props.onUseBeamSearchChange(e.target.checked)}
                />
                Use Beam Search for Generation
              </label>
            </div>

            {props.useBeamSearch && (
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Beam Width</div>
                <input
                  aria-label="Beam width"
                  type="number"
                  value={props.beamWidth}
                  onChange={(e) =>
                    props.onBeamWidthChange(
                      clamp(
                        parseInt(e.target.value || '4'),
                        HYPERPARAMETER_CONSTRAINTS.beamWidth.min,
                        HYPERPARAMETER_CONSTRAINTS.beamWidth.max
                      )
                    )
                  }
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: 6,
                    padding: 8,
                    color: 'white'
                  }}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Information Bottleneck Loss */}
      {props.useAdvanced && (
        <div style={{ marginTop: 12, marginBottom: 12 }}>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: 'pointer',
              fontSize: 12
            }}
          >
            <input
              type="checkbox"
              checked={props.useIB}
              onChange={(e) => props.onUseIBChange(e.target.checked)}
            />
            <span>📊 Enable Information Bottleneck Loss (Research)</span>
          </label>
        </div>
      )}

      {/* Information Bottleneck Configuration */}
      {props.useAdvanced && props.useIB && (
        <div
          style={{
            background: 'rgba(168, 85, 247, 0.1)',
            border: '1px solid rgba(168, 85, 247, 0.3)',
            borderRadius: 12,
            padding: 16,
            marginBottom: 16
          }}
        >
          <div style={{ fontSize: 13, fontWeight: 600, color: '#c4b5fd', marginBottom: 12 }}>
            📊 Information Bottleneck Configuration
          </div>

          <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 12, lineHeight: 1.5 }}>
            IB loss balances compression I(X;Z) vs prediction I(Z;Y) using β annealing. Loss: L =
            (1-α)·CE + α·[-I(Z;Y) + β·I(X;Z)]
          </div>

          <div style={{ display: 'grid', gap: 12 }}>
            {/* Beta Schedule */}
            <div>
              <div style={{ display: 'block', fontSize: 11, color: '#cbd5e1', marginBottom: 6 }}>
                β Schedule
              </div>
              <select
                value={props.betaSchedule}
                onChange={(e) =>
                  props.onBetaScheduleChange(
                    e.target.value as 'constant' | 'linear' | 'exponential' | 'cosine'
                  )
                }
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: '1px solid #4b5563',
                  background: '#1e293b',
                  color: '#e5e7eb',
                  fontSize: 11
                }}
              >
                <option value="constant">Constant (fixed β)</option>
                <option value="linear">Linear (gradual change)</option>
                <option value="exponential">Exponential (fast then slow)</option>
                <option value="cosine">Cosine (smooth annealing)</option>
              </select>
            </div>

            {/* Beta Start */}
            <div>
              <label style={{ display: 'block', fontSize: 11, color: '#cbd5e1', marginBottom: 6 }}>
                β Start: {props.betaStart.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.0001"
                max="10"
                step="0.001"
                value={props.betaStart}
                onChange={(e) => props.onBetaStartChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                Initial β (compression weight)
              </div>
            </div>

            {/* Beta End */}
            {props.betaSchedule !== 'constant' && (
              <div>
                <label
                  style={{ display: 'block', fontSize: 11, color: '#cbd5e1', marginBottom: 6 }}
                >
                  β End: {props.betaEnd.toFixed(4)}
                </label>
                <input
                  type="range"
                  min="0.0001"
                  max="10"
                  step="0.001"
                  value={props.betaEnd}
                  onChange={(e) => props.onBetaEndChange(Number(e.target.value))}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                  Final β (compression weight)
                </div>
              </div>
            )}

            {/* IB Alpha (hybrid loss weight) */}
            <div>
              <label style={{ display: 'block', fontSize: 11, color: '#cbd5e1', marginBottom: 6 }}>
                α (Hybrid Weight): {props.ibAlpha.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={props.ibAlpha}
                onChange={(e) => props.onIbAlphaChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                0 = pure CE loss, 1 = pure IB loss
              </div>
            </div>

            {/* Number of Bins */}
            <div>
              <label style={{ display: 'block', fontSize: 11, color: '#cbd5e1', marginBottom: 6 }}>
                MI Estimation Bins: {props.numBins}
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={props.numBins}
                onChange={(e) => props.onNumBinsChange(Number(e.target.value))}
                style={{ width: '100%' }}
              />
              <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>
                More bins = more accurate MI estimation (slower)
              </div>
            </div>
          </div>

          <div
            style={{
              marginTop: 12,
              padding: 8,
              background: 'rgba(139, 92, 246, 0.1)',
              borderRadius: 6,
              fontSize: 9,
              color: '#94a3b8',
              lineHeight: 1.5
            }}
          >
            💡 <strong style={{ color: '#c4b5fd' }}>Tip:</strong> Start with α=0.1, βStart=0.001,
            βEnd=1.0, linear schedule. Higher β = more compression (simpler representations).
          </div>
        </div>
      )}

      {/* Tokenizer Configuration */}
      <TokenizerConfig
        config={props.tokenizerConfig}
        customPattern={props.customTokenizerPattern}
        error={props.tokenizerError}
        onConfigChange={props.onTokenizerConfigChange}
        onCustomPatternChange={props.onCustomPatternChange}
        onError={props.onTokenizerError}
        onMessage={props.onMessage}
      />

      {/* Resume Checkbox & Action Buttons */}
      <div
        style={{
          display: 'flex',
          gap: 12,
          alignItems: 'center',
          flexWrap: 'wrap',
          marginBottom: 10
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
          <input
            type="checkbox"
            checked={props.resume}
            onChange={(e) => props.onResumeChange(e.target.checked)}
          />{' '}
          Resume training when possible
        </label>
        <div style={{ marginInlineStart: 'auto', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            onClick={props.onTrain}
            disabled={props.isTraining}
            aria-label={props.isTraining ? 'Training in progress' : 'Start model training'}
            aria-disabled={props.isTraining}
            style={{
              padding: '12px 20px',
              background: props.isTraining ? '#475569' : 'linear-gradient(90deg, #7c3aed, #059669)',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 700,
              cursor: props.isTraining ? 'not-allowed' : 'pointer',
              minWidth: 120
            }}
          >
            {props.isTraining ? '🔄 Training…' : '🚀 Train model'}
          </button>
          {props.isTraining && (
            <button
              onClick={props.onStop}
              aria-label="Stop training"
              style={{
                padding: '12px 16px',
                background: '#dc2626',
                border: 'none',
                borderRadius: 10,
                color: 'white',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              ⏹️ Stop
            </button>
          )}
          <button
            onClick={props.onReset}
            aria-label="Reset model"
            style={{
              padding: '12px 16px',
              background: '#374151',
              border: '1px solid #4b5563',
              borderRadius: 10,
              color: '#e5e7eb',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            🔄 Reset
          </button>
          <button
            onClick={props.onSave}
            aria-label="Save model to localStorage"
            style={{
              padding: '12px 16px',
              background: '#2563eb',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            💾 Save
          </button>
          <button
            onClick={props.onLoad}
            aria-label="Load model from localStorage"
            style={{
              padding: '12px 16px',
              background: '#4b5563',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            📀 Load
          </button>
          <button
            onClick={props.onExport}
            aria-label="Export model to JSON file"
            style={{
              padding: '12px 16px',
              background: '#16a34a',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            ⬇️ Export
          </button>
          <button
            onClick={props.onCompress}
            aria-label="Compress and export model"
            title="Compress model using quantization, distillation, or low-rank approximation"
            style={{
              padding: '12px 16px',
              background: 'linear-gradient(90deg, #3b82f6, #2563eb)',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            🗜️ Compress
          </button>
          <button
            onClick={() => importRef.current?.click()}
            aria-label="Import model from JSON file"
            style={{
              padding: '12px 16px',
              background: '#9333ea',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            ⬆️ Import
          </button>
          <input
            ref={importRef}
            type="file"
            accept="application/json"
            onChange={props.onImport}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {/* Progress Bar */}
      {props.isTraining && (
        <div style={{ marginTop: 8 }}>
          <div
            role="progressbar"
            aria-valuenow={Math.round(props.progress)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Training progress"
            style={{
              width: '100%',
              height: 12,
              background: '#334155',
              borderRadius: 6,
              overflow: 'hidden'
            }}
          >
            <div
              style={{
                width: `${props.progress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #a78bfa, #34d399)',
                transition: 'width 0.3s ease'
              }}
            />
          </div>
          <div
            style={{
              fontSize: 12,
              color: '#94a3b8',
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: 6
            }}
            aria-live="polite"
          >
            <span>Training… {props.progress.toFixed(0)}%</span>
            <span>
              Epoch: {props.currentEpoch + 1}/{props.epochs}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
