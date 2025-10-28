import React, { useRef } from 'react';
import type { Optimizer, TokenizerConfig as TokenizerConfigType } from '../lib/ProNeuralLM';
import type { ActivationFunction, LRSchedule, InitializationScheme } from '../lib/AdvancedNeuralLM';
import { clamp } from '../lib/ProNeuralLM';
import { TokenizerConfig } from './TokenizerConfig';
import { HYPERPARAMETER_CONSTRAINTS } from '../config/constants';

interface TrainingPanelProps {
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
  samplingMode: 'off' | 'topk' | 'topp';
  seed: number;
  resume: boolean;

  // Advanced features
  useAdvanced: boolean;
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

  // Tokenizer
  tokenizerConfig: TokenizerConfigType;
  customTokenizerPattern: string;
  tokenizerError: string | null;

  // Training state
  isTraining: boolean;
  progress: number;
  currentEpoch: number;

  // Callbacks
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
  onSamplingModeChange: (value: 'off' | 'topk' | 'topp') => void;
  onSeedChange: (value: number) => void;
  onResumeChange: (value: boolean) => void;

  // Advanced callbacks
  onUseAdvancedChange: (value: boolean) => void;
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
  onUseLayerNormChange: (value: boolean) => void;
  onUseBeamSearchChange: (value: boolean) => void;
  onBeamWidthChange: (value: number) => void;

  onTokenizerConfigChange: (config: TokenizerConfigType) => void;
  onCustomPatternChange: (pattern: string) => void;
  onTokenizerError: (error: string | null) => void;
  onTrain: () => void;
  onStop: () => void;
  onReset: () => void;
  onSave: () => void;
  onLoad: () => void;
  onExport: () => void;
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
                onChange={(e) => props.onInitializationChange(e.target.value as InitializationScheme)}
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

            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
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
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
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
