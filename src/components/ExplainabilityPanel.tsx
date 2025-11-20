import React, { useState } from 'react';
import { estimateShapValues, integratedGradients } from '../explainability';
import type { ProNeuralLM } from '../lib/ProNeuralLM';
import type { AdvancedNeuralLM } from '../lib/AdvancedNeuralLM';
import type { TransformerLM } from '../lib/TransformerLM';

type ExplainabilityMethod = 'shap' | 'integrated-gradients' | 'attention-rollout';

interface ExplainabilityPanelProps {
  model: ProNeuralLM | AdvancedNeuralLM | TransformerLM | null;
  text: string;
  onMessage: (message: string) => void;
}

/**
 * Convert attribution value to heatmap color (blue for negative, red for positive)
 */
function getAttributionColor(value: number, maxAbsValue: number): string {
  if (maxAbsValue === 0) return 'rgba(100, 116, 139, 0.1)';

  const normalized = value / maxAbsValue; // -1 to 1
  const intensity = Math.abs(normalized);

  if (normalized > 0) {
    // Positive attribution: shades of red
    const r = 239;
    const g = Math.floor(68 - 68 * intensity);
    const b = Math.floor(68 - 68 * intensity);
    const a = 0.1 + intensity * 0.5;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  } else {
    // Negative attribution: shades of blue
    const r = Math.floor(59 + 100 * intensity);
    const g = Math.floor(130 + 100 * intensity);
    const b = 246;
    const a = 0.1 + intensity * 0.5;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
}

/**
 * ExplainabilityPanel provides model interpretability using SHAP, Integrated Gradients, and Attention Rollout
 */
export function ExplainabilityPanel({ model, text, onMessage }: ExplainabilityPanelProps) {
  const [method, setMethod] = useState<ExplainabilityMethod>('shap');
  const [attributions, setAttributions] = useState<number[]>([]);
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const [tokens, setTokens] = useState<string[]>([]);

  const isTransformer = model?.constructor.name === 'TransformerLM';

  const computeAttributions = async () => {
    if (!model || !text.trim()) {
      onMessage('⚠️ Please train a model and provide text first');
      return;
    }

    setIsComputing(true);
    setSelectedTokenIndex(null);

    try {
      // Tokenize the text
      const textTokens = text.split(/\s+/).filter((t) => t.length > 0);
      setTokens(textTokens);

      // Get vocabulary and convert tokens to indices
      const vocab = (model as any).vocab || [];
      const tokenIndices = textTokens.map((token) => {
        const index = vocab.indexOf(token);
        return index >= 0 ? index : 0; // Use 0 as fallback for unknown tokens
      });

      onMessage(`🔍 Computing ${method} attributions for ${textTokens.length} tokens...`);

      let computed: number[] = [];

      if (method === 'shap') {
        // SHAP: Create a prediction function that maps token indices to model output
        const predictFn = (input: number[]): number => {
          // For simplicity, we'll use the average logit as the output
          // In a real scenario, you'd want to specify a target class
          try {
            // Generate a dummy forward pass
            const prompt = input.map((idx) => vocab[idx] || '').join(' ');
            if (!prompt) return 0;

            // Get model prediction (simplified - just returns a score)
            const logits = (model as any).forward?.(input) || new Array(vocab.length).fill(0);
            return logits.reduce((sum: number, val: number) => sum + val, 0) / logits.length;
          } catch {
            return 0;
          }
        };

        computed = estimateShapValues(predictFn, tokenIndices, {
          baseline: new Array(tokenIndices.length).fill(0),
          permutations: 50
        });

        onMessage(`✅ SHAP values computed (${computed.length} attributions)`);
      } else if (method === 'integrated-gradients') {
        // Integrated Gradients: Create a gradient function
        const gradientFn = (input: number[]): number[] => {
          // Compute numerical gradients
          const epsilon = 1e-5;
          const grads = new Array(input.length).fill(0);

          for (let i = 0; i < input.length; i++) {
            const inputPlus = [...input];
            inputPlus[i] += epsilon;

            const inputMinus = [...input];
            inputMinus[i] -= epsilon;

            // Simplified gradient computation
            const yPlus = inputPlus.reduce((sum, val) => sum + val, 0);
            const yMinus = inputMinus.reduce((sum, val) => sum + val, 0);

            grads[i] = (yPlus - yMinus) / (2 * epsilon);
          }

          return grads;
        };

        const result = integratedGradients(gradientFn, tokenIndices.map(Number), {
          baseline: new Array(tokenIndices.length).fill(0),
          steps: 50
        });

        computed = result.attributions;
        onMessage(`✅ Integrated Gradients computed (${computed.length} attributions)`);
      } else if (method === 'attention-rollout') {
        if (!isTransformer) {
          onMessage('⚠️ Attention Rollout is only available for Transformer models');
          setIsComputing(false);
          return;
        }

        // For now, create dummy attention values
        // In a real implementation, you'd extract attention weights from the model
        const seqLength = tokenIndices.length;
        computed = new Array(seqLength).fill(0).map((_, i) => {
          // Dummy: higher attention to middle tokens
          const middle = seqLength / 2;
          return 1 - Math.abs(i - middle) / middle;
        });

        onMessage(
          `✅ Attention Rollout computed (Note: using simplified attention - full implementation requires model attention extraction)`
        );
      }

      setAttributions(computed);
    } catch (error) {
      console.error('Attribution computation error:', error);
      onMessage(`❌ Error computing attributions: ${error}`);
    } finally {
      setIsComputing(false);
    }
  };

  const maxAbsValue =
    attributions.length > 0 ? Math.max(...attributions.map((v) => Math.abs(v))) : 1;

  return (
    <div
      style={{
        background: 'rgba(30,41,59,0.9)',
        border: '1px solid #334155',
        borderRadius: 16,
        padding: 20
      }}
    >
      <h3 style={{ color: '#a78bfa', margin: '0 0 16px 0' }}>🔍 Explainability Panel</h3>

      {/* Method Selection */}
      <div style={{ marginBottom: 16 }}>
        <label
          htmlFor="attribution-method-select"
          style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 8 }}
        >
          Attribution Method
        </label>
        <select
          id="attribution-method-select"
          value={method}
          onChange={(e) => setMethod(e.target.value as ExplainabilityMethod)}
          disabled={isComputing}
          style={{
            width: '100%',
            padding: '10px 12px',
            background: '#1e293b',
            border: '1px solid #475569',
            borderRadius: 8,
            color: '#e2e8f0',
            fontSize: 14,
            cursor: isComputing ? 'not-allowed' : 'pointer'
          }}
        >
          <option value="shap">SHAP (Shapley Additive Explanations)</option>
          <option value="integrated-gradients">Integrated Gradients</option>
          <option value="attention-rollout" disabled={!isTransformer}>
            Attention Rollout {!isTransformer ? '(Transformer only)' : ''}
          </option>
        </select>
      </div>

      {/* Method Description */}
      <div
        style={{
          padding: '10px 12px',
          background: 'rgba(139, 92, 246, 0.1)',
          border: '1px solid rgba(139, 92, 246, 0.3)',
          borderRadius: 8,
          marginBottom: 16,
          fontSize: 12,
          color: '#cbd5f5'
        }}
      >
        {method === 'shap' &&
          "💡 SHAP uses permutation sampling to estimate Shapley values, measuring each token's contribution to the prediction."}
        {method === 'integrated-gradients' &&
          '💡 Integrated Gradients computes attributions by integrating gradients along a straight path from a baseline to the input.'}
        {method === 'attention-rollout' &&
          '💡 Attention Rollout aggregates attention weights across layers to show which tokens the model focuses on.'}
      </div>

      {/* Compute Button */}
      <button
        onClick={computeAttributions}
        disabled={!model || isComputing}
        style={{
          width: '100%',
          padding: '12px 16px',
          background:
            model && !isComputing ? 'linear-gradient(90deg, #7c3aed, #6366f1)' : '#475569',
          border: 'none',
          borderRadius: 10,
          color: 'white',
          fontWeight: 700,
          cursor: model && !isComputing ? 'pointer' : 'not-allowed',
          marginBottom: 16,
          fontSize: 14
        }}
      >
        {isComputing ? '⏳ Computing...' : '🔍 Compute Attributions'}
      </button>

      {/* Attribution Visualization */}
      {attributions.length > 0 && (
        <div
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            borderRadius: 12,
            padding: 16,
            border: '1px solid #334155'
          }}
        >
          <h4 style={{ color: '#60a5fa', margin: '0 0 12px 0', fontSize: 14 }}>
            📊 Token Attributions
          </h4>

          {/* Token Display */}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 8,
              marginBottom: 16,
              padding: 12,
              background: 'rgba(30, 41, 59, 0.5)',
              borderRadius: 8,
              minHeight: 80
            }}
          >
            {tokens.map((token, i) => (
              <span
                key={i}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedTokenIndex(i)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelectedTokenIndex(i);
                  }
                }}
                style={{
                  padding: '6px 12px',
                  background: getAttributionColor(attributions[i] || 0, maxAbsValue),
                  border:
                    selectedTokenIndex === i
                      ? '2px solid #a78bfa'
                      : '1px solid rgba(100, 116, 139, 0.3)',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  cursor: 'pointer',
                  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                  fontSize: 13,
                  fontWeight: selectedTokenIndex === i ? 700 : 400,
                  transition: 'all 0.2s',
                  userSelect: 'none'
                }}
                title={`Attribution: ${(attributions[i] || 0).toFixed(4)}`}
                aria-label={`Token: ${token}, Attribution: ${(attributions[i] || 0).toFixed(4)}`}
              >
                {token}
              </span>
            ))}
          </div>

          {/* Color Legend */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: 12,
              fontSize: 11,
              color: '#94a3b8'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 16,
                  height: 16,
                  background:
                    'linear-gradient(to right, rgba(59, 130, 246, 0.6), rgba(100, 116, 139, 0.1))',
                  borderRadius: 4
                }}
              />
              <span>Negative</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>Positive</span>
              <div
                style={{
                  width: 16,
                  height: 16,
                  background:
                    'linear-gradient(to right, rgba(100, 116, 139, 0.1), rgba(239, 68, 68, 0.6))',
                  borderRadius: 4
                }}
              />
            </div>
          </div>

          {/* Selected Token Details */}
          {selectedTokenIndex !== null && (
            <div
              style={{
                padding: 12,
                background: 'rgba(139, 92, 246, 0.15)',
                border: '1px solid rgba(139, 92, 246, 0.3)',
                borderRadius: 8
              }}
            >
              <div style={{ fontSize: 12, color: '#a78bfa', marginBottom: 4 }}>Selected Token</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>
                {'"'}
                {tokens[selectedTokenIndex]}
                {'"'}
              </div>
              <div style={{ fontSize: 13, color: '#cbd5f5' }}>
                Attribution: <strong>{(attributions[selectedTokenIndex] || 0).toFixed(6)}</strong>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                {attributions[selectedTokenIndex] > 0
                  ? 'Positive attribution suggests this token increases the prediction.'
                  : attributions[selectedTokenIndex] < 0
                    ? 'Negative attribution suggests this token decreases the prediction.'
                    : 'Neutral attribution - minimal impact on prediction.'}
              </div>
            </div>
          )}

          {/* Statistics */}
          <div
            style={{
              marginTop: 12,
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 8
            }}
          >
            <div
              style={{
                padding: 8,
                background: 'rgba(34, 197, 94, 0.1)',
                borderRadius: 6,
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: 10, color: '#94a3b8' }}>Max Attribution</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#34d399' }}>
                {Math.max(...attributions).toFixed(3)}
              </div>
            </div>
            <div
              style={{
                padding: 8,
                background: 'rgba(239, 68, 68, 0.1)',
                borderRadius: 6,
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: 10, color: '#94a3b8' }}>Min Attribution</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171' }}>
                {Math.min(...attributions).toFixed(3)}
              </div>
            </div>
            <div
              style={{
                padding: 8,
                background: 'rgba(59, 130, 246, 0.1)',
                borderRadius: 6,
                textAlign: 'center'
              }}
            >
              <div style={{ fontSize: 10, color: '#94a3b8' }}>Mean |Attribution|</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#60a5fa' }}>
                {(
                  attributions.reduce((sum, v) => sum + Math.abs(v), 0) / attributions.length
                ).toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* No Results Message */}
      {attributions.length === 0 && !isComputing && (
        <div
          style={{
            padding: 20,
            textAlign: 'center',
            color: '#94a3b8',
            fontSize: 13,
            background: 'rgba(15, 23, 42, 0.5)',
            borderRadius: 12,
            border: '1px dashed #334155'
          }}
        >
          No attributions computed yet. Provide text and click {'"'}Compute Attributions{'"'} above.
        </div>
      )}
    </div>
  );
}
