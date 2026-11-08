import React, { useState } from 'react';
import { ProNeuralLM } from '../lib/ProNeuralLM';
import { TransformerLM } from '../lib/TransformerLM';
import { explainNextToken } from '../explainability/modelAttribution';

interface ExplainabilityPanelProps {
  model: ProNeuralLM | null;
}

type ExplainabilityMethod = 'shap' | 'integrated-gradients' | 'attention-rollout';

interface TokenAttribution {
  token: string;
  score: number;
  normalized: number; // Normalized to [0, 1] or [-1, 1]
}

export function ExplainabilityPanel({ model }: ExplainabilityPanelProps) {
  const [inputText, setInputText] = useState('');
  const [method, setMethod] = useState<ExplainabilityMethod>('shap');
  const [attributions, setAttributions] = useState<TokenAttribution[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explained, setExplained] = useState<{
    target: string;
    probability: number;
    totalEffect?: number;
  } | null>(null);

  const isTransformer = model instanceof TransformerLM;

  const computeExplainability = async () => {
    if (!model || !inputText.trim()) {
      setError('Please provide a trained model and input text');
      return;
    }

    setIsComputing(true);
    setError(null);

    try {
      const tokens = ProNeuralLM.tokenizeText(inputText, model.getTokenizerConfig());

      if (tokens.length === 0) {
        setError('No valid tokens found in input text');
        setIsComputing(false);
        return;
      }
      if (method === 'attention-rollout' && !isTransformer) {
        setError('Attention Rollout is only available for Transformer models');
        setIsComputing(false);
        return;
      }

      // Attributions of the model's next-token prediction to the tokens of its context window
      const explanation = await explainNextToken(model, inputText, method);
      const absMax = Math.max(...explanation.scores.map(Math.abs), 1e-12);
      setAttributions(
        explanation.tokens.map((token, i) => ({
          token,
          score: explanation.scores[i] ?? 0,
          normalized: (explanation.scores[i] ?? 0) / absMax
        }))
      );
      setExplained({
        target: explanation.target,
        probability: explanation.targetProbability,
        totalEffect: explanation.totalEffect
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Computation failed');
      console.error('Explainability error:', err);
    } finally {
      setIsComputing(false);
    }
  };

  const getColorForScore = (normalized: number): string => {
    // Map score to color: blue (negative) → white (0) → red (positive)
    const absValue = Math.abs(normalized);
    if (normalized > 0) {
      // Positive: white to red
      const intensity = Math.floor(255 * (1 - absValue));
      return `rgb(255, ${intensity}, ${intensity})`;
    } else {
      // Negative: white to blue
      const intensity = Math.floor(255 * (1 - absValue));
      return `rgb(${intensity}, ${intensity}, 255)`;
    }
  };

  const getMethodDescription = (m: ExplainabilityMethod): string => {
    switch (m) {
      case 'shap':
        return 'Shapley values of the context tokens for log p(predicted next token): exact over all token subsets for windows of up to 10 tokens, with a removed token replaced by BOS padding. Positive scores raise the probability of the prediction; the scores sum to the log-probability change from an all-padding context.';
      case 'integrated-gradients':
        return 'Integrated Gradients in embedding space, from an all-padding baseline to the actual context, using the model’s exact gradient of log p(predicted next token). Per-token scores sum to the same log-probability change (completeness).';
      case 'attention-rollout':
        return 'Attention Rollout over the transformer’s attention maps (heads averaged, residual connection modelled as ½A + ½I): the share of each context token in the representation that predicts the next token. Scores sum to 1.';
      default:
        return '';
    }
  };

  return (
    <div className="explainability-panel" style={{ padding: '20px', maxWidth: '900px' }}>
      <h2>🔍 Model Explainability</h2>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Understand how your model makes predictions by analyzing token importance and attention
        patterns.
      </p>

      {!model && (
        <div
          style={{
            padding: '15px',
            background: '#fff3cd',
            border: '1px solid #ffc107',
            borderRadius: '4px',
            marginBottom: '20px'
          }}
        >
          ⚠️ No model loaded. Please train a model first.
        </div>
      )}

      <div style={{ marginBottom: '20px' }}>
        <div style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
          Explainability Method:
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => setMethod('shap')}
            disabled={isComputing}
            style={{
              padding: '10px 20px',
              background: method === 'shap' ? '#007bff' : '#f0f0f0',
              color: method === 'shap' ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: isComputing ? 'not-allowed' : 'pointer',
              fontWeight: method === 'shap' ? 'bold' : 'normal'
            }}
          >
            SHAP Values
          </button>
          <button
            onClick={() => setMethod('integrated-gradients')}
            disabled={isComputing}
            style={{
              padding: '10px 20px',
              background: method === 'integrated-gradients' ? '#007bff' : '#f0f0f0',
              color: method === 'integrated-gradients' ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: isComputing ? 'not-allowed' : 'pointer',
              fontWeight: method === 'integrated-gradients' ? 'bold' : 'normal'
            }}
          >
            Integrated Gradients
          </button>
          <button
            onClick={() => setMethod('attention-rollout')}
            disabled={isComputing || !isTransformer}
            style={{
              padding: '10px 20px',
              background: method === 'attention-rollout' ? '#007bff' : '#f0f0f0',
              color: method === 'attention-rollout' ? 'white' : 'black',
              border: 'none',
              borderRadius: '4px',
              cursor: isComputing || !isTransformer ? 'not-allowed' : 'pointer',
              fontWeight: method === 'attention-rollout' ? 'bold' : 'normal',
              opacity: !isTransformer ? 0.5 : 1
            }}
            title={!isTransformer ? 'Only available for Transformer models' : ''}
          >
            Attention Rollout {!isTransformer && '(Transformer only)'}
          </button>
        </div>
        <p style={{ fontSize: '14px', color: '#666', marginTop: '10px', lineHeight: '1.6' }}>
          {getMethodDescription(method)}
        </p>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label
          htmlFor="explainability-input"
          style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}
        >
          Input Text:
        </label>
        <textarea
          id="explainability-input"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter text to analyze (e.g., 'the quick brown fox')"
          disabled={isComputing}
          style={{
            width: '100%',
            minHeight: '100px',
            padding: '10px',
            fontSize: '14px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontFamily: 'monospace',
            resize: 'vertical'
          }}
        />
      </div>

      <button
        onClick={computeExplainability}
        disabled={isComputing || !model || !inputText.trim()}
        style={{
          padding: '12px 24px',
          background: isComputing || !model || !inputText.trim() ? '#ccc' : '#28a745',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: isComputing || !model || !inputText.trim() ? 'not-allowed' : 'pointer',
          fontSize: '16px',
          fontWeight: 'bold',
          marginBottom: '20px'
        }}
      >
        {isComputing ? '⏳ Computing...' : '🔍 Analyze'}
      </button>

      {error && (
        <div
          style={{
            padding: '15px',
            background: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
            color: '#721c24',
            marginBottom: '20px'
          }}
        >
          ❌ {error}
        </div>
      )}

      {attributions.length > 0 && (
        <div>
          <h3 style={{ marginBottom: '15px' }}>Token Attribution Scores</h3>
          {explained && (
            <p style={{ fontSize: '14px', color: '#444', marginBottom: '12px' }}>
              Explaining the prediction <strong>“{explained.target}”</strong> (p ={' '}
              {explained.probability.toFixed(3)}) from the model&apos;s context window
              {explained.totalEffect !== undefined &&
                ` · total effect ${explained.totalEffect.toFixed(3)} nats`}
              .
            </p>
          )}
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '8px',
              padding: '15px',
              background: '#f8f9fa',
              borderRadius: '4px',
              marginBottom: '20px'
            }}
          >
            {attributions.map((attr, i) => (
              <div
                key={i}
                style={{
                  padding: '8px 12px',
                  background: getColorForScore(attr.normalized),
                  borderRadius: '4px',
                  border: '1px solid #ddd',
                  fontFamily: 'monospace',
                  fontSize: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  minWidth: '80px'
                }}
                title={`Score: ${attr.score.toFixed(4)}`}
              >
                <span style={{ fontWeight: 'bold', marginBottom: '4px' }}>{attr.token}</span>
                <span style={{ fontSize: '12px', color: '#666' }}>
                  {attr.score >= 0 ? '+' : ''}
                  {attr.score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>

          <div
            style={{
              padding: '15px',
              background: '#e7f3ff',
              border: '1px solid #b3d9ff',
              borderRadius: '4px'
            }}
          >
            <h4 style={{ marginTop: 0 }}>💡 Interpretation Guide</h4>
            <ul style={{ marginBottom: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
              <li>
                <strong style={{ color: '#dc3545' }}>Red tokens</strong> have positive attribution
                scores (increase model output)
              </li>
              <li>
                <strong style={{ color: '#007bff' }}>Blue tokens</strong> have negative attribution
                scores (decrease model output)
              </li>
              <li>
                <strong>Darker colors</strong> indicate stronger attribution (higher importance)
              </li>
              <li>
                <strong>White/light colors</strong> indicate near-zero attribution (less important)
              </li>
            </ul>
          </div>
        </div>
      )}

      {attributions.length === 0 && !error && !isComputing && (
        <div
          style={{
            padding: '20px',
            background: '#f8f9fa',
            border: '1px solid #dee2e6',
            borderRadius: '4px',
            textAlign: 'center',
            color: '#666'
          }}
        >
          Enter text and click &quot;Analyze&quot; to see token attributions
        </div>
      )}
    </div>
  );
}
