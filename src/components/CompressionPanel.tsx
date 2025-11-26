/**
 * CompressionPanel - UI for model compression options
 *
 * Allows users to compress models using:
 * - Int8 Quantization (4x size reduction)
 * - Knowledge Distillation (train smaller student model)
 * - Low-Rank Approximation (SVD-based compression)
 */

import React, { useState } from 'react';
import type { ProNeuralLM } from '../lib/ProNeuralLM';
import {
  compressWithQuantization,
  compressWithLowRank,
  compressWithDistillation,
  type CompressionResult,
  exportCompressedModel
} from '../compression/compress';
import { downloadBlob } from '../lib/utils';

interface CompressionPanelProps {
  model: ProNeuralLM | null;
  corpus: string;
  onClose: () => void;
  onMessage: (message: string) => void;
}

export function CompressionPanel({ model, corpus, onClose, onMessage }: CompressionPanelProps) {
  const [method, setMethod] = useState<'quantization' | 'lowrank' | 'distillation'>('quantization');
  const [rank, setRank] = useState(16);
  const [targetRatio, setTargetRatio] = useState(2.0);
  const [studentHiddenSize, setStudentHiddenSize] = useState(32);
  const [temperature, setTemperature] = useState(3.0);
  const [isCompressing, setIsCompressing] = useState(false);
  const [result, setResult] = useState<CompressionResult | null>(null);

  const handleCompress = async () => {
    if (!model) {
      onMessage('❌ No model to compress');
      return;
    }

    setIsCompressing(true);
    setResult(null);

    try {
      let compressionResult: CompressionResult;

      switch (method) {
        case 'quantization':
          compressionResult = compressWithQuantization(model);
          break;

        case 'lowrank':
          compressionResult = compressWithLowRank(model, {
            rank: rank,
            targetCompressionRatio: targetRatio
          });
          break;

        case 'distillation':
          if (!corpus || corpus.trim().length === 0) {
            onMessage('❌ Corpus required for knowledge distillation');
            setIsCompressing(false);
            return;
          }
          compressionResult = await compressWithDistillation(model, corpus, {
            studentHiddenSize,
            temperature,
            alpha: 0.7,
            epochs: 30,
            learningRate: 0.1,
            useHardLabels: true
          });
          break;
      }

      setResult(compressionResult);
      onMessage(
        `✅ Compression complete! Ratio: ${compressionResult.compressionRatio.toFixed(2)}x`
      );
    } catch (error) {
      onMessage(`❌ Compression failed: ${error}`);
      console.error('Compression error:', error);
    } finally {
      setIsCompressing(false);
    }
  };

  const handleExport = () => {
    if (!result) return;

    const json = exportCompressedModel(result);
    const blob = new Blob([json], { type: 'application/json' });
    const filename = `compressed-${method}-${Date.now()}.json`;

    downloadBlob(blob, filename);
    onMessage(`📦 Exported compressed model: ${filename}`);
  };

  return (
    <div
      role="presentation"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0,0,0,0.8)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
      }}
    >
      <div
        style={{
          background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
          border: '2px solid #3b82f6',
          borderRadius: 16,
          padding: 32,
          maxWidth: 600,
          width: '90%',
          maxHeight: '80vh',
          overflowY: 'auto',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
        }}
      >
        <div role="dialog" aria-modal="true" aria-labelledby="compression-title">
          {/* Header */}
          <div style={{ marginBottom: 24 }}>
            <h2
              id="compression-title"
              style={{
                margin: 0,
                fontSize: 24,
                fontWeight: 700,
                background: 'linear-gradient(90deg, #60a5fa, #a78bfa)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: 8
              }}
            >
              🗜️ Model Compression
            </h2>
            <p style={{ margin: 0, fontSize: 13, color: '#94a3b8' }}>
              Reduce model size for faster loading and smaller file sizes
            </p>
          </div>

          {/* Method Selection */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 12, fontWeight: 600 }}>
              Compression Method
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <button
                onClick={() => setMethod('quantization')}
                style={{
                  padding: '16px 12px',
                  background: method === 'quantization' ? '#3b82f6' : '#374151',
                  border: method === 'quantization' ? '2px solid #60a5fa' : '1px solid #4b5563',
                  borderRadius: 12,
                  color: 'white',
                  fontWeight: method === 'quantization' ? 700 : 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  fontSize: 13
                }}
              >
                <div>🎯 Quantization</div>
                <div style={{ fontSize: 10, color: '#cbd5e1', marginTop: 6 }}>4x smaller</div>
              </button>

              <button
                onClick={() => setMethod('lowrank')}
                style={{
                  padding: '16px 12px',
                  background: method === 'lowrank' ? '#3b82f6' : '#374151',
                  border: method === 'lowrank' ? '2px solid #60a5fa' : '1px solid #4b5563',
                  borderRadius: 12,
                  color: 'white',
                  fontWeight: method === 'lowrank' ? 700 : 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  fontSize: 13
                }}
              >
                <div>📐 Low-Rank</div>
                <div style={{ fontSize: 10, color: '#cbd5e1', marginTop: 6 }}>2-3x smaller</div>
              </button>

              <button
                onClick={() => setMethod('distillation')}
                style={{
                  padding: '16px 12px',
                  background: method === 'distillation' ? '#3b82f6' : '#374151',
                  border: method === 'distillation' ? '2px solid #60a5fa' : '1px solid #4b5563',
                  borderRadius: 12,
                  color: 'white',
                  fontWeight: method === 'distillation' ? 700 : 600,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  fontSize: 13
                }}
              >
                <div>🎓 Distillation</div>
                <div style={{ fontSize: 10, color: '#cbd5e1', marginTop: 6 }}>Variable</div>
              </button>
            </div>
          </div>

          {/* Method-specific controls */}
          <div
            style={{
              background: '#0f172a',
              borderRadius: 12,
              padding: 20,
              marginBottom: 24,
              border: '1px solid #1e293b'
            }}
          >
            {method === 'quantization' && (
              <div>
                <div style={{ fontSize: 13, color: '#e2e8f0', marginBottom: 8 }}>
                  <strong>Int8 Quantization</strong>
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>
                  Converts float32 weights to int8 (8-bit integers). Typically achieves 4x size
                  reduction with minimal accuracy loss (&lt;2%).
                </div>
                <div
                  style={{
                    marginTop: 12,
                    padding: 12,
                    background: '#1e293b',
                    borderRadius: 8,
                    fontSize: 11,
                    color: '#cbd5e1'
                  }}
                >
                  💡 <strong>Best for:</strong> General compression, fast export/import
                </div>
              </div>
            )}

            {method === 'lowrank' && (
              <div>
                <div style={{ fontSize: 13, color: '#e2e8f0', marginBottom: 12 }}>
                  <strong>Low-Rank Approximation (SVD)</strong>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <label
                    htmlFor="rank-input"
                    style={{ display: 'block', fontSize: 12, color: '#cbd5e1', marginBottom: 8 }}
                  >
                    Rank (lower = more compression)
                  </label>
                  <input
                    id="rank-input"
                    type="number"
                    value={rank}
                    onChange={(e) => setRank(Math.max(1, Number(e.target.value)))}
                    min={1}
                    max={128}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#1e293b',
                      border: '1px solid #374151',
                      borderRadius: 8,
                      color: 'white',
                      fontSize: 13
                    }}
                  />
                </div>

                <div style={{ marginBottom: 12 }}>
                  <label
                    htmlFor="target-ratio-input"
                    style={{ display: 'block', fontSize: 12, color: '#cbd5e1', marginBottom: 8 }}
                  >
                    Target Compression Ratio
                  </label>
                  <input
                    id="target-ratio-input"
                    type="number"
                    value={targetRatio}
                    onChange={(e) => setTargetRatio(Math.max(1.1, Number(e.target.value)))}
                    min={1.1}
                    max={10}
                    step={0.1}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#1e293b',
                      border: '1px solid #374151',
                      borderRadius: 8,
                      color: 'white',
                      fontSize: 13
                    }}
                  />
                </div>

                <div
                  style={{
                    padding: 12,
                    background: '#1e293b',
                    borderRadius: 8,
                    fontSize: 11,
                    color: '#cbd5e1'
                  }}
                >
                  💡 <strong>Best for:</strong> Weight matrices, customizable compression
                </div>
              </div>
            )}

            {method === 'distillation' && (
              <div>
                <div style={{ fontSize: 13, color: '#e2e8f0', marginBottom: 12 }}>
                  <strong>Knowledge Distillation</strong>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <label
                    htmlFor="student-hidden-size-input"
                    style={{ display: 'block', fontSize: 12, color: '#cbd5e1', marginBottom: 8 }}
                  >
                    Student Hidden Size
                  </label>
                  <input
                    id="student-hidden-size-input"
                    type="number"
                    value={studentHiddenSize}
                    onChange={(e) => setStudentHiddenSize(Math.max(8, Number(e.target.value)))}
                    min={8}
                    max={256}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#1e293b',
                      border: '1px solid #374151',
                      borderRadius: 8,
                      color: 'white',
                      fontSize: 13
                    }}
                  />
                </div>

                <div style={{ marginBottom: 12 }}>
                  <label
                    htmlFor="temperature-input"
                    style={{ display: 'block', fontSize: 12, color: '#cbd5e1', marginBottom: 8 }}
                  >
                    Temperature
                  </label>
                  <input
                    id="temperature-input"
                    type="number"
                    value={temperature}
                    onChange={(e) => setTemperature(Math.max(1, Number(e.target.value)))}
                    min={1}
                    max={10}
                    step={0.5}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      background: '#1e293b',
                      border: '1px solid #374151',
                      borderRadius: 8,
                      color: 'white',
                      fontSize: 13
                    }}
                  />
                </div>

                <div
                  style={{
                    padding: 12,
                    background: '#1e293b',
                    borderRadius: 8,
                    fontSize: 11,
                    color: '#cbd5e1'
                  }}
                >
                  💡 <strong>Best for:</strong> Creating smaller models, requires corpus
                </div>
              </div>
            )}
          </div>

          {/* Results Display */}
          {result && (
            <div
              style={{
                background: '#064e3b',
                border: '1px solid #059669',
                borderRadius: 12,
                padding: 20,
                marginBottom: 24
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 700, color: '#34d399', marginBottom: 12 }}>
                ✅ Compression Successful
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#6ee7b7' }}>Original Size</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'white' }}>
                    {(result.originalSize / 1024).toFixed(1)} KB
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: 11, color: '#6ee7b7' }}>Compressed Size</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'white' }}>
                    {(result.compressedSize / 1024).toFixed(1)} KB
                  </div>
                </div>

                <div>
                  <div style={{ fontSize: 11, color: '#6ee7b7' }}>Compression Ratio</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#34d399' }}>
                    {result.compressionRatio.toFixed(2)}x
                  </div>
                </div>

                {result.approximationError !== undefined && (
                  <div>
                    <div style={{ fontSize: 11, color: '#6ee7b7' }}>Approx. Error</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: 'white' }}>
                      {result.approximationError.toExponential(2)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              onClick={handleCompress}
              disabled={isCompressing || !model}
              style={{
                flex: 1,
                padding: '14px 24px',
                background: isCompressing ? '#374151' : 'linear-gradient(90deg, #3b82f6, #2563eb)',
                border: 'none',
                borderRadius: 10,
                color: 'white',
                fontWeight: 700,
                fontSize: 14,
                cursor: isCompressing || !model ? 'not-allowed' : 'pointer',
                opacity: isCompressing || !model ? 0.6 : 1,
                transition: 'all 0.2s'
              }}
            >
              {isCompressing ? '⏳ Compressing...' : '🗜️ Compress Model'}
            </button>

            {result && (
              <button
                onClick={handleExport}
                style={{
                  flex: 1,
                  padding: '14px 24px',
                  background: 'linear-gradient(90deg, #10b981, #059669)',
                  border: 'none',
                  borderRadius: 10,
                  color: 'white',
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                📦 Export Compressed
              </button>
            )}

            <button
              onClick={onClose}
              style={{
                padding: '14px 24px',
                background: '#374151',
                border: '1px solid #4b5563',
                borderRadius: 10,
                color: '#cbd5e1',
                fontWeight: 600,
                fontSize: 14,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
