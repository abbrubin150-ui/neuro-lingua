import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { ProNeuralLM } from '../lib/ProNeuralLM';
import {
  projectWithTSNE,
  projectWithUMAP,
  type ProjectionResult,
  type NormalisationMode,
  type TSNEProjectionOptions,
  type UMAPProjectionOptions
} from '../visualization/embeddings';

interface EmbeddingVisualizationPanelProps {
  model: ProNeuralLM | null;
  onClose?: () => void;
}

type ProjectionMethod = 'tsne' | 'umap';

interface CanvasViewState {
  offsetX: number;
  offsetY: number;
  scale: number;
}

export function EmbeddingVisualizationPanel({ model, onClose }: EmbeddingVisualizationPanelProps) {
  // Projection state
  const [method, setMethod] = useState<ProjectionMethod>('tsne');
  const [normalisation, setNormalisation] = useState<NormalisationMode>('zscore');
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // t-SNE parameters
  const [perplexity, setPerplexity] = useState<number>(30);
  const [iterations, setIterations] = useState<number>(500);
  const [epsilon, setEpsilon] = useState<number>(100);

  // UMAP parameters
  const [nNeighbors, setNNeighbors] = useState<number>(15);
  const [minDist, setMinDist] = useState<number>(0.1);
  const [spread, setSpread] = useState<number>(1.0);

  // Canvas state
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [viewState, setViewState] = useState<CanvasViewState>({
    offsetX: 0,
    offsetY: 0,
    scale: 1
  });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);

  // Vocabulary from model
  const [vocab, setVocab] = useState<string[]>([]);

  // Extract embeddings and vocabulary from model
  useEffect(() => {
    if (!model) {
      setProjection(null);
      setVocab([]);
      return;
    }

    try {
      const modelVocab = model.getVocab();
      setVocab(modelVocab);
      setError(null);
    } catch (err) {
      setError(`Failed to extract vocabulary: ${err}`);
      console.error(err);
    }
  }, [model]);

  // Compute projection
  const computeProjection = useCallback(async () => {
    if (!model) {
      setError('No model loaded');
      return;
    }

    setIsComputing(true);
    setError(null);

    try {
      const embeddings = model.getEmbeddings();

      if (embeddings.length === 0) {
        setError('Model has no embeddings (untrained model?)');
        setIsComputing(false);
        return;
      }

      if (embeddings.length < 3) {
        setError('Need at least 3 tokens for visualization');
        setIsComputing(false);
        return;
      }

      // Run projection in a setTimeout to allow UI to update
      await new Promise((resolve) => setTimeout(resolve, 10));

      let result: ProjectionResult;

      if (method === 'tsne') {
        const options: TSNEProjectionOptions = {
          perplexity: Math.min(perplexity, embeddings.length - 1),
          epsilon,
          iterations,
          normalise: normalisation
        };
        result = projectWithTSNE(embeddings, options);
      } else {
        const options: UMAPProjectionOptions = {
          nNeighbors: Math.min(nNeighbors, embeddings.length - 1),
          minDist,
          spread,
          normalise: normalisation
        };
        result = projectWithUMAP(embeddings, options);
      }

      setProjection(result);
      setViewState({ offsetX: 0, offsetY: 0, scale: 1 }); // Reset view
    } catch (err) {
      setError(`Projection failed: ${err}`);
      console.error(err);
    } finally {
      setIsComputing(false);
    }
  }, [model, method, normalisation, perplexity, epsilon, iterations, nNeighbors, minDist, spread]);

  // Auto-compute on parameter change (with debouncing)
  useEffect(() => {
    if (!model) return;

    const timer = setTimeout(() => {
      computeProjection();
    }, 300);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, method, normalisation, perplexity, epsilon, iterations, nNeighbors, minDist, spread]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !projection) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    const { coordinates, summary } = projection;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, width, height);

    // Calculate transformation
    const padding = 40;
    const dataWidth = summary.max[0] - summary.min[0] || 1;
    const dataHeight = summary.max[1] - summary.min[1] || 1;
    const scaleX = ((width - 2 * padding) / dataWidth) * viewState.scale;
    const scaleY = ((height - 2 * padding) / dataHeight) * viewState.scale;
    const scale = Math.min(scaleX, scaleY);

    const centerX = width / 2 + viewState.offsetX;
    const centerY = height / 2 + viewState.offsetY;

    // Transform data coordinates to canvas coordinates
    const toCanvasX = (x: number) => centerX + (x - (summary.min[0] + summary.max[0]) / 2) * scale;
    const toCanvasY = (y: number) => centerY - (y - (summary.min[1] + summary.max[1]) / 2) * scale;

    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Draw points
    coordinates.forEach(([x, y], i) => {
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);

      // Highlight hovered point
      if (i === hoveredPoint) {
        ctx.fillStyle = '#ff6b6b';
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Draw label
        const label = vocab[i] || `Token ${i}`;
        ctx.fillStyle = '#fff';
        ctx.font = '14px monospace';
        ctx.fillText(label, cx + 12, cy - 12);
      } else {
        ctx.fillStyle = '#4dabf7';
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw axis labels
    ctx.fillStyle = '#aaa';
    ctx.font = '12px sans-serif';
    ctx.fillText('Dimension 1', width - 80, height - 10);
    ctx.fillText('Dimension 2', 10, 20);

    // Draw info
    ctx.fillStyle = '#888';
    ctx.font = '11px monospace';
    ctx.fillText(`${coordinates.length} tokens`, 10, height - 10);
    ctx.fillText(`Scale: ${viewState.scale.toFixed(2)}x`, 10, height - 25);
  }, [projection, viewState, hoveredPoint, vocab]);

  // Canvas interaction handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !projection) return;

    // Handle dragging
    if (isDragging && dragStart) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      setViewState((prev) => ({
        ...prev,
        offsetX: prev.offsetX + dx,
        offsetY: prev.offsetY + dy
      }));
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    // Handle hover
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const { width, height } = canvas;
    const { coordinates, summary } = projection;

    const padding = 40;
    const dataWidth = summary.max[0] - summary.min[0] || 1;
    const dataHeight = summary.max[1] - summary.min[1] || 1;
    const scaleX = ((width - 2 * padding) / dataWidth) * viewState.scale;
    const scaleY = ((height - 2 * padding) / dataHeight) * viewState.scale;
    const scale = Math.min(scaleX, scaleY);

    const centerX = width / 2 + viewState.offsetX;
    const centerY = height / 2 + viewState.offsetY;

    const toCanvasX = (x: number) => centerX + (x - (summary.min[0] + summary.max[0]) / 2) * scale;
    const toCanvasY = (y: number) => centerY - (y - (summary.min[1] + summary.max[1]) / 2) * scale;

    // Find closest point
    let closestIdx = -1;
    let closestDist = Infinity;
    const hoverThreshold = 10;

    coordinates.forEach(([x, y], i) => {
      const cx = toCanvasX(x);
      const cy = toCanvasY(y);
      const dist = Math.sqrt((mouseX - cx) ** 2 + (mouseY - cy) ** 2);
      if (dist < hoverThreshold && dist < closestDist) {
        closestDist = dist;
        closestIdx = i;
      }
    });

    setHoveredPoint(closestIdx >= 0 ? closestIdx : null);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDragStart(null);
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setDragStart(null);
    setHoveredPoint(null);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setViewState((prev) => ({
      ...prev,
      scale: Math.max(0.1, Math.min(10, prev.scale * delta))
    }));
  };

  const resetView = () => {
    setViewState({ offsetX: 0, offsetY: 0, scale: 1 });
  };

  // Export functionality
  const exportAsImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `embedding-${method}-${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  const exportAsJSON = () => {
    if (!projection) return;

    const data = {
      ...projection,
      vocabulary: vocab,
      method,
      normalisation,
      timestamp: new Date().toISOString()
    };

    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `embedding-${method}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!model) {
    return (
      <div style={{ padding: '20px', background: '#2a2a2a', borderRadius: '8px' }}>
        <p style={{ color: '#aaa' }}>No model loaded. Train a model first.</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', background: '#2a2a2a', borderRadius: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
        <h3 style={{ margin: 0, color: '#fff' }}>Embedding Visualization</h3>
        {onClose && (
          <button onClick={onClose} style={{ padding: '5px 10px' }}>
            Close
          </button>
        )}
      </div>

      {/* Controls */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '10px',
          marginBottom: '15px'
        }}
      >
        {/* Method selector */}
        <div>
          <label style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}>
            Method
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as ProjectionMethod)}
              style={{ width: '100%', padding: '5px', marginTop: '5px', display: 'block' }}
              disabled={isComputing}
            >
              <option value="tsne">t-SNE</option>
              <option value="umap">UMAP</option>
            </select>
          </label>
        </div>

        {/* Normalization selector */}
        <div>
          <label style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}>
            Normalization
            <select
              value={normalisation}
              onChange={(e) => setNormalisation(e.target.value as NormalisationMode)}
              style={{ width: '100%', padding: '5px', marginTop: '5px', display: 'block' }}
              disabled={isComputing}
            >
              <option value="none">None</option>
              <option value="l2">L2</option>
              <option value="zscore">Z-score</option>
            </select>
          </label>
        </div>

        {/* t-SNE parameters */}
        {method === 'tsne' && (
          <>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Perplexity: {perplexity}
                <input
                  type="range"
                  min="2"
                  max="50"
                  value={perplexity}
                  onChange={(e) => setPerplexity(Number(e.target.value))}
                  style={{ width: '100%', marginTop: '5px', display: 'block' }}
                  disabled={isComputing}
                />
              </label>
            </div>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Iterations: {iterations}
              </label>
              <input
                type="range"
                min="100"
                max="1000"
                step="50"
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                style={{ width: '100%' }}
                disabled={isComputing}
              />
            </div>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Learning Rate: {epsilon}
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={epsilon}
                onChange={(e) => setEpsilon(Number(e.target.value))}
                style={{ width: '100%' }}
                disabled={isComputing}
              />
            </div>
          </>
        )}

        {/* UMAP parameters */}
        {method === 'umap' && (
          <>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Neighbors: {nNeighbors}
              </label>
              <input
                type="range"
                min="2"
                max="50"
                value={nNeighbors}
                onChange={(e) => setNNeighbors(Number(e.target.value))}
                style={{ width: '100%' }}
                disabled={isComputing}
              />
            </div>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Min Distance: {minDist.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={minDist}
                onChange={(e) => setMinDist(Number(e.target.value))}
                style={{ width: '100%' }}
                disabled={isComputing}
              />
            </div>
            <div>
              <label
                style={{ display: 'block', color: '#aaa', fontSize: '12px', marginBottom: '5px' }}
              >
                Spread: {spread.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.1"
                value={spread}
                onChange={(e) => setSpread(Number(e.target.value))}
                style={{ width: '100%' }}
                disabled={isComputing}
              />
            </div>
          </>
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <button onClick={computeProjection} disabled={isComputing} style={{ padding: '8px 15px' }}>
          {isComputing ? 'Computing...' : 'Recompute'}
        </button>
        <button onClick={resetView} disabled={!projection} style={{ padding: '8px 15px' }}>
          Reset View
        </button>
        <button onClick={exportAsImage} disabled={!projection} style={{ padding: '8px 15px' }}>
          Export PNG
        </button>
        <button onClick={exportAsJSON} disabled={!projection} style={{ padding: '8px 15px' }}>
          Export JSON
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div
          style={{
            padding: '10px',
            background: '#ff4444',
            color: '#fff',
            borderRadius: '5px',
            marginBottom: '15px'
          }}
        >
          {error}
        </div>
      )}

      {/* Status */}
      {isComputing && (
        <div
          style={{
            padding: '10px',
            background: '#4dabf7',
            color: '#fff',
            borderRadius: '5px',
            marginBottom: '15px'
          }}
        >
          Computing {method.toUpperCase()} projection... This may take a few seconds.
        </div>
      )}

      {/* Canvas */}
      <div style={{ position: 'relative', background: '#1a1a1a', borderRadius: '5px' }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
          style={{
            cursor: isDragging ? 'grabbing' : 'grab',
            display: 'block',
            borderRadius: '5px'
          }}
        />
        {!projection && !isComputing && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: '#666',
              fontSize: '14px',
              textAlign: 'center'
            }}
          >
            Waiting for projection...
            <br />
            <small>(Adjust parameters above to compute)</small>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div
        style={{
          marginTop: '15px',
          padding: '10px',
          background: '#333',
          borderRadius: '5px',
          fontSize: '12px',
          color: '#aaa'
        }}
      >
        <strong>Controls:</strong> Drag to pan • Scroll to zoom • Hover over points to see token
        labels
      </div>

      {/* Projection info */}
      {projection && (
        <div
          style={{
            marginTop: '15px',
            padding: '10px',
            background: '#333',
            borderRadius: '5px',
            fontSize: '12px',
            color: '#aaa'
          }}
        >
          <strong>Projection Info:</strong>
          <br />
          Method: {String(projection.metadata.method)}
          {' • '}
          Tokens: {projection.coordinates.length}
          {' • '}
          Normalization: {String(projection.metadata.normalise)}
          <br />
          {method === 'tsne' && (
            <>
              Perplexity: {String(projection.metadata.perplexity)}
              {' • '}
              Iterations: {String(projection.metadata.iterations)}
              {' • '}
              LR: {String(projection.metadata.epsilon)}
            </>
          )}
          {method === 'umap' && (
            <>
              Neighbors: {String(projection.metadata.nNeighbors)}
              {' • '}
              Min Dist: {String(projection.metadata.minDist)}
              {' • '}
              Spread: {String(projection.metadata.spread)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
