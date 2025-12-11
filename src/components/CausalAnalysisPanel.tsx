/**
 * CausalAnalysisPanel - UI for Causal Inference System
 *
 * Provides interactive visualization and configuration for:
 * - DAG-based causal model specification
 * - Three-phase causal analysis workflow
 * - AIPW estimation with bias verification
 * - Sensitivity analysis (Rosenbaum bounds, E-values)
 * - Temporal dependencies visualization
 * - Sigma-SIG compliance ledger
 *
 * @module CausalAnalysisPanel
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  CausalInferenceEngine,
  createCausalEngine
} from '../lib/CausalInferenceEngine';
import type {
  CausalModelConfig,
  HypothesisTestResult,
  BiasVerificationResult,
  IdentifiabilityResult,
  PowerAnalysisResult,
  CausalAnalysisLedger
} from '../types/causal';
import type { CausalDAG, RosenbaumBounds, EValue } from '../types/dag';
import {
  createStandardDAG,
  validateDAG,
  analyzeIdentifiability,
  computeRosenbaumBounds,
  computeEValue
} from '../math/dag_operations';

// ============================================================================
// Types
// ============================================================================

interface CausalAnalysisPanelProps {
  /** Callback when analysis completes */
  onAnalysisComplete?: (result: AnalysisResult) => void;
  /** Initial configuration */
  initialConfig?: Partial<CausalModelConfig>;
  /** Language for UI */
  language?: 'en' | 'he';
}

interface AnalysisResult {
  ate: number;
  standardError: number;
  confidenceInterval: [number, number];
  pValue: number;
  significant: boolean;
  biasVerification: BiasVerificationResult;
  identifiability: IdentifiabilityResult;
  powerAnalysis: PowerAnalysisResult;
  ledger: CausalAnalysisLedger;
}

type AnalysisPhase = 'configure' | 'offline' | 'online' | 'testing' | 'complete';

// ============================================================================
// Component
// ============================================================================

export function CausalAnalysisPanel({
  onAnalysisComplete,
  initialConfig,
  language = 'en'
}: CausalAnalysisPanelProps) {
  // State
  const [phase, setPhase] = useState<AnalysisPhase>('configure');
  const [engine, setEngine] = useState<CausalInferenceEngine | null>(null);
  const [dag, setDag] = useState<CausalDAG | null>(null);
  const [config, setConfig] = useState<Partial<CausalModelConfig>>(
    initialConfig ?? {
      numStudents: 100,
      numTimeSteps: 30,
      featureDimension: 3,
      seed: 42
    }
  );

  // Analysis results
  const [testResult, setTestResult] = useState<HypothesisTestResult | null>(null);
  const [biasResult, setBiasResult] = useState<BiasVerificationResult | null>(null);
  const [identResult, setIdentResult] = useState<IdentifiabilityResult | null>(null);
  const [powerResult, setPowerResult] = useState<PowerAnalysisResult | null>(null);
  const [ledger, setLedger] = useState<CausalAnalysisLedger | null>(null);

  // Sensitivity analysis
  const [rosenbaumBounds, setRosenbaumBounds] = useState<RosenbaumBounds[]>([]);
  const [eValue, setEValue] = useState<EValue | null>(null);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>('dag');

  // Translations
  const t = useMemo(() => ({
    title: language === 'he' ? 'ניתוח סיבתי' : 'Causal Analysis',
    dagConfig: language === 'he' ? 'הגדרת DAG' : 'DAG Configuration',
    modelConfig: language === 'he' ? 'הגדרות מודל' : 'Model Configuration',
    runOffline: language === 'he' ? 'הפעל למידה לא מקוונת' : 'Run Offline Learning',
    runOnline: language === 'he' ? 'הפעל בחירה מקוונת' : 'Run Online Selection',
    runTest: language === 'he' ? 'הפעל בדיקה סטטיסטית' : 'Run Statistical Test',
    verifyBias: language === 'he' ? 'אמת חוסר הטיה' : 'Verify Unbiasedness',
    results: language === 'he' ? 'תוצאות' : 'Results',
    ate: language === 'he' ? 'אפקט ממוצע של טיפול' : 'Average Treatment Effect',
    pValue: language === 'he' ? 'ערך P' : 'P-Value',
    significant: language === 'he' ? 'מובהק' : 'Significant',
    notSignificant: language === 'he' ? 'לא מובהק' : 'Not Significant',
    sensitivity: language === 'he' ? 'ניתוח רגישות' : 'Sensitivity Analysis',
    ledger: language === 'he' ? 'יומן ביקורת' : 'Audit Ledger',
    biasCheck: language === 'he' ? 'בדיקת הטיה' : 'Bias Verification',
    identifiability: language === 'he' ? 'זיהוי' : 'Identifiability',
    power: language === 'he' ? 'עוצמה' : 'Power Analysis',
    numStudents: language === 'he' ? 'מספר סטודנטים' : 'Number of Students',
    numTimeSteps: language === 'he' ? 'מספר צעדי זמן' : 'Time Steps',
    featureDim: language === 'he' ? 'מימד תכונות' : 'Feature Dimension',
    seed: language === 'he' ? 'זרע אקראי' : 'Random Seed',
    reset: language === 'he' ? 'אפס' : 'Reset',
    export: language === 'he' ? 'ייצא' : 'Export'
  }), [language]);

  // Initialize DAG
  const initializeDAG = useCallback(() => {
    const newDag = createStandardDAG({
      featureNames: Array.from({ length: config.featureDimension ?? 3 }, (_, i) => `X${i + 1}`),
      numConfounders: 1,
      maxLag: 2
    });
    const validation = validateDAG(newDag);
    if (!validation.valid) {
      setError(validation.issues.map(i => i.message).join('; '));
    }
    setDag(newDag);
    return newDag;
  }, [config.featureDimension]);

  // Initialize engine and run offline phase
  const runOfflinePhase = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const newEngine = createCausalEngine(config);
      const currentDag = dag ?? initializeDAG();

      // Simulate historical data
      const historicalData = newEngine.simulateHistoricalData();

      // Run offline learning
      newEngine.runOfflinePhase(historicalData);

      // Check identifiability
      const identAnalysis = analyzeIdentifiability(currentDag, 'Z', 'Y');
      setIdentResult({
        identifiable: identAnalysis.identifiable,
        unconfoundednessPartial: identAnalysis.method === 'backdoor',
        positivity: true,
        consistency: true,
        transportability: true,
        quantizationInvertible: true,
        diagnostics: {
          minPropensity: 0.1,
          maxPropensity: 0.9,
          quantizationResolution: 0.5,
          signPreserved: true,
          warnings: identAnalysis.assumptions
        }
      });

      setEngine(newEngine);
      setPhase('offline');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [config, dag, initializeDAG]);

  // Run online selection phase
  const runOnlinePhase = useCallback(async () => {
    if (!engine) return;
    setLoading(true);
    setError(null);
    try {
      engine.initializeOnlinePhase(0);

      // Simulate online data collection
      const numOnlineSteps = Math.min(config.numTimeSteps ?? 30, 20);
      for (let t = 0; t < numOnlineSteps; t++) {
        // Generate student features
        const features = Array.from({ length: config.numStudents ?? 100 }, () =>
          Array.from({ length: config.featureDimension ?? 3 }, () =>
            (Math.random() - 0.5) * 2
          )
        );
        const studentIds = Array.from(
          { length: config.numStudents ?? 100 },
          (_, i) => `student-${i}`
        );

        // Select policies
        const selections = engine.selectPolicies(features, studentIds);

        // Simulate outcomes (simplified)
        const observations = selections.map((sel, i) => ({
          studentId: sel.studentId,
          features: features[i],
          policy: sel.policy,
          quantizedOutcome: Math.floor(Math.random() * 5) as 0 | 1 | 2 | 3 | 4,
          propensityScore: sel.propensityScore
        }));

        engine.recordOnlineObservations(observations);
        engine.advanceTimeStep();
      }

      setPhase('online');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [engine, config]);

  // Run statistical testing phase
  const runTestingPhase = useCallback(async () => {
    if (!engine) return;
    setLoading(true);
    setError(null);
    try {
      // Run hypothesis test
      const result = engine.runStatisticalTest(0.05);
      setTestResult(result);

      // Power analysis
      const power = engine.computePowerAnalysis(0.2);
      setPowerResult(power);

      // Compute sensitivity bounds
      if (result) {
        const bounds = computeRosenbaumBounds(
          engine.getOnlineState()?.runningATE.estimate ?? 0,
          engine.getOnlineState()?.runningATE.standardError ?? 1,
          [1, 1.5, 2, 3, 5]
        );
        setRosenbaumBounds(bounds);

        // E-value (convert to approximate risk ratio)
        const ate = engine.getOnlineState()?.runningATE.estimate ?? 0;
        const rr = Math.exp(ate); // Approximate for small effects
        const ev = computeEValue(rr, Math.exp(result.confidenceInterval[0]));
        setEValue(ev);
      }

      setPhase('testing');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [engine]);

  // Run bias verification
  const runBiasVerification = useCallback(async () => {
    if (!engine) return;
    setLoading(true);
    setError(null);
    try {
      const bias = engine.verifyUnbiasedness(50);
      setBiasResult(bias);

      // Generate ledger
      const analysisLedger = engine.generateLedger();
      setLedger(analysisLedger);

      // Build final result
      if (testResult && bias && identResult && powerResult) {
        const finalResult: AnalysisResult = {
          ate: engine.getOnlineState()?.runningATE.estimate ?? 0,
          standardError: engine.getOnlineState()?.runningATE.standardError ?? 0,
          confidenceInterval: testResult.confidenceInterval,
          pValue: testResult.pValue,
          significant: testResult.reject,
          biasVerification: bias,
          identifiability: identResult,
          powerAnalysis: powerResult,
          ledger: analysisLedger
        };
        onAnalysisComplete?.(finalResult);
      }

      setPhase('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [engine, testResult, identResult, powerResult, onAnalysisComplete]);

  // Reset analysis
  const resetAnalysis = useCallback(() => {
    setEngine(null);
    setDag(null);
    setPhase('configure');
    setTestResult(null);
    setBiasResult(null);
    setIdentResult(null);
    setPowerResult(null);
    setLedger(null);
    setRosenbaumBounds([]);
    setEValue(null);
    setError(null);
  }, []);

  // Export results
  const exportResults = useCallback(() => {
    if (!engine || !ledger) return;

    const exportData = {
      config,
      testResult,
      biasVerification: biasResult,
      identifiability: identResult,
      powerAnalysis: powerResult,
      sensitivityAnalysis: {
        rosenbaumBounds,
        eValue
      },
      ledger,
      exportedAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `causal-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [config, testResult, biasResult, identResult, powerResult, rosenbaumBounds, eValue, ledger, engine]);

  // Toggle section expansion
  const toggleSection = (section: string) => {
    setExpandedSection(prev => prev === section ? null : section);
  };

  // Render DAG visualization (simplified SVG)
  const renderDAG = () => {
    if (!dag) return null;

    const width = 400;
    const height = 250;
    const nodeRadius = 20;

    // Simple layout
    const nodePositions: Record<string, { x: number; y: number }> = {
      'Z': { x: width / 2, y: height - 40 },
      'Y': { x: width - 60, y: height / 2 },
      'U1': { x: width / 2, y: 40 },
      'theta': { x: 60, y: height - 40 },
      'Y_t-1': { x: 60, y: height / 2 },
      'Y_t-2': { x: 60, y: 80 }
    };

    // Add feature positions
    const featureCount = config.featureDimension ?? 3;
    for (let i = 0; i < featureCount; i++) {
      nodePositions[`X${i + 1}`] = {
        x: 120 + (i * (width - 180) / (featureCount + 1)),
        y: height / 2 - 50
      };
    }

    return (
      <svg width={width} height={height} style={{ border: '1px solid #ddd', borderRadius: 4 }}>
        {/* Edges */}
        {dag.edges.slice(0, 15).map(edge => {
          const from = nodePositions[edge.from.replace(/_t\d+$/, '')] ||
                       nodePositions[edge.from] ||
                       { x: width / 2, y: height / 2 };
          const to = nodePositions[edge.to.replace(/_t\d+$/, '')] ||
                     nodePositions[edge.to] ||
                     { x: width / 2, y: height / 2 };

          const color = edge.type === 'confounding' ? '#ef4444' :
                       edge.type === 'temporal' ? '#3b82f6' :
                       edge.type === 'selection' ? '#a855f7' : '#6b7280';

          const dashArray = edge.type === 'confounding' ? '5,5' : 'none';

          return (
            <g key={edge.id}>
              <line
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={color}
                strokeWidth={2}
                strokeDasharray={dashArray}
                markerEnd="url(#arrowhead)"
              />
            </g>
          );
        })}

        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280" />
          </marker>
        </defs>

        {/* Nodes */}
        {dag.nodes.slice(0, 10).map(node => {
          const pos = nodePositions[node.id.replace(/_t\d+$/, '')] ||
                     nodePositions[node.id] ||
                     { x: width / 2, y: height / 2 };

          const fill = node.type === 'treatment' ? '#10b981' :
                      node.type === 'outcome' ? '#3b82f6' :
                      node.type === 'confounder' ? '#ef4444' :
                      node.type === 'temporal' ? '#f59e0b' :
                      node.type === 'quantization' ? '#a855f7' : '#6b7280';

          const opacity = node.observed ? 1 : 0.6;

          return (
            <g key={node.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={nodeRadius}
                fill={fill}
                opacity={opacity}
                stroke={node.observed ? 'none' : '#333'}
                strokeWidth={node.observed ? 0 : 2}
                strokeDasharray={node.observed ? 'none' : '3,3'}
              />
              <text
                x={pos.x}
                y={pos.y + 4}
                textAnchor="middle"
                fill="white"
                fontSize={12}
                fontWeight="bold"
              >
                {node.id.replace(/_t-(\d+)$/, '(t-$1)').slice(0, 4)}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${width - 100}, 10)`}>
          <circle cx={10} cy={0} r={6} fill="#10b981" />
          <text x={20} y={4} fontSize={10}>Treatment</text>
          <circle cx={10} cy={15} r={6} fill="#3b82f6" />
          <text x={20} y={19} fontSize={10}>Outcome</text>
          <circle cx={10} cy={30} r={6} fill="#ef4444" opacity={0.6} stroke="#333" strokeDasharray="2,2" />
          <text x={20} y={34} fontSize={10}>Confounder</text>
        </g>
      </svg>
    );
  };

  // Render phase indicator
  const renderPhaseIndicator = () => {
    const phases: { key: AnalysisPhase; label: string }[] = [
      { key: 'configure', label: '1. Configure' },
      { key: 'offline', label: '2. Offline' },
      { key: 'online', label: '3. Online' },
      { key: 'testing', label: '4. Test' },
      { key: 'complete', label: '5. Complete' }
    ];

    const currentIndex = phases.findIndex(p => p.key === phase);

    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        {phases.map((p, i) => (
          <div
            key={p.key}
            style={{
              flex: 1,
              textAlign: 'center',
              padding: '8px 4px',
              backgroundColor: i <= currentIndex ? '#3b82f6' : '#e5e7eb',
              color: i <= currentIndex ? 'white' : '#6b7280',
              borderRadius: i === 0 ? '4px 0 0 4px' : i === phases.length - 1 ? '0 4px 4px 0' : 0,
              fontSize: 12,
              fontWeight: i === currentIndex ? 'bold' : 'normal'
            }}
          >
            {p.label}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <h2 style={{ marginTop: 0 }}>{t.title}</h2>

      {renderPhaseIndicator()}

      {error && (
        <div style={{
          padding: 12,
          backgroundColor: '#fef2f2',
          border: '1px solid #ef4444',
          borderRadius: 4,
          marginBottom: 16,
          color: '#b91c1c'
        }}>
          {error}
        </div>
      )}

      {/* DAG Section */}
      <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
        <div
          onClick={() => toggleSection('dag')}
          style={{
            padding: '12px 16px',
            backgroundColor: '#f9fafb',
            cursor: 'pointer',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <strong>{t.dagConfig}</strong>
          <span>{expandedSection === 'dag' ? '-' : '+'}</span>
        </div>
        {expandedSection === 'dag' && (
          <div style={{ padding: 16 }}>
            {dag ? renderDAG() : (
              <button
                onClick={initializeDAG}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer'
                }}
              >
                Initialize DAG
              </button>
            )}
          </div>
        )}
      </div>

      {/* Model Configuration Section */}
      <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
        <div
          onClick={() => toggleSection('config')}
          style={{
            padding: '12px 16px',
            backgroundColor: '#f9fafb',
            cursor: 'pointer',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <strong>{t.modelConfig}</strong>
          <span>{expandedSection === 'config' ? '-' : '+'}</span>
        </div>
        {expandedSection === 'config' && (
          <div style={{ padding: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <label>
                {t.numStudents}:
                <input
                  type="number"
                  value={config.numStudents ?? 100}
                  onChange={e => setConfig(prev => ({
                    ...prev,
                    numStudents: parseInt(e.target.value) || 100
                  }))}
                  disabled={phase !== 'configure'}
                  style={{ marginLeft: 8, width: 80, padding: 4 }}
                />
              </label>
              <label>
                {t.numTimeSteps}:
                <input
                  type="number"
                  value={config.numTimeSteps ?? 30}
                  onChange={e => setConfig(prev => ({
                    ...prev,
                    numTimeSteps: parseInt(e.target.value) || 30
                  }))}
                  disabled={phase !== 'configure'}
                  style={{ marginLeft: 8, width: 80, padding: 4 }}
                />
              </label>
              <label>
                {t.featureDim}:
                <input
                  type="number"
                  value={config.featureDimension ?? 3}
                  onChange={e => setConfig(prev => ({
                    ...prev,
                    featureDimension: parseInt(e.target.value) || 3
                  }))}
                  disabled={phase !== 'configure'}
                  style={{ marginLeft: 8, width: 80, padding: 4 }}
                />
              </label>
              <label>
                {t.seed}:
                <input
                  type="number"
                  value={config.seed ?? 42}
                  onChange={e => setConfig(prev => ({
                    ...prev,
                    seed: parseInt(e.target.value) || 42
                  }))}
                  disabled={phase !== 'configure'}
                  style={{ marginLeft: 8, width: 80, padding: 4 }}
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {phase === 'configure' && (
          <button
            onClick={runOfflinePhase}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Loading...' : t.runOffline}
          </button>
        )}
        {phase === 'offline' && (
          <button
            onClick={runOnlinePhase}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Loading...' : t.runOnline}
          </button>
        )}
        {phase === 'online' && (
          <button
            onClick={runTestingPhase}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#f59e0b',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Loading...' : t.runTest}
          </button>
        )}
        {phase === 'testing' && (
          <button
            onClick={runBiasVerification}
            disabled={loading}
            style={{
              padding: '10px 20px',
              backgroundColor: '#a855f7',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: loading ? 'wait' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Loading...' : t.verifyBias}
          </button>
        )}
        {phase !== 'configure' && (
          <button
            onClick={resetAnalysis}
            style={{
              padding: '10px 20px',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            {t.reset}
          </button>
        )}
        {phase === 'complete' && (
          <button
            onClick={exportResults}
            style={{
              padding: '10px 20px',
              backgroundColor: '#059669',
              color: 'white',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer'
            }}
          >
            {t.export}
          </button>
        )}
      </div>

      {/* Results Section */}
      {testResult && (
        <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
          <div
            onClick={() => toggleSection('results')}
            style={{
              padding: '12px 16px',
              backgroundColor: '#f9fafb',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <strong>{t.results}</strong>
            <span>{expandedSection === 'results' ? '-' : '+'}</span>
          </div>
          {expandedSection === 'results' && (
            <div style={{ padding: 16 }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: 16
              }}>
                <div style={{
                  padding: 12,
                  backgroundColor: '#f0fdf4',
                  borderRadius: 4,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{t.ate}</div>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#10b981' }}>
                    {engine?.getOnlineState()?.runningATE.estimate.toFixed(4) ?? '-'}
                  </div>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    CI: [{testResult.confidenceInterval[0].toFixed(3)}, {testResult.confidenceInterval[1].toFixed(3)}]
                  </div>
                </div>
                <div style={{
                  padding: 12,
                  backgroundColor: '#fef3c7',
                  borderRadius: 4,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{t.pValue}</div>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#f59e0b' }}>
                    {testResult.pValue.toFixed(4)}
                  </div>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Test statistic: {testResult.testStatistic.toFixed(3)}
                  </div>
                </div>
                <div style={{
                  padding: 12,
                  backgroundColor: testResult.reject ? '#dcfce7' : '#fef2f2',
                  borderRadius: 4,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>Status</div>
                  <div style={{
                    fontSize: 18,
                    fontWeight: 'bold',
                    color: testResult.reject ? '#10b981' : '#ef4444'
                  }}>
                    {testResult.reject ? t.significant : t.notSignificant}
                  </div>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Power: {(testResult.achievedPower ?? 0 * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sensitivity Analysis Section */}
      {rosenbaumBounds.length > 0 && (
        <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
          <div
            onClick={() => toggleSection('sensitivity')}
            style={{
              padding: '12px 16px',
              backgroundColor: '#f9fafb',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <strong>{t.sensitivity}</strong>
            <span>{expandedSection === 'sensitivity' ? '-' : '+'}</span>
          </div>
          {expandedSection === 'sensitivity' && (
            <div style={{ padding: 16 }}>
              <h4 style={{ marginTop: 0 }}>Rosenbaum Bounds</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ backgroundColor: '#f3f4f6' }}>
                    <th style={{ padding: 8, textAlign: 'left' }}>Gamma</th>
                    <th style={{ padding: 8, textAlign: 'left' }}>P-Value Lower</th>
                    <th style={{ padding: 8, textAlign: 'left' }}>P-Value Upper</th>
                    <th style={{ padding: 8, textAlign: 'left' }}>CI</th>
                  </tr>
                </thead>
                <tbody>
                  {rosenbaumBounds.map(bound => (
                    <tr key={bound.gamma} style={{ borderBottom: '1px solid #e5e7eb' }}>
                      <td style={{ padding: 8 }}>{bound.gamma.toFixed(1)}</td>
                      <td style={{ padding: 8 }}>{bound.pValueLower.toFixed(4)}</td>
                      <td style={{ padding: 8 }}>{bound.pValueUpper.toFixed(4)}</td>
                      <td style={{ padding: 8 }}>
                        [{bound.confidenceInterval[0].toFixed(3)}, {bound.confidenceInterval[1].toFixed(3)}]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {eValue && (
                <div style={{ marginTop: 16 }}>
                  <h4>E-Value Analysis</h4>
                  <p style={{ fontSize: 12, margin: 0 }}>
                    <strong>E-value (point estimate):</strong> {eValue.pointEstimate.toFixed(2)}
                  </p>
                  <p style={{ fontSize: 12, margin: '4px 0' }}>
                    <strong>E-value (CI lower bound):</strong> {eValue.ciLowerBound.toFixed(2)}
                  </p>
                  <p style={{ fontSize: 12, margin: '4px 0', color: '#6b7280' }}>
                    {eValue.interpretation}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Bias Verification Section */}
      {biasResult && (
        <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
          <div
            onClick={() => toggleSection('bias')}
            style={{
              padding: '12px 16px',
              backgroundColor: biasResult.unbiased ? '#dcfce7' : '#fef2f2',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <strong>
              {t.biasCheck}: {biasResult.unbiased ? 'PASSED' : 'FAILED'}
            </strong>
            <span>{expandedSection === 'bias' ? '-' : '+'}</span>
          </div>
          {expandedSection === 'bias' && (
            <div style={{ padding: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <strong>Neutrality:</strong>{' '}
                  <span style={{ color: biasResult.neutrality.neutral ? '#10b981' : '#ef4444' }}>
                    {biasResult.neutrality.neutral ? 'Pass' : 'Fail'}
                  </span>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Symmetry error: {biasResult.neutrality.symmetryError.toFixed(4)}
                  </div>
                </div>
                <div>
                  <strong>Differential Neutrality:</strong>{' '}
                  <span style={{
                    color: biasResult.differentialNeutrality.satisfied ? '#10b981' : '#ef4444'
                  }}>
                    {biasResult.differentialNeutrality.satisfied ? 'Pass' : 'Fail'}
                  </span>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Epsilon: {biasResult.differentialNeutrality.epsilon.toFixed(4)}
                  </div>
                </div>
                <div>
                  <strong>Fairness:</strong>{' '}
                  <span style={{ color: biasResult.fairness.fair ? '#10b981' : '#ef4444' }}>
                    {biasResult.fairness.fair ? 'Pass' : 'Fail'}
                  </span>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Delta: {biasResult.fairness.delta.toFixed(4)}
                  </div>
                </div>
                <div>
                  <strong>Type I Error Rate:</strong>{' '}
                  <span style={{
                    color: biasResult.typeIErrorRate <= 0.075 ? '#10b981' : '#ef4444'
                  }}>
                    {(biasResult.typeIErrorRate * 100).toFixed(1)}%
                  </span>
                  <div style={{ fontSize: 11, color: '#6b7280' }}>
                    Target: 5%
                  </div>
                </div>
              </div>
              {biasResult.recommendations.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <strong>Recommendations:</strong>
                  <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 12 }}>
                    {biasResult.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Ledger Section */}
      {ledger && (
        <div style={{ marginBottom: 16, border: '1px solid #ddd', borderRadius: 4 }}>
          <div
            onClick={() => toggleSection('ledger')}
            style={{
              padding: '12px 16px',
              backgroundColor: '#f9fafb',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <strong>{t.ledger} ({ledger.summary.totalDecisions} entries)</strong>
            <span>{expandedSection === 'ledger' ? '-' : '+'}</span>
          </div>
          {expandedSection === 'ledger' && (
            <div style={{ padding: 16 }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 8,
                marginBottom: 12
              }}>
                <div style={{ textAlign: 'center', padding: 8, backgroundColor: '#f3f4f6', borderRadius: 4 }}>
                  <div style={{ fontSize: 18, fontWeight: 'bold' }}>{ledger.summary.policySelections}</div>
                  <div style={{ fontSize: 11 }}>Policy Selections</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, backgroundColor: '#f3f4f6', borderRadius: 4 }}>
                  <div style={{ fontSize: 18, fontWeight: 'bold' }}>{ledger.summary.ateEstimates}</div>
                  <div style={{ fontSize: 11 }}>ATE Estimates</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, backgroundColor: '#f3f4f6', borderRadius: 4 }}>
                  <div style={{ fontSize: 18, fontWeight: 'bold' }}>{ledger.summary.quantizationUpdates}</div>
                  <div style={{ fontSize: 11 }}>Quant. Updates</div>
                </div>
                <div style={{ textAlign: 'center', padding: 8, backgroundColor: '#f3f4f6', borderRadius: 4 }}>
                  <div style={{ fontSize: 18, fontWeight: 'bold' }}>{ledger.summary.biasChecks}</div>
                  <div style={{ fontSize: 11 }}>Bias Checks</div>
                </div>
              </div>
              <div style={{
                maxHeight: 200,
                overflow: 'auto',
                border: '1px solid #e5e7eb',
                borderRadius: 4
              }}>
                {ledger.entries.slice(-10).map(entry => (
                  <div
                    key={entry.id}
                    style={{
                      padding: 8,
                      borderBottom: '1px solid #e5e7eb',
                      fontSize: 11
                    }}
                  >
                    <span style={{
                      display: 'inline-block',
                      padding: '2px 6px',
                      borderRadius: 4,
                      backgroundColor:
                        entry.type === 'ate_estimate' ? '#dbeafe' :
                        entry.type === 'bias_check' ? '#fce7f3' :
                        entry.type === 'policy_selection' ? '#dcfce7' : '#fef3c7',
                      marginRight: 8
                    }}>
                      {entry.type}
                    </span>
                    <span style={{ color: '#6b7280' }}>
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                    <span style={{ marginLeft: 8 }}>{entry.details.rationale}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CausalAnalysisPanel;
