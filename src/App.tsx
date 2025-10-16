import React, { useEffect, useRef, useState } from 'react';

import {
  ProNeuralLM,
  type Optimizer,
  clamp,
  MODEL_VERSION,
  MODEL_STORAGE_KEY,
  MODEL_EXPORT_FILENAME
} from './lib/ProNeuralLM';

const LEGACY_STORAGE_KEYS = ['neuro-lingua-pro-v32'];

function loadLatestModel(): ProNeuralLM | null {
  const primary = ProNeuralLM.loadFromLocalStorage(MODEL_STORAGE_KEY);
  if (primary) return primary;

  for (const legacyKey of LEGACY_STORAGE_KEYS) {
    const legacy = ProNeuralLM.loadFromLocalStorage(legacyKey);
    if (legacy) {
      try {
        legacy.saveToLocalStorage(MODEL_STORAGE_KEY);
        localStorage.removeItem(legacyKey);
      } catch (err) {
        console.warn('Failed to migrate legacy model', err);
      }
      return legacy;
    }
  }

  return null;
}

type Msg = { type: 'system' | 'user' | 'assistant'; content: string; timestamp?: number };

export default function NeuroLinguaDomesticaV324() {
  const [trainingText, setTrainingText] = useState(
    'An advanced neural language model trains in the browser. The model learns patterns from the text and can draft fluent English responses.'
  );
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Msg[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stats, setStats] = useState({ loss: 0, acc: 0, ppl: 0 });
  const [info, setInfo] = useState({ V: 0, P: 0 });
  const [trainingHistory, setTrainingHistory] = useState<
    { loss: number; accuracy: number; timestamp: number }[]
  >([]);

  const [hiddenSize, setHiddenSize] = useState(64);
  const [epochs, setEpochs] = useState(20);
  const [lr, setLr] = useState(0.08);
  const [optimizer, setOptimizer] = useState<Optimizer>('momentum');
  const [momentum, setMomentum] = useState(0.9);
  const [dropout, setDropout] = useState(0.1);
  const [contextSize, setContextSize] = useState(3);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(20);
  const [topP, setTopP] = useState(0.9);
  const [samplingMode, setSamplingMode] = useState<'off' | 'topk' | 'topp'>('topp');
  const [seed, setSeed] = useState(1337);
  const [resume, setResume] = useState(true);

  const modelRef = useRef<ProNeuralLM | null>(null);
  const trainingRef = useRef({ running: false, currentEpoch: 0 });
  const importRef = useRef<HTMLInputElement>(null);

  function runSelfTests() {
    try {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'world'];
      const m = new ProNeuralLM(vocab, 16, 0.05, 3, 'momentum', 0.9, 0, 1234);
      const res = m.train('hello world hello', 1);
      console.assert(res.loss > 0 && res.accuracy >= 0 && res.accuracy <= 1, '[SelfTest] train metrics valid');
      const out = m.generate('hello', 5, 0.9, 0, 0.9);
      console.assert(typeof out === 'string' && out.length <= 5 * 10, '[SelfTest] generate output (bounded)');

      const m2: any = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0.3, 42);
      const fTrain = m2.forward([1, 1, 1], true);
      const fEval = m2.forward([1, 1, 1], false);
      console.assert(!!fTrain.dropMask && !fEval.dropMask, '[SelfTest] dropout train vs eval');

      const m3 = new ProNeuralLM(vocab, 8, 0.05, 3, 'adam', 0.9, 0.0, 7);
      const hLen0 = m3.getTrainingHistory().length;
      m3.train('hello world hello', 2);
      const hLen1 = m3.getTrainingHistory().length;
      console.assert(hLen1 === hLen0 + 2, '[SelfTest] history grows per epoch (adam)');

      const m4: any = new ProNeuralLM(vocab, 8, 0.05, 5, 'momentum', 0.9, 0.0, 9);
      const seqs = m4.createTrainingSequences('hello');
      console.assert(seqs.length > 0, '[SelfTest] sequences exist');
      console.assert(seqs[0][0].length === 5, '[SelfTest] context window = 5');

      const sK = m.generate('hello', 5, 0.9, 2, 0);
      const sP = m.generate('hello', 5, 0.9, 0, 0.8);
      console.assert(typeof sK === 'string' && typeof sP === 'string', '[SelfTest] sampling modes');

      const mA = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0.0, 111);
      const mB = new ProNeuralLM(vocab, 12, 0.05, 3, 'momentum', 0.9, 0.0, 111);
      const gA = mA.generate('hello', 6, 0.7, 0, 0.9);
      const gB = mB.generate('hello', 6, 0.7, 0, 0.9);
      console.assert(gA === gB, '[SelfTest] deterministic generation with same seed');

      const vocab2 = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'hello', 'friends'];
      const s1 = new ProNeuralLM(vocab).getVocabSignature();
      const s2 = new ProNeuralLM(vocab2).getVocabSignature();
      console.assert(s1 !== s2, '[SelfTest] vocab signature reflects vocab');

      const m5 = new ProNeuralLM(vocab, 8, 0.05, 3, 'momentum', 0.9, 0.0, 3);
      const r5 = m5.train('hello world', 1);
      const ppl = Math.exp(Math.max(1e-8, r5.loss));
      console.assert(!Number.isNaN(ppl), '[SelfTest] perplexity finite');

      console.log('[SelfTest] OK');
    } catch (e) {
      console.warn('[SelfTest] failed', e);
    }
  }

  useEffect(() => {
    runSelfTests();
    const saved = loadLatestModel();
    if (saved) {
      modelRef.current = saved;
      setInfo({ V: saved.getVocabSize(), P: saved.getParametersCount() });
      setTrainingHistory(saved.getTrainingHistory());
      setMessages((m) => [
        ...m,
        {
          type: 'system',
          content: `📀 Model v${MODEL_VERSION} loaded from local storage`,
          timestamp: Date.now()
        }
      ]);
    }
  }, []);

  function buildVocab(text: string): string[] {
    const toks = text
      .toLowerCase()
      .replace(/[^\p{L}\d\s'-]/gu, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 0);
    const uniq = Array.from(new Set(toks));
    const specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
    return Array.from(new Set([...specials, ...uniq]));
  }

  async function onTrain() {
    if (!trainingText.trim() || trainingRef.current.running) return;

    trainingRef.current = { running: true, currentEpoch: 0 };
    setIsTraining(true);
    setProgress(0);

    const vocab = buildVocab(trainingText);
    if (vocab.length < 8) {
      setMessages((m) => [
        ...m,
        { type: 'system', content: '❌ Need more training text (at least 8 unique tokens).', timestamp: Date.now() }
      ]);
      setIsTraining(false);
      trainingRef.current.running = false;
      return;
    }

    const sigNew = vocab.join('\u241F');
    const sigOld = modelRef.current?.getVocabSignature();
    const shouldReinit = !resume || !modelRef.current || sigNew !== sigOld;

    if (shouldReinit) {
      modelRef.current = new ProNeuralLM(
        vocab,
        hiddenSize,
        lr,
        clamp(contextSize, 2, 6),
        optimizer,
        momentum,
        clamp(dropout, 0, 0.5),
        seed
      );
      setMessages((m) => [
        ...m,
        { type: 'system', content: `🎯 Starting fresh training with ${vocab.length} vocabulary tokens…`, timestamp: Date.now() }
      ]);
    } else {
      setMessages((m) => [
        ...m,
        { type: 'system', content: '🔁 Continuing training on the current model…', timestamp: Date.now() }
      ]);
    }

    const total = Math.max(1, epochs);
    let aggLoss = 0;
    let aggAcc = 0;

    for (let e = 0; e < total; e++) {
      if (!trainingRef.current.running) break;
      trainingRef.current.currentEpoch = e;

      const res = modelRef.current!.train(trainingText, 1);
      aggLoss += res.loss;
      aggAcc += res.accuracy;
      const meanLoss = aggLoss / (e + 1);
      setStats({ loss: meanLoss, acc: aggAcc / (e + 1), ppl: Math.exp(Math.max(1e-8, meanLoss)) });
      setTrainingHistory(modelRef.current!.getTrainingHistory());
      setProgress(((e + 1) / total) * 100);
      await new Promise((r) => setTimeout(r, 16));
    }

    if (trainingRef.current.running) {
      setInfo({ V: modelRef.current!.getVocabSize(), P: modelRef.current!.getParametersCount() });
      setMessages((m) => [
        ...m,
        {
          type: 'system',
          content: `✅ Training complete! Average accuracy: ${(aggAcc / total * 100).toFixed(1)}%`,
          timestamp: Date.now()
        }
      ]);
      modelRef.current!.saveToLocalStorage(MODEL_STORAGE_KEY);
    }

    setIsTraining(false);
    trainingRef.current.running = false;
  }

  function onStopTraining() {
    trainingRef.current.running = false;
    setIsTraining(false);
    setMessages((m) => [...m, { type: 'system', content: '⏹️ Training stopped', timestamp: Date.now() }]);
  }

  function onSave() {
    modelRef.current?.saveToLocalStorage(MODEL_STORAGE_KEY);
    setMessages((m) => [...m, { type: 'system', content: '💾 Saved locally', timestamp: Date.now() }]);
  }

  function onLoad() {
    const m = loadLatestModel();
    if (m) {
      modelRef.current = m;
      setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
      setTrainingHistory(m.getTrainingHistory());
      setMessages((s) => [...s, { type: 'system', content: '📀 Loaded from local storage', timestamp: Date.now() }]);
    }
  }

  function onExport() {
    if (!modelRef.current) return;
    const blob = new Blob([JSON.stringify(modelRef.current.toJSON(), null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = MODEL_EXPORT_FILENAME;
    a.click();
    URL.revokeObjectURL(url);
  }

  function onImport(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const d = JSON.parse(String(reader.result));
        localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(d));
        const m = loadLatestModel();
        if (m) {
          modelRef.current = m;
          setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
          setTrainingHistory(m.getTrainingHistory());
          setMessages((s) => [...s, { type: 'system', content: '📥 Imported model file', timestamp: Date.now() }]);
        }
      } catch {
        setMessages((s) => [...s, { type: 'system', content: '❌ Failed to import model file', timestamp: Date.now() }]);
      }
    };
    reader.readAsText(file);
  }

  function onReset() {
    modelRef.current = null;
    localStorage.removeItem(MODEL_STORAGE_KEY);
    for (const legacyKey of LEGACY_STORAGE_KEYS) localStorage.removeItem(legacyKey);
    setInfo({ V: 0, P: 0 });
    setStats({ loss: 0, acc: 0, ppl: 0 });
    setTrainingHistory([]);
    setMessages((m) => [...m, { type: 'system', content: '🔄 Model reset. Ready to train again.', timestamp: Date.now() }]);
  }

  function onGenerate() {
    if (!modelRef.current || !input.trim()) {
      setMessages((m) => [...m, { type: 'system', content: '❌ Please train the model first.', timestamp: Date.now() }]);
      return;
    }
    setMessages((m) => [...m, { type: 'user', content: input, timestamp: Date.now() }]);
    const k = samplingMode === 'topk' ? topK : 0;
    const p = samplingMode === 'topp' ? topP : 0;
    const txt = modelRef.current.generate(input, 25, temperature, k, p);
    setMessages((m) => [...m, { type: 'assistant', content: txt, timestamp: Date.now() }]);
    setInput('');
  }

  function onExample() {
    setTrainingText(
      `Machine learning and artificial intelligence are reshaping the technology landscape. Advanced algorithms learn from data patterns and improve their performance over time.

Artificial neural models emulate the human brain using layers of digital neurons. These technologies enable smart systems that understand language, recognize images, and support decision making.

English-language research communities share open tools, papers, and tutorials so builders everywhere can prototype intelligent assistants.`
    );
  }

  const TrainingChart = () => {
    if (trainingHistory.length === 0) return null;
    const maxLoss = Math.max(...trainingHistory.map((h) => h.loss), 1e-6);
    return (
      <div
        style={{
          background: 'rgba(30,41,59,0.9)',
          borderRadius: 12,
          padding: 16,
          marginTop: 16,
          border: '1px solid #334155'
        }}
      >
        <h4 style={{ color: '#a78bfa', margin: '0 0 12px 0' }}>📈 Training Progress</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, height: 140 }}>
          <div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Loss</div>
            <div
              style={{
                display: 'flex',
                alignItems: 'end',
                gap: 2,
                height: 70,
                borderLeft: '1px solid #334155',
                borderBottom: '1px solid #334155',
                padding: '4px 0'
              }}
            >
              {trainingHistory.map((h, i) => (
                <div
                  key={i}
                  title={`Epoch ${i + 1}: ${h.loss.toFixed(4)}`}
                  style={{
                    flex: 1,
                    height: `${(h.loss / maxLoss) * 100}%`,
                    background: 'linear-gradient(to top, #ef4444, #dc2626)',
                    borderRadius: 2,
                    minHeight: 1
                  }}
                />
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Accuracy</div>
            <div
              style={{
                display: 'flex',
                alignItems: 'end',
                gap: 2,
                height: 70,
                borderLeft: '1px solid #334155',
                borderBottom: '1px solid #334155',
                padding: '4px 0'
              }}
            >
              {trainingHistory.map((h, i) => (
                <div
                  key={i}
                  title={`Epoch ${i + 1}: ${(h.accuracy * 100).toFixed(1)}%`}
                  style={{
                    flex: 1,
                    height: `${h.accuracy * 100}%`,
                    background: 'linear-gradient(to top, #10b981, #059669)',
                    borderRadius: 2,
                    minHeight: 1
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)',
        color: '#e2e8f0',
        padding: 20,
        fontFamily: "'Segoe UI', system-ui, sans-serif"
      }}
    >
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        <header style={{ textAlign: 'center', marginBottom: 32 }}>
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
            🧠 Neuro‑Lingua DOMESTICA — v{MODEL_VERSION}
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '1.05rem' }}>
            Advanced neural language model with Momentum/Adam, training-only dropout, real-time charts, and flexible context windows.
          </p>
        </header>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 20,
            marginBottom: 20
          }}
        >
          <div
            style={{
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 16,
              padding: 20
            }}
          >
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
                  type="number"
                  value={hiddenSize}
                  onChange={(e) => setHiddenSize(clamp(parseInt(e.target.value || '64'), 16, 256))}
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
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(clamp(parseInt(e.target.value || '20'), 1, 200))}
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
                  type="number"
                  step="0.01"
                  value={lr}
                  onChange={(e) => setLr(clamp(parseFloat(e.target.value || '0.08'), 0.001, 1))}
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
                  type="number"
                  value={contextSize}
                  onChange={(e) => setContextSize(clamp(parseInt(e.target.value || '3'), 2, 6))}
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
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as Optimizer)}
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
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Dropout</div>
                <input
                  type="number"
                  step="0.01"
                  value={dropout}
                  onChange={(e) => setDropout(clamp(parseFloat(e.target.value || '0.1'), 0, 0.5))}
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
                  type="number"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(clamp(parseFloat(e.target.value || '0.8'), 0.05, 5))}
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
                    <input type="radio" checked={samplingMode === 'off'} onChange={() => setSamplingMode('off')} /> off
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input type="radio" checked={samplingMode === 'topk'} onChange={() => setSamplingMode('topk')} /> top‑k
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input type="radio" checked={samplingMode === 'topp'} onChange={() => setSamplingMode('topp')} /> top‑p
                  </label>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8', opacity: samplingMode === 'topk' ? 1 : 0.5 }}>Top‑K</div>
                <input
                  type="number"
                  value={topK}
                  disabled={samplingMode !== 'topk'}
                  onChange={(e) => setTopK(clamp(parseInt(e.target.value || '20'), 0, 1000))}
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
                <div style={{ fontSize: 12, color: '#94a3b8', opacity: samplingMode === 'topp' ? 1 : 0.5 }}>Top‑P</div>
                <input
                  type="number"
                  step="0.01"
                  value={topP}
                  disabled={samplingMode !== 'topp'}
                  onChange={(e) => setTopP(clamp(parseFloat(e.target.value || '0.9'), 0, 0.99))}
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
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value || '1337'))}
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

            <div
              style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}
            >
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                <input type="checkbox" checked={resume} onChange={(e) => setResume(e.target.checked)} /> Resume training when possible
              </label>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  onClick={onTrain}
                  disabled={isTraining}
                  style={{
                    padding: '12px 20px',
                    background: isTraining
                      ? '#475569'
                      : 'linear-gradient(90deg, #7c3aed, #059669)',
                    border: 'none',
                    borderRadius: 10,
                    color: 'white',
                    fontWeight: 700,
                    cursor: isTraining ? 'not-allowed' : 'pointer',
                    minWidth: 120
                  }}
                >
                  {isTraining ? '🔄 Training…' : '🚀 Train model'}
                </button>
                {isTraining && (
                  <button
                    onClick={onStopTraining}
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
                  onClick={onReset}
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
                  onClick={onSave}
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
                  onClick={onLoad}
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
                  onClick={onExport}
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
                <input ref={importRef} type="file" accept="application/json" onChange={onImport} style={{ display: 'none' }} />
              </div>
            </div>

            {isTraining && (
              <div style={{ marginTop: 8 }}>
                <div style={{ width: '100%', height: 12, background: '#334155', borderRadius: 6, overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${progress}%`,
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
                >
                  <span>Training… {progress.toFixed(0)}%</span>
                  <span>Epoch: {trainingRef.current.currentEpoch + 1}/{epochs}</span>
                </div>
              </div>
            )}
          </div>

          <div
            style={{
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 16,
              padding: 20,
              display: 'flex',
              flexDirection: 'column'
            }}
          >
            <h3 style={{ color: '#34d399', marginTop: 0, marginBottom: 16 }}>📊 Advanced Statistics</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Loss</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#ef4444' }}>{stats.loss.toFixed(4)}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Accuracy</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#10b981' }}>{(stats.acc * 100).toFixed(1)}%</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Perplexity</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#f59e0b' }}>{stats.ppl.toFixed(2)}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Vocab Size</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#a78bfa' }}>{info.V}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Parameters</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#60a5fa' }}>{info.P.toLocaleString()}</div>
              </div>
            </div>
            <TrainingChart />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
          <div
            style={{
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 16,
              padding: 20,
              display: 'flex',
              flexDirection: 'column'
            }}
          >
            <h3 style={{ color: '#a78bfa', marginTop: 0, marginBottom: 16 }}>🎓 Training</h3>
            <textarea
              value={trainingText}
              onChange={(e) => setTrainingText(e.target.value)}
              placeholder="Enter training text (ideally 200+ English words)..."
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
              <span>Characters: {trainingText.length}</span>
              <span>Words: {trainingText.split(/\s+/).filter((w) => w.length > 0).length}</span>
            </div>
            <div style={{ marginTop: 12 }}>
              <button
                onClick={onExample}
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
                📚 Example
              </button>
            </div>
          </div>

          <div
            style={{
              background: 'rgba(30,41,59,0.9)',
              border: '1px solid #334155',
              borderRadius: 16,
              padding: 20,
              display: 'flex',
              flexDirection: 'column',
              height: 600
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 16
              }}
            >
              <h3 style={{ color: '#60a5fa', margin: 0 }}>💬 Chat Console</h3>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                {messages.filter((m) => m.type === 'assistant').length} replies
              </div>
            </div>

            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 12,
                marginBottom: 16
              }}
            >
              {messages.map((m, i) => (
                <div
                  key={i}
                  style={{
                    padding: '12px 16px',
                    borderRadius: 12,
                    background:
                      m.type === 'user'
                        ? 'linear-gradient(90deg, #3730a3, #5b21b6)'
                        : m.type === 'assistant'
                        ? 'linear-gradient(90deg, #1e293b, #334155)'
                        : 'linear-gradient(90deg, #065f46, #059669)',
                    border: '1px solid #475569',
                    wordWrap: 'break-word',
                    position: 'relative'
                  }}
                >
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                    {m.type === 'user' ? '👤 You' : m.type === 'assistant' ? '🤖 Model' : '⚙️ System'}
                    {m.timestamp && <span style={{ marginLeft: 8 }}>{new Date(m.timestamp).toLocaleTimeString('en-US')}</span>}
                  </div>
                  {m.content}
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    onGenerate();
                  }
                }}
                placeholder={modelRef.current ? 'Type a message for the model…' : 'Train the model first…'}
                style={{
                  flex: 1,
                  padding: '12px 16px',
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: 12,
                  color: '#e2e8f0',
                  fontSize: 14,
                  resize: 'none',
                  minHeight: 60,
                  fontFamily: 'inherit'
                }}
                rows={2}
              />
              <button
                onClick={onGenerate}
                disabled={!modelRef.current}
                style={{
                  padding: '12px 20px',
                  background: modelRef.current ? 'linear-gradient(90deg, #2563eb, #4f46e5)' : '#475569',
                  border: 'none',
                  borderRadius: 12,
                  color: 'white',
                  fontWeight: 700,
                  cursor: modelRef.current ? 'pointer' : 'not-allowed',
                  alignSelf: 'flex-end',
                  minWidth: 100
                }}
              >
                ✨ Generate
              </button>
            </div>
          </div>
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
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            <div>
              <strong>🎯 Training Tips</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>• 200–500 words • 20–50 epochs • LR: 0.05–0.1 • Context: 3–5</div>
            </div>
            <div>
              <strong>🎲 Text Generation</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>• Temperature: 0.7–1.0 • Choose top‑k or top‑p (top‑p ≈ 0.85–0.95)</div>
            </div>
            <div>
              <strong>⚡ Performance</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>• Momentum: 0.9 or Adam • Stable seed • Hidden: 32–128</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
