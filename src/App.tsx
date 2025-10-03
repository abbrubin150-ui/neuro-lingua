import React, { useEffect, useRef, useState } from 'react';

type Msg = { type: 'system' | 'user' | 'assistant'; content: string; timestamp?: number };

type Optimizer = 'momentum' | 'adam';

function makeRng(seed: number) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(x: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, x));
}

function argTopK(arr: number[], k: number): number[] {
  const idx = arr.map((_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx.slice(0, Math.max(1, k));
}

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const ex = logits.map((x) => Math.exp(x - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((x) => x / (s || 1));
}

function vectorDot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function l2norm(a: number[]) {
  return Math.sqrt(vectorDot(a, a));
}

function clipVectorInPlace(a: number[], maxNorm: number) {
  const n = l2norm(a);
  if (n > maxNorm && n > 0) {
    const s = maxNorm / n;
    for (let i = 0; i < a.length; i++) a[i] *= s;
  }
}

class ProNeuralLM {
  private vocab: string[];
  private wordToIdx: Map<string, number>;
  private idxToWord: Map<number, string>;

  private embedding: number[][] = [];
  private wHidden: number[][] = [];
  private wOutput: number[][] = [];
  private bHidden: number[] = [];
  private bOutput: number[] = [];

  private hiddenSize: number;
  private learningRate: number;
  private contextSize: number;
  private rng: () => number;

  private dropout: number;
  private optimizer: Optimizer;

  private mEmbedding: number[][] = [];
  private mWHidden: number[][] = [];
  private mWOutput: number[][] = [];
  private mBHidden: number[] = [];
  private mBOutput: number[] = [];
  private momentum: number;

  private adamBeta1 = 0.9;
  private adamBeta2 = 0.999;
  private adamEps = 1e-8;
  private adamT = 0;
  private aEmbedding: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aWHidden: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aWOutput: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aBHidden: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };
  private aBOutput: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };

  private trainingHistory: { loss: number; accuracy: number; timestamp: number }[] = [];

  private bos = '<BOS>';
  private eos = '<EOS>';
  private unk = '<UNK>';

  constructor(
    vocab: string[],
    hiddenSize = 64,
    learningRate = 0.08,
    contextSize = 3,
    optimizer: Optimizer = 'momentum',
    momentum = 0.9,
    dropout = 0.0,
    seed = 1337
  ) {
    this.vocab = vocab;
    this.hiddenSize = hiddenSize;
    this.learningRate = learningRate;
    this.contextSize = contextSize;
    this.optimizer = optimizer;
    this.momentum = momentum;
    this.dropout = clamp(dropout, 0, 0.5);
    this.rng = makeRng(seed);

    this.wordToIdx = new Map(vocab.map((w, i) => [w, i]));
    this.idxToWord = new Map(vocab.map((w, i) => [i, w]));
    this.initializeParameters();
  }

  private randn(scale = 0.1) {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.rng();
    while (v === 0) v = this.rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
  }

  private randomArray(size: number, scale: number): number[] {
    const a = new Array(size);
    for (let i = 0; i < size; i++) a[i] = this.randn(scale);
    return a;
  }

  private zerosVec(n: number) {
    return new Array(n).fill(0);
  }

  private zerosMat(r: number, c: number) {
    return new Array(r).fill(0).map(() => new Array(c).fill(0));
  }

  private initializeParameters() {
    const V = this.vocab.length;
    const H = this.hiddenSize;

    this.embedding = new Array(V);
    for (let i = 0; i < V; i++) this.embedding[i] = this.randomArray(H, 0.05);

    this.wHidden = new Array(H);
    for (let i = 0; i < H; i++) this.wHidden[i] = this.randomArray(H, 0.05);

    this.wOutput = new Array(H);
    for (let i = 0; i < H; i++) this.wOutput[i] = this.randomArray(V, 0.05);

    this.bHidden = new Array(H).fill(0);
    this.bOutput = new Array(V).fill(0);

    this.mEmbedding = this.zerosMat(V, H);
    this.mWHidden = this.zerosMat(H, H);
    this.mWOutput = this.zerosMat(H, V);
    this.mBHidden = this.zerosVec(H);
    this.mBOutput = this.zerosVec(V);

    this.aEmbedding = { m: this.zerosMat(V, H), v: this.zerosMat(V, H) };
    this.aWHidden = { m: this.zerosMat(H, H), v: this.zerosMat(H, H) };
    this.aWOutput = { m: this.zerosMat(H, V), v: this.zerosMat(H, V) };
    this.aBHidden = { m: this.zerosVec(H), v: this.zerosVec(H) };
    this.aBOutput = { m: this.zerosVec(V), v: this.zerosVec(V) };
  }

  private relu(x: number) {
    return Math.max(0, x);
  }

  private matrixVectorMul(A: number[][], x: number[]): number[] {
    const y = new Array(A.length).fill(0);
    for (let i = 0; i < A.length; i++) {
      let s = 0;
      const row = A[i];
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      y[i] = s;
    }
    return y;
  }

  private matrixVectorMulTranspose(A: number[][], x: number[]): number[] {
    if (A.length === 0) return [];
    const cols = A[0].length;
    const y = new Array(cols).fill(0);
    for (let i = 0; i < A.length; i++) {
      const row = A[i];
      const xi = x[i];
      for (let j = 0; j < cols; j++) y[j] += row[j] * xi;
    }
    return y;
  }

  private averageVectors(vectors: number[][]): number[] {
    const H = this.hiddenSize;
    const y = new Array(H).fill(0);
    const n = vectors.length || 1;
    for (const v of vectors) for (let i = 0; i < H; i++) y[i] += v[i];
    for (let i = 0; i < H; i++) y[i] /= n;
    return y;
  }

  private forward(inputs: number[]) {
    const emb = this.averageVectors(inputs.map((i) => this.embedding[i]));

    const hPre = this.matrixVectorMul(this.wHidden, emb);
    let h = hPre.map((v, i) => this.relu(v + this.bHidden[i]));

    let dropMask: number[] | null = null;
    if (this.dropout > 0) {
      const scale = 1 / (1 - this.dropout);
      dropMask = new Array(h.length).fill(0).map(() => (this.rng() > this.dropout ? scale : 0));
      h = h.map((v, i) => v * (dropMask ? dropMask[i] : 1));
    }

    const logits = this.matrixVectorMulTranspose(this.wOutput, h).map((v, i) => v + this.bOutput[i]);
    const probs = softmax(logits);
    return { h, logits, probs, avgEmb: emb, dropMask };
  }

  private applyMomentumMatrix(W: number[][], G: number[][], M: number[][]) {
    const lr = this.learningRate;
    const mom = this.momentum;
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        M[i][j] = mom * M[i][j] + lr * G[i][j];
        W[i][j] -= M[i][j];
      }
    }
  }

  private applyMomentumVector(b: number[], g: number[], m: number[]) {
    const lr = this.learningRate;
    const mom = this.momentum;
    for (let i = 0; i < b.length; i++) {
      m[i] = mom * m[i] + lr * g[i];
      b[i] -= m[i];
    }
  }

  private applyAdamMatrix(W: number[][], G: number[][], M: number[][], V: number[][]) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    this.adamT++;
    const b1t = 1 - Math.pow(b1, this.adamT);
    const b2t = 1 - Math.pow(b2, this.adamT);
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) {
        const g = G[i][j];
        M[i][j] = b1 * M[i][j] + (1 - b1) * g;
        V[i][j] = b2 * V[i][j] + (1 - b2) * (g * g);
        const mhat = M[i][j] / b1t;
        const vhat = V[i][j] / b2t;
        W[i][j] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
      }
    }
  }

  private applyAdamVector(b: number[], g: number[], M: number[], V: number[]) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    this.adamT++;
    const b1t = 1 - Math.pow(b1, this.adamT);
    const b2t = 1 - Math.pow(b2, this.adamT);
    for (let i = 0; i < b.length; i++) {
      const gi = g[i];
      M[i] = b1 * M[i] + (1 - b1) * gi;
      V[i] = b2 * V[i] + (1 - b2) * (gi * gi);
      const mhat = M[i] / b1t;
      const vhat = V[i] / b2t;
      b[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }

  private applyAdamRow(W: number[][], rowIdx: number, gRow: number[], M: number[][], V: number[][]) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    this.adamT++;
    const b1t = 1 - Math.pow(b1, this.adamT);
    const b2t = 1 - Math.pow(b2, this.adamT);
    const row = W[rowIdx];
    const mRow = M[rowIdx];
    const vRow = V[rowIdx];
    for (let i = 0; i < row.length; i++) {
      const g = gRow[i];
      mRow[i] = b1 * mRow[i] + (1 - b1) * g;
      vRow[i] = b2 * vRow[i] + (1 - b2) * (g * g);
      const mhat = mRow[i] / b1t;
      const vhat = vRow[i] / b2t;
      row[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }

  private backward(
    inputs: number[],
    target: number,
    cache: { h: number[]; probs: number[]; avgEmb: number[]; dropMask: number[] | null }
  ) {
    const { h, probs, avgEmb, dropMask } = cache;
    const V = this.vocab.length;
    const H = this.hiddenSize;

    const dLogits = probs.map((p, i) => p - (i === target ? 1 : 0));

    const dWout = this.zerosMat(H, V);
    const dBout = this.zerosVec(V);
    for (let i = 0; i < H; i++) {
      const hi = h[i];
      for (let j = 0; j < V; j++) dWout[i][j] += hi * dLogits[j];
    }
    for (let j = 0; j < V; j++) dBout[j] += dLogits[j];

    const dHidden = new Array(H).fill(0);
    for (let i = 0; i < H; i++) {
      let s = 0;
      for (let j = 0; j < V; j++) s += this.wOutput[i][j] * dLogits[j];
      s *= h[i] > 0 ? 1 : 0;
      if (dropMask) s *= dropMask[i];
      dHidden[i] = s;
    }

    clipVectorInPlace(dHidden, 5);

    const dWh = this.zerosMat(H, H);
    const dBh = this.zerosVec(H);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < H; j++) dWh[i][j] += dHidden[i] * avgEmb[j];
      dBh[i] += dHidden[i];
    }

    const wHiddenSnap = this.wHidden.map((r) => r.slice());

    if (this.optimizer === 'adam') {
      this.applyAdamMatrix(this.wOutput, dWout, this.aWOutput.m, this.aWOutput.v);
      this.applyAdamVector(this.bOutput, dBout, this.aBOutput.m, this.aBOutput.v);
      this.applyAdamMatrix(this.wHidden, dWh, this.aWHidden.m, this.aWHidden.v);
      this.applyAdamVector(this.bHidden, dBh, this.aBHidden.m, this.aBHidden.v);
    } else {
      this.applyMomentumMatrix(this.wOutput, dWout, this.mWOutput);
      this.applyMomentumVector(this.bOutput, dBout, this.mBOutput);
      this.applyMomentumMatrix(this.wHidden, dWh, this.mWHidden);
      this.applyMomentumVector(this.bHidden, dBh, this.mBHidden);
    }

    const scale = 1 / Math.max(1, inputs.length);
    const dEmb = new Array(H).fill(0);
    for (let i = 0; i < H; i++) {
      let s = 0;
      for (let k = 0; k < H; k++) s += wHiddenSnap[k][i] * dHidden[k];
      dEmb[i] = s * scale;
    }
    for (const idx of inputs) {
      if (this.optimizer === 'adam') {
        this.applyAdamRow(this.embedding, idx, dEmb, this.aEmbedding.m, this.aEmbedding.v);
      } else {
        for (let i = 0; i < H; i++) {
          this.mEmbedding[idx][i] = this.momentum * this.mEmbedding[idx][i] + this.learningRate * dEmb[i];
          this.embedding[idx][i] -= this.mEmbedding[idx][i];
        }
      }
    }
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\u0590-\u05FF\w\s'-]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 0);
  }

  private toIndex(tok: string) {
    return this.wordToIdx.get(tok) ?? this.wordToIdx.get(this.unk)!;
  }

  private createTrainingSequences(text: string): [number[], number][] {
    const toks = [this.bos, this.bos, this.bos, ...this.tokenize(text), this.eos];
    const seqs: [number[], number][] = [];
    for (let i = this.contextSize; i < toks.length; i++) {
      const ctx = toks.slice(i - this.contextSize, i).map((t) => this.toIndex(t));
      const tgt = this.toIndex(toks[i]);
      seqs.push([ctx, tgt]);
    }
    return seqs;
  }

  private shuffleInPlace<T>(arr: T[]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  train(text: string, epochs = 10) {
    const seqs = this.createTrainingSequences(text);
    if (seqs.length === 0) return { loss: 0, accuracy: 0, history: this.trainingHistory };

    let totalLoss = 0;
    let correct = 0;
    let count = 0;

    for (let e = 0; e < epochs; e++) {
      this.shuffleInPlace(seqs);
      let epochLoss = 0;
      let epochCorrect = 0;
      for (const [ctx, tgt] of seqs) {
        const cache = this.forward(ctx);
        const loss = -Math.log(cache.probs[tgt] + 1e-8);
        epochLoss += loss;
        totalLoss += loss;
        const pred = cache.probs.indexOf(Math.max(...cache.probs));
        if (pred === tgt) {
          epochCorrect++;
          correct++;
        }
        count++;
        this.backward(ctx, tgt, cache);
      }
      const avgLoss = epochLoss / seqs.length;
      const accuracy = epochCorrect / seqs.length;
      this.trainingHistory.push({ loss: avgLoss, accuracy, timestamp: Date.now() });
    }

    return {
      loss: totalLoss / Math.max(1, count),
      accuracy: correct / Math.max(1, count),
      history: this.trainingHistory
    };
  }

  private sampleFromLogits(logits: number[], temperature = 1.0, topK = 0, topP = 0): number {
    const T = clamp(temperature, 0.05, 5);
    const scaled = logits.map((z) => z / T);
    let p = softmax(scaled);

    if (topP && topP > 0 && topP < 1) {
      const idx = p
        .map((v, i) => i)
        .sort((a, b) => p[b] - p[a]);
      let c = 0;
      const keep: number[] = [];
      for (const i of idx) {
        keep.push(i);
        c += p[i];
        if (c >= topP) break;
      }
      const set = new Set(keep);
      const sum = keep.reduce((acc, i) => acc + p[i], 0);
      p = p.map((v, i) => (set.has(i) ? v / (sum || 1) : 0));
    } else if (topK && topK > 0 && topK < p.length) {
      const keep = new Set(argTopK(p, topK));
      const sum = p.reduce((acc, v, i) => acc + (keep.has(i) ? v : 0), 0);
      p = p.map((v, i) => (keep.has(i) ? v / (sum || 1) : 0));
    }

    let r = Math.max(0, Math.min(0.999999, this.rng()));
    for (let i = 0; i < p.length; i++) {
      r -= p[i];
      if (r <= 0) return i;
    }
    return p.length - 1;
  }

  generate(seedText: string, maxLen = 25, temperature = 0.9, topK = 0, topP = 0): string {
    const seedToks = this.tokenize(seedText).map((t) => this.toIndex(t));
    const ctx: number[] = new Array(this.contextSize).fill(this.toIndex(this.bos));
    for (const t of seedToks) ctx.push(t);

    const out: string[] = [];
    while (out.length < maxLen) {
      const window = ctx.slice(-this.contextSize);
      const { logits } = this.forward(window);
      const idx = this.sampleFromLogits(logits, temperature, topK, topP);
      const tok = this.idxToWord.get(idx)!;
      if (tok === this.eos) break;
      out.push(tok);
      ctx.push(idx);
    }
    return out.join(' ');
  }

  getVocabSize() {
    return this.vocab.length;
  }

  getParametersCount() {
    const V = this.vocab.length;
    const H = this.hiddenSize;
    return V * H + H * H + H * V + H + V;
  }

  getTrainingHistory() {
    return this.trainingHistory;
  }

  toJSON() {
    return {
      version: '3.2',
      vocab: this.vocab,
      hiddenSize: this.hiddenSize,
      learningRate: this.learningRate,
      contextSize: this.contextSize,
      optimizer: this.optimizer,
      momentum: this.momentum,
      dropout: this.dropout,
      embedding: this.embedding,
      wHidden: this.wHidden,
      wOutput: this.wOutput,
      bHidden: this.bHidden,
      bOutput: this.bOutput,
      wordToIdx: Array.from(this.wordToIdx.entries()),
      idxToWord: Array.from(this.idxToWord.entries()),
      trainingHistory: this.trainingHistory
    } as const;
  }

  saveToLocalStorage(key: string) {
    try {
      localStorage.setItem(key, JSON.stringify(this.toJSON()));
    } catch (e) {
      console.warn('Failed to save model', e);
    }
  }

  static loadFromLocalStorage(key: string): ProNeuralLM | null {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const d = JSON.parse(raw);
      const m = new ProNeuralLM(
        d.vocab,
        d.hiddenSize,
        d.learningRate ?? 0.08,
        d.contextSize ?? 3,
        (d.optimizer as Optimizer) ?? 'momentum',
        d.momentum ?? 0.9,
        d.dropout ?? 0,
        1337
      );
      m.embedding = d.embedding;
      m.wHidden = d.wHidden;
      m.wOutput = d.wOutput;
      m.bHidden = d.bHidden;
      m.bOutput = d.bOutput;
      m.wordToIdx = new Map(d.wordToIdx);
      m.idxToWord = new Map(d.idxToWord);
      m.trainingHistory = d.trainingHistory || [];
      return m;
    } catch (e) {
      console.warn('Failed to load model', e);
      return null;
    }
  }
}

export default function NeuroLinguaDomesticaV32() {
  const [trainingText, setTrainingText] = useState(
    'מודל שפה עצבי מתקדם מתאמן בדפדפן. המודל לומד דפוסים מהטקסט, ויכול ליצור ניסוחים בעברית.'
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

  function runSelfTests() {
    try {
      const vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>', 'שלום', 'עולם'];
      const m = new ProNeuralLM(vocab, 16, 0.05, 3, 'momentum', 0.9, 0, 1234);
      const res = m.train('שלום עולם שלום', 1);
      console.assert(res.loss > 0, '[SelfTest] Loss should be positive');
      const out = m.generate('שלום', 5, 0.9, 0, 0.9);
      console.assert(typeof out === 'string', '[SelfTest] Generate returns string');
      console.log('[SelfTest] OK');
    } catch (e) {
      console.warn('[SelfTest] failed', e);
    }
  }

  useEffect(() => {
    runSelfTests();
    const saved = ProNeuralLM.loadFromLocalStorage('neuro-lingua-pro-v32');
    if (saved) {
      modelRef.current = saved;
      setInfo({ V: saved.getVocabSize(), P: saved.getParametersCount() });
      setTrainingHistory(saved.getTrainingHistory());
      setMessages((m) => [
        ...m,
        { type: 'system', content: '📀 מודל v3.2 נטען מהזיכרון המקומי', timestamp: Date.now() }
      ]);
    }
  }, []);

  function buildVocab(text: string): string[] {
    const toks = text
      .toLowerCase()
      .replace(/[^\u0590-\u05FF\w\s'-]/g, ' ')
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
        {
          type: 'system',
          content: '❌ צריך יותר טקסט לאימון (לפחות 8 מילים שונות).',
          timestamp: Date.now()
        }
      ]);
      setIsTraining(false);
      trainingRef.current.running = false;
      return;
    }

    const shouldReinit =
      !resume ||
      !modelRef.current ||
      modelRef.current.getVocabSize() !== vocab.length;
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
        {
          type: 'system',
          content: `🎯 התחלת אימון חדש עם ${vocab.length} מילים באוצר…`,
          timestamp: Date.now()
        }
      ]);
    } else {
      setMessages((m) => [
        ...m,
        { type: 'system', content: '🔁 ממשיך אימון על המודל הנוכחי…', timestamp: Date.now() }
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
      setInfo({
        V: modelRef.current!.getVocabSize(),
        P: modelRef.current!.getParametersCount()
      });
      setMessages((m) => [
        ...m,
        {
          type: 'system',
          content: `✅ אימון הושלם! דיוק ממוצע: ${(aggAcc / total * 100).toFixed(1)}%`,
          timestamp: Date.now()
        }
      ]);
    }

    setIsTraining(false);
    trainingRef.current.running = false;
  }

  function onStopTraining() {
    trainingRef.current.running = false;
    setIsTraining(false);
    setMessages((m) => [
      ...m,
      { type: 'system', content: '⏹️ האימון הופסק', timestamp: Date.now() }
    ]);
  }

  function onSave() {
    modelRef.current?.saveToLocalStorage('neuro-lingua-pro-v32');
    setMessages((m) => [
      ...m,
      { type: 'system', content: '💾 נשמר מקומית', timestamp: Date.now() }
    ]);
  }

  function onLoad() {
    const m = ProNeuralLM.loadFromLocalStorage('neuro-lingua-pro-v32');
    if (m) {
      modelRef.current = m;
      setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
      setTrainingHistory(m.getTrainingHistory());
      setMessages((s) => [
        ...s,
        { type: 'system', content: '📀 נטען מקומית', timestamp: Date.now() }
      ]);
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
    a.download = `neuro-lingua-v32.json`;
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
        localStorage.setItem('neuro-lingua-pro-v32', JSON.stringify(d));
        const m = ProNeuralLM.loadFromLocalStorage('neuro-lingua-pro-v32');
        if (m) {
          modelRef.current = m;
          setInfo({ V: m.getVocabSize(), P: m.getParametersCount() });
          setTrainingHistory(m.getTrainingHistory());
          setMessages((s) => [
            ...s,
            { type: 'system', content: '📥 יובא קובץ מודל', timestamp: Date.now() }
          ]);
        }
      } catch (error) {
        console.error(error);
        setMessages((s) => [
          ...s,
          { type: 'system', content: '❌ כשל בייבוא הקובץ', timestamp: Date.now() }
        ]);
      }
    };
    reader.readAsText(file);
  }

  function onReset() {
    modelRef.current = null;
    localStorage.removeItem('neuro-lingua-pro-v32');
    setInfo({ V: 0, P: 0 });
    setStats({ loss: 0, acc: 0, ppl: 0 });
    setTrainingHistory([]);
    setMessages((m) => [
      ...m,
      { type: 'system', content: '🔄 מודל אופס. אפשר לאמן מחדש.', timestamp: Date.now() }
    ]);
  }

  function onGenerate() {
    if (!modelRef.current || !input.trim()) {
      setMessages((m) => [
        ...m,
        { type: 'system', content: '❌ צריך לאמן את המודל קודם.', timestamp: Date.now() }
      ]);
      return;
    }
    setMessages((m) => [
      ...m,
      { type: 'user', content: input, timestamp: Date.now() }
    ]);
    const k = samplingMode === 'topk' ? topK : 0;
    const p = samplingMode === 'topp' ? topP : 0;
    const txt = modelRef.current.generate(input, 25, temperature, k, p);
    setMessages((m) => [
      ...m,
      { type: 'assistant', content: txt, timestamp: Date.now() }
    ]);
    setInput('');
  }

  function onExample() {
    setTrainingText(
      `למידת מכונה ובינה מלאכותית משנות את העולם הטכנולוגי. אלגוריתמים מתקדמים לומדים מדפוסים בנתונים ומשפרים את ביצועיהם עם הזמן.

מודלים עצביים מלאכותיים מחקים את פעולת המוח האנושי באמצעות שכבות של נוירונים דיגיטליים. טכנולוגיות אלו מאפשרות יצירת מערכות חכמות שמבינות שפה, מזהות תמונות, ומקבלות החלטות.

בעברית אפשר לפתח תכניות מתקדמות בתחום הבינה המלאכותית. קהילת המפתחים הישראלית תורמת רבות לקידום התחום במחקר ופיתוח.`
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
        <h4 style={{ color: '#a78bfa', margin: '0 0 12px 0' }}>📈 התקדמות אימון</h4>
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
            🧠 Neuro‑Lingua DOMESTICA — v3.2
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '1.05rem' }}>
            מודל שפה עצבי מתקדם עם Momentum/Adam, Dropout, גרפים בזמן אמת, ו‑context גמיש
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
                  onChange={(e) =>
                    setTemperature(clamp(parseFloat(e.target.value || '0.8'), 0.05, 5))
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
                      checked={samplingMode === 'off'}
                      onChange={() => setSamplingMode('off')}
                    />
                    off
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                      type="radio"
                      checked={samplingMode === 'topk'}
                      onChange={() => setSamplingMode('topk')}
                    />
                    top‑k
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                      type="radio"
                      checked={samplingMode === 'topp'}
                      onChange={() => setSamplingMode('topp')}
                    />
                    top‑p
                  </label>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: '#94a3b8', opacity: samplingMode === 'topk' ? 1 : 0.5 }}>
                  Top‑K
                </div>
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
                <div style={{ fontSize: 12, color: '#94a3b8', opacity: samplingMode === 'topp' ? 1 : 0.5 }}>
                  Top‑P
                </div>
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
                <input
                  type="checkbox"
                  checked={resume}
                  onChange={(e) => setResume(e.target.checked)}
                />
                המשך אימון (אם אפשר)
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
                  {isTraining ? '🔄 מאמן…' : '🚀 אמן מודל'}
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
                    ⏹️ עצור
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
                  🔄 אפס
                </button>
                <button
                  onClick={onExample}
                  style={{
                    padding: '12px 16px',
                    background: '#6366f1',
                    border: 'none',
                    borderRadius: 10,
                    color: 'white',
                    fontWeight: 600,
                    cursor: 'pointer'
                  }}
                >
                  📚 דוגמה
                </button>
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
                  <span>Epoch: {trainingRef.current.currentEpoch + 1}/{epochs}</span>
                  <span>
                    אורך טקסט: {trainingText.length} • מילים: {
                      trainingText.split(/\s+/).filter((w) => w.length > 0).length
                    }
                  </span>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
              <button
                onClick={onSave}
                style={{
                  padding: '8px 12px',
                  background: '#334155',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  color: '#e5e7eb'
                }}
              >
                💾 Save
              </button>
              <button
                onClick={onLoad}
                style={{
                  padding: '8px 12px',
                  background: '#334155',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  color: '#e5e7eb'
                }}
              >
                📀 Load
              </button>
              <button
                onClick={onExport}
                style={{
                  padding: '8px 12px',
                  background: '#334155',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  color: '#e5e7eb'
                }}
              >
                📤 Export
              </button>
              <label
                style={{
                  padding: '8px 12px',
                  background: '#334155',
                  border: '1px solid #475569',
                  borderRadius: 8,
                  color: '#e5e7eb',
                  cursor: 'pointer'
                }}
              >
                📥 Import
                <input type="file" accept="application/json" onChange={onImport} style={{ display: 'none' }} />
              </label>
            </div>
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
            <h3 style={{ color: '#34d399', marginTop: 0, marginBottom: 16 }}>📊 סטטיסטיקות</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Loss</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#ef4444' }}>{stats.loss.toFixed(4)}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>דיוק</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#10b981' }}>
                  {(stats.acc * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>Perplexity</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#f59e0b' }}>{stats.ppl.toFixed(2)}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>אוצר מילים</div>
                <div style={{ fontSize: 24, fontWeight: 800, color: '#a78bfa' }}>{info.V}</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: '#94a3b8' }}>פרמטרים</div>
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
            <h3 style={{ color: '#a78bfa', marginTop: 0, marginBottom: 16 }}>🎓 אימון</h3>
            <textarea
              value={trainingText}
              onChange={(e) => setTrainingText(e.target.value)}
              placeholder="הזן טקסט לאימון (עדיף 200+ מילים בעברית)..."
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
              <span>אורך טקסט: {trainingText.length} תווים</span>
              <span>
                מילים: {trainingText.split(/\s+/).filter((w) => w.length > 0).length}
              </span>
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
              <h3 style={{ color: '#60a5fa', margin: 0 }}>💬 צ'אט</h3>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                {messages.filter((m) => m.type === 'assistant').length} תשובות
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
                    {m.type === 'user' ? '👤 אתה' : m.type === 'assistant' ? '🤖 המודל' : '⚙️ מערכת'}
                    {m.timestamp && <span style={{ marginLeft: 8 }}>{new Date(m.timestamp).toLocaleTimeString('he-IL')}</span>}
                  </div>
                  {m.content}
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 12, alignItems: 'stretch' }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={modelRef.current ? 'הקלד הודעה למודל…' : 'אמן את המודל קודם…'}
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
                  background: modelRef.current
                    ? 'linear-gradient(90deg, #2563eb, #4f46e5)'
                    : '#475569',
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
              <strong>🎯 אימון אופטימלי</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                • 200–500 מילים • 20–50 epochs • LR: 0.05–0.1 • Context: 3–5
              </div>
            </div>
            <div>
              <strong>🎲 יצירת טקסט</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                • Temperature: 0.7–1.0 • בחרו מצב: top‑k או top‑p (מומלץ top‑p=0.85–0.95)
              </div>
            </div>
            <div>
              <strong>⚡ ביצועים</strong>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                • Momentum: 0.9 או Adam • Seed קבוע • Hidden: 32–128
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
