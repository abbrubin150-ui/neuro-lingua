import {
  applyUpdateVector,
  computeDiagonalHessian,
  createSecondOrderState,
  dampedNewtonStep,
  flattenGradients,
  flattenStructure,
  quasiNewtonStep,
  type SecondOrderConfig,
  type SecondOrderState
} from '../training/optimizer';

export type Optimizer = 'momentum' | 'adam' | 'newton' | 'bfgs';
export type TokenizerMode = 'unicode' | 'ascii' | 'custom';

export type TokenizerConfig = {
  mode: TokenizerMode;
  pattern?: string;
};

const DEFAULT_TOKENIZER_CONFIG: TokenizerConfig = { mode: 'unicode' };

export const MODEL_VERSION = '3.2.4';
export const MODEL_COMPACT_VERSION = MODEL_VERSION.replace(/\./g, '');
export const MODEL_STORAGE_KEY = `neuro-lingua-pro-v${MODEL_COMPACT_VERSION}`;
export const MODEL_EXPORT_FILENAME = `neuro-lingua-v${MODEL_COMPACT_VERSION}.json`;

type Rng = {
  next(): number;
  getState(): number;
};

function makeRng(seed: number, state?: number): Rng {
  const baseSeed = seed >>> 0;
  let t = (state !== undefined ? state : baseSeed) >>> 0;
  return {
    next() {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    },
    getState() {
      return t >>> 0;
    }
  };
}

export function clamp(x: number, lo: number, hi: number) {
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

export class ProNeuralLM {
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
  private rng: Rng;
  private rngSeed: number;
  private rngState: number;

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
  private aEmbedding: { m: number[][]; v: number[][] } = {
    m: [] as number[][],
    v: [] as number[][]
  };
  private aWHidden: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aWOutput: { m: number[][]; v: number[][] } = { m: [] as number[][], v: [] as number[][] };
  private aBHidden: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };
  private aBOutput: { m: number[]; v: number[] } = { m: [] as number[], v: [] as number[] };

  private trainingHistory: { loss: number; accuracy: number; timestamp: number }[] = [];

  private tokenizerConfig: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG;
  private lastUpdatedAt: number | null = null;

  private secondOrderConfig: SecondOrderConfig = {
    damping: 1e-4,
    epsilon: 1e-9,
    maxHistory: 7
  };
  private secondOrderState: SecondOrderState = createSecondOrderState();

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
    seed = 1337,
    tokenizerConfig: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG
  ) {
    this.vocab = vocab;
    this.hiddenSize = hiddenSize;
    this.learningRate = learningRate;
    this.contextSize = contextSize;
    this.optimizer = optimizer;
    this.momentum = momentum;
    this.dropout = clamp(dropout, 0, 0.5);
    this.rngSeed = seed >>> 0;
    this.rng = makeRng(this.rngSeed);
    this.rngState = this.rng.getState();
    this.setTokenizerConfig(tokenizerConfig);

    this.wordToIdx = new Map(vocab.map((w, i) => [w, i]));
    this.idxToWord = new Map(vocab.map((w, i) => [i, w]));
    this.initializeParameters();
  }

  private static normalizeTokenizerConfig(config?: TokenizerConfig): TokenizerConfig {
    if (!config) return { ...DEFAULT_TOKENIZER_CONFIG };
    if (config.mode === 'ascii') return { mode: 'ascii' };
    if (config.mode === 'custom') {
      if (config.pattern && config.pattern.length > 0) {
        return { mode: 'custom', pattern: config.pattern };
      }
      return { ...DEFAULT_TOKENIZER_CONFIG };
    }
    return { mode: 'unicode' };
  }

  private static tokenizerRegexFromConfig(config: TokenizerConfig): RegExp {
    try {
      if (config.mode === 'ascii') {
        return /[^a-z0-9\s'-]/g;
      }
      if (config.mode === 'custom' && config.pattern) {
        try {
          return new RegExp(config.pattern, 'gu');
        } catch {
          return new RegExp(config.pattern, 'g');
        }
      }
      return /[^\p{L}\d\s'-]/gu;
    } catch (err) {
      console.warn('Failed to compile tokenizer regex. Falling back to Unicode pattern.', err);
      return /[^\p{L}\d\s'-]/gu;
    }
  }

  static tokenizeText(text: string, config?: TokenizerConfig): string[] {
    const normalized = ProNeuralLM.normalizeTokenizerConfig(config);
    const regex = ProNeuralLM.tokenizerRegexFromConfig(normalized);
    return text
      .toLowerCase()
      .replace(regex, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 0);
  }

  setTokenizerConfig(config: TokenizerConfig) {
    const normalized = ProNeuralLM.normalizeTokenizerConfig(config);
    this.tokenizerConfig = normalized;
  }

  getTokenizerConfig(): TokenizerConfig {
    return { ...this.tokenizerConfig };
  }

  exportTokenizerConfig() {
    return this.getTokenizerConfig();
  }

  importTokenizerConfig(config: TokenizerConfig) {
    this.setTokenizerConfig(config);
  }

  getLastUpdatedAt() {
    if (this.lastUpdatedAt) return this.lastUpdatedAt;
    const last = this.trainingHistory[this.trainingHistory.length - 1];
    return last?.timestamp ?? null;
  }

  private nextRandom() {
    const value = this.rng.next();
    this.rngState = this.rng.getState();
    return value;
  }

  private randn(scale = 0.1) {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.nextRandom();
    while (v === 0) v = this.nextRandom();
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

    this.resetSecondOrderState();
  }

  private resetSecondOrderState() {
    const maxHistory = this.secondOrderConfig.maxHistory ?? 7;
    this.secondOrderState = createSecondOrderState(maxHistory);
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

  private forward(inputs: number[], train = false) {
    const emb = this.averageVectors(inputs.map((i) => this.embedding[i]));

    const hPreCore = this.matrixVectorMul(this.wHidden, emb);
    const preAct = hPreCore.map((v, i) => v + this.bHidden[i]);
    let h = preAct.map((v) => this.relu(v));

    let dropMask: number[] | null = null;
    if (train && this.dropout > 0) {
      const scale = 1 / (1 - this.dropout);
      dropMask = new Array(h.length)
        .fill(0)
        .map(() => (this.nextRandom() > this.dropout ? scale : 0));
      h = h.map((v, i) => v * dropMask![i]);
    }

    const logits = this.matrixVectorMulTranspose(this.wOutput, h).map(
      (v, i) => v + this.bOutput[i]
    );
    const probs = softmax(logits);
    return { h, logits, probs, avgEmb: emb, dropMask, preAct };
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

  private applyAdamMatrix(
    W: number[][],
    G: number[][],
    M: number[][],
    V: number[][],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
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

  private applyAdamVector(
    b: number[],
    g: number[],
    M: number[],
    V: number[],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
    for (let i = 0; i < b.length; i++) {
      const gi = g[i];
      M[i] = b1 * M[i] + (1 - b1) * gi;
      V[i] = b2 * V[i] + (1 - b2) * (gi * gi);
      const mhat = M[i] / b1t;
      const vhat = V[i] / b2t;
      b[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }

  private applyAdamRow(
    W: number[][],
    rowIdx: number,
    gRow: number[],
    M: number[][],
    V: number[][],
    b1t: number,
    b2t: number
  ) {
    const lr = this.learningRate;
    const b1 = this.adamBeta1;
    const b2 = this.adamBeta2;
    const eps = this.adamEps;
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
    cache: {
      h: number[];
      probs: number[];
      avgEmb: number[];
      dropMask: number[] | null;
      preAct: number[];
    }
  ) {
    const { h, probs, avgEmb, dropMask, preAct } = cache;
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
      s *= preAct[i] > 0 ? 1 : 0;
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

    const clipRow = (row: number[], m = 5) => {
      const n = l2norm(row);
      if (n > m) {
        const s = m / n;
        for (let k = 0; k < row.length; k++) row[k] *= s;
      }
    };
    for (let i = 0; i < dWout.length; i++) clipRow(dWout[i]);
    for (let i = 0; i < dWh.length; i++) clipRow(dWh[i]);

    const wHiddenSnap = this.wHidden.map((r) => r.slice());

    const scale = 1 / Math.max(1, inputs.length);
    const dEmb = new Array(H).fill(0);
    for (let i = 0; i < H; i++) {
      let s = 0;
      for (let k = 0; k < H; k++) s += wHiddenSnap[k][i] * dHidden[k];
      dEmb[i] = s * scale;
    }
    let adamB1t: number | null = null;
    let adamB2t: number | null = null;

    if (this.optimizer === 'newton' || this.optimizer === 'bfgs') {
      const paramStruct = {
        wOutput: this.wOutput,
        bOutput: this.bOutput,
        wHidden: this.wHidden,
        bHidden: this.bHidden
      };
      const gradStruct = {
        wOutput: dWout,
        bOutput: dBout,
        wHidden: dWh,
        bHidden: dBh
      };
      const { vector: paramVec, meta } = flattenStructure(paramStruct);
      const gradVec = flattenGradients(gradStruct, meta);
      if (this.optimizer === 'newton') {
        const diag = computeDiagonalHessian(gradVec, this.secondOrderConfig.epsilon ?? 1e-9);
        const step = dampedNewtonStep(gradVec, diag, {
          ...this.secondOrderConfig,
          learningRate: this.learningRate
        });
        applyUpdateVector(paramStruct, step, meta);
      } else {
        const step = quasiNewtonStep(paramVec, gradVec, this.secondOrderState, {
          ...this.secondOrderConfig,
          learningRate: this.learningRate
        });
        applyUpdateVector(paramStruct, step, meta);
      }
    } else if (this.optimizer === 'adam') {
      this.adamT += 1;
      adamB1t = 1 - Math.pow(this.adamBeta1, this.adamT);
      adamB2t = 1 - Math.pow(this.adamBeta2, this.adamT);
      this.applyAdamMatrix(this.wOutput, dWout, this.aWOutput.m, this.aWOutput.v, adamB1t, adamB2t);
      this.applyAdamVector(this.bOutput, dBout, this.aBOutput.m, this.aBOutput.v, adamB1t, adamB2t);
      this.applyAdamMatrix(this.wHidden, dWh, this.aWHidden.m, this.aWHidden.v, adamB1t, adamB2t);
      this.applyAdamVector(this.bHidden, dBh, this.aBHidden.m, this.aBHidden.v, adamB1t, adamB2t);
    } else {
      this.applyMomentumMatrix(this.wOutput, dWout, this.mWOutput);
      this.applyMomentumVector(this.bOutput, dBout, this.mBOutput);
      this.applyMomentumMatrix(this.wHidden, dWh, this.mWHidden);
      this.applyMomentumVector(this.bHidden, dBh, this.mBHidden);
    }

    if (this.optimizer === 'adam' && adamB1t !== null && adamB2t !== null) {
      for (const idx of inputs) {
        this.applyAdamRow(
          this.embedding,
          idx,
          dEmb,
          this.aEmbedding.m,
          this.aEmbedding.v,
          adamB1t,
          adamB2t
        );
      }
    } else if (this.optimizer === 'momentum') {
      for (const idx of inputs) {
        for (let i = 0; i < H; i++) {
          this.mEmbedding[idx][i] =
            this.momentum * this.mEmbedding[idx][i] + this.learningRate * dEmb[i];
          this.embedding[idx][i] -= this.mEmbedding[idx][i];
        }
      }
    } else if (this.optimizer === 'newton' || this.optimizer === 'bfgs') {
      for (const idx of inputs) {
        for (let i = 0; i < H; i++) {
          this.embedding[idx][i] -= this.learningRate * dEmb[i];
        }
      }
    }
  }

  private tokenize(text: string): string[] {
    return ProNeuralLM.tokenizeText(text, this.tokenizerConfig);
  }

  private toIndex(tok: string) {
    return this.wordToIdx.get(tok) ?? this.wordToIdx.get(this.unk)!;
  }

  private createTrainingSequences(text: string): [number[], number][] {
    const bosArr = Array(this.contextSize).fill(this.bos);
    const toks = [...bosArr, ...this.tokenize(text), this.eos];
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
      const j = Math.floor(this.nextRandom() * (i + 1));
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
        const cache = this.forward(ctx, true);
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

    const payload = {
      loss: totalLoss / Math.max(1, count),
      accuracy: correct / Math.max(1, count),
      history: this.trainingHistory
    } as const;
    this.lastUpdatedAt = Date.now();
    return payload;
  }

  private sampleFromLogits(logits: number[], temperature = 1.0, topK = 0, topP = 0): number {
    const T = clamp(temperature, 0.05, 5);
    const scaled = logits.map((z) => z / T);
    let p = softmax(scaled);

    if (topP && topP > 0 && topP < 1) {
      const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
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

    let r = Math.max(0, Math.min(0.999999, this.nextRandom()));
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
      const { logits } = this.forward(window, false);
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

  getVocabSignature() {
    return this.vocab.join('\u241F');
  }

  toJSON() {
    this.rngState = this.rng.getState();
    return {
      version: MODEL_VERSION,
      vocab: this.vocab,
      hiddenSize: this.hiddenSize,
      learningRate: this.learningRate,
      contextSize: this.contextSize,
      optimizer: this.optimizer,
      momentum: this.momentum,
      dropout: this.dropout,
      rngSeed: this.rngSeed,
      rngState: this.rngState,
      embedding: this.embedding,
      wHidden: this.wHidden,
      wOutput: this.wOutput,
      bHidden: this.bHidden,
      bOutput: this.bOutput,
      adamT: this.adamT,
      mEmbedding: this.mEmbedding,
      mWHidden: this.mWHidden,
      mWOutput: this.mWOutput,
      mBHidden: this.mBHidden,
      mBOutput: this.mBOutput,
      aEmbedding: this.aEmbedding,
      aWHidden: this.aWHidden,
      aWOutput: this.aWOutput,
      aBHidden: this.aBHidden,
      aBOutput: this.aBOutput,
      wordToIdx: Array.from(this.wordToIdx.entries()),
      idxToWord: Array.from(this.idxToWord.entries()),
      trainingHistory: this.trainingHistory,
      tokenizerConfig: this.tokenizerConfig,
      lastUpdatedAt: this.lastUpdatedAt
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
      const hasSeed = typeof d.rngSeed === 'number';
      const hasState = typeof d.rngState === 'number';
      const seed = hasSeed ? (d.rngSeed as number) : undefined;
      const state = hasState ? (d.rngState as number) : undefined;
      const m = new ProNeuralLM(
        d.vocab,
        d.hiddenSize,
        d.learningRate ?? 0.08,
        d.contextSize ?? 3,
        (d.optimizer as Optimizer) ?? 'momentum',
        d.momentum ?? 0.9,
        d.dropout ?? 0,
        seed,
        ProNeuralLM.normalizeTokenizerConfig(d.tokenizerConfig)
      );
      m.embedding = d.embedding;
      m.wHidden = d.wHidden;
      m.wOutput = d.wOutput;
      m.bHidden = d.bHidden;
      m.bOutput = d.bOutput;
      if (typeof d.adamT === 'number') m.adamT = d.adamT;
      if (d.mEmbedding) m.mEmbedding = d.mEmbedding;
      if (d.mWHidden) m.mWHidden = d.mWHidden;
      if (d.mWOutput) m.mWOutput = d.mWOutput;
      if (d.mBHidden) m.mBHidden = d.mBHidden;
      if (d.mBOutput) m.mBOutput = d.mBOutput;
      if (d.aEmbedding) m.aEmbedding = { m: d.aEmbedding.m, v: d.aEmbedding.v };
      if (d.aWHidden) m.aWHidden = { m: d.aWHidden.m, v: d.aWHidden.v };
      if (d.aWOutput) m.aWOutput = { m: d.aWOutput.m, v: d.aWOutput.v };
      if (d.aBHidden) m.aBHidden = { m: d.aBHidden.m, v: d.aBHidden.v };
      if (d.aBOutput) m.aBOutput = { m: d.aBOutput.m, v: d.aBOutput.v };
      m.wordToIdx = new Map(d.wordToIdx);
      m.idxToWord = new Map(d.idxToWord);
      m.trainingHistory = d.trainingHistory || [];
      if (typeof d.lastUpdatedAt === 'number') {
        m.lastUpdatedAt = d.lastUpdatedAt;
      } else if (m.trainingHistory.length > 0) {
        m.lastUpdatedAt = m.trainingHistory[m.trainingHistory.length - 1]?.timestamp ?? null;
      }
      if (hasSeed) {
        m.rngSeed = seed! >>> 0;
      }
      m.rng = makeRng(m.rngSeed, hasState ? state! >>> 0 : undefined);
      m.rngState = m.rng.getState();
      m.resetSecondOrderState();
      return m;
    } catch (e) {
      console.warn('Failed to load model', e);
      return null;
    }
  }
}
