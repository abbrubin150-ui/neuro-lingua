/**
 * ParamOptimizer — a small first-order optimizer over named tensors (number[] / number[][]),
 * used for parameters that live outside ProNeuralLM's built-in optimizer state: the transformer
 * stack (TransformerLM) and LayerNorm gains/biases (AdvancedNeuralLM).
 *
 * Supported rules (all with in-place updates):
 *   adam      m ← β₁m + (1−β₁)g,  v ← β₂v + (1−β₂)g²,  θ ← θ − η·m̂/(√v̂ + ε)
 *   momentum  m ← μm + g,                               θ ← θ − η·m
 *   lion      c ← β₁m + (1−β₁)g,  θ ← θ − η·sign(c),     m ← β₂m + (1−β₂)g
 */
export type ParamOptimizerKind = 'adam' | 'momentum' | 'lion';

type Tensor = number[] | number[][];

interface SlotState {
  m: Tensor;
  v: Tensor;
}

function zerosLike(t: Tensor): Tensor {
  return Array.isArray(t[0])
    ? (t as number[][]).map((r) => new Array(r.length).fill(0))
    : new Array((t as number[]).length).fill(0);
}

function shapeMatches(a: Tensor, b: Tensor): boolean {
  if (a.length !== b.length) return false;
  if (Array.isArray(a[0]) !== Array.isArray(b[0])) return false;
  if (Array.isArray(a[0])) {
    return (a as number[][]).every((row, i) => row.length === (b as number[][])[i].length);
  }
  return true;
}

/**
 * Minimal first-order optimizer over named tensors (in-place updates).
 * State is created lazily and reset automatically when a tensor changes shape
 * (e.g. after Cerebro neuron injection).
 */
export class ParamOptimizer {
  private slots = new Map<string, SlotState>();
  private t = 0;

  constructor(
    private kind: ParamOptimizerKind,
    private momentum = 0.9,
    private beta1 = 0.9,
    private beta2 = 0.999,
    private eps = 1e-8
  ) {}

  setKind(kind: ParamOptimizerKind, momentum: number): void {
    if (kind !== this.kind) this.reset();
    this.kind = kind;
    this.momentum = momentum;
  }

  reset(): void {
    this.slots.clear();
    this.t = 0;
  }

  /** Advance the shared step counter (call once per optimisation step). */
  step(): void {
    this.t++;
  }

  update(name: string, param: Tensor, grad: Tensor, lr: number, rows?: Iterable<number>): void {
    let slot = this.slots.get(name);
    if (!slot || !shapeMatches(slot.m, param)) {
      slot = { m: zerosLike(param), v: zerosLike(param) };
      this.slots.set(name, slot);
    }
    if (Array.isArray(param[0])) {
      const P = param as number[][];
      const G = grad as number[][];
      const M = slot.m as number[][];
      const Vs = slot.v as number[][];
      const rowIter = rows ?? P.keys();
      for (const i of rowIter) this.updateRow(P[i], G[i], M[i], Vs[i], lr);
    } else {
      this.updateRow(
        param as number[],
        grad as number[],
        slot.m as number[],
        slot.v as number[],
        lr
      );
    }
  }

  private updateRow(p: number[], g: number[], m: number[], v: number[], lr: number): void {
    const t = Math.max(1, this.t);
    if (this.kind === 'adam') {
      const b1 = this.beta1;
      const b2 = this.beta2;
      const c1 = 1 - Math.pow(b1, t);
      const c2 = 1 - Math.pow(b2, t);
      for (let j = 0; j < p.length; j++) {
        m[j] = b1 * m[j] + (1 - b1) * g[j];
        v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j];
        p[j] -= (lr * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + this.eps);
      }
    } else if (this.kind === 'lion') {
      const b1 = 0.9;
      const b2 = 0.99;
      for (let j = 0; j < p.length; j++) {
        const c = b1 * m[j] + (1 - b1) * g[j];
        p[j] -= lr * Math.sign(c);
        m[j] = b2 * m[j] + (1 - b2) * g[j];
      }
    } else {
      for (let j = 0; j < p.length; j++) {
        m[j] = this.momentum * m[j] + g[j];
        p[j] -= lr * m[j];
      }
    }
  }
}

/** Global L2 norm of a collection of tensors. */
export function globalNorm(tensors: Tensor[]): number {
  let ss = 0;
  for (const t of tensors) {
    if (Array.isArray(t[0])) {
      for (const row of t as number[][]) for (const v of row) ss += v * v;
    } else {
      for (const v of t as number[]) ss += v * v;
    }
  }
  return Math.sqrt(ss);
}

export function scaleTensors(tensors: Tensor[], factor: number): void {
  for (const t of tensors) {
    if (Array.isArray(t[0])) {
      for (const row of t as number[][]) for (let j = 0; j < row.length; j++) row[j] *= factor;
    } else {
      const v = t as number[];
      for (let j = 0; j < v.length; j++) v[j] *= factor;
    }
  }
}
