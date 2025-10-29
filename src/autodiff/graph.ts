/**
 * Minimal dynamic computation graph with reverse-mode automatic differentiation.
 * The goal is to replace hand-derived gradients in experimental layers with a
 * transparent graph structure that records operations and exposes `backward()`.
 */

type BackwardFn = (grad: number) => void;

export class Variable {
  public grad = 0;
  public children: Array<{ node: Variable; backward: BackwardFn }> = [];

  constructor(
    public value: number,
    public name?: string
  ) {}

  add(other: Variable | number): Variable {
    const rhs = ensureVariable(other);
    const out = new Variable(this.value + rhs.value, `(${this.label()}+${rhs.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += grad;
      }
    });
    out.children.push({
      node: rhs,
      backward: (grad) => {
        rhs.grad += grad;
      }
    });
    return out;
  }

  mul(other: Variable | number): Variable {
    const rhs = ensureVariable(other);
    const out = new Variable(this.value * rhs.value, `(${this.label()}*${rhs.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += rhs.value * grad;
      }
    });
    out.children.push({
      node: rhs,
      backward: (grad) => {
        rhs.grad += this.value * grad;
      }
    });
    return out;
  }

  sub(other: Variable | number): Variable {
    return this.add(ensureVariable(other).neg());
  }

  div(other: Variable | number): Variable {
    const rhs = ensureVariable(other);
    const out = new Variable(this.value / rhs.value, `(${this.label()}/${rhs.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += grad / rhs.value;
      }
    });
    out.children.push({
      node: rhs,
      backward: (grad) => {
        rhs.grad -= (this.value / (rhs.value * rhs.value)) * grad;
      }
    });
    return out;
  }

  pow(exponent: number): Variable {
    const out = new Variable(this.value ** exponent, `${this.label()}^${exponent}`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += exponent * this.value ** (exponent - 1) * grad;
      }
    });
    return out;
  }

  tanh(): Variable {
    const value = Math.tanh(this.value);
    const out = new Variable(value, `tanh(${this.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += (1 - value * value) * grad;
      }
    });
    return out;
  }

  exp(): Variable {
    const value = Math.exp(this.value);
    const out = new Variable(value, `exp(${this.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += value * grad;
      }
    });
    return out;
  }

  log(): Variable {
    const out = new Variable(Math.log(this.value), `log(${this.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad += grad / this.value;
      }
    });
    return out;
  }

  neg(): Variable {
    const out = new Variable(-this.value, `(-${this.label()})`);
    out.children.push({
      node: this,
      backward: (grad) => {
        this.grad -= grad;
      }
    });
    return out;
  }

  backward(gradient = 1): void {
    const topo = topologicalSort(this);
    this.grad += gradient;
    for (const node of topo) {
      for (const child of node.children) {
        child.backward(node.grad);
      }
    }
  }

  zeroGrad(): void {
    const topo = topologicalSort(this);
    for (const node of topo) {
      node.grad = 0;
    }
  }

  private label(): string {
    return this.name ?? this.value.toFixed(3);
  }
}

function ensureVariable(value: Variable | number): Variable {
  if (value instanceof Variable) return value;
  return new Variable(value);
}

function topologicalSort(root: Variable): Variable[] {
  const visited = new Set<Variable>();
  const order: Variable[] = [];
  const stack: Variable[] = [root];

  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || visited.has(node)) continue;
    visited.add(node);
    order.unshift(node);
    for (const child of node.children) {
      stack.push(child.node);
    }
  }
  return order;
}

export function meanSquaredError(predictions: number[], targets: number[]): Variable {
  if (predictions.length !== targets.length) {
    throw new Error('MSE expects predictions and targets with matching length.');
  }
  let loss = new Variable(0, 'mse');
  for (let i = 0; i < predictions.length; i++) {
    const diff = new Variable(predictions[i], `pred_${i}`).sub(targets[i]);
    loss = loss.add(diff.pow(2));
  }
  return loss.div(predictions.length);
}
