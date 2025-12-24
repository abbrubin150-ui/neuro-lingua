# Mathematical Guarantees for Neuro-Lingua

> **Version**: 4.3.0
> **Last Updated**: 2025-12-14
> **Purpose**: Comprehensive documentation of mathematical guarantees, proofs, and error bounds

---

## Table of Contents

1. [Introduction](#introduction)
2. [Numerical Stability](#numerical-stability)
3. [Optimizer Convergence](#optimizer-convergence)
4. [Information Theory](#information-theory)
5. [Sampling Methods](#sampling-methods)
6. [Attention Mechanisms](#attention-mechanisms)
7. [Low-Rank Approximations](#low-rank-approximations)
8. [Complexity Analysis](#complexity-analysis)

---

## Introduction

This document provides formal mathematical guarantees for the algorithms implemented in Neuro-Lingua. Each algorithm includes:

- **Assumptions**: Conditions under which the guarantees hold
- **Theorems**: Formal statements of correctness/convergence
- **Error Bounds**: Quantitative bounds on approximation error
- **Complexity**: Time and space complexity analysis

---

## Numerical Stability

### Kahan Summation

**Location**: `src/math/numerics.ts`

**Problem**: Naive summation of n floating-point numbers accumulates error O(nε).

**Theorem (Kahan, 1965)**:
For n floating-point numbers with unit roundoff ε, Kahan summation computes:

```
|fl(Σxᵢ) - Σxᵢ| ≤ 2ε × Σ|xᵢ| + O(nε²)
```

**Proof Sketch**:
1. Maintain compensation term c = (t - sum) - y
2. Algebraically c = 0, but captures rounding error
3. Next iteration adds compensation back
4. Error accumulates at O(ε) instead of O(nε)

**Usage**:
```typescript
import { kahanSum } from './math/numerics';
const result = kahanSum([1e-16, 1, 1e-16]); // Accurate
```

### Stable Softmax

**Location**: `src/lib/MathUtils.ts`

**Problem**: Naive softmax exp(xᵢ) / Σexp(xⱼ) overflows for large logits.

**Algorithm**: Log-sum-exp trick
```
softmax(x)ᵢ = exp(xᵢ - max(x)) / Σexp(xⱼ - max(x))
```

**Guarantee**: No overflow for logits in float32 range (-3.4e38, 3.4e38).

**Theorem**:
```
|softmax(x)ᵢ - true_value| ≤ nε × softmax(x)ᵢ
```

where n is vocabulary size and ε is machine epsilon.

### Condition Number Estimation

**Location**: `src/math/numerics.ts`

**Definition**: κ(A) = ||A|| × ||A⁻¹||

**Hager's Algorithm**:
- Estimates ||A⁻¹||₁ without computing inverse
- O(n²) complexity
- Accurate within factor of 10 typically

**Theorem (Hager, 1984)**:
The estimate κ̂ satisfies:
```
κ(A) / n ≤ κ̂ ≤ κ(A)
```

**Warning Thresholds**:
- κ < 10⁶: Well-conditioned
- 10⁶ ≤ κ < 10¹²: Moderately ill-conditioned
- κ ≥ 10¹²: Severely ill-conditioned

---

## Optimizer Convergence

### Sophia Optimizer

**Location**: `src/training/SophiaOptimizer.ts`, `src/math/convergence.ts`

**Algorithm**:
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ           (momentum)
hₜ = β₂hₜ₋₁ + (1-β₂)gₜ²          (Hessian diagonal)
θₜ₊₁ = θₜ - η × clip(mₜ/hₜ, -ρ, ρ)
```

**Assumptions**:
- L-smooth objective: ||∇f(x) - ∇f(y)|| ≤ L||x-y||
- Bounded variance: E[||gₜ - ∇f(θₜ)||²] ≤ σ²
- Bounded gradients: ||gₜ|| ≤ G

**Theorem (Liu et al., 2023)**:
Under the above assumptions, for non-convex f:
```
(1/T) Σₜ E[||∇f(θₜ)||²] ≤ O((f(θ₀) - f*) / ηT) + O(σ²η)
```

**Convergence Rate**: O(1/√T) with optimal η ~ 1/√T

**Practical Recommendations**:
- Learning rate: η = 1e-4 (lower than Adam)
- Momentum: β₁ = 0.965
- Hessian EMA: β₂ = 0.99
- Clip bound: ρ = 1.0

### Lion Optimizer

**Location**: `src/training/LionOptimizer.ts`, `src/math/convergence.ts`

**Algorithm**:
```
update = sign(β₁mₜ + (1-β₁)gₜ)
θₜ₊₁ = θₜ - η × update - η × λ × θₜ
mₜ₊₁ = β₂mₜ + (1-β₂)gₜ
```

**Key Insight**: sign() normalizes gradients, providing implicit clipping.

**Theorem (Chen et al., 2023)**:
For L-smooth, bounded gradient objectives:
```
(1/T) Σₜ E[||∇f(θₜ)||] ≤ O((f(θ₀) - f*)/√T)
```

**Memory**: 50% less than Adam (1 buffer vs 2)

**Comparison**:
| Optimizer | Memory | Convergence | Best For |
|-----------|--------|-------------|----------|
| SGD       | 1×     | O(1/√T)    | Simple cases |
| Adam      | 2×     | O(1/√T)    | General use |
| Lion      | 1×     | O(1/√T)    | Memory-constrained |
| Sophia    | 2×     | O(1/T)     | Curvature-aware |

---

## Information Theory

### KSG Mutual Information Estimator

**Location**: `src/math/information_theory.ts`

**Estimator (Kraskov et al., 2004)**:
```
I(X;Y) = ψ(k) + ψ(N) - ⟨ψ(nₓ+1) + ψ(nᵧ+1)⟩
```

where:
- ψ is digamma function
- k is number of neighbors
- nₓ, nᵧ are counts in marginal spaces

**Assumptions**:
- Continuous random variables
- N samples i.i.d.
- k << N

**Theorem**:
```
E[Î(X;Y)] → I(X;Y) as N → ∞
Var[Î(X;Y)] = O(1/N)
```

**Standard Error**:
```
SE ≈ √((ψ(k) - ψ(1) + 1/k) / N)
```

**Practical k Selection**:
- k = 3-7 for most cases
- Larger k reduces variance but increases bias
- Default: k = 3

### Information Bottleneck

**Location**: `src/losses/information_bottleneck.ts`

**Principle (Tishby et al., 1999)**:
Find representation Z that minimizes:
```
L_IB = I(X;Z) - β × I(Z;Y)
```

**Lagrangian**:
```
minimize I(X;Z) subject to I(Z;Y) ≥ I₀
```

**Beta Schedule**:
- Constant: β = β₀
- Linear: β(t) = β₀ + (β₁ - β₀) × t/T
- Cosine: β(t) = β₁ + (β₀ - β₁)(1 + cos(πt/T))/2

**Rate-Distortion Trade-off**:
```
R(β) = I(X;Z) at optimum
D(β) = E[d(X, X̂)]
```

Pareto frontier traces optimal compression-accuracy trade-off.

---

## Sampling Methods

### Typical Sampling

**Location**: `src/generation/sampling.ts`, `src/math/sampling_analysis.ts`

**Definition**: Token x is ε-typical if:
```
|−log p(x) − H(P)| ≤ ε
```

where H(P) is entropy of model distribution.

**Theorem (Cover & Thomas)**:
For typical set Aε⁽ⁿ⁾:
1. P(Aε⁽ⁿ⁾) > 1 - ε for large n
2. |Aε⁽ⁿ⁾| ≤ 2^(n(H+ε))
3. |Aε⁽ⁿ⁾| ≥ (1-ε) × 2^(n(H-ε))

**Implementation**:
```typescript
function typicalSample(logits, tau = 0.9) {
  const H = computeEntropy(softmax(logits));
  const surprise = logits.map(l => -log(softmax(l)));
  const typical = surprise.filter(s => |s - H| < tau);
  return sampleFrom(typical);
}
```

### Mirostat v2

**Location**: `src/generation/sampling.ts`

**Algorithm**:
```
μₜ₊₁ = μₜ + η × (surpriseₜ - τ)
```

where:
- μ is adaptive threshold
- τ is target entropy (perplexity)
- η is learning rate

**Convergence Theorem**:
Under bounded surprises, μₜ converges to steady state:
```
E[μ_∞] = τ + O(σ²/η)
```

**Analysis Metrics**:
- Time constant: τ_mix ≈ -1/log(ρ₁)
- Mixing time: T_mix ≈ 3 × τ_mix
- Coefficient of variation: CV = σ_μ / μ̄

---

## Attention Mechanisms

### Spectral Analysis

**Location**: `src/math/spectral_graph.ts`

**Graph Laplacian**:
- Unnormalized: L = D - A
- Normalized: L = I - D^(-1/2)AD^(-1/2)

**Algebraic Connectivity** (Fiedler value):
- λ₂(L) = second smallest eigenvalue
- λ₂ > 0 ⟺ graph is connected

**Cheeger Inequality**:
```
λ₂/2 ≤ h(G) ≤ √(2λ₂)
```

where h(G) is edge expansion (Cheeger constant).

**Mixing Time Bound**:
```
T_mix ≤ O(log(n) / λ₂)
```

**Expander Properties**:
- Good expander: λ₂ ≥ Ω(1/log n)
- Ramanujan graph: λ₂ ≥ 2√(d-1)

### Grouped-Query Attention (GQA)

**Configuration**:
- MHA: numKVHeads = numHeads (baseline)
- GQA: numKVHeads < numHeads (efficient)
- MQA: numKVHeads = 1 (maximum efficiency)

**Complexity**:
```
MHA: O(n² × d × numHeads)
GQA: O(n² × d × numHeads) forward, O(n × d × numKVHeads) KV cache
MQA: O(n² × d × numHeads) forward, O(n × d) KV cache
```

**Memory Savings**:
```
GQA savings = 1 - numKVHeads/numHeads
```

**Quality vs Efficiency Trade-off**:
| Config | KV Cache | Quality |
|--------|----------|---------|
| 8:8    | 100%     | 100%    |
| 8:4    | 50%      | ~99%    |
| 8:2    | 25%      | ~98%    |
| 8:1    | 12.5%    | ~95%    |

---

## Low-Rank Approximations

### Wedin's Sin-Theta Theorem

**Location**: `src/math/approximation.ts`

**Setup**:
- A = UΣV^T (original SVD)
- Ã = A + E (perturbed matrix)
- Ã = ŨΣ̃Ṽ^T (perturbed SVD)

**Theorem (Wedin, 1972)**:
```
sin θ(V_k, Ṽ_k) ≤ ||E||₂ / δ
```

where:
- θ is canonical angle between subspaces
- δ = σ_k(A) - σ_{k+1}(Ã) is singular value gap

**Implications**:
- Small perturbation + large gap → stable subspace
- Meaningful bound when ||E||₂ < δ

### Nyström Approximation

**Location**: `src/math/approximation.ts`

**Algorithm**:
Given PSD kernel matrix K, sample m landmarks:
```
K̃ = K_{nm} K_{mm}^{-1} K_{mn}
```

**Error Bound (Williams & Seeger, 2001)**:
```
||K - K̃||_F ≤ ||K - K_k||_F + (n/m)||K_{mm}^{-1}||₂ × ||K - K_k||²_F
```

where K_k is best rank-k approximation.

**Simplified**:
```
||K - K̃||_F ≈ O(n/m × σ_{m+1})
```

**Recommendations**:
- Use m ≥ √n landmarks for reasonable accuracy
- k-means++ for landmark selection
- Check condition number of K_{mm}

### Randomized SVD

**Location**: `src/math/approximation.ts`

**Algorithm (Halko et al., 2011)**:
```
1. Y = A × Ω (random projection, Ω is n×(k+p))
2. Y = (AA^T)^q × Y (power iterations)
3. Q = QR(Y) (orthonormalize)
4. B = Q^T × A (project to lower dim)
5. [Ũ, Σ, V] = SVD(B)
6. U = Q × Ũ
```

**Error Bound**:
```
E[||A - UΣV^T||] ≤ (1 + 4√((k+p)/(p-1))) × σ_{k+1}
```

**Complexity**:
- Standard SVD: O(mn min(m,n))
- Randomized: O(mnk) for rank-k

---

## Complexity Analysis

### Training Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Forward (ProNeuralLM) | O(V×H + H²) | O(V + H) |
| Forward (Transformer) | O(n²d + nd²) | O(n² + nd) |
| Backward | Same as forward | Same |
| Adam update | O(params) | O(2 × params) |
| Lion update | O(params) | O(params) |
| Sophia update | O(params) | O(2 × params) |

where:
- V = vocabulary size
- H = hidden size
- n = sequence length
- d = embedding dimension

### Generation Complexity

| Method | Time per Token | Space |
|--------|---------------|-------|
| Greedy | O(V) | O(1) |
| Top-k | O(V + k log k) | O(k) |
| Top-p | O(V log V) | O(V) |
| Typical | O(V) | O(V) |
| Mirostat | O(V) | O(1) |
| Beam Search | O(b × V) | O(b × n) |

where b = beam width.

### Compression Complexity

| Method | Time | Compression Ratio |
|--------|------|-------------------|
| Quantization (int8) | O(params) | 4× |
| Distillation | O(epochs × data × model) | Variable |
| Low-rank (SVD) | O(params^1.5) | 2-10× |
| Nyström | O(n × m²) | n/m |

---

## References

### Numerical Analysis
- Higham, N.J. (2002) "Accuracy and Stability of Numerical Algorithms"
- Kahan, W. (1965) "Pracniques: Further remarks on reducing truncation errors"

### Optimization
- Liu et al. (2023) "Sophia: A Scalable Stochastic Second-order Optimizer"
- Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
- Bottou et al. (2018) "Optimization Methods for Large-Scale Machine Learning"

### Information Theory
- Kraskov et al. (2004) "Estimating mutual information"
- Tishby et al. (1999) "The Information Bottleneck Method"
- Cover & Thomas (2006) "Elements of Information Theory"

### Sampling
- Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration"
- Basu et al. (2021) "Mirostat: A Neural Text Decoding Algorithm"

### Graph Theory
- Chung, F. (1997) "Spectral Graph Theory"
- Von Luxburg (2007) "A Tutorial on Spectral Clustering"

### Approximation Theory
- Wedin (1972) "Perturbation bounds in connection with SVD"
- Halko et al. (2011) "Finding Structure with Randomness"
- Williams & Seeger (2001) "Using the Nyström Method"

---

*This document is maintained as part of the Neuro-Lingua mathematical rigor initiative.*
