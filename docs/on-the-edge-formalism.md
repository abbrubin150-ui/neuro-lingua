# "On the Edge" Learning Principle — Final Unified Formalism

## Production-Ready with Critical Corrections & Defaults

---

## Part I: Critical Corrections

### 1.1 Mean-Squared Error (No Double-Counting)

**Error:** $\text{MSE} + b^2$ when $\text{MSE} = \text{Var} + b^2$ already.

**Correction:**
\[
E_b(\theta,n) = n \cdot \mathrm{tr}\big(I_1(\theta)^{1/2} \, \operatorname{MSE}(\hat\theta\mid\theta,n) \, I_1(\theta)^{1/2}\big)
\]

where $\operatorname{MSE} = \operatorname{Var} + \text{bias}^2$ is computed once.

---

### 1.2 Smooth $\lambda_{\max}$ via Log-Sum-Exp

**Problem:** $\lambda_{\max}$ is non-differentiable at eigenvalue multiplicities.

**Solution (LogSumExp approximation):**
\[
\lambda_{\max}^{\text{LSE}} \approx \frac{1}{\tau}\log\sum_i e^{\tau\lambda_i}
\]

with temperature $\tau \in [10, 50]$ (higher $\tau$ $\rightarrow$ closer to true max, but less smooth).

**Implementation:** Use in $E_{\max}$ and $B$ wherever $\lambda_{\max}$ appears. Differentiable everywhere.

---

### 1.3 Conservative & Stable Choice for $B$

**Instead of:** $B(\theta,n) = 1/\sqrt{n \cdot \lambda_{\max}(I_1)}$ (may be too large if one eigenvalue is tiny)

**Use:**
\[
B(\theta,n) = \frac{1}{\sqrt{n}} \cdot \frac{1}{\sqrt{\frac{1}{p}\mathrm{tr}(I_1(\theta))}} = \frac{1}{\sqrt{n}} \cdot s(\theta)^{-1}
\]

where:
\[
s(\theta) = \sqrt{\frac{1}{p}\mathrm{tr}(I_1(\theta))}
\]

is the **average-scale** Fisher information. More stable; avoids over-inflating for unidentified directions.

**Optional:** If you want to be conservative, use:
\[
B(\theta,n) = \frac{1}{\sqrt{n}} \cdot \lambda_{\max}^{\text{LSE}}(\theta)^{-1/2} \quad (\text{with clipping at } \epsilon_{\min}=10^{-6})
\]

---

### 1.4 Operational Criterion: $\lVert\nabla_\theta\Psi\rVert \neq 0$

**Not a constraint**, but a **diagnostic check:**

Compute $\lVert\nabla_\theta\Psi\rVert_2$ across the $\theta$ grid at each $n$. Over a window of $n$ values, require:
\[
\operatorname{Percentile}_{25}(\lVert\nabla_\theta\Psi\rVert) \geq \eta
\]

with small $\eta > 0$ (e.g., $\eta = 10^{-4}$). If violated, raise warning: "flat region detected; increase $\lambda_H$ or adjust bandwidths."

**Ensures:** System doesn't get stuck in accidental flat plateaus.

---

### 1.5 FFT Padding & Boundary Handling

When applying 1D convolution via FFT:

1. **Pad** the arrays (in $\theta$ and $\log n$ axes) with **reflect padding** to $\pm 3\sigma$ beyond boundaries
2. Apply FFT-based convolution on padded arrays
3. **Extract** (slice back) to original domain

**Pseudocode:**
```python
Ψ_padded = pad(H * B * V, "reflect", pad_width=3*σ_θ)
Ψ_conv = fft_convolve(Ψ_padded, K_θ)
Ψ_result = Ψ_conv[original_indices]
```

This avoids wrap-around artifacts.

---

## Part II: Unified Multivariate Formalism

### 2.1 Efficiency Measures

**Trace-based (orthogonal-invariant):**
\[
E_p(\theta,n) = \frac{n}{p}\,\mathrm{tr}\big(I_1^{1/2}(\theta)\,\Sigma(\theta,n)\,I_1^{1/2}(\theta)\big)
\]

**Eigenvalue-based (conservative):**
\[
E_{\max}(\theta,n) = n \cdot \lambda_{\max}^{\text{LSE}}\big(I_1^{1/2}\Sigma I_1^{1/2}\big)
\]

where $\Sigma$ is one of:
- **Variance:** $\operatorname{Var}(\hat\theta\mid\theta,n)$ (unbiased estimators)
- **MSE:** $\operatorname{Var}(\hat\theta) + \text{bias}(\hat\theta)^2$ (potentially biased estimators)
- **Sandwich:** $I^{-1}JI^{-1}$ (misspecified models, where $J = \mathbb{E}[\nabla\ell \, \nabla\ell^T]$)

---

### 2.2 Component Functions

**Entropy** (choose one):
\[
H(\theta) = H_R^{(2)}(\theta) = -\log\int p(x\mid\theta)^2\,dx
\]
or
\[
H(\theta) = \int p(x\mid\theta) \log\frac{p(x\mid\theta)}{q(x)}\,dx \quad (\text{relative, vs. reference } q)
\]

**Cramér–Rao Bound Term:**
\[
B(\theta,n) = \frac{1}{\sqrt{n}}\cdot s(\theta)^{-1}
\]

**Variance Constraint (smoothed):**
\[
V(\theta,n) = \exp\big(-\alpha[E_p(\theta,n)-1]^2\big) \cdot \sigma_s(E_p(\theta,n) - 1 + \delta)
\]

where $\sigma_s(z) = 1/(1+e^{-\lambda z})$ is a sigmoid with steepness $\lambda$.

**Separable Coupling Kernel:**
\[
K(\Delta\theta, \Delta\log n) = K_\theta(\Delta\theta) \cdot K_{\log n}(\Delta\log n)
\]

with:
\[
K_\theta(\Delta\theta) = \frac{1}{\sqrt{2\pi}\sigma_\theta}\exp\left(-\frac{(\Delta\theta)^2}{2\sigma_\theta^2}\right)
\]
\[
K_{\log n}(\Delta\log n) = \frac{1}{\sqrt{2\pi}\sigma_{\log n}}\exp\left(-\frac{(\Delta\log n)^2}{2\sigma_{\log n}^2}\right)
\]

---

### 2.3 Convolution & Objective

**Convolution:**
\[
\Psi(\theta,n) = \iint H(\theta')\,B(\theta',n')\,V(\theta',n')\,K_\theta(\theta-\theta')\,K_{\log n}(\log n-\log n')\,d\theta'\,d(\log n')
\]

(Computed via separable 1D convolutions with FFT + padding.)

**Derivative on log-scale:**
\[
\frac{\partial E_p}{\partial \log n} = n \cdot \frac{\partial E_p}{\partial n}
\]

(Use central differences on log-grid; apply EMA smoothing over small window to reduce noise.)

**Objective (Maximize):**
\[
\boxed{\mathcal{J}(\theta,n) = \Psi(\theta,n) + \lambda_H\,H(\theta) - \lambda_E\,(E_p(\theta,n)-1)^2 - \lambda_S\left(\frac{\partial E_p}{\partial\log n}\right)^2 - \mu\,\mathrm{softplus}(1-\delta-E_p(\theta,n))^2}
\]

**Soft constraint:** $E_{\max}(\theta,n) \geq 1 - \delta$ (enforced via softplus penalty).

---

## Part III: Default Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Entropy type | — | $H_R^{(2)}$ | Rényi (more stable than differential) |
| Fisher scale | $s(\theta)$ | $\sqrt{\frac{1}{p}\mathrm{tr}(I_1)}$ | Average information; use LSE $\lambda_{\max}$ if very conservative |
| $\theta$-bandwidth | $\sigma_\theta$ | Silverman on grid | $\approx 1.06 \, \sigma_{\hat\theta} \, n^{-1/(p+4)}$ |
| $\log n$ bandwidth | $\sigma_{\log n}$ | 0.3–0.5 | Cross-validate or fix (log-scale insensitive to $n$ range) |
| Offset | $\delta$ | 0.01 | Soft barrier start |
| Penalty weight | $\alpha$ | 10 | Strength of $(E_p-1)^2$ term in $V$ |
| LSE temperature | $\tau$ | 20 | For $\lambda_{\max}^{\text{LSE}}$; higher = closer to exact max |
| Eigenvalue floor | $\epsilon_{\min}$ | $10^{-6}$ | Clipping for small $\lambda_i$ |
| Sigmoid steepness | $\lambda$ | 10–20 | Higher = sharper transition at $E_p=1-\delta$ |
| Hyperparameter weights | $(\lambda_H, \lambda_E, \lambda_S, \mu)$ | [0.1–1] | Normalize by z-score effect on validation set (see §IV) |

---

## Part IV: Hyperparameter Normalization (Crucial)

**On a validation grid** $\{(\theta_j, n_k)\}$:

1. Compute independently:
   - $\Delta_H = \mathbb{E}[|H(\theta)|]$
   - $\Delta_E = \mathbb{E}[(E_p-1)^2]$
   - $\Delta_S = \mathbb{E}[(\partial E_p/\partial\log n)^2]$
   - $\Delta_{\text{bar}} = \mathbb{E}[\mathrm{softplus}(1-\delta-E_p)^2]$
2. Scale:
\[
\lambda_H' = \lambda_H / \Delta_H, \quad \lambda_E' = \lambda_E / \Delta_E, \quad \lambda_S' = \lambda_S / \Delta_S, \quad \mu' = \mu / \Delta_{\text{bar}}
\]
3. Choose final $(\lambda_H', \lambda_E', \lambda_S', \mu') \in [0.1, 1]$ so relative contributions to $\Delta\mathcal{J}$ are balanced.

**Prevents:** One term dominating (e.g., $\Psi$ alone drowning out entropy or smoothness).

---

## Part V: Minimal Pseudocode (Direct Implementation)

```python
# Precompute once
Θ = linspace(θ_min, θ_max, n_grid)
L = linspace(log(n_min), log(n_max), m_grid)

for θ in Θ:
  I1[θ] = fisher_per_sample(θ)  # Hessian/score; clipped eigenvalues
  H_val[θ] = renyi2_entropy(θ)  # or relative entropy

for ℓ in L:
  n = exp(ℓ)
  for θ in Θ:
    Σ = estimator_covariance(θ, n)  # Var or MSE or Sandwich
    E_p[θ, ℓ] = (n/p) * trace(sqrtm(I1[θ]) @ Σ @ sqrtm(I1[θ]))
    V[θ, ℓ] = exp(-α*(E_p-1)^2) * sigmoid(E_p-1+δ)
    B[θ, ℓ] = (1/sqrt(n)) / s(I1[θ])  # s = sqrt(tr(I1)/p)

  # Separable convolution with padding
  temp = H_val * B[:, ℓ] * V[:, ℓ]
  Ψ[:, ℓ] = convolve_1d(temp, K_θ, mode="reflect_pad")

# Derivative on log-axis
dE_dlogn = diff(E_p, axis=1)  # central differences on ℓ
dE_dlogn = ema_smooth(dE_dlogn, window=3)  # light smoothing

# Objective at each (θ, ℓ)
J = Ψ + λ_H*H_val - λ_E*(E_p-1)^2 - λ_S*(dE_dlogn)^2 - μ*softplus(1-δ-E_p)^2

# Check gradient magnitude
∇J_θ = autograd_or_finite_diff(J, w.r.t. θ)
if percentile(|∇J_θ|, 25) < η:
  print("Warning: flat region detected")

# Optimization: gradient ascent in θ (n fixed per step)
for step in range(max_steps):
  ∇J = grad(J, θ)
  θ_new = θ + step_size * ∇J

  # Diagnostics
  monitor: J, E_p, H_val, Ψ
```

---

## Part VI: Unit Tests (Before Full Run)

### Test 1: Gaussian (Known Variance)
```
Model: x_i ~ N(θ, 1)
Expected: E_p(θ, n) ≡ 1 for all n
Check:
  ✓ V(θ,n) ≈ 1 (on edge)
  ✓ ∂E_p/∂ log n ≈ 0 (flat trajectory)
  ✓ |∇_θ Ψ| > η everywhere (non-flat)
```

### Test 2: Bernoulli (Eigenvalue Collapse)
```
Model: x_i ~ Bernoulli(p)
Challenge: I_1(p) = 1/(p(1-p)) → ∞ as p → 0 or 1
Check:
  ✓ Clipped I_1 prevents overflow
  ✓ E_max ≥ 1-δ maintained
  ✓ No oscillations along log n
```

### Test 3: Misspecification (Godambe Sandwich)
```
Data: x_i ~ N(θ, 1)
Model: Assume Laplace (wrong)
Check:
  ✓ E_G (Godambe) > E_p (naive)
  ✓ Edge band still contains E_G
  ✓ Robust against model error
```

**Success criterion:** All three tests stable and within expected ranges → proceed to real experiments.

---

## Part VII: Why This Formalism is Closed

- **Mathematically:** Well-defined assumptions (A1–A4), finite convolution, smooth objective
- **Numerically:** FFT with padding, LSE smoothing for $\lambda_{\max}$, clipping for eigenvalues
- **Computationally:** Separable kernel (FFT on $\theta$, EMA on $\log n$), pre-computed tables
- **Optimizable:** Fully differentiable via autodiff; gradient ascent stable
- **Robust:** MSE (not doubled), $E_p$ scale-invariant, hyperparameters normalized

---

## Part VIII: Checklist Before Coding

- [ ] Implement Fisher information computation with eigen-clipping ($\epsilon_{\min}=10^{-6}$)
- [ ] Code $E_p$ (trace) and $E_{\max}$ (LSE-based $\lambda_{\max}$)
- [ ] Test that $E_p$ equals 1 on Gaussian with $n \to \infty$
- [ ] Implement $H$ (Rényi or relative entropy) and test for constant vs. varying signal
- [ ] Code $V$ with sigmoid; ensure differentiability
- [ ] Set up FFT-based separable convolution with reflect padding
- [ ] Implement $\partial E_p/\partial\log n$ via central diff + EMA
- [ ] Normalize hyperparameters on validation set
- [ ] Run three unit tests
- [ ] Visualize: heatmaps of $E_p(\theta,n)$, $\Psi(\theta,n)$, $\mathcal{J}(\theta,n)$ with edge band overlay
- [ ] Log and save: trajectory of all components per optimization step
