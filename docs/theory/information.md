# Information-Theoretic Training Diagnostics

## The Information Bottleneck Principle

The **information bottleneck (IB)** formalises representation learning as a
trade-off between compressing the input $X$ and preserving the labels $Y$.
Given a representation $Z = f_\theta(X)$, the IB objective minimises
\[
\mathcal{L}_{\text{IB}}(\theta) = I(X; Z) - \beta I(Z; Y),
\]
where $I(\cdot; \cdot)$ denotes mutual information and $\beta > 0$ controls the
degree of compression. Minimising $I(X; Z)$ drives $Z$ towards a minimal
sufficient statistic of $X$, while maximising $I(Z; Y)$ ensures the
representation keeps the predictive signal.

In practice we rarely have access to the full joint distribution, so we work
with tractable bounds:

- **Variational upper bound on $I(X; Z)$**.
  Let $q_\phi(z)$ be a learnable prior over representations. Using the
  non-negativity of the Kullback–Leibler divergence we obtain
  \[
  I(X; Z) = \mathrm{KL}\big(p(z \mid x) \Vert p(z)\big)
  \leq \mathbb{E}_{p(x)}\big[\mathrm{KL}\big(p(z \mid x) \Vert q_\phi(z)\big)\big].
  \]
  The expectation can be estimated with mini-batches and Monte Carlo samples.
- **Variational lower bound on $I(Z; Y)$**.
  Introducing a decoder $q_\psi(y \mid z)$ gives
  \[
  I(Z; Y) \geq \mathbb{E}_{p(x, y)}\mathbb{E}_{p(z \mid x)}\big[\log q_\psi(y \mid z)\big] + H(Y),
  \]
  which coincides with maximising the conditional log-likelihood of the labels
  under the decoder. The entropy term $H(Y)$ does not depend on the model
  parameters and can be ignored during optimisation.

The resulting loss resembles a VAE objective with an explicit information
budget. In `neuro-lingua` the trade-off coefficient $\beta$ is exposed in the
second-order optimiser configuration, enabling annealing schedules that start
with strong compression to stabilise training and gradually release capacity to
recover task performance.

### Training Dynamics Insight

Monitoring the two information terms during optimisation reveals characteristic
phases:

1. **Fitting regime**: $I(Z; Y)$ increases rapidly while $I(X; Z)$ remains high,
   signalling that the representation memorises both signal and noise.
2. **Compression regime**: curvature-aware updates (e.g. natural gradient)
   reduce $I(X; Z)$ without sacrificing $I(Z; Y)$, pushing the network towards a
   minimal sufficient representation.
3. **Over-compression warning**: if $I(Z; Y)$ starts to drop, the optimiser is
   removing too much task-relevant information; scheduling $\beta$ or switching
   to a Fisher preconditioner mitigates the issue.

These diagnostics align with the spectral and Lyapunov analysis utilities in
`src/math/analysis.ts`, offering a complementary perspective rooted in
information geometry.

## Entropy Gradient and Flat-Minima Bias

For a model with predictive distribution $p_\theta(y \mid x)$, the entropy of
the output is
\[
H_\theta(Y \mid X = x) = - \sum_y p_\theta(y \mid x) \log p_\theta(y \mid x).
\]
The gradient of the entropy with respect to the parameters is
\[
\nabla_\theta H_\theta(Y \mid X = x) = - \sum_y \nabla_\theta p_\theta(y \mid x)
\big(1 + \log p_\theta(y \mid x)\big).
\]
Replacing the probability gradient with the score function yields a compact
expression:
\[
\nabla_\theta H_\theta(Y \mid X = x) = - \mathbb{E}_{p_\theta(y \mid x)}
\big[\nabla_\theta \log p_\theta(y \mid x) \log p_\theta(y \mid x)\big].
\]

This gradient highlights how entropy regularisation biases learning towards
flat minima:

- **Large negative log-probabilities** correspond to confident predictions;
  their contribution is small because the score function vanishes near mode
  peaks.
- **Near-uniform predictions** yield high entropy. The gradient then nudges the
  parameters in the direction of increasing uncertainty, acting as an implicit
  temperature control on the logits.

### Entropy Gradient in Optimisation

To incorporate the entropy gradient into curvature-aware training we combine it
with the empirical Fisher information matrix $F(\theta)$:
\[
F(\theta) = \mathbb{E}_{p_\theta(x, y)}\big[\nabla_\theta \log p_\theta(y \mid x)
\nabla_\theta \log p_\theta(y \mid x)^\top\big].
\]
Projecting the entropy gradient through $F(\theta)^{-1}$ produces a natural
entropy step that respects the geometry induced by the model's likelihood:
\[
\Delta\theta_{\text{entropy}} = -\eta F(\theta)^{-1}
\nabla_\theta H_\theta(Y \mid X),
\]
where $\eta$ is a learning-rate-like coefficient. When the optimiser uses the
empirical Fisher diagonal (as implemented in `computeDiagonalHessian`), this
reduces to a coordinate-wise rescaling that emphasises directions with low
curvature and deemphasises sharp ones, reinforcing the flat-minima bias of the
entropy regulariser.

### Practical Monitoring Recipe

1. Track the running average of $H_\theta(Y \mid X)$ alongside the training
   loss.
2. Estimate $\nabla_\theta H_\theta(Y \mid X)$ using the same automatic
   differentiation tape used for the loss, and accumulate its norm.
3. Compare the entropy-gradient norm with the Fisher-induced quadratic form
   $\nabla_\theta H_\theta(Y \mid X)^\top F(\theta)^{-1}\nabla_\theta H_\theta(Y \mid X)$.
   A large discrepancy indicates misalignment between the entropy and likelihood
   geometry, suggesting that either the Fisher estimate is noisy or the model is
   moving into brittle regions of parameter space.

The combination of IB tracking and entropy-gradient diagnostics provides a
holistic view of how representations evolve, complementing the existing spectral
and stability tools in this repository.
