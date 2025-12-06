# Cerebro Neuron Injection

The Cerebro mode introduces adaptive neuron growth based on bubble residuals in embedding space. It is intended for research-grade, reversible experiments and keeps a lightweight footprint so it can ship alongside the existing training UI.

## Flow
1. **Observe** bubbles derived from embeddings, attention rollout, or saliency maps.
2. **Propose** an injection based on residual energy orthogonal to the current layer basis.
3. **Inject** neurons into an `InjectableLayer` implementation and push the event to the Decision Ledger.
4. **Undo** restores the previous weight snapshot for the layer.

## Math in short
- Residual per bubble: `r = e - W W^T e` where `W` is the current neuron basis.
- Weighted covariance over bubbles: `Σ = Σ_i a_i (e_i - μ)(e_i - μ)^T`.
- Orthogonal projector: `P_⊥ = I - W W^T` for an orthonormal basis.
- Candidate neurons: top eigenvectors of `Σ_⊥ = P_⊥ Σ P_⊥`.

## API slices
- `InjectableLayer` defines a minimal surface area: export/import weights, feasibility checks, and an `inject` entrypoint.
- `InjectionEngine` performs diagnostics, proposals, execution (with rollback), and vector materialisation for UI previews.
- `InjectionRunSession` (training hook) handles snapshots, ledger logging, and undo.

## UI cues
- The Cerebro panel shows a bubble ring, minimal controls (Propose/Inject/Undo), ledger deltas, and advanced knobs for epsilon, orthogonality penalty, and gain thresholds.

## Next steps
- Add GPU power-iteration to `injection_math` for large d-models.
- Wire `InjectionRunSession` into real training loops and Run/Decision ledger persistence.
- Extend target types to adapters once available.
