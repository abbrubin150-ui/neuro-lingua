# Mathematical Analysis Module

This module provides rigorous mathematical analysis tools for the Neuro-Lingua neural language model. All algorithms include mathematical guarantees and error bounds.

## Module Overview

| File | Purpose | Key Functions |
|------|---------|---------------|
| `numerics.ts` | Numerical stability operations | Kahan/Neumaier summation, stable norms, condition number |
| `convergence.ts` | Optimizer convergence analysis | Sophia/Lion/SGD theorems, Lyapunov analysis |
| `information_theory.ts` | Information-theoretic analysis | KSG MI estimation, rate-distortion, Fisher information |
| `sampling_analysis.ts` | Sampling quality analysis | Entropy tests, Mirostat, nucleus threshold |
| `spectral_graph.ts` | Spectral graph analysis | Laplacian, attention patterns, expander properties |
| `approximation.ts` | Approximation theory bounds | Wedin bounds, Nystrom approximation, randomized SVD |
| `analysis.ts` | Core analysis utilities | Spectral radius, Lyapunov stability |
| `statistics.ts` | Statistical analysis | Fisher information, Hessian statistics |
| `bias_verification.ts` | Bias verification for estimators | Bias bounds, variance analysis |
| `causal_math.ts` | Causal inference mathematics | DAG operations, intervention calculus |
| `dag_operations.ts` | DAG operations | Topological sort, cycle detection, path finding |
| `ntk_analysis.ts` | Neural Tangent Kernel analysis | NTK computation, infinite-width limits |

## Quick Start

```typescript
import {
  kahanSum,
  sophiaConvergenceTheorem,
  ksgMutualInformation,
  computeEntropy,
  analyzeAttentionGraph
} from '../math';
```

## Module Details

### numerics.ts - Numerical Stability

Provides numerically stable implementations of common operations.

**Key Functions:**
- `kahanSum(values)` - Kahan summation for floating-point accuracy
- `neumaierSum(values)` - Improved Kahan summation
- `stableFrobeniusNorm(matrix)` - Stable Frobenius norm computation
- `estimateConditionNumber(matrix)` - Matrix condition number estimation
- `checkNumericalStability(matrix)` - Comprehensive stability check

### convergence.ts - Convergence Analysis

Analyzes convergence properties of optimizers.

**Key Functions:**
- `sophiaConvergenceTheorem(config)` - Sophia optimizer convergence rate
- `lionConvergenceTheorem(config)` - Lion optimizer convergence rate
- `sgdStronglyConvexTheorem(config)` - SGD on strongly convex functions
- `analyzeConvergence(losses)` - Practical convergence analysis
- `analyzeOptimizationStability(gradients)` - Lyapunov stability analysis

### information_theory.ts - Information Theory

Advanced information-theoretic measures and analysis.

**Key Functions:**
- `ksgMutualInformation(x, y, k)` - KSG mutual information estimator
- `knnEntropy(data, k)` - k-NN entropy estimation
- `computeRateDistortionCurve(data, distortions)` - Rate-distortion analysis
- `computeInformationPlane(inputs, hidden, outputs)` - Information plane
- `estimateFisherInformation(gradients)` - Fisher information matrix

### sampling_analysis.ts - Sampling Quality

Analyzes quality of text generation sampling methods.

**Key Functions:**
- `computeEntropy(distribution)` - Shannon entropy
- `computeKLDivergence(p, q)` - KL divergence
- `entropyDistributionTest(samples)` - Statistical entropy test
- `analyzeMirostatConvergence(history)` - Mirostat mu convergence
- `analyzeSamplingQuality(samples, reference)` - Overall sampling quality

### spectral_graph.ts - Spectral Graph Analysis

Analyzes attention patterns using spectral graph theory.

**Key Functions:**
- `computeLaplacian(adjacency)` - Graph Laplacian matrix
- `analyzeAttentionGraph(attention)` - Attention pattern analysis
- `checkExpanderProperties(graph)` - Expander graph verification
- `analyzeSparseAttentionPattern(pattern)` - Sparse attention analysis

### approximation.ts - Approximation Theory

Bounds and algorithms for low-rank approximations.

**Key Functions:**
- `wedinBound(U1, U2, sigma)` - Wedin perturbation bound
- `nystromApproximation(matrix, landmarks)` - Nystrom approximation
- `randomizedSVD(matrix, rank, oversampling)` - Randomized SVD
- `analyzeLowRankApproximation(original, approximation)` - Error analysis

### analysis.ts - Core Analysis

Core mathematical analysis utilities.

**Key Functions:**
- `spectralRadius(matrix)` - Spectral radius via power iteration
- `analyzeLyapunov(matrix)` - Lyapunov stability analysis

### statistics.ts - Statistical Analysis

Statistical analysis tools for training diagnostics.

**Key Functions:**
- `empiricalFisherFromGradients(gradients)` - Empirical Fisher matrix
- `fisherHessianStatistics(gradients)` - Fisher information statistics
- `fisherQuadraticForm(fisher, vector)` - Quadratic form evaluation

## Mathematical References

- **Kahan Summation**: Kahan (1965) "Pracniques: Further Remarks on Reducing Truncation Errors"
- **KSG Estimator**: Kraskov et al. (2004) "Estimating Mutual Information"
- **Wedin Bounds**: Wedin (1972) "Perturbation Bounds in Connection with SVD"
- **Information Bottleneck**: Tishby et al. (1999) "The Information Bottleneck Method"
- **Neural Tangent Kernel**: Jacot et al. (2018) "Neural Tangent Kernel"

## Testing

Tests are located in `/tests/math/`:
- `analysis.test.ts` - Core analysis tests
- `numerics.test.ts` - Numerical stability tests

Run tests with:
```bash
pnpm test math
```
