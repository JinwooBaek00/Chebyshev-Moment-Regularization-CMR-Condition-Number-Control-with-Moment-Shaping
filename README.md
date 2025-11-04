# Chebyshev Moment Regularization (CMR)

A drop-in spectral regularizer that directly optimizes layer spectra for stable, accurate training — log-condition (log-κ) proxy + Chebyshev moments; architecture-agnostic.

## Features
- **Condition proxy**: differentiable log-condition surrogate with strict descent under gradient flow
- **Moment shaping**: Chebyshev trace moments \(s_k, k \ge 3\) on a normalized Gram; shapes interior spectral mass
- **Decoupled capped mixing**: preserves task gradients while bounding spectral intervention (warm-up friendly)
- **Orthogonal invariance**: invariant under \(QWR\); compatible with standard layers/optimizers
- **κ-stress recovery (illustrative)**: reduces mean layer κ from ~3.9e3 to ~3.4 in 5 epochs and restores test accuracy (~10% → ~86%) on a 15-layer MLP (MNIST) [see paper]

Runs as a single notebook.  
Tested on Python 3.11.
