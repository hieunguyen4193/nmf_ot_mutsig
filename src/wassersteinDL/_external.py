"""Stand-ins for functions wasserstein_DL.m calls but the repository does not ship.

* ``fkmeans`` (MATLAB File Exchange "fast k-means") and ``pairwiseDistance``
  are only used by ``initialValues.method='kmeans'``; neither file is in the
  repository, so the original MATLAB code cannot run that branch as-is.
* ``nnmf`` is from MATLAB's Statistics and Machine Learning Toolbox and is
  used by ``initialValues.method='euclidean'``.

These are plain NumPy implementations of the same ideas. Because they draw
random numbers from a NumPy generator, their output will not match MATLAB
bit-for-bit; they only provide a starting point for the optimisation.
"""

import numpy as np


def pairwiseDistance(A, B):
    """Squared Euclidean distances between the columns of A and of B.

    Returns a size(A,2)-by-size(B,2) matrix.
    """
    sqA = np.sum(A * A, axis=0)[:, None]
    sqB = np.sum(B * B, axis=0)[None, :]
    return np.maximum(sqA + sqB - 2 * A.T @ B, 0)


def fkmeans(X, k, rng, maxiter=100):
    """k-means (k-means++ seeding, Lloyd iterations) on the rows of X.

    Returns (labels, centroids) with centroids of size k-by-size(X,2),
    the same convention as fkmeans.m.
    """
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]))
    centroids[0] = X[rng.integers(n)]
    d2 = np.sum((X - centroids[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centroids[j] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((X - centroids[j]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(maxiter):
        dist = pairwiseDistance(X.T, centroids.T)
        newLabels = np.argmin(dist, axis=1)
        if _ > 0 and np.array_equal(newLabels, labels):
            break
        labels = newLabels
        for j in range(k):
            if np.any(labels == j):
                centroids[j] = X[labels == j].mean(axis=0)
    return labels, centroids


def nnmf(X, k, rng, maxiter=100, tolx=1e-4, tolfun=1e-4):
    """Alternating least squares NMF, X ~ W*H, as in MATLAB nnmf's 'als'."""
    n, m = X.shape
    W = rng.random((n, k))
    H = np.maximum(np.linalg.lstsq(W, X, rcond=None)[0], 0)
    dnorm0 = np.inf
    sqrteps = np.sqrt(np.finfo(float).eps)
    for _ in range(maxiter):
        W0, H0 = W, H
        H = np.maximum(np.linalg.lstsq(W, X, rcond=None)[0], 0)
        W = np.maximum(np.linalg.lstsq(H.T, X.T, rcond=None)[0].T, 0)
        dnorm = np.sqrt(np.sum((X - W @ H) ** 2) / X.size)
        dw = np.max(np.abs(W - W0)) / (sqrteps + np.max(np.abs(W0)))
        dh = np.max(np.abs(H - H0)) / (sqrteps + np.max(np.abs(H0)))
        delta = max(dw, dh)
        if _ > 0 and (dnorm0 - dnorm <= tolfun * max(1, dnorm0) or delta <= tolx):
            break
        dnorm0 = dnorm
    # Like MATLAB nnmf: normalise columns of W to unit norm and sort the
    # factors by decreasing norm of the rows of H.
    hlen = np.sqrt(np.sum(W ** 2, axis=0))
    hlen[hlen == 0] = 1
    W = W / hlen
    H = H * hlen[:, None]
    order = np.argsort(-np.sum(H ** 2, axis=1))
    return W[:, order], H[order, :]
