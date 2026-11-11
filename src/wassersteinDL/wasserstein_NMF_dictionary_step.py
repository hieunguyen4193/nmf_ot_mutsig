"""Translation of toolbox/wasserstein_NMF_dictionary_step.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import numpy as np

from ._matlab import gather, isempty, isfield, norm_fro
from .buildMultipliers import buildMultipliers
from .checkOptionsWasserteinProjection import checkOptionsWasserteinProjection
from .computeWassersteinLegendre import computeWassersteinLegendre
from .computeWassersteinLegendreWeighted import computeWassersteinLegendreWeighted
from .isHistogram import isHistogram
from .matrixEntropy import matrixEntropy
from .accelerated_gradient import accelerated_gradient


def wasserstein_NMF_dictionary_step(X, K, lambda_, gamma, rho, H=None, options=None, isHist=None):
    """Dictionary step of Wasserstein NMF.

    This function solves

                   min_{D} W_gamma(X,D*lambda) + rho*E(D)

    Inputs:
          - X, the input data (n-by-m matrix)
          - K, exp(-M/gamma) where M is the ground metric (n-by-t matrix)
          - lambda_, the coefficients (k-by-m matrix)
          - gamma, the regularization parameter of Wasserstein (scalar)
          - rho, the regularization parameter for D (scalar)
    Optional inputs:
          - H, the initial dual variables (t-by-m matrix)
          - options, an option dict (see wasserstein_DL)

    Returns (D, H, obj, gradient).
    """
    if options is None:
        options = {}
    options = checkOptionsWasserteinProjection(options)

    n = X.shape[0]
    m = X.shape[1]
    if options['Kmultiplication'] == 'symmetric':
        if not ((K.shape[1] == n) and (np.sum(np.abs(K - K.T)) / np.sum(K) < 1e-3)):
            options['Kmultiplication'] = 'asymmetric'
    if options['Kmultiplication'] == 'asymmetric':
        n = K.shape[1]

    if isHist is None or not isHist:
        if not isHistogram(X):
            raise ValueError('Input vectors are not histograms. Check that they are non-negative, '
                             'sum to one and do not contain NaN of inf')

    multiplyK, multiplyKt = buildMultipliers(K, gamma, options, X.shape)

    # Precompute the entropy of X
    if not isfield(options, 'pX') or isempty(options['pX']):
        pX = matrixEntropy(X)
    else:
        pX = options['pX']

    bigMatrices = options['bigMatrices']

    k = lambda_.shape[0]

    # Function that compute the entropy part of the objective and its gradient
    def entropyObj(H):
        with np.errstate(over='ignore'):
            expL = np.exp(-H @ lambda_.T / rho)
        if np.any(np.isinf(expL)):
            expL[np.isinf(expL)] = np.max(expL[~np.isinf(expL)])
        sumL = np.sum(expL, axis=0, keepdims=True)
        obj = rho * np.sum(np.log(sumL))
        grad = -expL @ (lambda_ / sumL.T)
        return obj, grad

    # Function handler that compute the Wasserstein part of the objective and its gradient
    if isempty(options['weights']):
        wassersteinObj = lambda H: computeWassersteinLegendre(X, H, gamma, pX, multiplyK, multiplyKt)
    else:
        wassersteinObj = lambda H: computeWassersteinLegendreWeighted(X, H, gamma, pX, multiplyK, multiplyKt, options['weights'])

    # Function that computes the objective and its gradient
    def computeObj(H):
        # Compute the Wasserstein part of the gradient and objective
        obj, grad = wassersteinObj(H)

        # Compute the entropy part of the gradient
        objE, gradE = entropyObj(H)

        obj = obj + objE
        grad = grad + gradE

        # Compute the norm
        gradNorm = gather(norm_fro(grad))
        grad = grad / gradNorm
        return obj, grad, gradNorm

    # Initialize
    if isempty(H):
        H = options['createZeroArray']((n, m))

    # Launch the solver to get the dual optimizer
    H, obj, gradient = accelerated_gradient(
        H, lambda G: computeObj(G), lambda G: G, options['dual_descent_stop'], options['t0'],
        options['alpha'], options['beta'], options['verbose'])
    obj = -obj

    # Recover D from the dual variable
    D = np.exp(-H @ lambda_.T / rho)
    D = D / np.sum(D, axis=0, keepdims=True)

    return D, H, obj, gradient
