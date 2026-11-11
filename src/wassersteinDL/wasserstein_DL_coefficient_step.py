"""Translation of toolbox/wasserstein_DL_coefficient_step.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import numpy as np

from ._matlab import gather, isempty, isfield, mldivide, norm_fro, pinv
from .buildMultipliers import buildMultipliers
from .checkOptionsWasserteinProjection import checkOptionsWasserteinProjection
from .computeWassersteinLegendre import computeWassersteinLegendre
from .computeWassersteinLegendreWeighted import computeWassersteinLegendreWeighted
from .isHistogram import isHistogram
from .matrixEntropy import matrixEntropy
from .linear_projected_gradient_descent import linear_projected_gradient_descent


def wasserstein_DL_coefficient_step(X, K, D, gamma, H=None, options=None, isHist=None):
    """Wasserstein projection on the span of D.

    This function solves

                   min_{lambda} W_gamma(X,D*lambda)

    Inputs:
          - X, the input data (n-by-m matrix)
          - K, exp(-M/gamma) where M is the ground metric (n-by-t matrix)
          - D, the dictionary (t-by-k matrix)
          - gamma, the regularization parameter of Wasserstein (scalar)
    Optional inputs:
          - H, the initial dual variables (t-by-m matrix)
          - options, an option dict (see wasserstein_DL)

    Returns (lambda, H, obj, gradient):
          - lambda: the minimizer (k-by-m matrix)
          - H: the dual optimal variable (t-by-m matrix)
          - obj: the sequence of (negated) dual objective values
          - gradient: the gradient of the dual on H (t-by-m matrix)
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

    # Projection operator
    pinvD = D @ pinv(D)

    def projection(gradient):
        gradNorm = norm_fro(gradient)
        if gradNorm > 0:
            projected = (gradient - pinvD @ gradient) / gradNorm
            gradNorm = np.sqrt(np.sum(projected * gradient))
        else:
            projected = gradient
        return projected, gradNorm

    # Function handler that compute the Wasserstein part of the objective and its gradient
    if isempty(options['weights']):
        wassersteinObj = lambda H: computeWassersteinLegendre(X, H, gamma, pX, multiplyK, multiplyKt)
    else:
        wassersteinObj = lambda H: computeWassersteinLegendreWeighted(X, H, gamma, pX, multiplyK, multiplyKt, options['weights'])

    # Function that computes the objective and its gradient
    def computeObj(H):
        obj, grad = wassersteinObj(H)

        # Compute the norm
        gradNorm = gather(norm_fro(grad))
        return obj, grad, gradNorm

    # Initialize
    if isempty(H):
        H = np.zeros((n, m))
    else:
        H, _ = projection(H)

    # Launch the solver to get the dual optimizer
    H, obj, gradient = linear_projected_gradient_descent(
        H, lambda G: computeObj(G), lambda G: projection(G), options['dual_descent_stop'],
        options['t0'], options['alpha'], options['beta'], options['verbose'])
    obj = -obj

    # Recover D from the dual variable
    lambda_ = mldivide(D, gradient)

    return lambda_, H, obj, gradient
