"""Translation of toolbox/computeWassersteinLegendre.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import numpy as np


def computeWassersteinLegendre(X, H, gamma, pX, multiplyK, multiplyKt):
    """Legendre transform of Wasserstein distance.

    This function computes \\sum W^(x_i,h_i) and its gradient where
    W^ is the Legendre transform of the regularized wasserstein distance.
    """
    # Compute the Wasserstein part of the gradient
    with np.errstate(over='ignore'):
        alphaTmp = np.exp(H / gamma)
    grad = multiplyK(alphaTmp)
    if np.any(grad == 0):
        grad[grad == 0] = np.min(grad[grad > 0])
    if np.any(np.isinf(grad)):
        grad[np.isinf(grad)] = np.max(grad[~np.isinf(grad)])
    # MATLAB silently returns Inf/NaN here; the line search rejects such points
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        obj = (np.sum(X * np.log(grad)) + pX) * gamma
        grad = alphaTmp * (multiplyKt(X / grad))

    return obj, grad
