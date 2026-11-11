"""Translation of toolbox/buildMultipliers.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import math

import numpy as np

from .gaussianConv import gaussianConv


def buildMultipliers(K, gamma, options, sizeX):
    """Prepare the multipliers for Wasserstein Dictionary Learning.

    Function that creates handlers for multiplication by K and K^T for
    computing a regularized wasserstein distance.

    For 'convolution', K is a dict with keys 'grid_dimensions' and
    'kernelSize'. Reshapes use Fortran (column-major) order, as in MATLAB.
    """
    if options['Kmultiplication'] == 'symmetric':
        multiplyK = lambda M: K @ M
        multiplyKt = multiplyK
    elif options['Kmultiplication'] == 'asymmetric':
        multiplyK = lambda M: K @ M
        if options['bigMatrices']:
            multiplyKt = lambda M: K.T @ M
        else:
            Kt = np.ascontiguousarray(K.T)
            multiplyKt = lambda M: Kt @ M
    elif options['Kmultiplication'] == 'convolution':
        kernelSize = np.atleast_1d(K['kernelSize'])
        grid_dimensions = tuple(int(g) for g in np.atleast_1d(K['grid_dimensions']))
        filters = [None] * kernelSize.size
        for dimIndex in range(1, kernelSize.size + 1):   # A gaussian convolution can be done one dimension at a time
            kSize = math.ceil(kernelSize[dimIndex - 1] / 2)
            f = np.exp(-(np.arange(-kSize, kSize + 1) ** 2) / gamma)
            if dimIndex == 1:
                filters[dimIndex - 1] = f.reshape(-1, 1)
            else:
                filters[dimIndex - 1] = f.reshape((1,) * (dimIndex - 1) + (2 * kSize + 1,))
        sizeX = tuple(int(s) for s in sizeX)
        multiplyK = lambda M: np.reshape(
            gaussianConv(np.reshape(M, grid_dimensions + (sizeX[1],), order='F'), filters),
            sizeX, order='F')
        multiplyKt = multiplyK
    else:
        raise ValueError('Unknown multiplication type')
    return multiplyK, multiplyKt
