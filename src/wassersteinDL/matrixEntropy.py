"""Translation of toolbox/matrixEntropy.m"""

import numpy as np


def matrixEntropy(X):
    """Compute the entropy of a matrix: -sum(X.*log(X)), with 0*log(0)=0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        pX = X * np.log(X)
    pX[np.isnan(pX)] = 0
    pX = -np.sum(pX)
    return pX
