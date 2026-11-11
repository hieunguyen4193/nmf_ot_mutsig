"""Translation of toolbox/isHistogram.m"""

import numpy as np

from ._matlab import EPS, isreal


def isHistogram(X):
    """Check whether columns of a matrix are histograms.

    Checks whether all columns of X are real valued, non-negative and sum to
    one.
    """
    isHist = (not np.any(np.isnan(X) | np.isinf(X))) and allPositive(X) and sumsToOne(X)
    return bool(isHist)


def allPositive(X):
    allPos = isreal(X) and bool(np.all(X >= 0))
    return allPos


def sumsToOne(X):
    n = X.shape[0]

    sums = bool(np.all(np.abs(1 - np.sum(X, axis=0)) < EPS * n * 2))
    return sums
