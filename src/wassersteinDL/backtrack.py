"""Translation of toolbox/backtrack.m"""

import numpy as np

from ._matlab import isreal


def backtrack(phi, f, U, dir, alpha, beta=None, t=1, sumFunction=None):
    """Backtracking line search.

    Performs a backtracking line search on convex function phi with initial
    position U in the decrease direction dir. The initial value is f.

    Returns (t, obj, H, grad, gradNorm).
    """
    # if nargin<6, t=1; end   (handled by the default value of t)
    H = sumFunction(U, -t, dir)
    obj, grad, gradNorm = phi(H)
    objs = [obj]
    test = obj + gradNorm
    while (not isreal(obj)) or (not isreal(test)) or np.isnan(test) or np.isinf(test) or obj > f + alpha * t:
        t = beta * t
        H = sumFunction(U, -t, dir)
        obj, grad, gradNorm = phi(H)
        objs.append(obj)
        test = obj + gradNorm

    return t, obj, H, grad, gradNorm
