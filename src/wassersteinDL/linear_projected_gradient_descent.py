"""Translation of toolbox/linear_projected_gradient_descent.m"""

import math

import numpy as np

from ._matlab import display, gather, isempty, num2str
from .backtrack import backtrack


def linear_projected_gradient_descent(H, computeObj, projection, stop=None, t0=None,
                                      alpha=None, beta=None, verbose=None):
    """Projected gradient descent with backtracking line search.

    Returns (H, obj, gradient); obj is the 1-D array of objective values.
    """
    if isempty(t0):
        t0 = 1
    if isempty(alpha):
        alpha = 0
    if isempty(beta):
        beta = 0.8

    if isempty(verbose):
        verbose = 0

    if isempty(stop):
        stop = 1e-5

    objective, gradient, _ = computeObj(H)
    obj = [gather(objective)]
    last = math.inf
    niter = 0
    t = t0
    numberEval = 1
    if verbose:
        display('    Dual initialization, Objective : ' + num2str(obj[-1]) + ' Current step size ' + num2str(t))
    while abs(obj[-1] - last) > stop * (1 + abs(obj[-1])) * t or abs(obj[-1] - last) > 1:
        last = obj[-1]
        projected, gradNorm = projection(gradient)
        prevt = t
        t, objective, H, gradient, _ = backtrack(lambda x: computeObj(x), last, H, projected,
                                                 -alpha * gradNorm, beta, t,
                                                 lambda a, coeff, b: a + coeff * b)
        numberEval = numberEval + math.log(t / prevt) / math.log(beta) + 1
        obj.append(gather(objective))

        if verbose and niter % 20 == 0:
            display('    Dual iteration : ' + num2str(niter + 1) + ' Objective : ' + num2str(obj[-1]) + ' Current step size ' + num2str(t))

        t = t / math.sqrt(beta)
        niter = niter + 1

    if verbose:
        display('    Finished, number of iterations : ' + num2str(niter) + ' Objective : ' + num2str(obj[-1]) + ' Current step size ' + num2str(t))

    return H, np.array(obj, dtype=float), gradient
