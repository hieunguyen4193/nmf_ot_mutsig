"""Translation of toolbox/accelerated_gradient.m"""

import math

import numpy as np

from ._matlab import display, gather, isempty, norm_fro
from .backtrack import backtrack


def accelerated_gradient(H0, computeObjGrad, proximal, stop=None, t0=None, alpha=None,
                         beta=None, verbose=None, sumFunction=None, computeNorm=None):
    """Accelerated gradient.

    Function that computes the minimum of a convex function using an
    accelerated gradient with the update described in:
      Chambolle, A., & Dossal, C. (2015). On the convergence of the iterates
      of the "fast iterative shrinkage/thresholding algorithm". Journal of
      Optimization Theory and Applications, 166(3), 968-982.

    Returns (H, obj, gradient); obj is the 1-D array of objective values.
    """
    if isempty(t0):
        t0 = 1
    if isempty(alpha):
        alpha = .5
    if isempty(beta):
        beta = 0.8

    if isempty(verbose):
        verbose = 0

    if isempty(stop):
        stop = 1e-5

    if sumFunction is None:
        sumFunction = lambda x, t, y: x + t * y

    if computeNorm is None:
        computeNorm = lambda matr: gather(norm_fro(matr))

    H = H0
    objective, gradient, gradNorm = computeObjGrad(H)

    obj = [gather(objective)]
    checkObj = obj[0]
    niter = 0
    t = t0
    numberEval = 1

    Glast = H
    Hlast = H
    if verbose:
        display('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, obj[-1], t))
        display('\t\tGradnorm=%f stop=%f' % (gradNorm, stop))

    tol = stop * (1 + computeNorm(H))

    while gradNorm > tol:
        niter = niter + 1
        last = obj[-1]
        prevt = t
        t, objective, H, _, _ = backtrack(lambda G: computeObjGrad(G), last, H, gradient,
                                          -alpha * gradNorm, beta, t, sumFunction)
        numberEval = numberEval + math.log(t / prevt) / math.log(beta) + 1

        G = proximal(H)
        H = sumFunction(G, (niter - 2) / (niter + 1), sumFunction(G, -1, Glast))
        objective, gradient, gradNorm = computeObjGrad(H)

        obj.append(gather(objective))
        Glast = G
        numberEval = numberEval + 1

        tol = stop * (1 + computeNorm(H))

        if (niter - 1) % 20 == 0:
            if verbose:
                display('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, obj[-1], t))
                display('\t\tGradnorm=%f tol=%f' % (gradNorm, tol))
            Hlast = H

        if checkObj < obj[-1]:
            niter = 0
            H = Hlast
            objective, gradient, gradNorm = computeObjGrad(H)
            obj.append(gather(objective))
        checkObj = obj[-1]

        t = min([t / math.sqrt(beta), t0])

    if verbose:
        display('\tDual iteration : %d Objective : %f, Current step size %f' % (niter, obj[-1], t))
        display('\t\tGradnorm=%f stop=%f' % (gradNorm, tol))

    return H, np.array(obj, dtype=float), gradient
