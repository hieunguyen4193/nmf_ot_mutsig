"""Translation of toolbox/wasserstein_DL.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.

The Initial Developers of the Original Code is Antoine Rolet.
"""

import numpy as np

from ._external import fkmeans, nnmf, pairwiseDistance
from ._matlab import display, isempty, isfield, num2str
from .isHistogram import isHistogram
from .matrixEntropy import matrixEntropy
from .wasserstein_DL_coefficient_step import wasserstein_DL_coefficient_step
from .wasserstein_DL_dictionary_step import wasserstein_DL_dictionary_step
from .wasserstein_NMF_coefficient_step import wasserstein_NMF_coefficient_step
from .wasserstein_NMF_dictionary_step import wasserstein_NMF_dictionary_step


def wasserstein_DL(X, p, M, gamma, rhoL=None, rhoD=None, options=None, initialValues=None,
                   rng=None):
    """Dictionary learning with regularized wasserstein cost.

    This function solves

          min_{lambda,D} W_gamma(X,D*lambda) + rhoL*E(lambda) + rhoD*E(D)

    where E is the sum of entropy on the columns. If rho_1, rho_2 > 0,
    this performs NMF. Otherwise D or lambda can have negative values, but
    not D*lambda.

    Inputs:
          - X, the input data (n-by-m matrix)
          - p, the number of dictionary elements to learn (scalar)
          - M, the ground metric (n-by-t matrix). If M is symmetric, consider
          using options['Kmultiplication']='symmetric'. If M is a squared
          Euclidean distance on a regular grid, consider using
          options['Kmultiplication']='convolution' (see optional inputs for more
          information)
          - gamma, the regularization parameter of Wasserstein (scalar)

    Optional inputs:
          - rhoL, the regularization parameter for lambda (scalar). Default 0
          - rhoD, the regularization parameter for D (scalar). Default 0
          - options, an option dict:
              * 'verbose': verbose level, 0 (default) for no output, 1
              for an output at each outer loop iteration, 2 for additional
              output at each inner loop iteration.
              * 't0': the initial step size for gradient updates. Default 1
              * 'alpha': parameter for the backtracking line search.
              * 'beta': parameter for the backtracking line search. Default .8
              * 'stop': outer stopping criterion. Default 1e-3
              * 'D_step_stop': inner stopping criterion for the D step.
              Default 1e-5
              * 'lambda_step_stop': inner stopping criterion for the lambda
              step. Default 1e-5
              * 'bigMatrices': if true K' is not precomputed. Default false.
              * 'weights': weights on the wasserstein distance (length m).
              Defaults to None which is equivalent to ones(1,m).
              * 'Kmultiplication': 'symmetric', 'asymmetric' or 'convolution'.
              For 'convolution', M should be a dict with keys
              'grid_dimensions' and 'kernelSize'.
              * 'GPU': must be 0 (GPU is not supported in the translation).
              * 'plothandler': called as plothandler(objL, objD, objectives, D)
              after each outer iteration when verbose.
          - initialValues, a dict containing starting points or
          initialization method:
              * 'D': the initial dictionary (t-by-p matrix)
              * 'isInitLambda': if 'D' is present and this is true, then 'D'
              holds the initial coefficients (p-by-m matrix)
              * 'method': if no 'D', which initialization method to use:
              'random' (default), 'fromData', 'kmeans' or 'euclidean'
              * 'HD', 'Hlambda': initial dual variables
          - rng, a numpy.random.Generator used for random initialisations
          (stands in for MATLAB's global random stream).

    Outputs (D, lambda, objectives, HD, Hlambda):
          - D: the optimal dictionary (t-by-p matrix)
          - lambda: the optimal coefficients (p-by-m matrix)
          - objectives: the sequence of objective values
          - HD: the dual optimal variable for the D step (t-by-m matrix).
          - Hlambda: the dual optimal variable (t-by-m matrix)
    """
    if options is None:
        options = {}
    if rng is None:
        rng = np.random.default_rng()

    options = checkOptions(options)

    verbose = options['verbose']
    if verbose:
        options['verbose'] = verbose - 1

    if not isHistogram(X):
        raise ValueError('Input vectors are not histograms. Check that they are non-negative, sum to one and do not contain NaN of inf')

    # Load GPU device if applicable
    if options['GPU']:
        raise NotImplementedError('options.GPU is not supported by the Python translation')

    # Create optimizers
    if isempty(rhoL):
        rhoL = 0
    if isempty(rhoD):
        rhoD = 0
    optimizeLambda, optimizeD, options, sizeD = createHandlers(options, X, M, p, gamma, rhoL, rhoD)

    # Create or assign starting values
    if isempty(initialValues):
        initialValues = {}
    D, HD, Hlambda = initialValue(options, optimizeD, X, p, sizeD, initialValues, rng)

    del initialValues
    biglast = 0
    bigobj = -1
    objectives = []
    niter = 0
    lambda_ = None

    # Loop that optimizes alternatively with D or lambda fixed until
    # convergence
    while abs(bigobj - biglast) > options['stop'] * (1 + abs(bigobj)):
        niter = niter + 1

        if verbose:
            display('Optimize with respect to lambda')

        # Optimize with respect to lambda, D is fixed
        lambda_, Hlambda, objL = optimizeLambda(D, Hlambda)

        # Store objective values
        biglast = bigobj
        tmpObj = objL[-1]
        if rhoD > 0:
            tmpObj = tmpObj - rhoD * matrixEntropy(D)
        objectives.append(tmpObj)

        if verbose:
            display('Optimize with respect to D')

        # Optimize with respect to D, lambda is fixed
        D, HD, objD, _ = optimizeD(lambda_, HD)

        # Store objective values
        tmpObj = objD[-1]
        if rhoL > 0:
            tmpObj = tmpObj - rhoL * matrixEntropy(lambda_)
        else:
            sumD = np.sum(np.abs(D), axis=0, keepdims=True)
            sumD[sumD == 0] = 1
            D = D / sumD
        objectives.append(tmpObj)

        if verbose:
            display('Outer iteration : ' + num2str(niter) + ' Objective : ' + num2str(objectives[-1]) + ' gamma=' + num2str(gamma))
            options['plothandler'](objL, objD, np.array(objectives), D)

        bigobj = objectives[-1]

    return D, lambda_, np.array(objectives, dtype=float), HD, Hlambda


def checkOptions(options):
    options = dict(options)   # MATLAB structs are passed by value

    if not isfield(options, 'stop') or isempty(options['stop']):
        options['stop'] = 1e-3

    if not isfield(options, 'D_step_stop') or isempty(options['D_step_stop']):
        options['D_step_stop'] = 1e-5

    if not isfield(options, 'lambda_step_stop') or isempty(options['lambda_step_stop']):
        options['lambda_step_stop'] = 1e-5

    if not isfield(options, 'alpha') or isempty(options['alpha']):
        options['alpha'] = None
    if not isfield(options, 'beta') or isempty(options['beta']):
        options['beta'] = None

    if not isfield(options, 't0') or isempty(options['t0']):
        options['t0'] = None

    if not isfield(options, 'verbose') or isempty(options['verbose']):
        options['verbose'] = 0
    if not isfield(options, 'GPU') or isempty(options['GPU']):
        options['GPU'] = 0

    if not isfield(options, 'bigMatrices') or isempty(options['bigMatrices']):
        options['bigMatrices'] = 0

    if options['GPU']:
        raise NotImplementedError('options.GPU is not supported by the Python translation')
    else:
        options['createZeroArray'] = lambda siz: np.zeros(siz)

    if not isfield(options, 'plothandler') or isempty(options['plothandler']):
        options['plothandler'] = lambda obj1, obj2, obj3, D: 1

    return options


def createHandlers(options, X, M, p, gamma, rhoL, rhoD):
    """Create the solvers for sub-problems on either D or lambda."""
    options = dict(options)

    if not isfield(options, 'Kmultiplication') or (options['Kmultiplication'] != 'convolution' and options['Kmultiplication'] != 'graphLapalacian'):
        K = np.exp(-M / gamma)
        K[K < 1e-200] = 1e-200
        options['convolution'] = False
        if (K.shape[0] == K.shape[1]) and (np.sum(np.abs(K - K.T)) / np.sum(K) < 1e-5):
            options['Kmultiplication'] = 'symmetric'
            sizeD = X.shape
        else:
            options['Kmultiplication'] = 'asymmetric'
            sizeD = (K.shape[1], X.shape[1])
    elif options['Kmultiplication'] == 'convolution':
        M = dict(M)
        grid_dimensions = np.atleast_1d(M['grid_dimensions'])
        if np.prod(grid_dimensions) != X.shape[0]:
            raise ValueError('When using a convolution to compute the gradient, M.grid_dimensions should be the grid dimensions of the original data the product of its components should be size(X,1)')
        kernelSize = np.atleast_1d(M['kernelSize'])
        if kernelSize.size == 1:
            M['kernelSize'] = kernelSize * np.ones(grid_dimensions.size)
        if np.any(np.atleast_1d(M['kernelSize']) > grid_dimensions):
            raise ValueError('Kernel size sould be smaller than dimension')
        K = M
        sizeD = X.shape
    elif options['Kmultiplication'] == 'graphLapalacian':
        K = M
        sizeD = X.shape

    # Precompute the entropy of X
    options['pX'] = matrixEntropy(X)

    optionsD = dict(options)
    options['dual_descent_stop'] = options['lambda_step_stop']
    optionsD['dual_descent_stop'] = options['D_step_stop']

    # MATLAB anonymous functions capture values at creation time; freeze copies.
    optionsL = dict(options)

    # Create solvers depending on options.algorithms
    if rhoL:
        optimizeLambda = lambda D, Hlambda: wasserstein_NMF_coefficient_step(X, K, D, gamma, rhoL, Hlambda, optionsL, 1)[:3]
    else:
        optimizeLambda = lambda D, Hlambda: wasserstein_DL_coefficient_step(X, K, D, gamma, Hlambda, optionsL, 1)[:3]
    if rhoD:
        optimizeDcomplete = lambda lambda_, HD: wasserstein_NMF_dictionary_step(X, K, lambda_, gamma, rhoD, HD, optionsD, 1)[:3]
    else:
        optimizeDcomplete = lambda lambda_, HD: wasserstein_DL_dictionary_step(X, K, lambda_, gamma, HD, optionsD, 1)[:3]
    optimizeD = lambda lambda_, HD: optimizeDWithUnused(lambda_, HD, optimizeDcomplete, p)

    return optimizeLambda, optimizeD, options, tuple(sizeD)


def initialValue(options, optimizeD, X, p, sizeD, initialValues, rng):
    # NOTE: in the MATLAB original, HD/Hlambda are left unassigned when they
    # are supplied in initialValues (which would make MATLAB error out). The
    # evident intent, using the supplied values, is implemented here.
    if not isfield(initialValues, 'HD'):
        HD = options['createZeroArray'](sizeD)
    else:
        HD = initialValues['HD']
    if not isfield(initialValues, 'Hlambda'):
        Hlambda = options['createZeroArray'](sizeD)
    else:
        Hlambda = initialValues['Hlambda']
    if not isfield(initialValues, 'D') or isempty(initialValues['D']):   # No initial value provided
        if not isfield(initialValues, 'method') or initialValues['method'] == 'random':
            D = randomInitializationD(options, sizeD, p, rng)
        elif initialValues['method'] == 'fromData':
            if sizeD[0] != X.shape[0]:
                raise ValueError('Initialization from data is only possible when the dictionary has the same dimension as the input')
            D = randomInitializationFromData(X, options, sizeD, p, rng)
        elif initialValues['method'] == 'kmeans':
            _, D = fkmeans(X.T, p, rng)
            dist = pairwiseDistance(D.T, X)
            lambda_ = np.exp(-dist)
            lambda_ = lambda_ / np.sum(lambda_, axis=0, keepdims=True)
            D, _ = initializeFromLambda(lambda_, p, optimizeD, X, HD, options, rng)
        elif initialValues['method'] == 'euclidean':
            _, lambda_ = nnmf(X, p, rng)
            lambda_ = lambda_ / np.sum(lambda_, axis=0, keepdims=True)
            D, _ = initializeFromLambda(lambda_, p, optimizeD, X, HD, options, rng)
        else:
            raise ValueError('initialValues.method should be either "kmeans", "euclidean" or "random"')
    else:                        # Initial value provided
        D = np.array(initialValues['D'], dtype=float)

        if isfield(initialValues, 'isInitLambda') and initialValues['isInitLambda']:   # Initial value provided is lambda
            # NOTE: the MATLAB original passes an extra sizeD argument here,
            # which errors with "Too many input arguments"; the intended call
            # is made instead.
            D, _ = initializeFromLambda(D, p, optimizeD, X, HD, options, rng)

    return D, HD, Hlambda


def initializeFromLambda(lambda_, p, optimizeD, X, HD, options, rng):
    D, HD, obj, Qind = optimizeD(lambda_, HD)

    hasUnused = np.sum(Qind) < p
    if hasUnused:
        # Initialize columns of D that couldn't be optimized
        # NOTE: m is undefined in the MATLAB original; m=size(X,2) is meant.
        m = X.shape[1]
        if D.shape[1] < p:   # MATLAB grows D with zero columns on assignment
            D = np.hstack([D, np.zeros((D.shape[0], p - D.shape[1]))])
        initQ = rng.permutation(m)
        D[:, ~Qind] = X[:, initQ[:(p - int(np.sum(Qind)))]] + 1e-7
        D = D / np.sum(D, axis=0, keepdims=True)
    return D, HD


def randomInitializationFromData(X, options, sizeD, p, rng):
    initIndices = rng.permutation(sizeD[1])[:p]
    D = X[:, initIndices]
    return D


def optimizeDWithUnused(lambda_, HD, optimizeD, p):
    Qind = np.sum(np.abs(lambda_), axis=1) > 0
    hasUnused = np.sum(Qind) < p
    if hasUnused:                   # Some rows of lambda are always 0, corresponding columns of D cannot be optimized

        tmpLambda = lambda_[Qind, :]

        Dpart, HD, obj = optimizeD(tmpLambda, HD)
        # MATLAB creates D on assignment D(:,Qind)=Dpart, zero-filled up to
        # the last true index of Qind.
        last = int(np.nonzero(Qind)[0][-1]) + 1 if np.any(Qind) else 0
        D = np.zeros((Dpart.shape[0], last))
        D[:, Qind[:last]] = Dpart

    else:
        D, HD, obj = optimizeD(lambda_, HD)
    return D, HD, obj, Qind


def randomInitializationD(options, sizeD, p, rng):
    D = rng.integers(1, 100000, size=(sizeD[0], p), endpoint=True).astype(float)
    D = D / np.sum(D, axis=0, keepdims=True)
    return D
