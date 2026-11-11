"""Translation of toolbox/checkOptionsWasserteinProjection.m

This code is (c) Antoine Rolet 2016.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import numpy as np

from ._matlab import isempty, isfield


def checkOptionsWasserteinProjection(options):
    """Check the optional inputs of Wasserstein projection.

    MATLAB structs are passed by value, so the dict is copied first.
    """
    options = dict(options)
    if not isfield(options, 't0') or isempty(options['t0']):
        options['t0'] = 1
    if not isfield(options, 'alpha'):
        options['alpha'] = None
    if not isfield(options, 'beta') or isempty(options['beta']):
        options['beta'] = .8

    if not isfield(options, 'verbose') or isempty(options['verbose']):
        options['verbose'] = 0

    if not isfield(options, 'dual_descent_stop') or isempty(options['dual_descent_stop']):
        options['dual_descent_stop'] = 1e-5
    if not isfield(options, 'bigMatrices') or isempty(options['bigMatrices']):
        options['bigMatrices'] = False
    if not isfield(options, 'weights'):
        options['weights'] = None
    if isfield(options, 'GPU') and options['GPU']:
        raise NotImplementedError('options.GPU is not supported by the Python translation')
    else:
        options['createZeroArray'] = lambda d: np.zeros(d)
    return options
