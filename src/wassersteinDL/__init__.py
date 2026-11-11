"""Python (NumPy/SciPy) translation of the Wasserstein Dictionary Learning toolbox.

Line-by-line translation of the MATLAB code accompanying

    Antoine Rolet, Marco Cuturi and Gabriel Peyre. Fast Dictionary Learning
    with a Smoothed Wasserstein Loss. AISTATS 2016.

Each module corresponds to the .m file of the same name in ../toolbox.
Original code (c) Antoine Rolet 2016, Mozilla Public License 2.0.
"""

from .accelerated_gradient import accelerated_gradient
from .backtrack import backtrack
from .buildMultipliers import buildMultipliers
from .checkOptionsWasserteinProjection import checkOptionsWasserteinProjection
from .computeWassersteinLegendre import computeWassersteinLegendre
from .computeWassersteinLegendreWeighted import computeWassersteinLegendreWeighted
from .gaussianConv import gaussianConv
from .isHistogram import isHistogram
from .linear_projected_gradient_descent import linear_projected_gradient_descent
from .matrixEntropy import matrixEntropy
from .wasserstein_DL import wasserstein_DL
from .wasserstein_DL_coefficient_step import wasserstein_DL_coefficient_step
from .wasserstein_DL_dictionary_step import wasserstein_DL_dictionary_step
from .wasserstein_NMF_coefficient_step import wasserstein_NMF_coefficient_step
from .wasserstein_NMF_dictionary_step import wasserstein_NMF_dictionary_step

__all__ = [
    'accelerated_gradient', 'backtrack', 'buildMultipliers',
    'checkOptionsWasserteinProjection', 'computeWassersteinLegendre',
    'computeWassersteinLegendreWeighted', 'gaussianConv', 'isHistogram',
    'linear_projected_gradient_descent', 'matrixEntropy', 'wasserstein_DL',
    'wasserstein_DL_coefficient_step', 'wasserstein_DL_dictionary_step',
    'wasserstein_NMF_coefficient_step', 'wasserstein_NMF_dictionary_step',
]
