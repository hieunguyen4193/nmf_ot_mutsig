"""Translation of toolbox/gaussianConv.m"""

from scipy import ndimage


def gaussianConv(M, filters):
    """Apply each filter in turn, like imfilter(M, f, 'same', 'replicate').

    imfilter performs correlation (not convolution) by default, with the
    kernel centre at floor((size+1)/2); for the odd-sized kernels built in
    buildMultipliers this is the default centre of ndimage.correlate.
    'replicate' padding is mode='nearest'. A filter with fewer dimensions
    than M is padded with trailing singleton dimensions, as in MATLAB.
    """
    for i in range(len(filters)):
        f = filters[i]
        f = f.reshape(f.shape + (1,) * (M.ndim - f.ndim))
        M = ndimage.correlate(M, f, mode='nearest')

    return M
