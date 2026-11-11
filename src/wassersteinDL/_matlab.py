"""Small NumPy equivalents of MATLAB built-ins used by the toolbox.

Each helper reproduces the MATLAB semantics that matter for the
translation (tolerances, edge cases), so the translated toolbox files can
stay line-by-line close to the original .m files.
"""

import math

import numpy as np

EPS = np.finfo(float).eps  # eps(1.0) in MATLAB


def gather(x):
    """MATLAB gather(): a no-op for CPU arrays."""
    return x


def norm_fro(A):
    """MATLAB norm(A,'fro')."""
    return float(np.sqrt(np.sum(np.abs(A) ** 2)))


def pinv(A):
    """MATLAB pinv(A) with its default tolerance max(size(A))*norm(A)*eps."""
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * (s[0] if s.size else 0.0) * EPS
    r = int(np.sum(s > tol))
    if r == 0:
        return np.zeros((A.shape[1], A.shape[0]))
    return (Vt[:r, :].T / s[:r]) @ U[:, :r].T


def mldivide(A, B):
    """MATLAB A\\B: exact solve for square A, least squares otherwise."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape[0] == A.shape[1]:
        return np.linalg.solve(A, B)
    return np.linalg.lstsq(A, B, rcond=None)[0]


def mrdivide(B, A):
    """MATLAB B/A, i.e. (A'\\B')'."""
    return mldivide(np.asarray(A).T, np.asarray(B).T).T


def isreal(x):
    """MATLAB isreal(): False only for complex-typed values."""
    return not np.iscomplexobj(x)


def num2str(x):
    """Approximation of MATLAB num2str for a real scalar."""
    x = float(x)
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Inf' if x > 0 else '-Inf'
    if x == round(x) and abs(x) < 1e15:
        return '%d' % int(round(x))
    digits = math.floor(math.log10(abs(x))) if x != 0 else 0
    precision = min(max(digits + 5, 5), 16)
    return '%.*g' % (precision, x)


def display(s):
    """MATLAB display() of a character vector."""
    print(s)


def isfield(s, name):
    """MATLAB isfield(s, name) for an options dict."""
    return isinstance(s, dict) and name in s


def isempty(v):
    """MATLAB isempty(v); None stands in for MATLAB's []."""
    if v is None:
        return True
    if isinstance(v, np.ndarray):
        return v.size == 0
    if isinstance(v, (list, tuple, str, dict)):
        return len(v) == 0
    return False
