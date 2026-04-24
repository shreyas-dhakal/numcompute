"""
stats.py

Statistical utilities with emphasis on:
- Vectorisation
- Numerical stability
- Streaming (Welford)
- Robust handling of NaNs

Functions:
- mean
- variance
- welford (streaming mean/variance)
- histogram
- quantile
"""

import numpy as np


# -----
# Mean
# -----

def mean(x, axis=None, ignore_nan=True):
    """
    Compute mean.

    Parameters:
        x : np.ndarray
        axis : int or None
        ignore_nan : bool

    Returns:
        np.ndarray or float
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Empty array")

    if ignore_nan:
        return np.nanmean(x, axis=axis)
    else:
        return np.mean(x, axis=axis)


# ---------
# Variance
# ---------

def variance(x, axis=None, ddof=0, ignore_nan=True):
    """
    Compute variance.

    Parameters:
        x : np.ndarray
        axis : int or None
        ddof : int (0 = population, 1 = sample)
        ignore_nan : bool

    Returns:
        np.ndarray or float
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Empty array")

    if ignore_nan:
        return np.nanvar(x, axis=axis, ddof=ddof)
    else:
        return np.var(x, axis=axis, ddof=ddof)


# ------------------------------------
# Welford (Streaming Mean & Variance)
# ------------------------------------

def welford(x):
    """
    Compute mean and variance using Welford’s algorithm.

    Parameters:
        x : iterable or np.ndarray

    Returns:
        mean : float
        variance : float (sample variance)
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("Empty input")

    mean = 0.0
    M2 = 0.0
    n = 0

    for value in x:
        if np.isnan(value):
            continue

        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean
        M2 += delta * delta2

    if n < 2:
        return mean, 0.0

    variance = M2 / (n - 1)
    return mean, variance


# ----------
# Histogram
# ----------

def histogram(x, bins=10, range=None):
    """
    Compute histogram.

    Parameters:
        x : np.ndarray
        bins : int
        range : tuple (min, max)

    Returns:
        hist : counts
        bin_edges : array
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Empty array")

    hist, bin_edges = np.histogram(x, bins=bins, range=range)
    return hist, bin_edges


# --------
# Quantile
# --------
def quantile(x, q, axis=None):
    """
    Compute quantiles.

    Parameters:
        x : np.ndarray
        q : float or array-like in [0, 1]
        axis : int or None

    Returns:
        quantiles
    """
    x = np.asarray(x)

    if x.size == 0:
        raise ValueError("Empty array")

    if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
        raise ValueError("q must be in [0, 1]")

    return np.quantile(x, q, axis=axis)
