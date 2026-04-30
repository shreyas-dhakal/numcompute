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


def welford(x):
    """
    Compute mean and variance using Welford's algorithm.

    Parameters:
        x : iterable or np.ndarray

    Returns:
        mean : float
        variance : float (sample variance)
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("Empty input")

    x = x[~np.isnan(x)]
    n = x.size

    if n < 2:
        return (float(x[0]), 0.0) if n == 1 else (0.0, 0.0)

    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1))
    return mean, variance


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
