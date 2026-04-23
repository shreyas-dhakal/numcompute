from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np


def mean(data: np.ndarray, axis: Optional[int] = None, skipna: bool = False) -> Union[float, np.ndarray]:
    """Compute mean with optional NaN skipping."""
    arr = np.asarray(data)
    if skipna:
        return np.nanmean(arr, axis=axis)
    return np.mean(arr, axis=axis)


def median(data: np.ndarray, axis: Optional[int] = None, skipna: bool = False) -> Union[float, np.ndarray]:
    """Compute median with optional NaN skipping."""
    arr = np.asarray(data)
    if skipna:
        return np.nanmedian(arr, axis=axis)
    return np.median(arr, axis=axis)


def std(
        data: np.ndarray,
        axis: Optional[int] = None,
        ddof: int = 0,
        skipna: bool = False,
) -> Union[float, np.ndarray]:
    """Compute standard deviation with optional NaN skipping."""
    arr = np.asarray(data)
    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)
    return np.std(arr, axis=axis, ddof=ddof)


def min(data: np.ndarray, axis: Optional[int] = None, skipna: bool = False) -> Union[float, np.ndarray]:
    """Compute minimum with optional NaN skipping."""
    arr = np.asarray(data)
    if skipna:
        return np.nanmin(arr, axis=axis)
    return np.min(arr, axis=axis)


def max(data: np.ndarray, axis: Optional[int] = None, skipna: bool = False) -> Union[float, np.ndarray]:
    """Compute maximum with optional NaN skipping."""
    arr = np.asarray(data)
    if skipna:
        return np.nanmax(arr, axis=axis)
    return np.max(arr, axis=axis)


def histogram(
        data: np.ndarray,
        bins: Union[int, Sequence[float]] = 10,
        range: Optional[Tuple[float, float]] = None,
        density: bool = False,
        skipna: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram for the input data."""
    arr = np.asarray(data)
    if arr.ndim > 1:
        arr = arr.ravel()

    if skipna and np.issubdtype(arr.dtype, np.number):
        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        raise ValueError("histogram expects at least one valid value.")

    return np.histogram(arr, bins=bins, range=range, density=density)


def quantile(
        data: np.ndarray,
        q: Union[float, np.ndarray],
        axis: Optional[int] = None,
        interpolation: Literal["linear", "lower", "higher", "midpoint"] = "linear",
        skipna: bool = False,
) -> Union[float, np.ndarray]:
    """Compute quantile(s) with optional NaN skipping."""
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("interpolation must be one of: 'linear', 'lower', 'higher', 'midpoint'.")

    q_arr = np.asarray(q)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in the range [0, 1].")

    arr = np.asarray(data)

    quantile_fn = np.nanquantile if skipna else np.quantile
    try:
        return quantile_fn(arr, q, axis=axis, method=interpolation)
    except TypeError:
        return quantile_fn(arr, q, axis=axis, interpolation=interpolation)
