from typing import Literal, Union

import numpy as np
def rank(
        data: np.ndarray,
        method: Literal["average", "dense", "ordinal"] = "average",
) -> np.ndarray:
    """
    Rank values in a 1D array, handling ties by the selected method.
    """
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("rank expects a 1D array.")
    if arr.size == 0:
        raise ValueError("rank expects a non-empty array.")

    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be one of: 'average', 'dense', 'ordinal'.")

    order = np.argsort(arr, kind="stable")
    sorted_vals = arr[order]
    n = arr.size

    ranks_sorted = np.empty(n, dtype=float)

    if method == "ordinal":
        ranks_sorted = np.arange(1, n + 1, dtype=float)
    else:
        group_start_mask = np.r_[True, sorted_vals[1:] != sorted_vals[:-1]]
        group_starts = np.flatnonzero(group_start_mask)
        group_ends = np.r_[group_starts[1:], n]
        group_sizes = group_ends - group_starts

        if method == "average":
            group_ranks = (group_starts + 1 + group_ends) / 2.0
        else:
            group_ranks = np.arange(1, group_starts.size + 1, dtype=float)

        ranks_sorted = np.repeat(group_ranks, group_sizes)

    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def percentile(
        data: np.ndarray,
        q: Union[float, np.ndarray],
        interpolation: Literal["linear", "lower", "higher", "midpoint"] = "linear",
) -> Union[float, np.ndarray]:
    """
    Compute percentile(s) for a 1D array.
    """
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("percentile expects a 1D array.")
    if arr.size == 0:
        raise ValueError("percentile expects a non-empty array.")

    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("interpolation must be one of: 'linear', 'lower', 'higher', 'midpoint'.")

    q_arr = np.asarray(q)
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be in the range [0, 100].")

    try:
        return np.percentile(arr, q, method=interpolation)
    except TypeError:
        return np.percentile(arr, q, interpolation=interpolation)
