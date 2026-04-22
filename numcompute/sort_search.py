import numpy as np
from typing import Sequence, Tuple, Union

def stable_sort(data, axis=-1):
    """
    Perform a stable sort on a NumPy array.
    """
    data = np.sort(data, axis=axis, kind="stable")
    return data


def multi_key_sort(
        data: np.ndarray,
        keys: Sequence[int],
        return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stable multi-key sort for 2D arrays by column index priority.
    Keys are interpreted from highest to lowest priority. Example:
    keys=[0, 2] means sort by column 0 first, then break ties by column 2.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("multi_key_sort expects a 2D array.")
    if len(keys) == 0:
        raise ValueError("keys must contain at least one column index.")

    n_rows, n_cols = arr.shape
    indices = np.arange(n_rows)

    for key in reversed(keys):
        if key < 0 or key >= n_cols:
            raise IndexError(f"Column index {key} is out of bounds for shape {arr.shape}.")
        order = np.argsort(arr[indices, key], kind="stable")
        indices = indices[order]

    sorted_data = arr[indices]
    if return_indices:
        return sorted_data, indices
    return sorted_data



def topk(values, k, largest=True, return_indices=True):
    """
    Return the top-k elements from a 1D array.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("topk expects a 1D array.")
    if k <= 0 or k > values.size:
        raise ValueError("k must be between 1 and len(values)")

    if largest:
        partition_idx = values.size - k
        idx = np.argpartition(values, partition_idx)[partition_idx:]
        order = np.argsort(values[idx])[::-1]
    else:
        partition_idx = k
        idx = np.argpartition(values, partition_idx)[:k]
        order = np.argsort(values[idx])

    idx = idx[order]

    if return_indices:
        return values[idx], idx
    return values[idx]


def quickselect(values: np.ndarray, k: int, largest: bool = False) -> float:
    """
    Educational in-place quickselect for 1D arrays.
    k is a zero-based rank. With largest=False, k=0 is minimum.
    With largest=True, k=0 is maximum.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")
    if k < 0 or k >= arr.size:
        raise ValueError(f"k must be in [0, {arr.size - 1}], got {k}.")

    work = arr.copy()
    target = arr.size - 1 - k if largest else k

    left = 0
    right = work.size - 1

    while True:
        if left == right:
            return float(work[left])

        pivot = work[(left + right) // 2]
        i, j = left, right

        while i <= j:
            while work[i] < pivot:
                i += 1
            while work[j] > pivot:
                j -= 1
            if i <= j:
                work[i], work[j] = work[j], work[i]
                i += 1
                j -= 1

        if target <= j:
            right = j
        elif target >= i:
            left = i
        else:
            return float(work[target])


def binary_search(sorted_array: np.ndarray, x: float) -> Tuple[int, bool]:
    """
    Binary search on a sorted 1D array.
    Returns (insertion_index, exists).
    """
    arr = np.asarray(sorted_array)
    if arr.ndim != 1:
        raise ValueError("binary_search expects a 1D array.")

    idx = int(np.searchsorted(arr, x, side="left"))
    exists = idx < arr.size and arr[idx] == x
    return idx, bool(exists)