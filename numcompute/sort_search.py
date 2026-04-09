import numpy as np


def stable_sort(data):
    """
    Perform a stable sort on a 1D NumPy array.
    """
    data = np.sort(data, kind="stable")
    return data


def multi_key_sort(data, keys):
    """
    Perform a multi-key sort on a 2D NumPy array.
    """
    for key in reversed(keys):
        data = data[data[:, key].argsort(kind="stable")]
    return data


def topk(values, k, largest=True, return_indices=True):
    """
    Return the top-k elements from a 1D array.
    """
    values = np.asarray(values)
    if k <= 0 or k > values.size:
        raise ValueError("invalid k value (must be between 1 and len(values))")

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
