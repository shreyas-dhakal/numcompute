import numpy as np
from typing import Callable, Literal
from typing import Iterator, Tuple


def euchlidean_distance(a, b):
    """
    Compute the Euclidean distance between two 1D arrays.
    """
    return np.sqrt(np.sum((a - b) ** 2))


def pairwise_euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between rows of X and Y."""
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError("X and Y must have the same feature dimension.")

    x2 = np.sum(X_arr ** 2, axis=1, keepdims=True)
    y2 = np.sum(Y_arr ** 2, axis=1, keepdims=True).T
    sq = np.maximum(x2 + y2 - 2.0 * (X_arr @ Y_arr.T), 0.0)
    return np.sqrt(sq)



def manhattan_distance(a, b):
    """
    Compute the Manhattan distance between two 1D arrays.
    """
    return np.sum(np.abs(a - b))


def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two 1D arrays.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def jaccard_similarity(a, b):
    """
    Compute the Jaccard similarity between two 1D binary arrays.
    """
    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))
    if union == 0:
        return 0.0
    return intersection / union


def hamming_distance(a, b):
    """
    Compute the Hamming distance between two 1D arrays.
    """
    return np.sum(a != b)


def relu(x):
    """
    Apply the ReLU activation function to a 1D array.
    """
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along axis."""
    x_arr = np.asarray(x, dtype=float)
    shift = x_arr - np.max(x_arr, axis=axis, keepdims=True)
    exps = np.exp(shift)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x_arr = np.asarray(x, dtype=float)
    m = np.max(x_arr, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x_arr - m), axis=axis, keepdims=True))
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)


def topk_indices(values: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Return sorted top-k indices using argpartition."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if k <= 0 or k > arr.size:
        raise ValueError("k must be within [1, len(values)].")

    if largest:
        pivot = arr.size - k
        idx = np.argpartition(arr, pivot)[pivot:]
        return idx[np.argsort(-arr[idx], kind="stable")]
    idx = np.argpartition(arr, k - 1)[:k]
    return idx[np.argsort(arr[idx], kind="stable")]



def batch_slices(n_samples: int, batch_size: int, drop_last: bool = False) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) slice boundaries for mini-batches."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    start = 0
    while start < n_samples:
        end = min(start + batch_size, n_samples)
        if drop_last and (end - start) < batch_size:
            break
        yield start, end
        start = end