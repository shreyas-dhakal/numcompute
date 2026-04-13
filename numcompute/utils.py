import numpy as np


def euchlidean_distance(a, b):
    """
    Compute the Euclidean distance between two 1D arrays.
    """
    return np.sqrt(np.sum((a - b) ** 2))


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


def sigmoid(x):
    """
    Apply the sigmoid activation function to a 1D array.
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Apply the softmax function to a 1D array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def logsumexp(x):
    """
    Compute the log-sum-exp of a 1D array for numerical stability.
    """
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))
