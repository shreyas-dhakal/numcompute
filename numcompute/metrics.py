from typing import Optional, Sequence, Tuple
import numpy as np


def _validate_1d_pair(y_true: np.ndarray, y_pred: np.ndarray, name_pred: str = "y_pred") -> Tuple[np.ndarray, np.ndarray]:
    """Validate two 1D arrays with matching shapes.

    Parameters
    y_true : np.ndarray
        Ground truth values of shape (n,).
    y_pred : np.ndarray
        Predicted values of shape (n,).
    name_pred : str, optional
        Name for y_pred used in error messages.

    Returns
    Tuple[np.ndarray, np.ndarray]
        Tuple of validated arrays (y_true, y_pred), both shape (n,).

    Raises
    ValueError
        If inputs are not 1D, empty, or have mismatched lengths.

    Time Complexity
    O(n)

    Space Complexity
    O(n), due to array conversion.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true and {name_pred} must have the same length.")
    return yt, yp


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Parameters
    y_true : np.ndarray
        True labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).

    Returns
    float
        Accuracy score in [0, 1].

    Raises
    ValueError
        If input validation fails.

    Time Complexity
    O(n)

    Space Complexity
    O(n)
    """
    yt, yp = _validate_1d_pair(y_true, y_pred)
    return float(np.mean(yt == yp))


def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[Sequence] = None,
) -> np.ndarray:
    """Compute the confusion matrix.

    Parameters
    y_true : np.ndarray
        True labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    labels : Sequence, optional
        List of label values to index the matrix. If None, inferred
        from sorted unique values in y_true and y_pred.

    Returns
    np.ndarray
        Confusion matrix of shape (k, k), where k is number of labels.
        Entry (i, j) counts instances with true label i and predicted label j.

    Raises
    ValueError
        If inputs are invalid or labels are malformed.

    Time Complexity
    O(n + k), where n is number of samples and k is number of labels.

    Space Complexity
    O(k^2)
    """
    yt, yp = _validate_1d_pair(y_true, y_pred)

    if labels is None:
        label_values = np.unique(np.concatenate([yt, yp]))
    else:
        label_values = np.asarray(labels)
        if label_values.ndim != 1 or label_values.size == 0:
            raise ValueError("labels must be a non-empty 1D sequence when provided.")

    n = label_values.size
    matrix = np.zeros((n, n), dtype=int)
    index = {label: i for i, label in enumerate(label_values.tolist())}

    for t, p in zip(yt, yp):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1

    return matrix


def precision(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1, zero_division: float = 0.0) -> float:
    """
    Compute precision for binary classification.

    Parameters
    y_true : np.ndarray
        True labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    pos_label : Any, optional
        Label considered as positive class.
    zero_division : float, optional
        Value to return when denominator is zero.

    Returns
    float
        Precision score in [0, 1].

    Raises
    ValueError
        If input validation fails.

    Time Complexity
    O(n)

    Space Complexity
    O(n)
    """

    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fp = np.sum((yt != pos_label) & (yp == pos_label))
    denom = tp + fp
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def recall(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1, zero_division: float = 0.0) -> float:
    """
    Compute recall for binary classification.

    Parameters
    y_true : np.ndarray
        True labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    pos_label : Any, optional
        Label considered as positive class.
    zero_division : float, optional
        Value to return when denominator is zero.

    Returns
    float
        Recall score in [0, 1].

    Raises
    ValueError
        If input validation fails.

    Time Complexity
    O(n)

    Space Complexity
    O(n)
    """
    yt, yp = _validate_1d_pair(y_true, y_pred)
    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fn = np.sum((yt == pos_label) & (yp != pos_label))
    denom = tp + fn
    if denom == 0:
        return float(zero_division)
    return float(tp / denom)


def f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1, zero_division: float = 0.0) -> float:
    """
    Compute F1 score for binary classification.

    Parameters
    y_true : np.ndarray
        True labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    pos_label : Any, optional
        Label considered as positive class.
    zero_division : float, optional
        Value to return when precision + recall is zero.

    Returns
    float
        F1 score in [0, 1].

    Raises
    ValueError
        If input validation fails.

    Time Complexity
    O(n)

    Space Complexity
    O(n)
    """

    p = precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    r = recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    denom = p + r
    if denom == 0.0:
        return float(zero_division)
    return float(2.0 * p * r / denom)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.

    Parameters
    y_true : np.ndarray
        True values of shape (n,).
    y_pred : np.ndarray
        Predicted values of shape (n,).

    Returns
    float
        Mean squared error.

    Raises
    ValueError
        If input validation fails.

    Time Complexity
    O(n)

    Space Complexity
    O(n)
    """

    yt, yp = _validate_1d_pair(y_true, y_pred)
    diff = yt.astype(float) - yp.astype(float)
    return float(np.mean(diff ** 2))


def roc_curve(y_true: np.ndarray, y_score: np.ndarray, pos_label=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters
    y_true : np.ndarray
        True binary labels of shape (n,).
    y_score : np.ndarray
        Predicted scores/probabilities of shape (n,).
    pos_label : Any, optional
        Label considered as positive class.

    Returns
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - fpr : False positive rates of shape (k,)
        - tpr : True positive rates of shape (k,)
        - thresholds : Thresholds of shape (k,)

    Raises
    ValueError
        If inputs are invalid or dataset lacks positive/negative samples.

    Time Complexity
    O(n log n), dominated by sorting.

    Space Complexity
    O(n)
    """
    yt, ys = _validate_1d_pair(y_true, y_score, name_pred="y_score")
    ys = ys.astype(float)

    y_pos = yt == pos_label
    positives = int(np.sum(y_pos))
    negatives = int(yt.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("roc_curve requires both positive and negative samples.")

    order = np.argsort(-ys, kind="stable")
    ys_sorted = ys[order]
    y_sorted = y_pos[order]

    distinct_idx = np.where(np.diff(ys_sorted))[0]
    threshold_idx = np.r_[distinct_idx, yt.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, ys_sorted[threshold_idx]]

    tpr = tps / positives
    fpr = fps / negatives
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Area Under Curve (AUC) using trapezoidal rule.

    Parameters
    x : np.ndarray
        X-coordinates (e.g., FPR) of shape (n,).
    y : np.ndarray
        Y-coordinates (e.g., TPR) of shape (n,).

    Returns
    float
        Area under the curve.

    Raises
    ValueError
        If inputs are invalid or contain fewer than two points.

    Time Complexity
    O(n log n), due to sorting.

    Space Complexity
    O(n)
    """
    x_arr, y_arr = _validate_1d_pair(x, y, name_pred="y")
    if x_arr.size < 2:
        raise ValueError("x and y must contain at least two points.")

    order = np.argsort(x_arr, kind="stable")
    x_sorted = x_arr[order].astype(float)
    y_sorted = y_arr[order].astype(float)
    return float(np.trapezoid(y_sorted, x_sorted))
