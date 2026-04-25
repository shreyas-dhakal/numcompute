import numpy as np
from typing import Sequence, Tuple


def _as_2d_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty along both axes.")
    return arr


class StandardScaler:
    """Scale features to zero mean and unit variance column-wise."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        x_arr = _as_2d_array(X).astype(float)
        self.mean_ = np.mean(x_arr, axis=0)
        std = np.std(x_arr, axis=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must be fitted before transform.")

        x_arr = _as_2d_array(X).astype(float)
        if x_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError("X has different number of features than in fit.")
        return (x_arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features to a target range column-wise."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        lo, hi = feature_range
        if hi <= lo:
            raise ValueError("feature_range must satisfy max > min.")

        self.feature_range = (float(lo), float(hi))
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        x_arr = _as_2d_array(X).astype(float)
        self.data_min_ = np.min(x_arr, axis=0)
        self.data_max_ = np.max(x_arr, axis=0)
        ranges = self.data_max_ - self.data_min_
        self.data_range_ = np.where(ranges == 0.0, 1.0, ranges)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_range_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform.")

        x_arr = _as_2d_array(X).astype(float)
        if x_arr.shape[1] != self.data_min_.shape[0]:
            raise ValueError("X has different number of features than in fit.")

        lo, hi = self.feature_range
        scaled_01 = (x_arr - self.data_min_) / self.data_range_
        return scaled_01 * (hi - lo) + lo

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    """One-hot encode categorical columns of a 2D input array."""

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] | None = None

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        x_arr = _as_2d_array(X)
        self.categories_ = [np.unique(x_arr[:, i]) for i in range(x_arr.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform.")

        x_arr = _as_2d_array(X)
        if x_arr.shape[1] != len(self.categories_):
            raise ValueError("X has different number of features than in fit.")

        blocks = []
        for i, cats in enumerate(self.categories_):
            block = (x_arr[:, [i]] == cats.reshape(1, -1)).astype(float)
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder"]
