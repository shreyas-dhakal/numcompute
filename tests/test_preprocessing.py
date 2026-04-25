from pathlib import Path
import numpy as np
import pytest
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer


#sTANDARDScaler
def test_standard_scaler_mean_zero_std_pne() -> None:
        X = np.array([[2.0, 4.0],
                  [6.0, 8.0],
                  [10.0, 12.0]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(np.mean(X_scaled, axis=0), 0.0)
        assert np.allclose(np.std(X_scaled, axis=0), 1.0)


def test_standard_scaler_all_equal_values_no_crash() -> None:
    X = np.array([[5.0, 5.0],
                  [5.0, 5.0],
                  [5.0, 5.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.any(np.isnan(X_scaled))
    assert not np.any(np.isinf(X_scaled))


def test_standard_scaler_not_fitted_raises() -> None:
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    scaler = StandardScaler()
    with pytest.raises(ValueError, match="not fitted"):
        scaler.transform(X)


def test_standard_scaler_wrong_features_raises() -> None:
    X_train = np.array([[1.0, 2.0],
                        [3.0, 4.0]])
    X_test = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(X_train)
    with pytest.raises(ValueError, match="features"):
        scaler.transform(X_test)

def test_minmax_scaler_default_range() -> None:
    X = np.array([[2.0, 4.0],
                  [6.0, 8.0],
                  [10.0, 12.0]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.min(X_scaled, axis=0), 0.0)
    assert np.allclose(np.max(X_scaled, axis=0), 1.0)