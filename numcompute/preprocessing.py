import numpy as np

def _validate_array(X: np.ndarray, name: str = "X", check_nan: bool = False) -> np.ndarray:
   
    """
    Convert input to a 2D NumPy array of floats and validate it.

    Parameters
    X : np.ndarray
    name : str, optional
    check_nan : bool, optional

    Returns
    np.ndarray
        Array of shape (n_samples, n_features).

    Raises
    ValueError
        If not 2D, empty, or contains NaNs (if check_nan=True).

    Time Complexity
    O(n * d)

    Space Complexity
    O(n * d)
  """
   
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
      raise ValueError(f"{name} must be 2D.")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if check_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values. Use SimpleImputer first.")
    return arr




#Formula: z = (x - mean) / std
class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    Applies z-score normalization per feature column:

    Formula : z = (X - u) / s
    """
    def __init__(self) -> None:
      """
      Initialize StandardScaler with no learned parameters.
      Time Complexity
      O(1)

      Space Complexity
      O(1)
      """
      self.mean_ = None
      self.std_ = None
        
    def fit(self, X: np.ndarray) -> "StandardScaler":
      """
        Compute mean and std from data.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        StandardScaler

        Raises
        ValueError
            If input is invalid.

        Time Complexity
        O(n * d)

        Space Complexity
        O(d)
      """
      arr = _validate_array(X, name="X", check_nan=True)
      self.mean_ = np.mean(arr, axis=0)
      std = np.std(arr, axis=0)
      self.std_ = np.where(std == 0.0, 1.0, std)    
      return self                    

    def transform(self, X: np.ndarray) -> np.ndarray:
      """
        Apply standardization.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples, n_features)

        Raises
        ValueError
            If not fitted or shape mismatch.

        Time Complexity
        O(n * d)

        Space Complexity
        O(n * d)
        """
      if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler is not fitted yet. Call fit() first.")
      arr = _validate_array(X, name="X", check_nan=True)
      if arr.shape[1] != self.mean_.shape[0]:
          raise ValueError(f"Expected {self.mean_.shape[0]} features, got {arr.shape[1]}.")
      X_out = (arr - self.mean_)/self.std_
      return X_out
    
    

    def fit_transform(self,X: np.ndarray) -> np.ndarray:
      """
        Fit and transform.

        Parameters
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        np.ndarray of shape (n_samples, n_features)

        Time Complexity
        O(n * d)

        Space Complexity
        O(n * d)
        """
      X_out = self.fit(X).transform(X)
      return X_out
        

class MinMaxScaler():
  """
  Scale features to a given range.
  z = (X - min) / (max - min) * (b - a) + a
  """
  def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
    """
    Initialize scaler.

    Parameters
    feature_range : tuple[float, float]

    Raises
    ValueError
        If max <= min.

    Time Complexity
    O(1)

    Space Complexity
    O(1)
    """
    a, b = feature_range
    if b <= a:
            raise ValueError(f"feature_range max ({b}) must be greater than min ({a}).")
    self.feature_range = (float(a), float(b))
    self.min_ = None    
    self.max_ = None    
    self.range_ = None
  
  def fit(self,X: np.ndarray) -> "MinMaxScaler":
    """
    Compute min and max.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    MinMaxScaler

    Raises
    ValueError
        If input invalid.

    Time Complexity
    O(n * d)

    Space Complexity
    O(d)
    """
    arr = _validate_array(X, name="X", check_nan=True)
    self.min_ = np.min(arr, axis=0)
    self.max_ = np.max(arr, axis=0)
    ranges = self.max_ - self.min_
    self.range_ = np.where(ranges == 0.0, 1.0, ranges)
    return self
  
 
  def transform(self,X: np.ndarray) -> np.ndarray:
    """
    Apply scaling.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    np.ndarray of shape (n_samples, n_features)

    Raises
    ValueError
        If not fitted or shape mismatch.

    Time Complexity
    O(n * d)

    Space Complexity
    O(n * d)
    """
    if self.min_ is None or self.range_ is None:
        raise ValueError("MinMaxScaler is not fitted yet. Call fit() first.")
    arr = _validate_array(X, name="X", check_nan=True)
    if arr.shape[1] != self.min_.shape[0]:
            raise ValueError(f"Expected {self.min_.shape[0]} features, got {arr.shape[1]}.")
    a,b = self.feature_range
    X_out = (arr - self.min_) / self.range_ * (b - a) + a
    return X_out

  def fit_transform(self,X:np.ndarray) -> np.ndarray:
    """
    Fit and transform.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    np.ndarray of shape (n_samples, n_features)

    Time Complexity
    O(n * d)

    Space Complexity
    O(n * d)
    """
    X_out = self.fit(X).transform(X)
    return X_out
  

class OneHotEncoder():
  """
  Encode categorical features as a one-hot numeric array.
  Each unique category becomes its own binary column.
  """
  def __init__(self) -> None:
    """
    Initialize encoder.

    Time Complexity
    O(1)

    Space Complexity
    O(1)
    """
    self.categories_ = None

  def fit(self, X: np.ndarray) -> "OneHotEncoder":
    """
    Learn categories.

    Parameters
    X : np.ndarray of shape (n_samples,)

    Returns
    OneHotEncoder

    Raises
    ValueError
        If empty.

    Time Complexity
    O(n log n)

    Space Complexity
    O(k)
        """
    arr = np.asarray(X)
    if arr.size == 0:
        raise ValueError("X cannot be empty.")
    self.categories_ = np.unique(arr)
    return self
  
  def transform(self, X: np.ndarray) -> np.ndarray:
    """
    Encode data.

    Parameters
    X : np.ndarray of shape (n_samples,)

    Returns
    np.ndarray of shape (n_samples, n_categories)

    Raises
    ValueError
        If not fitted.

    Time Complexity
    O(n * k)

    Space Complexity
    O(n * k)
        """
    if self.categories_ is None:
        raise ValueError("OneHotEncoder is not fitted yet. Call fit() first.")
    arr = np.asarray(X)
    X_out = (arr[:,None] == self.categories_).astype(int)
    return X_out
  
  def fit_transform(self, X:np.ndarray) -> np.ndarray:
    """
    Fit and transform.

    Parameters
    X : np.ndarray of shape (n_samples,)

    Returns
    np.ndarray of shape (n_samples, n_categories)

    Time Complexity
    O(n log n + n * k)

    Space Complexity
    O(n * k)
    """
    X_out = self.fit(X).transform(X)
    return X_out



class SimpleImputer():
  """
  Imputer for completing missing values with simple strategies like  mean, median
  along each column, or using a constant value.
  """

  def __init__(self,strategy : str="mean", fill_value:float = 0):
    """
    Initialize imputer.

    Parameters
    strategy : str
    fill_value : float

    Time Complexity
    O(1)

    Space Complexity
    O(1)
    """
    self.strategy = strategy
    self.fill_value = fill_value
    self.statistics_ = None

  def fit(self, X: np.ndarray) -> "SimpleImputer":
    """
    Compute imputation statistics.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    SimpleImputer

    Raises
    ValueError
        If strategy invalid.

    Time Complexity
    O(n * d)

    Space Complexity
    O(d)
    """
    arr = _validate_array(X, name="X", check_nan=False)
    if self.strategy == "mean":
      self.statistics_ = np.nanmean(arr,axis=0)
    elif self.strategy == "median":
      self.statistics_ = np.nanmedian(arr,axis=0)
    elif self.strategy == "constant":
      self.statistics_ = self.fill_value
    else:
        raise ValueError(f"Invalid strategy: {self.strategy}. Use 'mean', 'median', or 'constant'.")
    return self

  def transform(self, X: np.ndarray) ->np.ndarray:
    """
    Replace NaNs.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    np.ndarray of shape (n_samples, n_features)

    Raises
    ValueError
        If not fitted.

    Time Complexity
    O(n * d)

    Space Complexity
    O(n * d)
    """
    if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted yet. Call fit() first.")
    arr = _validate_array(X, name="X", check_nan=False)
    X_out = np.where(np.isnan(arr), self.statistics_ , arr)
    return X_out
  
  def fit_transform(self, X:np.ndarray) ->np.ndarray:
    """
    Fit and transform.

    Parameters
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    np.ndarray of shape (n_samples, n_features)

    Time Complexity
    O(n * d)

    Space Complexity
    O(n * d)
    """
    X_out  = self.fit(X).transform(X)
    return X_out

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]