import numpy as np

def _validate_array(X: np.ndarray, name: str = "X", check_nan: bool = False) -> np.ndarray:
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
    Formula : z = (X - u) / s
    """
    def __init__(self) -> None:
      """
      Initialize StandardScaler with no learned parameters.
      """
      self.mean_ = None
      self.std_ = None
        
    def fit(self, X: np.ndarray) -> "StandardScaler":
      """
      Compute and store mean and std from training data.
      :param X: 2D training data array of shape (n_samples, n_features).
      :return: self
      """
      arr = _validate_array(X, name="X", check_nan=True)
      self.mean_ = np.mean(arr, axis=0)
      std = np.std(arr, axis=0)
      self.std_ = np.where(std == 0.0, 1.0, std)    
      return self                    

    def transform(self, X: np.ndarray) -> np.ndarray:
      """
      Apply z-score standardization using stored mean and std.
      :param X: 2D array of shape (n_samples, n_features).
      :return: Standardized array of same shape as X.
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
      Fit to data then transform it in one step.
      :param X: 2D training data array of shape (n_samples, n_features).
      :return: Standardized array of same shape as X.
      """
      X_out = self.fit(X).transform(X)
      return X_out
        

class MinMaxScaler():
  """
  Scale features to a given range using min-max normalization.
  Formula: z = (X - min) / (max - min) * (b - a) + a
  """
  def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
    """
    Initialize MinMaxScaler with desired output range.
    :param feature_range: Tuple (a, b) defining the target range (default: (0, 1))
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
    Compute and store min and max from training data.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: self
    """
    arr = _validate_array(X, name="X", check_nan=True)
    self.min_ = np.min(arr, axis=0)
    self.max_ = np.max(arr, axis=0)
    ranges = self.max_ - self.min_
    self.range_ = np.where(ranges == 0.0, 1.0, ranges)
    return self
  
 
  def transform(self,X: np.ndarray) -> np.ndarray:
    """
    Apply min-max scaling using stored min and max.
    :param X: 2D array of shape (n_samples, n_features).
    :return: Scaled array of same shape as X.
    """
    if self.min_ is None or self.range_ is None:
        raise ValueError("MinMaxScaler is not fitted yet. Call fit() first.")
    arr = _validate_array(X, name="X", check_nan=True)
    if arr.shape[1] != self.min_.shape[0]:
            raise ValueError(f"Expected {self.min_.shape[0]} features, got {arr.shape[1]}.")
    a,b = self.feature_range
    X_out = (arr - self.min_) / (self.max_ - self.min_) * (b - a) + a
    return X_out

  def fit_transform(self,X:np.ndarray) -> np.ndarray:
    """
    Fit to data then transform it in one step.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: Scaled array of same shape as X.
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
    Initialize OneHotEncoder with no learned parameters.
    """
    self.categories_ = None

  def fit(self, X: np.ndarray) -> "OneHotEncoder":
    """
    Learn unique categories from training data.
    :param X: 1D array of categorical values.
    :return: self
    """
    arr = np.asarray(X)
    if arr.size == 0:
        raise ValueError("X cannot be empty.")
    self.categories_ = np.unique(arr)
    return self
  
  def transform(self, X: np.ndarray) -> np.ndarray:
    """
    Convert categories to one-hot encoded binary array.
    :param X: 1D array of categorical values.
    :return: 2D binary array of shape (n_samples, n_categories).
    """
    if self.categories_ is None:
        raise ValueError("OneHotEncoder is not fitted yet. Call fit() first.")
    arr = np.asarray(X)
    X_out = (arr[:,None] == self.categories_).astype(int)
    return X_out
  
  def fit_transform(self, X:np.ndarray) -> np.ndarray:
    """
    Fit to data then transform it in one step.
    :param X: 1D array of categorical values.
    :return: 2D binary array of shape (n_samples, n_categories).
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
    Initialize SimpleImputer with a given strategy.
    :param strategy: Imputation strategy - 'mean', 'median', or 'constant' (default: 'mean').
    :param fill_value: Constant value to use when strategy is 'constant' (default: 0).
    """
    self.strategy = strategy
    self.fill_value = fill_value
    self.statistics_ = None

  def fit(self, X: np.ndarray) -> "SimpleImputer":
    """
    Compute and store the imputation statistic from training data.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: self
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
    Replace NaN values with the stored imputation statistic.
    :param X: 2D array of shape (n_samples, n_features).
    :return: Array of same shape as X with no NaN values.
    """
    if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted yet. Call fit() first.")
    arr = _validate_array(X, name="X", check_nan=False)
    X_out = np.where(np.isnan(arr), self.statistics_ , arr)
    return X_out
  
  def fit_transform(self, X:np.ndarray) ->np.ndarray:
    """
    Fit to data then transform it in one step.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: Array of same shape as X with no NaN values.
    """
    X_out  = self.fit(X).transform(X)
    return X_out

__all__ = ["StandardScaler", "MinMaxScaler", "OneHotEncoder", "SimpleImputer"]