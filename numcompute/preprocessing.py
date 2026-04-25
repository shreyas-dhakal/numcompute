""" 

### `preprocessing.py`
- Implement:
  - `StandardScaler` (z-score standardization)
  - `MinMaxScaler` (scale to a given range)
  - `OneHotEncoder` (for categorical variables)
  - *(Optional)* Simple imputer to replace NaN with a constant
- API:
  - `fit(X) -> self`
  - `transform(X) -> X_out`
  - `fit_transform(X) -> X_out`
- Fully vectorized using NumPy; avoid Python loops

"""

import numpy as np

#Formula: z = (x - mean) / std
class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    Formula : z = (X - u) / s
    """
    def __init__(self):
      """
      Initialize StandardScaler with no learned parameters.
      """
      self.mean = None
      self.std = None
        
    def fit(self,X):
      """
      Compute and store mean and std from training data.
      :param X: 2D training data array of shape (n_samples, n_features).
      :return: self
      """
     
      self.mean = np.mean(X,axis=0)  
      self.std = np.std(X,axis=0)    
      return self                    

    def transform(self,X):
      """
      Apply z-score standardization using stored mean and std.
      :param X: 2D array of shape (n_samples, n_features).
      :return: Standardized array of same shape as X.
      """
      X_out = (X - self.mean)/self.std
      return X_out
    

    def fit_transform(self,X):
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
  def __init__(self,feature_range: tuple =(0,1)):
    """
    Initialize MinMaxScaler with desired output range.
    :param feature_range: Tuple (a, b) defining the target range (default: (0, 1))
    """
    self.max_ = None
    self.min_ = None
    self.feature_range = feature_range
  
  def fit(self,X: np.ndarray) -> "MinMaxScaler":
    """
    Compute and store min and max from training data.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: self
    """
    self.max_ = np.max(X,axis=0)
    self.min_ = np.min(X,axis=0)
    return self
  
 
  def transform(self,X: np.ndarray) -> np.ndarray:
    """
    Apply min-max scaling using stored min and max.
    :param X: 2D array of shape (n_samples, n_features).
    :return: Scaled array of same shape as X.
    """
    a,b = self.feature_range
    X_out = (X - self.min_) / (self.max_ - self.min_) * (b - a) + a
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
  def __init__(self):
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
    self.categories_ = np.unique(X)
    return self
  
  def transform(self, X: np.ndarray) -> np.ndarray:
    """
    Convert categories to one-hot encoded binary array.
    :param X: 1D array of categorical values.
    :return: 2D binary array of shape (n_samples, n_categories).
    """
    X_out = (X[:,None] == self.categories_).astype(int)
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
    if self.strategy == "mean":
      self.statistics_ = np.nanmean(X,axis=0)
    elif self.strategy == "median":
      self.statistics_ = np.nanmedian(X,axis=0)
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
    X_out = np.where(np.isnan(X), self.statistics_ , X)
    return X_out
  
  def fit_transform(self, X:np.ndarray) ->np.ndarray:
    """
    Fit to data then transform it in one step.
    :param X: 2D training data array of shape (n_samples, n_features).
    :return: Array of same shape as X with no NaN values.
    """
    X_out  = self.fit(X).transform(X)
    return X_out

