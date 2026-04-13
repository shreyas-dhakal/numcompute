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
    def __init__(self):
      self.mean = None
      self.std = None
        

    #scaler = StandardScaler()
    #scaler.fit(X_train)
    def fit(self,X):
      #compute and store mean,std from X 
      #axis = 0, hence computer per column
      self.mean = np.mean(X,axis=0)  #stores mean of each column
      self.std = np.std(X,axis=0)    #stores standard deviation column
      return self                    #to allow fit().transform()

    def transform(self,X):
      #z-score formula using stored mean and std
      X_out = (X - self.mean)/self.std
      return X_out
    

    def fit_transform(self,X):
      #fit and transform in one step
      #same as calling fit(X) and transform(X)
      X_out = self.fit(X).transform(X)
      return X_out
        


#test
X_train = np.array([[2, 4],
                    [6, 8],
                    [10, 12]])

scaler = StandardScaler()
scaler.fit(X_train)

print(scaler.mean)
print(scaler.std)

X_scaled = scaler.transform(X_train)
print(X_scaled)

print(np.mean(X_scaled, axis=0))  # [0, 0]
print(np.std(X_scaled, axis=0))   # [1, 1]