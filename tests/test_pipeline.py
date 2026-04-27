import numpy as np
import pytest
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer

#Normal cases
def test_pipeline_imputer_then_scaler()-> None:
     X = np.array([[1.0,    2.0],
                  [np.nan, 4.0],
                  [5.0,    6.0]])
     
     pipe = Pipeline([
          ("imputer", SimpleImputer(strategy="mean")),
          ("scaler", StandardScaler())
     ])

     X_out = pipe.fit_transform(X)
     

     assert not np.any(np.isnan(X_out))
     assert np.allclose(np.mean(X_out, axis=0), 0.0)
     assert np.allclose(np.std(X_out, axis=0), 1.0)

#8 testcases for 8 valueErrors in Pipeline.py
