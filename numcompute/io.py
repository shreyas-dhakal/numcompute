import numpy as np


def load_csv(filepath, delimiter=",", fill_value=np.nan, skip_missing=False):
    """
    Reads a CSV file and loads it into a Numpy array.
    """
    data = np.genfromtxt(filepath, delimiter=delimiter, filling_values=fill_value)
    if skip_missing:
        data = data[~np.isnan(data).any(axis=1)]
    return data
