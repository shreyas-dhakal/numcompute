import numpy as np
from pathlib import Path
from typing import Union


def load_csv(
        filepath: Union[str, Path],
        delimiter: str = ",",
        missing_strategy: str = "fill",
        fill_value: float = np.nan,
        skip_rows: int = 1
) -> np.ndarray:
    """
    Load a CSV file into a NumPy array with basic missing value handling.

    Parameters
    filepath : Union[str, Path]
        Path to the CSV file.
    delimiter : str, optional
        Field delimiter used in the file (default is ',').
    missing_strategy : {"fill", "skip"}, optional
        Strategy for handling missing values:
        - "fill": Replace missing values with `fill_value`.
        - "skip": Remove rows containing any missing values.
    fill_value : float, optional
        Value used to replace missing entries when `missing_strategy="fill"`.
        If set to np.nan, missing values remain unchanged.
    skip_rows : int, optional
        Number of rows to skip at the beginning of the file
        (default is 1, typically to skip a header row).

    Returns
    np.ndarray
        Loaded data as a 2D NumPy array of shape (n_samples, n_features).
        If the input is a single column, it is reshaped to (n_samples, 1).

    Raises
    ValueError
        If `missing_strategy` is not one of {"fill", "skip"}.

    Time Complexity
    O(n * m), where n is the number of rows and m is the number of columns.

    Space Complexity
    O(n * m), for storing the dataset in memory.
    """
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        missing_values="",
        filling_values=np.nan,
        skip_header=skip_rows
    )
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if missing_strategy == "fill":
        if not np.isnan(fill_value):
            data = np.where(np.isnan(data), fill_value, data)
    elif missing_strategy == "skip":
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]
    else:
        raise ValueError(f"Invalid missing_strategy: {missing_strategy}. Use 'fill' or 'skip'.")

    return data
