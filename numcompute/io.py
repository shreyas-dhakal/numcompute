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
    Load a CSV file into a NumPy array.
    :param filepath: Path to the CSV file.
    :param delimiter: The string used to separate values (default: ',').
    :param missing_strategy: skip - skip rows with missing data; fill - fill missing data with fill_value (default: 'skip').
    :param fill_value: The value to fill missing data with (default: np.nan) if handle_missing is 'fill'.
    :param skip_rows: The number of rows to skip at the beginning of the file (default: 1, to skip header).
    :return: The loaded data as a float array.
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
