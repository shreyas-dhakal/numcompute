# NumCompute

**Modular, end-to-end machine learning framework, built from scratch with Python and NumPy.**

NumCompute is a machine learning library designed to simulate the core functionalities of ML frameworks like scikit-learn. Developed with emphasis on deep algorithmic understanding, numerical computation, and clean software engineering, NumCompute provides essential modules for data handling, model training, evaluation, and pipeline, all implemented without using any external ML/DL libraries. It only utilises plain Python and NumPy library.

## Features

*   **Data Handling**: Efficient CSV reading with streaming/chunking and flexible dtype handling.
*   **Preprocessing**: Includes `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, and `OneHotEncoder`.
*   **Core Algorithms**: Implementations of sorting, searching (e.g., `top-k`, `quickselect`, `binary_search`), and ranking with tie handling and percentiles.
*   **Statistical Utilities**: Streaming statistics (Welford's algorithm), histograms, and quantiles.
*   **Model Evaluation**: Metrics such as Accuracy, Precision, Recall, F1-score, Mean Squared Error (MSE), Confusion Matrix, and ROC/AUC.
*   **Optimization**: Finite-difference gradients and line search algorithms.
*   **Pipeline Abstraction**: A flexible `Pipeline` for chaining transformers and estimators.
*   **Benchmarking**: Tools for benchmarking and performance comparisons against native Python implementations.

## Installation

To install NumCompute, clone the repository and install it using pip:

```bash
git clone https://github.com/shreyas-dhakal/numcompute
cd numcompute
pip install .
```

## Usage Examples

### Data Loading and Preprocessing

```python
import numpy as np
from numcompute.io import load_csv
from numcompute.preprocessing import StandardScaler, SimpleImputer
from numcompute.pipeline import Pipeline

# Load data with missing values
data = load_csv("iris_missing.csv", missing_strategy="fill", fill_value=np.nan)

# Create a preprocessing pipeline
pipeline = Pipeline([
    (\'imputer\', SimpleImputer(strategy=\'mean\')),
    (\'scaler\', StandardScaler())
])

# Fit and transform the data
processed_data = pipeline.fit_transform(data)
print("Processed Data (first 5 rows):\n", processed_data[:5])
```

### Sorting and Searching

```python
from numcompute.sort_search import topk, binary_search

values = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

# Get top-3 largest elements
top_values, top_indices = topk(values, k=3, largest=True)
print(f"Top 3 largest values: {top_values}, indices: {top_indices}")

# Binary search for a value
sorted_values = np.sort(values)
index, exists = binary_search(sorted_values, 5)
print(f"Value 5 in sorted array: index {index}, exists: {exists}")
```

### Statistics

```python
from numcompute.stats import mean, variance, welford, quantile

data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

print(f"Mean: {mean(data)}")
print(f"Variance: {variance(data)}")

w_mean, w_var = welford(data)
print(f"Welford Mean: {w_mean}, Welford Variance: {w_var}")

print(f"25th percentile: {quantile(data, 0.25)}")
```

## API Overview

NumCompute modules are designed to be efficient and consistent with usage of vectorization, replicating the scikit-learn API. Key modules include:

*   `numcompute.io`: Functions for efficient data input and loading.
*   `numcompute.preprocessing`: Classes for data transformation (e.g., scaling, imputation, one-hot encoding).
*   `numcompute.sort_search`: Algorithms for sorting, selection, and searching.
*   `numcompute.rank`: Functions for ranking data, including percentile calculations.
*   `numcompute.stats`: Statistical computations, including streaming algorithms.
*   `numcompute.metrics`: Performance evaluation metrics for machine learning models.
*   `numcompute.optim`: Numerical optimization routines, such as gradient and Jacobian estimation.
*   `numcompute.pipeline`: Tools for constructing and managing machine learning workflows.
*   `numcompute.utils`: General utility functions like similarity scores and other utilities.
*   `numcompute.benchmarking`: Utilities for performance analysis between vectorization and traditional Python looping.

## Authors

* Avishkar Waikar
* Jianghan Sun
* Shreyas Dhakal
* Sukarna Paul
