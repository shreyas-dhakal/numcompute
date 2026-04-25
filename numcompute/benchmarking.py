"""
benchmarking.py

Utilities for benchmarking vectorised NumPy implementations against
pure Python loop-based implementations.

Features:
- Reproducible benchmarks (fixed random seed)
- Multiple runs with best/mean timing
- Comparison table output
- Simple API for extensibility
"""

import time
import numpy as np


def benchmark(func, *args, repeat=5, warmup=1):
    """
    Benchmark a function by measuring execution time.

    Parameters:
        func   : callable
        *args  : arguments to pass to the function
        repeat : (int) Number of timed runs
        warmup : (int) Number of warmup runs (not timed)

    Returns:
        float : best execution time (seconds)
    """
    # Warmup (avoids cold-start bias)
    for _ in range(warmup):
        func(*args)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)  # best time


def compare_functions(funcs, args, labels=None, repeat=5):
    """
    Compare multiple functions on the same input.

    Parameters:
        funcs  : list of callables
        args   : tuple
        labels : list of str
        repeat : int

    Returns:
        list of (label, time)
    """
    results = []

    if labels is None:
        labels = [f.__name__ for f in funcs]

    for func, label in zip(funcs, labels):
        t = benchmark(func, *args, repeat=repeat)
        results.append((label, t))

    return results


def print_results(results, title="Benchmark Results"):
    """
    Print formatted benchmark results.

    Parameters:
        results : list of (label, time)
    """
    print(f"\n{title}")
    print("-" * 40)
    print(f"{'Function':<20} {'Time (ms)':>10}")
    print("-" * 40)

    for label, t in results:
        print(f"{label:<20} {t * 1000:>10.4f}")

    print("-" * 40)


def mean_loop(x):
    total = 0.0
    for val in x:
        total += val
    return total / len(x)


def mean_vectorised(x):
    return np.mean(x)


def top_k_loop(x, k):
    return sorted(x)[-k:]


def top_k_vectorised(x, k):
    idx = np.argpartition(x, -k)[-k:]
    return x[idx]


def run_all_benchmarks(n=1_000_000, k=10, seed=42):
    """
    Run a suite of benchmarks comparing loop vs vectorised versions.

    Parameters:
        n    : (int) Size of input array
        k    : (int) Top-k parameter
        seed : (int) Random seed for reproducibility
    """
    np.random.seed(seed)
    x = np.random.rand(n)

    print("\nEnvironment:")
    print(f"Array size: {n}")
    print(f"Top-k: {k}")
    print(f"Seed: {seed}")

    mean_results = compare_functions(
        funcs=[mean_vectorised, mean_loop],
        args=(x,),
        labels=["Mean (NumPy)", "Mean (Loop)"],
    )
    print_results(mean_results, title="Mean Benchmark")

    topk_results = compare_functions(
        funcs=[top_k_vectorised, top_k_loop],
        args=(x, k),
        labels=["Top-k (NumPy)", "Top-k (Loop)"],
    )
    print_results(topk_results, title="Top-k Benchmark")


if __name__ == "__main__":
    run_all_benchmarks()
