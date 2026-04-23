from pathlib import Path
import numpy as np
from numcompute.stats import histogram, max, mean, median, min, quantile, std

DATA_DIR = Path(__file__).resolve().parent / "data" / "stats"

def test_basic_stats_1d_match_numpy() -> None:
    x = np.load(DATA_DIR / "stats_values_1d.npy")

    assert mean(x) == float(np.mean(x))
    assert median(x) == float(np.median(x))
    assert std(x) == float(np.std(x))
    assert min(x) == float(np.min(x))
    assert max(x) == float(np.max(x))


def test_axis_wise_stats_2d_match_numpy() -> None:
    x = np.load(DATA_DIR / "stats_values_2d.npy")

    assert np.array_equal(mean(x, axis=0), np.mean(x, axis=0))
    assert np.array_equal(median(x, axis=1), np.median(x, axis=1))
    assert np.array_equal(std(x, axis=0, ddof=1), np.std(x, axis=0, ddof=1))
    assert np.array_equal(min(x, axis=1), np.min(x, axis=1))
    assert np.array_equal(max(x, axis=0), np.max(x, axis=0))


def test_skipna_stats_ignore_nans() -> None:
    x = np.load(DATA_DIR / "stats_values_with_nan.npy")

    assert np.isclose(mean(x, skipna=True), np.nanmean(x))
    assert np.isclose(median(x, skipna=True), np.nanmedian(x))
    assert np.isclose(std(x, skipna=True), np.nanstd(x))
    assert np.isclose(min(x, skipna=True), np.nanmin(x))
    assert np.isclose(max(x, skipna=True), np.nanmax(x))


def test_quantile_scalar_and_vector() -> None:
    x = np.load(DATA_DIR / "stats_values_1d.npy")

    assert quantile(x, 0.25) == float(np.quantile(x, 0.25))
    qs = np.array([0.1, 0.5, 0.9])
    out = quantile(x, qs, interpolation="midpoint")
    expected = np.quantile(x, qs, method="midpoint")
    assert np.array_equal(out, expected)


def test_quantile_with_nan_skipna() -> None:
    x = np.load(DATA_DIR / "stats_values_with_nan.npy")

    out = quantile(x, 0.5, skipna=True)
    expected = float(np.nanquantile(x, 0.5))
    assert out == expected


def test_histogram_counts_and_edges() -> None:
    x = np.load(DATA_DIR / "stats_values_1d.npy")
    counts, edges = histogram(x, bins=5)
    expected_counts, expected_edges = np.histogram(x, bins=5)
    assert np.array_equal(counts, expected_counts)
    assert np.array_equal(edges, expected_edges)


def test_histogram_skipna_filters_nan() -> None:
    x = np.load(DATA_DIR / "stats_values_with_nan.npy")
    counts, edges = histogram(x, bins=4, skipna=True)
    expected_counts, expected_edges = np.histogram(x[~np.isnan(x)], bins=4)
    assert np.array_equal(counts, expected_counts)
    assert np.array_equal(edges, expected_edges)
