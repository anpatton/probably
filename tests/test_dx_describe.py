import math

import numpy as np
import pytest

from probably.dx import describe
from probably.dx.describe import LAYER_1_KEYS, LAYER_2_KEYS


def test_depth_1_returns_only_the_first_layer():
    assert set(describe([1, 2, 3, 4, 5])) == set(LAYER_1_KEYS)


def test_depth_2_returns_both_layers():
    assert set(describe([1, 2, 3, 4, 5], depth=2)) == set(LAYER_1_KEYS) | set(LAYER_2_KEYS)


def test_depth_2_does_not_change_the_first_layer():
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    shallow = describe(values)
    deep = describe(values, depth=2)
    assert all(deep[key] == shallow[key] for key in LAYER_1_KEYS)


def test_layer_1_values():
    summary = describe([1, 2, 3, 4, 5])
    assert summary["n"] == 5
    assert summary["mean"] == 3.0
    assert summary["median"] == 3.0
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["range"] == 4.0
    assert summary["q1"] == 2.0
    assert summary["q3"] == 4.0
    assert summary["std"] == pytest.approx(math.sqrt(2.5))
    assert summary["variance"] == pytest.approx(2.5)


def test_layer_2_values():
    summary = describe([1, 2, 3, 4, 5], depth=2)
    assert summary["iqr"] == 2.0
    assert summary["mad"] == 1.0
    assert summary["skew"] == pytest.approx(0.0)
    assert summary["cv"] == pytest.approx(math.sqrt(2.5) / 3.0)
    assert summary["sem"] == pytest.approx(math.sqrt(2.5) / math.sqrt(5))


def test_excess_kurtosis_puts_a_normal_near_zero():
    values = np.random.default_rng(0).normal(size=20_000)
    assert describe(values, depth=2)["kurtosis"] == pytest.approx(0.0, abs=0.1)


def test_skew_follows_the_tail():
    right = describe([1, 1, 1, 2, 10], depth=2)["skew"]
    left = describe([-10, -2, -1, -1, -1], depth=2)["skew"]
    assert right > 0
    assert left < 0


def test_accepts_any_vector_type():
    expected = describe([1.0, 2.0, 3.0])
    assert describe(np.array([1.0, 2.0, 3.0])) == expected
    assert describe((1.0, 2.0, 3.0)) == expected


def test_pandas_and_polars_series_work():
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    expected = describe([1.0, 2.0, 3.0])
    assert describe(pd.Series([1.0, 2.0, 3.0])) == expected
    assert describe(pl.Series([1.0, 2.0, 3.0])) == expected


@pytest.mark.parametrize("size", [1, 2, 3])
def test_small_samples_keep_their_keys_and_use_nan(size):
    summary = describe(list(range(size)), depth=2)
    assert set(summary) == set(LAYER_1_KEYS) | set(LAYER_2_KEYS)
    assert summary["n"] == size


def test_single_value_has_no_spread_or_shape():
    summary = describe([7.0], depth=2)
    assert summary["mean"] == 7.0
    assert summary["range"] == 0.0
    for key in ("std", "variance", "skew", "kurtosis", "sem"):
        assert math.isnan(summary[key])


def test_zero_mean_reports_cv_as_nan():
    # cv is a ratio to the mean; at mean 0 it is undefined, not enormous
    assert math.isnan(describe([-2.0, -1.0, 1.0, 2.0], depth=2)["cv"])


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="at least 1 value"):
        describe([])


@pytest.mark.parametrize("bad", [[1.0, np.nan, 3.0], [1.0, np.inf, 3.0]])
def test_rejects_non_finite_input(bad):
    with pytest.raises(ValueError, match="finite"):
        describe(bad)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        describe([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("depth", [0, 3, "2", None])
def test_rejects_unknown_depth(depth):
    with pytest.raises(ValueError, match="depth must be 1 or 2"):
        describe([1.0, 2.0, 3.0], depth=depth)


@pytest.mark.parametrize("depth", [1, 2])
def test_values_are_plain_floats(depth):
    # numpy scalars leak into reprs and break the dict[str, float] contract
    for value in describe([1.0, 2.0, 3.0, 4.0], depth=depth).values():
        assert type(value) is float
