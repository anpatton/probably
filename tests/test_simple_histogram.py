import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_histogram

matplotlib.use("Agg")


def test_returns_axes():
    axes = simple_histogram([1, 2, 2, 3, 3, 3])
    assert isinstance(axes, Axes)


def test_accepts_numpy_array():
    axes = simple_histogram(np.array([1.0, 2.0, 3.0]))
    assert len(axes.patches) > 0


def test_accepts_pandas_series():
    pd = pytest.importorskip("pandas")
    axes = simple_histogram(pd.Series([1, 2, 3, 4]))
    assert len(axes.patches) > 0


def test_accepts_polars_series():
    pl = pytest.importorskip("polars")
    axes = simple_histogram(pl.Series([1, 2, 3, 4]))
    assert len(axes.patches) > 0


def test_sets_labels_and_title():
    axes = simple_histogram([1, 2, 3], xlabel="value", title="my histogram")
    assert axes.get_xlabel() == "value"
    assert axes.get_title() == "my histogram"


def test_rejects_2d_input():
    with pytest.raises(ValueError):
        simple_histogram([[1, 2], [3, 4]])
