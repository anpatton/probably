import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_kde_2d

matplotlib.use("Agg")


def test_returns_axes():
    rng = np.random.default_rng(0)
    axes = simple_kde_2d(rng.normal(size=100), rng.normal(size=100))
    assert isinstance(axes, Axes)


def test_sets_labels_and_title():
    rng = np.random.default_rng(0)
    axes = simple_kde_2d(
        rng.normal(size=100), rng.normal(size=100), xlabel="x", ylabel="y", title="my 2d kde"
    )
    assert axes.get_xlabel() == "x"
    assert axes.get_ylabel() == "y"
    assert axes.get_title() == "my 2d kde"


def test_rejects_mismatched_lengths():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simple_kde_2d(rng.normal(size=100), rng.normal(size=50))
