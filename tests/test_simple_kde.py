import matplotlib
import numpy as np
from matplotlib.axes import Axes

from probably.viz import simple_kde

matplotlib.use("Agg")


def test_returns_axes():
    rng = np.random.default_rng(0)
    axes = simple_kde(rng.normal(size=100))
    assert isinstance(axes, Axes)


def test_sets_labels_and_title():
    rng = np.random.default_rng(0)
    axes = simple_kde(rng.normal(size=100), xlabel="value", title="my kde")
    assert axes.get_xlabel() == "value"
    assert axes.get_title() == "my kde"
    assert axes.get_ylabel() == "density"
