import matplotlib
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_bar

matplotlib.use("Agg")


def test_returns_axes():
    axes = simple_bar(["a", "b", "c"], [3, 1, 2])
    assert isinstance(axes, Axes)


def test_sorts_descending_top_to_bottom():
    axes = simple_bar(["a", "b", "c"], [3, 1, 2])
    labels = [tick.get_text() for tick in axes.get_yticklabels()]
    # matplotlib draws the first category at the bottom, so the largest
    # value should be last (i.e. plotted at the top).
    assert labels[-1] == "a"
    assert labels[0] == "b"


def test_sets_xlabel_and_title():
    axes = simple_bar(["a", "b"], [1, 2], xlabel="count", title="my bar chart")
    assert axes.get_xlabel() == "count"
    assert axes.get_title() == "my bar chart"


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simple_bar(["a", "b", "c"], [1, 2])
