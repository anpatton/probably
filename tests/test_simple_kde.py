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
    assert axes.get_ylabel() == "Density"


def test_single_series_draws_one_curve_and_no_legend():
    rng = np.random.default_rng(0)
    axes = simple_kde(rng.normal(size=100))
    assert len(axes.get_lines()) == 1
    assert axes.get_legend() is None


def test_mapping_draws_one_curve_per_key_labelled_by_key():
    rng = np.random.default_rng(0)
    axes = simple_kde(
        {
            "a": rng.normal(0, 1, 60),
            "b": rng.normal(5, 1, 60),
            "c": rng.normal(10, 1, 60),
        }
    )
    assert len(axes.get_lines()) == 3
    legend = axes.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b", "c"]


def test_list_of_lists_is_labelled_by_position():
    rng = np.random.default_rng(0)
    axes = simple_kde([rng.normal(0, 1, 40), rng.normal(5, 1, 40)])
    legend = axes.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["Series 1", "Series 2"]


def test_2d_array_is_treated_as_multiple_series():
    rng = np.random.default_rng(0)
    axes = simple_kde(rng.normal(size=(3, 40)))
    assert len(axes.get_lines()) == 3


def test_series_may_differ_in_length():
    rng = np.random.default_rng(0)
    axes = simple_kde({"short": rng.normal(size=20), "long": rng.normal(size=300)})
    assert len(axes.get_lines()) == 2


def test_groups_get_distinct_colors():
    rng = np.random.default_rng(0)
    axes = simple_kde([rng.normal(0, 1, 40), rng.normal(4, 1, 40)])
    colors = {line.get_color() for line in axes.get_lines()}
    assert len(colors) == 2


def test_groups_share_one_grid():
    rng = np.random.default_rng(0)
    axes = simple_kde([rng.normal(0, 1, 40), rng.normal(6, 1, 40)])
    grids = [line.get_xdata() for line in axes.get_lines()]
    assert np.allclose(grids[0], grids[1])


def test_grid_spans_every_series():
    axes = simple_kde({"low": [0.0, 0.5, 1.0, 1.5], "high": [8.0, 8.5, 9.0, 10.0]})
    grid = axes.get_lines()[0].get_xdata()
    assert grid.min() == 0.0
    assert grid.max() == 10.0
