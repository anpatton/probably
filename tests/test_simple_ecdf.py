import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_ecdf
from probably.viz._utils import CATEGORICAL_COLORS, SERIES_COLORS

matplotlib.use("Agg")


def test_one_series_draws_one_line_and_no_legend():
    axes = simple_ecdf(np.random.default_rng(0).normal(size=50))
    assert isinstance(axes, Axes)
    assert len(axes.lines) == 1
    assert axes.get_legend() is None


def test_a_mapping_draws_one_line_per_key_with_a_legend():
    rng = np.random.default_rng(0)
    axes = simple_ecdf({"control": rng.normal(size=50), "treated": rng.normal(2, size=50)})
    assert len(axes.lines) == 2
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["control", "treated"]


def test_a_list_of_vectors_is_labelled_by_position():
    rng = np.random.default_rng(0)
    axes = simple_ecdf([rng.normal(size=50), rng.normal(size=50)])
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["Series 1", "Series 2"]


def test_series_may_be_ragged():
    rng = np.random.default_rng(0)
    axes = simple_ecdf({"a": rng.normal(size=50), "b": rng.normal(size=17)})
    assert len(axes.lines) == 2


@pytest.mark.parametrize("flag", ["r", "y", "b"])
def test_color_applies_to_a_single_series(flag):
    axes = simple_ecdf([1.0, 2.0, 3.0], color=flag)
    assert axes.lines[0].get_color() == SERIES_COLORS[flag]


def test_several_series_take_the_categorical_palette():
    rng = np.random.default_rng(0)
    axes = simple_ecdf({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    assert [line.get_color() for line in axes.lines] == list(CATEGORICAL_COLORS[:2])


def test_the_curve_climbs_from_zero_to_one():
    axes = simple_ecdf([3.0, 1.0, 2.0, 4.0])
    _, cumulative = axes.lines[0].get_data()
    assert cumulative[0] == pytest.approx(0.25)
    assert cumulative[-1] == pytest.approx(1.0)
    assert list(cumulative) == sorted(cumulative)


def test_values_are_plotted_in_order():
    axes = simple_ecdf([3.0, 1.0, 2.0])
    observed, _ = axes.lines[0].get_data()
    assert list(observed) == [1.0, 2.0, 3.0]


def test_labels_and_title():
    axes = simple_ecdf([1.0, 2.0, 3.0], xlabel="Yield", title="Batch")
    assert axes.get_xlabel() == "Yield"
    assert axes.get_ylabel() == "Cumulative Probability"
    assert axes.get_title() == "Batch"


def test_rejects_a_vector_too_short_to_plot():
    with pytest.raises(ValueError, match="at least 2 values"):
        simple_ecdf([1.0])
