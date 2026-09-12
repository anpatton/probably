import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_table

matplotlib.use("Agg")


def _rows(axes):
    """The table's body rows as lists of cell text, header row excluded."""
    cells = axes.tables[0].get_celld()
    row_count = max(row for row, _ in cells) + 1
    column_count = max(column for _, column in cells) + 1
    return [
        [cells[(row, column)].get_text().get_text() for column in range(column_count)]
        for row in range(1, row_count)
    ]


def _lookup(axes, label):
    for row in _rows(axes):
        if row[0] == label:
            return row
    raise AssertionError(f"no row labelled {label!r}")


def test_returns_axes():
    rng = np.random.default_rng(0)
    assert isinstance(simple_table(rng.normal(size=50)), Axes)


def test_depth_1_reports_descriptives_only():
    rng = np.random.default_rng(0)
    labels = [row[0] for row in _rows(simple_table(rng.normal(size=50)))]
    assert "Mean" in labels
    assert "Std. dev." in labels
    assert not any("Shapiro" in label for label in labels)


def test_depth_1_has_two_columns():
    rng = np.random.default_rng(0)
    assert len(_rows(simple_table(rng.normal(size=50)))[0]) == 2


def test_depth_2_keeps_descriptives_and_adds_tests():
    rng = np.random.default_rng(0)
    labels = [row[0] for row in _rows(simple_table(rng.normal(size=50), depth=2))]
    assert "Mean" in labels
    assert "Distribution tests" in labels
    assert any("Shapiro" in label for label in labels)


def test_descriptive_values_are_correct():
    axes = simple_table([1.0, 2.0, 3.0, 4.0])
    assert _lookup(axes, "n")[1] == "4"
    assert _lookup(axes, "Mean")[1] == "2.5"
    assert _lookup(axes, "Median")[1] == "2.5"
    assert _lookup(axes, "Min")[1] == "1"
    assert _lookup(axes, "Max")[1] == "4"


def test_normal_sample_retains_normality():
    rng = np.random.default_rng(0)
    axes = simple_table(rng.normal(size=400), depth=2)
    assert _lookup(axes, "Shapiro–Wilk (normal)")[3] == "fail to reject"


def test_exponential_sample_rejects_normality_and_retains_exponential():
    rng = np.random.default_rng(0)
    axes = simple_table(rng.exponential(2.0, size=400), depth=2)
    assert _lookup(axes, "Shapiro–Wilk (normal)")[3] == "reject"
    assert _lookup(axes, "Kolmogorov–Smirnov (exponential)")[3] == "fail to reject"


def test_lognormal_sample_is_not_mistaken_for_normal():
    rng = np.random.default_rng(0)
    axes = simple_table(rng.lognormal(0, 0.6, size=400), depth=2)
    assert _lookup(axes, "Kolmogorov–Smirnov (lognormal)")[3] == "fail to reject"
    assert _lookup(axes, "Shapiro–Wilk (normal)")[3] == "reject"


def test_positive_support_tests_are_na_on_negative_data():
    rng = np.random.default_rng(0)
    axes = simple_table(rng.normal(-10, 1, size=100), depth=2)
    assert _lookup(axes, "Kolmogorov–Smirnov (lognormal)")[1:] == ["n/a", "n/a", "n/a"]


def test_tiny_sample_degrades_to_na_without_a_verdict():
    axes = simple_table([1.0, 2.0, 3.0, 4.0], depth=2)
    # too few points for D'Agostino-Pearson; it must not claim a verdict
    assert _lookup(axes, "D'Agostino–Pearson (normal)")[1:] == ["n/a", "n/a", "n/a"]


def test_sets_title():
    rng = np.random.default_rng(0)
    axes = simple_table(rng.normal(size=50), title="my table")
    assert axes.get_title() == "my table"


@pytest.mark.parametrize("depth", [0, 3, "1", None])
def test_rejects_invalid_depth(depth):
    with pytest.raises(ValueError):
        simple_table([1.0, 2.0, 3.0], depth=depth)
