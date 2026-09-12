import matplotlib
import pytest

from probably.viz import simple_2x2, simple_bar, simple_histogram
from probably.viz._utils import format_number, format_tick

matplotlib.use("Agg")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.7580000001, "3.76"),
        (0.03374892, "0.0337"),
        (1765.4321, "1770"),  # rounds above the decimal point, not to 1765
        (123456.7, "123000"),
        (0.9971234, "0.997"),
        (-1.396, "-1.4"),
        (0.0, "0"),
        (0.00004821, "4.82e-05"),
        (float("nan"), "n/a"),
        (None, "n/a"),
    ],
)
def test_format_number(value, expected):
    assert format_number(value) == expected


@pytest.mark.parametrize("value", [150, 1765, -42, 1000000])
def test_exact_integers_are_never_rounded(value):
    # a count must not be blurred to 3 significant figures
    assert format_number(value) == str(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (7, "7"),
        (0.25, "0.25"),
        (2500, "2.5k"),
        (10000, "10k"),
        (1_200_000, "1.2M"),
        (3.4e9, "3.4B"),
        (5e12, "5T"),
    ],
)
def test_format_tick(value, expected):
    assert format_tick(value) == expected


def test_numeric_axis_uses_compact_ticks():
    axes = simple_histogram([1000.0 * i for i in range(1, 40)])
    labels = [format_tick(t) for t in axes.get_xticks()]
    assert any(label.endswith("k") for label in labels)


def test_large_numbers_leave_no_offset_label():
    axes = simple_histogram([1e6 * i for i in range(1, 40)])
    assert axes.xaxis.get_offset_text().get_text() == ""


def test_category_axis_labels_survive_the_tick_formatter():
    axes = simple_bar(["setosa", "versicolor"], [10, 20])
    labels = [tick.get_text() for tick in axes.get_yticklabels()]
    assert set(labels) == {"setosa", "versicolor"}


def test_2x2_class_labels_survive_the_tick_formatter():
    axes = simple_2x2([0, 0, 1, 1], [0, 1, 1, 1])
    labels = [tick.get_text() for tick in axes.get_xticklabels()]
    assert labels == ["0", "1"]
