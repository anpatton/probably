import matplotlib
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_2x2

matplotlib.use("Agg")


def _counts(axes):
    """Map each quadrant label to the count drawn next to it."""
    texts = [t for t in axes.texts]
    quadrants = {}
    for text in texts:
        if text.get_text() in {"TN", "FP", "FN", "TP"}:
            match = [
                t
                for t in texts
                if t.get_position()[0] == text.get_position()[0]
                and t.get_text() not in {"TN", "FP", "FN", "TP"}
                and abs(t.get_position()[1] - text.get_position()[1]) < 0.5
            ]
            quadrants[text.get_text()] = int(match[0].get_text())
    return quadrants


def test_returns_axes():
    axes = simple_2x2([0, 0, 1, 1], [0, 1, 1, 1])
    assert isinstance(axes, Axes)


def test_counts_each_quadrant():
    # actual:    0  0  1  1
    # predicted: 0  1  1  1  -> TN=1, FP=1, FN=0, TP=2
    axes = simple_2x2([0, 0, 1, 1], [0, 1, 1, 1])
    assert _counts(axes) == {"TN": 1, "FP": 1, "FN": 0, "TP": 2}


def test_draws_four_cells():
    axes = simple_2x2([0, 1], [0, 1])
    assert len(axes.patches) == 4


def test_accepts_string_labels():
    axes = simple_2x2(["no", "no", "yes"], ["no", "yes", "yes"])
    assert _counts(axes) == {"TN": 1, "FP": 1, "FN": 0, "TP": 1}


def test_axis_labels_are_fixed():
    axes = simple_2x2([0, 1], [0, 1])
    assert axes.get_xlabel() == "Predicted"
    assert axes.get_ylabel() == "Actual"


def test_sets_title():
    axes = simple_2x2([0, 1], [0, 1], title="my matrix")
    assert axes.get_title() == "my matrix"


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simple_2x2([0, 1, 1], [0, 1])


def test_rejects_non_binary_truth():
    with pytest.raises(ValueError):
        simple_2x2([0, 1, 2], [0, 1, 1])


def test_rejects_prediction_outside_known_classes():
    with pytest.raises(ValueError):
        simple_2x2([0, 0, 1], [0, 1, 2])
