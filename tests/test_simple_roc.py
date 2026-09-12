import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes

from probably.viz import simple_roc

matplotlib.use("Agg")


def test_returns_axes():
    axes = simple_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert isinstance(axes, Axes)


def test_draws_curve_and_black_chance_line():
    axes = simple_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    lines = axes.get_lines()
    assert len(lines) == 2
    assert lines[0].get_color() == "black"


def test_caption_reports_auc():
    axes = simple_roc([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    captions = [t.get_text() for t in axes.figure.texts]
    # a perfectly separable split scores an AUC of 1
    assert any("AUC = 1.000" in caption for caption in captions)


def test_accepts_string_labels():
    axes = simple_roc(["no", "no", "yes", "yes"], [0.1, 0.2, 0.7, 0.9])
    captions = [t.get_text() for t in axes.figure.texts]
    assert any("AUC = 1.000" in caption for caption in captions)


def test_axis_labels_are_fixed():
    axes = simple_roc([0, 1], [0.2, 0.8])
    assert axes.get_xlabel() == "False Positive Rate"
    assert axes.get_ylabel() == "True Positive Rate"


def test_sets_title():
    axes = simple_roc([0, 1], [0.2, 0.8], title="my roc")
    assert axes.get_title() == "my roc"


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        simple_roc([0, 1, 1], [0.2, 0.8])


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        simple_roc([0, 1, 2], [0.2, 0.5, 0.8])


def test_accepts_numpy_arrays():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=50)
    scores = rng.random(size=50)
    assert isinstance(simple_roc(labels, scores), Axes)


def test_single_series_has_no_legend():
    axes = simple_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert axes.get_legend() is None


def test_mapping_of_pairs_draws_one_curve_per_model():
    axes = simple_roc(
        {
            "model a": ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
            "model b": ([0, 0, 1, 1], [0.2, 0.3, 0.6, 0.9]),
        }
    )
    # two model curves plus the black chance line
    assert len(axes.get_lines()) == 3
    assert axes.get_legend() is not None


def test_legend_reports_auc_per_model():
    axes = simple_roc(
        {
            "perfect": ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]),
            "inverted": ([0, 0, 1, 1], [0.9, 0.7, 0.2, 0.1]),
        }
    )
    entries = [t.get_text() for t in axes.get_legend().get_texts()]
    assert entries == ["perfect (AUC = 1.000)", "inverted (AUC = 0.000)"]


def test_series_may_come_from_different_splits():
    # different lengths, and each pair carries its own labels
    axes = simple_roc(
        {
            "split a": ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]),
            "split b": ([0, 1, 1, 0, 1, 0], [0.3, 0.8, 0.6, 0.1, 0.9, 0.2]),
        }
    )
    assert len(axes.get_lines()) == 3


def test_multi_series_caption_drops_auc():
    axes = simple_roc({"a": ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])})
    captions = [t.get_text() for t in axes.figure.texts]
    assert captions == ["Black line: chance"]


def test_list_of_pairs_is_labelled_by_position():
    axes = simple_roc(
        [
            ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]),
            ([0, 0, 1, 1], [0.3, 0.1, 0.8, 0.6]),
        ]
    )
    entries = [t.get_text() for t in axes.get_legend().get_texts()]
    assert entries[0].startswith("Series 1")
    assert entries[1].startswith("Series 2")


def test_rejects_pair_whose_scores_do_not_match_its_labels():
    with pytest.raises(ValueError, match="model b"):
        simple_roc(
            {
                "model a": ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]),
                "model b": ([0, 0, 1, 1], [0.1, 0.2]),
            }
        )


def test_rejects_missing_y_score():
    with pytest.raises(ValueError, match="y_score is required"):
        simple_roc([0, 0, 1, 1])


def test_rejects_series_that_is_not_a_pair():
    with pytest.raises(ValueError, match="pair"):
        simple_roc({"model a": [0.1, 0.2, 0.7, 0.9, 0.5]})
