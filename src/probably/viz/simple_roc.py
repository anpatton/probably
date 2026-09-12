"""Simple ROC curve plot."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve

from probably.viz._utils import (
    CATEGORICAL_COLORS,
    CHART_COLOR,
    INK_COLOR,
    apply_legend_style,
    apply_simple_style,
    coerce_binary_labels,
    coerce_to_1d_array,
    split_labelled_items,
)

_REFERENCE_COLOR = "black"


def _format_auc(auc: float) -> str:
    """AUC always carries its decimals, so a perfect score reads 1.000, not 1.

    This is the one number the package does not round smartly: it is bounded
    on [0, 1], and a column of AUCs is easier to compare at a fixed width.
    """
    return f"{auc:.3f}"


def _roc_series(
    y_true: Any, y_score: Any
) -> list[tuple[str | None, NDArray[Any], NDArray[np.floating[Any]], Any]]:
    """Resolve the inputs into one (label, labels, scores, positive) per curve."""
    if y_score is not None:
        pairs: list[tuple[str | None, Any, Any]] = [(None, y_true, y_score)]
    else:
        items = split_labelled_items(y_true)
        if items[0][0] is None:
            raise ValueError(
                "y_score is required unless y_true holds several (y_true, y_score) pairs"
            )
        pairs = []
        for label, item in items:
            try:
                truth, scores = item
            except (TypeError, ValueError):
                raise ValueError(
                    f"each series must be a (y_true, y_score) pair, got {item!r} for {label!r}"
                ) from None
            pairs.append((label, truth, scores))

    series = []
    for label, truth, scores in pairs:
        where = "y_true" if label is None else f"series {label!r}"
        labels, _, positive_label = coerce_binary_labels(truth, name=where)
        score_values = coerce_to_1d_array(scores, name=f"{where} scores")
        if score_values.shape != labels.shape:
            raise ValueError(
                f"scores and labels must be the same length in {where}, "
                f"got {score_values.shape} and {labels.shape}"
            )
        series.append((label, labels, score_values, positive_label))
    return series


def simple_roc(
    y_true: Any,
    y_score: Any = None,
    title: str | None = None,
) -> Axes:
    """Plot a ROC curve for one or more binary classifiers.

    Parameters
    ----------
    y_true : array-like
        A single vector of binary labels, paired with ``y_score``. To compare
        several classifiers instead, pass their ``(y_true, y_score)`` pairs
        here as the only data argument and leave ``y_score`` unset -- each
        pair carries its own labels, so the classifiers need not have been
        evaluated on the same split. A mapping takes its legend labels from
        the keys; a list of pairs is labelled by position. In every case the
        lower of a pair's two distinct labels is the negative class.
    y_score : array-like, optional
        Scores or predicted probabilities for the positive class, the same
        length as ``y_true``. Omitted when passing pairs.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the ROC curve was drawn on.

    Examples
    --------
    >>> axes = simple_roc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> compared = simple_roc(
    ...     {
    ...         "model a": ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
    ...         "model b": ([0, 1, 1], [0.2, 0.6, 0.9]),
    ...     }
    ... )
    """
    series = _roc_series(y_true, y_score)

    _, axes = plt.subplots()
    axes.plot([0, 1], [0, 1], color=_REFERENCE_COLOR)

    caption = "Black line: chance"

    for index, (label, labels, scores, positive_label) in enumerate(series):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            labels, scores, pos_label=positive_label
        )
        auc = roc_auc_score(labels == positive_label, scores)

        if label is None:
            # A lone curve has no legend, so its AUC goes in the caption.
            color = CHART_COLOR
            caption = f"AUC = {_format_auc(auc)} · {caption}"
        else:
            color = CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)]
            label = f"{label} (AUC = {_format_auc(auc)})"

        axes.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2,
            label=label,
        )

    if series[0][0] is not None:
        apply_legend_style(axes.legend(loc="lower right"))

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    apply_simple_style(axes)

    axes.figure.text(
        0.5,
        0.0,
        caption,
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK_COLOR,
    )

    if title is not None:
        axes.set_title(title)

    return axes
