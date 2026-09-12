"""Simple 2x2 confusion matrix."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from probably.viz._utils import (
    INK_COLOR,
    MUTED_COLOR,
    apply_simple_style,
    check_image_path,
    coerce_binary_labels,
    coerce_to_1d_list,
    save_figure,
)

# A shade deeper than the parchment ground so the cells read as cells.
_CELL_COLOR = "#F1E6D2"


def _format_label(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def simple_2x2(
    y_true: Any,
    y_pred: Any,
    title: str | None = None,
    filepath: str | Path | None = None,
) -> Axes:
    """Plot a 2x2 confusion matrix for a binary classifier.

    Rows are the actual class and columns the predicted class, with each
    quadrant labelled TN, FP, FN, or TP. Cells are left neutral so the
    counts themselves carry the chart.

    Parameters
    ----------
    y_true : array-like
        A single vector of binary labels. The lower of the two distinct
        labels is treated as the negative class.
    y_pred : array-like
        A single vector of predicted labels, drawn from the same two
        classes and the same length as ``y_true``.
    title : str, optional
        Plot title.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the confusion matrix was drawn on.

    Examples
    --------
    >>> axes = simple_2x2([0, 0, 1, 1], [0, 1, 1, 1])
    """
    path = check_image_path(filepath)
    labels, negative_label, positive_label = coerce_binary_labels(y_true, name="y_true")
    predictions = np.asarray(coerce_to_1d_list(y_pred, name="y_pred"))
    if labels.shape != predictions.shape:
        raise ValueError(
            f"y_true and y_pred must be the same length, "
            f"got {labels.shape} and {predictions.shape}"
        )

    unexpected = set(predictions.tolist()) - {negative_label, positive_label}
    if unexpected:
        raise ValueError(
            f"y_pred must only contain the classes in y_true "
            f"({negative_label!r}, {positive_label!r}), also got {sorted(unexpected)}"
        )

    actual_negative = labels == negative_label
    predicted_negative = predictions == negative_label

    # (column, row, quadrant, count); row 1 is the top (actual negative).
    cells = [
        (0, 1, "TN", int(np.sum(actual_negative & predicted_negative))),
        (1, 1, "FP", int(np.sum(actual_negative & ~predicted_negative))),
        (0, 0, "FN", int(np.sum(~actual_negative & predicted_negative))),
        (1, 0, "TP", int(np.sum(~actual_negative & ~predicted_negative))),
    ]

    _, axes = plt.subplots()

    for column, row, quadrant, count in cells:
        axes.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=_CELL_COLOR,
                edgecolor=MUTED_COLOR,
                linewidth=1,
            )
        )
        axes.text(
            column + 0.5,
            row + 0.55,
            str(count),
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=INK_COLOR,
        )
        axes.text(
            column + 0.5,
            row + 0.25,
            quadrant,
            ha="center",
            va="center",
            fontsize=11,
            color=MUTED_COLOR,
        )

    axes.set_xlim(0, 2)
    axes.set_ylim(0, 2)
    axes.set_aspect("equal")
    axes.set_xticks([0.5, 1.5])
    axes.set_xticklabels([_format_label(negative_label), _format_label(positive_label)])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels([_format_label(positive_label), _format_label(negative_label)])
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")

    apply_simple_style(axes)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(length=0)

    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
