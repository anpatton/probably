"""Simple horizontal bar chart."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from probably.viz._utils import (
    CHART_COLOR,
    apply_simple_style,
    coerce_to_1d_array,
    coerce_to_1d_list,
)


def simple_bar(
    categories: Any,
    values: Any,
    xlabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a horizontal bar chart of category values.

    Bars are horizontal (rather than vertical) so they are never confused
    with a histogram, and are sorted with the largest value at the top.

    Parameters
    ----------
    categories : array-like
        A single vector of category labels.
    values : array-like
        A single vector of values, one per category.
    xlabel : str, optional
        Label for the x-axis (the value axis).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the bar chart was drawn on.

    Examples
    --------
    >>> axes = simple_bar(["a", "b", "c"], [3, 1, 2])
    """
    category_labels = coerce_to_1d_list(categories, name="categories")
    value_array = coerce_to_1d_array(values, name="values")
    if len(category_labels) != len(value_array):
        raise ValueError(
            f"categories and values must be the same length, "
            f"got {len(category_labels)} and {len(value_array)}"
        )

    order = np.argsort(value_array)
    sorted_labels = [category_labels[i] for i in order]
    sorted_values = value_array[order]

    _, axes = plt.subplots()

    axes.barh(sorted_labels, sorted_values, color=CHART_COLOR)
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    return axes
