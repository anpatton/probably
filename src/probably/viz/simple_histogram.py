"""Simple histogram plot."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from probably.viz._utils import CHART_COLOR, apply_simple_style, coerce_to_1d_array


def simple_histogram(
    x: Any,
    bins: int | str = "auto",
    xlabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a histogram of a single vector.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    bins : int or str, default "auto"
        Passed through to ``matplotlib.axes.Axes.hist``.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the histogram was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_histogram(np.random.default_rng(0).normal(size=100))
    """
    values = coerce_to_1d_array(x)

    _, axes = plt.subplots()

    axes.hist(values, bins=bins, color=CHART_COLOR, edgecolor="white")
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    return axes
