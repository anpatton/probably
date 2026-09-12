"""Simple histogram plot."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from probably.viz._utils import (
    BACKGROUND_COLOR,
    apply_simple_style,
    check_image_path,
    coerce_to_1d_array,
    resolve_color,
    save_figure,
)


def simple_histogram(
    x: Any,
    bins: int | str = "auto",
    xlabel: str | None = None,
    title: str | None = None,
    color: Literal["r", "y", "b"] = "r",
    filepath: str | Path | None = None,
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
    color : {"r", "y", "b"}, default "r"
        Bar color, drawn from the package palette: rust, mustard, or petrol
        teal. Single-series charts each default to a different one.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the histogram was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_histogram(np.random.default_rng(0).normal(size=100))
    """
    path = check_image_path(filepath)
    bar_color = resolve_color(color)
    values = coerce_to_1d_array(x)

    _, axes = plt.subplots()

    axes.hist(values, bins=bins, color=bar_color, edgecolor=BACKGROUND_COLOR)
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
