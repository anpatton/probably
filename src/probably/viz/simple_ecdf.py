"""Simple empirical cumulative distribution plot."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from probably.viz._diagnostics import sorted_values
from probably.viz._utils import (
    apply_simple_style,
    check_image_path,
    coerce_to_1d_array,
    resolve_color,
    save_figure,
)


def simple_ecdf(
    x: Any,
    xlabel: str | None = None,
    title: str | None = None,
    color: Literal["r", "y", "b"] = "y",
    filepath: str | Path | None = None,
) -> Axes:
    """Plot the empirical cumulative distribution of a single vector.

    Every observation is shown as one step, so unlike a histogram or a
    density there is nothing to smooth and no bin width to choose -- the
    chart is the data. Read a quantile off the vertical axis.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.
    color : {"r", "y", "b"}, default "y"
        Line color, drawn from the package palette: rust, mustard, or
        petrol teal.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the step curve was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_ecdf(np.random.default_rng(0).normal(size=100))
    """
    path = check_image_path(filepath)
    line_color = resolve_color(color)
    observed = sorted_values(coerce_to_1d_array(x))

    cumulative = np.arange(1, observed.size + 1) / observed.size

    _, axes = plt.subplots()

    axes.step(observed, cumulative, where="post", color=line_color, linewidth=2)
    axes.set_ylim(0, 1.02)

    apply_simple_style(axes)
    axes.set_ylabel("Cumulative Probability")

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
