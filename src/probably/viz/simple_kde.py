"""Simple 1-D kernel density estimate plot."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from probably.viz._utils import CHART_COLOR, apply_simple_style, coerce_to_1d_array


def simple_kde(
    x: Any,
    xlabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a 1-D kernel density estimate of a single vector.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the density curve was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_kde(np.random.default_rng(0).normal(size=200))
    """
    values = coerce_to_1d_array(x)
    kde = gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), 200)
    density = kde(grid)

    _, axes = plt.subplots()

    axes.plot(grid, density, color=CHART_COLOR)
    axes.fill_between(grid, density, color=CHART_COLOR, alpha=0.15)
    axes.set_ylabel("density")
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    return axes
