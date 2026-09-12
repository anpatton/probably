"""Simple 2-D kernel density estimate plot."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from probably.viz._utils import apply_simple_style, coerce_to_1d_array

_GRID_SIZE = 100


def simple_kde_2d(
    x: Any,
    y: Any,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a 2-D kernel density estimate of two vectors.

    Parameters
    ----------
    x, y : array-like
        Single vectors of equal length, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the density contours were drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> axes = simple_kde_2d(rng.normal(size=200), rng.normal(size=200))
    """
    x_values = coerce_to_1d_array(x, name="x")
    y_values = coerce_to_1d_array(y, name="y")
    if x_values.shape != y_values.shape:
        raise ValueError(
            f"x and y must be the same length, got {x_values.shape} and {y_values.shape}"
        )

    kde = gaussian_kde(np.vstack([x_values, y_values]))
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_values.min(), x_values.max(), _GRID_SIZE),
        np.linspace(y_values.min(), y_values.max(), _GRID_SIZE),
    )
    density = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

    _, axes = plt.subplots()

    axes.contourf(x_grid, y_grid, density, cmap="Blues")
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)

    return axes
