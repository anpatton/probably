"""Simple 1-D kernel density estimate plot."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from probably.viz._utils import (
    CATEGORICAL_COLORS,
    CHART_COLOR,
    apply_legend_style,
    apply_simple_style,
    coerce_to_series,
)

_GRID_POINTS = 200
_FILL_ALPHA = 0.15


def _draw_density(
    axes: Axes,
    values: NDArray[np.floating[Any]],
    grid: NDArray[np.floating[Any]],
    color: str,
    label: str | None,
) -> None:
    density = gaussian_kde(values)(grid)
    axes.plot(grid, density, color=color, label=label)
    axes.fill_between(grid, density, color=color, alpha=_FILL_ALPHA)


def simple_kde(
    x: Any,
    xlabel: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a 1-D kernel density estimate.

    Parameters
    ----------
    x : array-like
        Either a single vector of values -- a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, ``polars.Series`` -- or an input holding several
        of them, in which case each becomes its own overlapping curve. A
        mapping takes its legend labels from the keys; a list of lists or a
        2-D array is labelled by position. Series may differ in length.
        Groups are colored with the package's muted retro palette, the same
        colors ``simple_scatter`` uses, and are not configurable.
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
    >>> rng = np.random.default_rng(0)
    >>> axes = simple_kde(rng.normal(size=200))
    >>> grouped = simple_kde({"a": rng.normal(size=50), "b": rng.normal(3, 1, 80)})
    """
    series = coerce_to_series(x)

    # One grid across every series, so overlapping curves stay comparable.
    all_values = np.concatenate([values for _, values in series])
    grid = np.linspace(all_values.min(), all_values.max(), _GRID_POINTS)

    _, axes = plt.subplots()

    if series[0][0] is None:
        _draw_density(axes, series[0][1], grid, CHART_COLOR, None)
    else:
        for index, (label, values) in enumerate(series):
            _draw_density(
                axes,
                values,
                grid,
                CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)],
                label,
            )
        apply_legend_style(axes.legend())

    axes.set_ylabel("Density")
    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    return axes
