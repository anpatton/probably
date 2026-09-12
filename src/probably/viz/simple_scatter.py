"""Simple scatter plot."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from probably.viz._utils import (
    BACKGROUND_COLOR,
    CATEGORICAL_COLORS,
    CHART_COLOR,
    INK_COLOR,
    MUTED_COLOR,
    SEQUENTIAL_COLORMAP,
    apply_legend_style,
    apply_simple_style,
    check_image_path,
    coerce_to_1d_array,
    coerce_to_1d_list,
    save_figure,
)

_LINE_COLOR = "black"
_LOESS_FRAC = 0.3
_LINE_NUM_POINTS = 200

_LINE_CAPTIONS = {
    "ab": "Black line: 1:1 reference",
    "lm": "Black line: OLS fit",
    "loess": "Black line: LOESS smooth",
}


def _ab_line(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    grid = np.linspace(low, high, 2)
    return grid, grid


def _lm_line(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    slope, intercept = np.polyfit(x, y, deg=1)
    grid = np.linspace(x.min(), x.max(), 2)
    return grid, slope * grid + intercept


def _loess_line(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    grid = np.linspace(x.min(), x.max(), _LINE_NUM_POINTS)
    bandwidth = max(int(np.ceil(_LOESS_FRAC * len(x))), 2)

    fitted = np.empty_like(grid)
    for i, point in enumerate(grid):
        distances = np.abs(x - point)
        radius = np.partition(distances, bandwidth - 1)[bandwidth - 1]
        radius = radius if radius > 0 else np.finfo(float).eps
        weights = np.clip(1 - (distances / radius) ** 3, 0, None) ** 3

        design = np.column_stack([np.ones_like(x), x])
        weighted_design = design * weights[:, None]
        coeffs, *_ = np.linalg.lstsq(weighted_design.T @ design, weighted_design.T @ y, rcond=None)
        fitted[i] = coeffs[0] + coeffs[1] * point

    return grid, fitted


_LINE_FUNCS = {"ab": _ab_line, "lm": _lm_line, "loess": _loess_line}


def _scatter_colored_by(
    axes: Axes,
    x_values: NDArray[np.floating[Any]],
    y_values: NDArray[np.floating[Any]],
    color_by: Any,
) -> None:
    try:
        numeric_values = coerce_to_1d_array(color_by, name="color_by")
    except (TypeError, ValueError):
        numeric_values = None

    if numeric_values is not None:
        scatter = axes.scatter(
            x_values,
            y_values,
            c=numeric_values,
            cmap=SEQUENTIAL_COLORMAP,
            edgecolor=BACKGROUND_COLOR,
        )
        colorbar = axes.figure.colorbar(scatter, ax=axes)
        colorbar.outline.set_edgecolor(MUTED_COLOR)
        colorbar.ax.tick_params(color=MUTED_COLOR, labelcolor=MUTED_COLOR)
        return

    labels = coerce_to_1d_list(color_by, name="color_by")
    categories = sorted(set(labels), key=labels.index)
    colors = {
        category: CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        for i, category in enumerate(categories)
    }
    for category in categories:
        mask = [label == category for label in labels]
        axes.scatter(
            x_values[mask],
            y_values[mask],
            color=colors[category],
            edgecolor=BACKGROUND_COLOR,
            label=str(category),
        )

    apply_legend_style(axes.legend())


def simple_scatter(
    x: Any,
    y: Any,
    line: Literal["ab", "lm", "loess"] | None = None,
    color_by: Any = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    filepath: str | Path | None = None,
) -> Axes:
    """Plot a scatter of two vectors.

    Parameters
    ----------
    x, y : array-like
        Single vectors of equal length, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    line : {"ab", "lm", "loess"}, optional
        Reference line to draw in black: ``"ab"`` for a 1:1 line, ``"lm"``
        for an ordinary least squares fit, or ``"loess"`` for a locally
        weighted smooth. The line type is noted in a caption below the plot.
    color_by : array-like, optional
        A vector, the same length as ``x`` and ``y``, used to color each
        point. Categorical values are colored with the package's muted
        retro palette (with a legend); numeric values are colored with its
        warm sequential ramp (with a colorbar). Colors are not configurable.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    title : str, optional
        Plot title.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the scatter was drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> axes = simple_scatter(rng.normal(size=50), rng.normal(size=50), line="lm")
    """
    path = check_image_path(filepath)
    x_values = coerce_to_1d_array(x, name="x")
    y_values = coerce_to_1d_array(y, name="y")
    if x_values.shape != y_values.shape:
        raise ValueError(
            f"x and y must be the same length, got {x_values.shape} and {y_values.shape}"
        )
    if line is not None and line not in _LINE_FUNCS:
        raise ValueError(f"line must be one of {sorted(_LINE_FUNCS)}, got {line!r}")
    if color_by is not None and len(color_by) != len(x_values):
        raise ValueError(
            f"color_by must be the same length as x and y, "
            f"got {len(color_by)} and {len(x_values)}"
        )

    _, axes = plt.subplots()

    if color_by is None:
        axes.scatter(x_values, y_values, color=CHART_COLOR, edgecolor=BACKGROUND_COLOR, alpha=0.85)
    else:
        _scatter_colored_by(axes, x_values, y_values, color_by)

    if line is not None:
        line_x, line_y = _LINE_FUNCS[line](x_values, y_values)
        axes.plot(line_x, line_y, color=_LINE_COLOR)
        axes.figure.text(
            0.5,
            0.0,
            _LINE_CAPTIONS[line],
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_COLOR,
        )

    apply_simple_style(axes)

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
