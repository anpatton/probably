"""Shared helpers for the viz subpackage."""

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

_SPINE_COLOR = "#888888"

# Shared bar/line color for every simple_* chart, so they read as one family.
CHART_COLOR = "#2563EB"

# RColorBrewer's Set1 palette, used to color categorical `color_by` values.
CATEGORICAL_COLORS = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
    "#999999",
]

# Colormap used to color continuous `color_by` values.
SEQUENTIAL_COLORMAP = "viridis"


def apply_simple_style(axes: Axes) -> None:
    """Apply the shared, minimal look used by every ``simple_*`` chart.

    Drops the top/right spines, lightens the remaining axis lines, and
    keeps tick labels plain -- so every ``simple_*`` function produces a
    chart with the same clean feel.
    """
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(_SPINE_COLOR)
    axes.spines["bottom"].set_color(_SPINE_COLOR)
    axes.tick_params(colors=_SPINE_COLOR)


def coerce_to_1d_array(x: Any, name: str = "x") -> NDArray[np.floating[Any]]:
    """Coerce an array-like input to a 1-D numpy float array.

    Accepts anything numpy can convert via ``np.asarray`` -- a ``list``,
    ``numpy.ndarray``, ``pandas.Series``, ``polars.Series``, etc.

    Examples
    --------
    >>> coerce_to_1d_array([1, 2, 3])
    array([1., 2., 3.])
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a single (1-D) vector, got shape {arr.shape}")
    return arr


def coerce_to_1d_list(x: Any, name: str = "x") -> list[Any]:
    """Coerce an array-like input of labels to a plain list.

    Like :func:`coerce_to_1d_array`, but for non-numeric vectors (e.g. bar
    chart category labels) where forcing a float dtype would fail.

    Examples
    --------
    >>> coerce_to_1d_list(["a", "b", "c"])
    ['a', 'b', 'c']
    """
    values = list(x)
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in values):
        raise ValueError(f"{name} must be a single (1-D) vector")
    return values
