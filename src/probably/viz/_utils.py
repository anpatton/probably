"""Shared helpers for the viz subpackage."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend
from numpy.typing import NDArray

# Warm parchment ground, dark ink, and a muted rule color -- the base of the
# lightly retro look every simple_* chart shares.
BACKGROUND_COLOR = "#FAF4E8"
INK_COLOR = "#3E3228"
MUTED_COLOR = "#9A8C7C"
GRID_COLOR = "#DCD4C6"

# Single-series charts and the first categorical slot share this rust.
CHART_COLOR = "#C4562F"

# Muted 1970s palette for categorical `color_by` values, ordered so the
# earliest slots -- the ones most charts actually use -- stay far apart.
CATEGORICAL_COLORS = [
    CHART_COLOR,  # rust
    "#2E6F7E",  # petrol teal
    "#D9A441",  # mustard
    "#4A5B8C",  # slate blue
    "#6B7F3A",  # olive
    "#A3586B",  # clay rose
    "#8A5A38",  # warm brown
    "#6E4B6E",  # plum
    "#8C8279",  # warm gray
]

# Warm ramp for continuous point colors. It stops at a light tan rather than
# fading to near-white, so the low end stays visible against the parchment.
SEQUENTIAL_COLORMAP = LinearSegmentedColormap.from_list(
    "probably_retro", ["#E9C9A3", CHART_COLOR, "#4A2416"]
)

# The same ramp for filled densities, but starting at the parchment ground so
# empty regions dissolve into the page instead of painting it a flat tan.
DENSITY_COLORMAP = LinearSegmentedColormap.from_list(
    "probably_retro_density", [BACKGROUND_COLOR, CHART_COLOR, "#4A2416"]
)


def apply_simple_style(axes: Axes) -> None:
    """Apply the shared, lightly retro look used by every ``simple_*`` chart.

    Lays the plot on a warm parchment ground over a light gray grid, drops
    the top/right spines, and settles the remaining rules and tick labels
    into a muted warm gray with darker ink for the labels and title -- so
    every ``simple_*`` function produces a chart with the same feel.
    """
    axes.figure.set_facecolor(BACKGROUND_COLOR)
    axes.set_facecolor(BACKGROUND_COLOR)

    axes.grid(True, color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)  # keep the grid behind the data, never over it

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(MUTED_COLOR)
    axes.spines["bottom"].set_color(MUTED_COLOR)
    axes.tick_params(color=MUTED_COLOR, labelcolor=MUTED_COLOR)

    # Set on the persistent label objects so a later set_xlabel/set_title keeps them.
    axes.xaxis.label.set_color(INK_COLOR)
    axes.yaxis.label.set_color(INK_COLOR)
    axes.title.set_color(INK_COLOR)


def apply_legend_style(legend: Legend) -> None:
    """Settle a legend onto the parchment ground, matching the chart around it."""
    legend.get_frame().set_facecolor(BACKGROUND_COLOR)
    legend.get_frame().set_edgecolor(MUTED_COLOR)
    for text in legend.get_texts():
        text.set_color(INK_COLOR)


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


def _is_nested(x: Any) -> bool:
    """Is this input a sequence of sub-sequences rather than a flat vector?"""
    try:
        first = next(iter(x))
    except (TypeError, StopIteration):
        return False
    return not isinstance(first, (str, bytes)) and hasattr(first, "__len__")


def split_labelled_items(x: Any) -> list[tuple[str | None, Any]]:
    """Split an input into labelled sub-items, or one unlabelled item.

    This carries the labelling half of the multi-series convention, shared by
    every ``simple_*`` chart that accepts several series. A mapping takes its
    labels from the keys; any other nested input -- a list of lists, a 2-D
    array -- is labelled by position; anything else is a single unlabelled
    item.

    Examples
    --------
    >>> split_labelled_items([1, 2, 3])
    [(None, [1, 2, 3])]
    >>> [label for label, _ in split_labelled_items({"a": [1, 2], "b": [3]})]
    ['a', 'b']
    >>> [label for label, _ in split_labelled_items([[1, 2], [3, 4]])]
    ['Series 1', 'Series 2']
    """
    if isinstance(x, Mapping):
        return [(str(key), value) for key, value in x.items()]

    if _is_nested(x):
        return [(f"Series {index + 1}", item) for index, item in enumerate(x)]

    return [(None, x)]


def coerce_to_series(x: Any, name: str = "x") -> list[tuple[str | None, NDArray[np.floating[Any]]]]:
    """Coerce an input into one or more labelled series of values.

    This is the standard way ``simple_*`` charts accept multiple series: a
    flat vector is one unlabelled series, while an input holding several
    sub-items becomes one series each, labelled by
    :func:`split_labelled_items`. Sub-items may differ in length.

    Examples
    --------
    >>> [(label, values.tolist()) for label, values in coerce_to_series([1, 2, 3])]
    [(None, [1.0, 2.0, 3.0])]
    >>> [label for label, _ in coerce_to_series({"a": [1, 2], "b": [3, 4, 5]})]
    ['a', 'b']
    >>> [label for label, _ in coerce_to_series([[1, 2], [3, 4]])]
    ['Series 1', 'Series 2']
    """
    return [
        (label, coerce_to_1d_array(item, name=name if label is None else f"{name}[{label!r}]"))
        for label, item in split_labelled_items(x)
    ]


def coerce_binary_labels(y: Any, name: str = "y") -> tuple[NDArray[Any], Any, Any]:
    """Coerce an array-like of binary labels to an array plus its two classes.

    The lower of the two distinct labels is treated as the negative class and
    the higher as the positive class, so ``0``/``1``, ``False``/``True``, and
    ``"no"``/``"yes"`` all work without configuration.

    Examples
    --------
    >>> values, negative, positive = coerce_binary_labels([0, 1, 1, 0])
    >>> negative, positive
    (0, 1)
    """
    values = np.asarray(coerce_to_1d_list(y, name=name))
    classes = sorted(set(values.tolist()))
    if len(classes) != 2:
        raise ValueError(f"{name} must have exactly 2 distinct values, got {len(classes)}")
    return values, classes[0], classes[1]
