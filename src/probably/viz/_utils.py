"""Shared helpers for the viz subpackage."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter
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

    _format_axis_ticks(axes.xaxis)
    _format_axis_ticks(axes.yaxis)

    # Set on the persistent label objects so a later set_xlabel/set_title keeps them.
    axes.xaxis.label.set_color(INK_COLOR)
    axes.yaxis.label.set_color(INK_COLOR)
    axes.title.set_color(INK_COLOR)


SIGNIFICANT_DIGITS = 3

# Largest first, so the biggest applicable suffix wins.
_SUFFIXES = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k"))


def _strip_trailing_zeros(text: str) -> str:
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_number(value: Any, digits: int = SIGNIFICANT_DIGITS) -> str:
    """Round a number for display: few digits, no noise, nothing misleading.

    Rounds to ``digits`` significant figures and drops trailing zeros, but
    leaves an exact integer exact -- rounding a count to 3 figures would
    turn 1765 into 1770, which is wrong for something that was counted.
    Magnitudes too small or large to write plainly fall back to scientific
    notation.

    Examples
    --------
    >>> [format_number(v) for v in (3.7580001, 0.03374892, 1765.4321)]
    ['3.76', '0.0337', '1770']
    >>> [format_number(v) for v in (150.0, 0.0, -1.396)]
    ['150', '0', '-1.4']
    >>> format_number(0.00004821)
    '4.82e-05'
    """
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    if number == 0:
        return "0"
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))

    magnitude = abs(number)
    if magnitude >= 1e15 or magnitude < 1e-4:
        mantissa, exponent_text = f"{number:.{digits - 1}e}".split("e")
        return f"{_strip_trailing_zeros(mantissa)}e{exponent_text}"

    exponent = math.floor(math.log10(magnitude))
    # Negative places round above the decimal point (1765.4 -> 1770), which is
    # what makes this significant figures rather than decimal places. Only the
    # display width is clamped at zero.
    places = -(exponent - digits + 1)
    return _strip_trailing_zeros(f"{round(number, places):.{max(0, places)}f}")


def format_tick(value: float, _position: Any = None) -> str:
    """Format an axis tick, abbreviating large numbers as 1.2k / 3.4M / 1.1B.

    Formatting every tick in full also drops matplotlib's shared "1e6" offset
    label, which otherwise floats in the corner of the axes.

    Examples
    --------
    >>> [format_tick(v) for v in (0, 2500, 1_200_000, 0.25)]
    ['0', '2.5k', '1.2M', '0.25']
    """
    if not math.isfinite(value):
        return ""
    if value == 0:
        return "0"

    magnitude = abs(value)
    for threshold, suffix in _SUFFIXES:
        if magnitude >= threshold:
            return format_number(value / threshold) + suffix
    return format_number(value)


def _format_axis_ticks(axis: Axis) -> None:
    """Apply the tick formatter, unless the axis carries caller-set labels.

    A FixedLocator means the ticks were pinned by hand (``set_xticks``), and
    categorical units mean matplotlib is labelling strings. In both cases the
    labels are text, not numbers, and formatting them would replace them with
    their underlying positions. Note that ``set_xticklabels`` installs a
    FuncFormatter rather than a FixedFormatter, so the locator -- not the
    formatter -- is the dependable signal.
    """
    if isinstance(axis.get_major_locator(), FixedLocator):
        return
    if isinstance(axis.get_major_formatter(), FixedFormatter):
        return
    if axis.units is not None:
        return
    axis.set_major_formatter(FuncFormatter(format_tick))


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
