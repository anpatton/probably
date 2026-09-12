"""Simple normal quantile-quantile plot."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm

from probably.viz._diagnostics import REFERENCE_COLOR, plotting_positions, sorted_values
from probably.viz._utils import (
    BACKGROUND_COLOR,
    apply_simple_style,
    check_image_path,
    coerce_to_1d_array,
    resolve_color,
    save_figure,
)


def simple_qq_normal(
    x: Any,
    title: str | None = None,
    color: Literal["r", "y", "b"] = "r",
    filepath: str | Path | None = None,
) -> Axes:
    """Plot sample quantiles against the quantiles of a fitted normal.

    Points falling along the black 1:1 line mean the values are close to
    normal; a curve, an S-shape, or drifting tails mean they are not. The
    normal is fitted to the data, so no reference parameters are needed.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    title : str, optional
        Plot title.
    color : {"r", "y", "b"}, default "r"
        Point color, drawn from the package palette: rust, mustard, or
        petrol teal.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the quantiles were drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_qq_normal(np.random.default_rng(0).normal(size=100))
    """
    path = check_image_path(filepath)
    point_color = resolve_color(color)
    observed = sorted_values(coerce_to_1d_array(x))

    theoretical = norm.ppf(plotting_positions(observed.size), observed.mean(), observed.std(ddof=1))

    _, axes = plt.subplots()

    axes.scatter(theoretical, observed, color=point_color, edgecolor=BACKGROUND_COLOR, alpha=0.85)

    low = min(theoretical.min(), observed.min())
    high = max(theoretical.max(), observed.max())
    axes.plot([low, high], [low, high], color=REFERENCE_COLOR)

    apply_simple_style(axes)
    axes.set_xlabel("Theoretical Quantiles")
    axes.set_ylabel("Sample Quantiles")

    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
