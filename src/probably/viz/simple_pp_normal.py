"""Simple normal probability-probability plot."""

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
    resolve_color,
    save_figure,
)


def simple_pp_normal(
    x: Any,
    title: str | None = None,
    color: Literal["r", "y", "b"] = "b",
    filepath: str | Path | None = None,
) -> Axes:
    """Plot empirical against theoretical probabilities for a fitted normal.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    title : str, optional
        Plot title.
    color : {"r", "y", "b"}, default "b"
        Point color, drawn from the package palette: rust, mustard, or
        petrol teal.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.


    Returns
    -------
    matplotlib.axes.Axes
        The axes the probabilities were drawn on.


    Examples
    --------
    >>> import numpy as np
    >>> axes = simple_pp_normal(np.random.default_rng(0).normal(size=100))
    """
    path = check_image_path(filepath)
    point_color = resolve_color(color)
    observed = sorted_values(x)

    theoretical = norm.cdf(observed, observed.mean(), observed.std(ddof=1))
    empirical = plotting_positions(observed.size)

    _, axes = plt.subplots()

    axes.scatter(theoretical, empirical, color=point_color, edgecolor=BACKGROUND_COLOR, alpha=0.85)
    axes.plot([0, 1], [0, 1], color=REFERENCE_COLOR)

    apply_simple_style(axes)
    axes.set_xlabel("Theoretical Probability")
    axes.set_ylabel("Empirical Probability")

    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
