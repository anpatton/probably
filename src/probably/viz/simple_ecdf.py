"""Simple empirical cumulative distribution plot."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from probably.viz._diagnostics import sorted_values
from probably.viz._utils import (
    CATEGORICAL_COLORS,
    apply_legend_style,
    apply_simple_style,
    check_image_path,
    coerce_to_series,
    resolve_color,
    save_figure,
)


def _draw_steps(
    axes: Axes, values: NDArray[np.floating[Any]], color: str, label: str | None
) -> None:
    observed = sorted_values(values)
    cumulative = np.arange(1, observed.size + 1) / observed.size
    axes.step(observed, cumulative, where="post", color=color, linewidth=2, label=label)


def simple_ecdf(
    x: Any,
    xlabel: str | None = None,
    title: str | None = None,
    color: Literal["r", "y", "b"] = "y",
    filepath: str | Path | None = None,
) -> Axes:
    """Plot the empirical cumulative distribution of one or more vectors.

    Every observation is shown as one step, so unlike a histogram or a
    density there is nothing to smooth and no bin width to choose -- the
    chart is the data. Read a quantile off the vertical axis.

    Passing several series overlays one curve each, which is the clearest
    picture of how two distributions differ: the widest vertical gap between
    two curves is exactly what a Kolmogorov-Smirnov test measures.

    Parameters
    ----------
    x : array-like or mapping
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``. An input holding several
        sub-items becomes one curve each: a mapping takes its legend labels
        from the keys, and a list of lists is labelled by position.
    xlabel : str, optional
        Label for the x-axis.
    title : str, optional
        Plot title.
    color : {"r", "y", "b"}, default "y"
        Line color, drawn from the package palette: rust, mustard, or
        petrol teal. Applies only when there is one series -- several take
        the package's categorical palette so they stay distinguishable.
    filepath : str or pathlib.Path, optional
        Write the figure to this path. Must end in ``.png``, ``.jpg``, or
        ``.jpeg``. The axes are returned either way.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the step curves were drawn on.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> axes = simple_ecdf(rng.normal(size=100))

    One curve per group, labelled by the mapping's keys:

    >>> axes = simple_ecdf({"control": rng.normal(size=100),
    ...                     "treated": rng.normal(loc=2, size=100)})
    >>> [text.get_text() for text in axes.get_legend().get_texts()]
    ['control', 'treated']
    """
    path = check_image_path(filepath)
    line_color = resolve_color(color)
    series = coerce_to_series(x)

    _, axes = plt.subplots()

    if series[0][0] is None:
        _draw_steps(axes, series[0][1], line_color, None)
    else:
        for index, (label, values) in enumerate(series):
            _draw_steps(axes, values, CATEGORICAL_COLORS[index % len(CATEGORICAL_COLORS)], label)
        apply_legend_style(axes.legend())

    axes.set_ylim(0, 1.02)

    apply_simple_style(axes)
    axes.set_ylabel("Cumulative Probability")

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if title is not None:
        axes.set_title(title)

    save_figure(axes, path)

    return axes
