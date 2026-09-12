"""Shared pieces of the distribution-diagnostic charts."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Blom's plotting positions. Subtracting 3/8 keeps the extreme points off 0 and
# 1, where a normal quantile is infinite and the plot would have nothing to draw.
_PLOTTING_OFFSET = 0.375

# Both diagnostics compare against a normal fitted to the data itself, so the
# reference line is always 1:1 -- unlike simple_scatter, where the line could be
# one of several fits, there is nothing to caption.
REFERENCE_COLOR = "black"


def sorted_values(x: NDArray[np.floating[Any]], name: str = "x") -> NDArray[np.floating[Any]]:
    """Sort a vector ascending, rejecting one too short to diagnose.

    Examples
    --------
    >>> sorted_values(np.array([3.0, 1.0, 2.0]))
    array([1., 2., 3.])
    """
    if x.size < 2:
        raise ValueError(f"{name} must have at least 2 values, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be all finite values")
    return np.sort(x)


def plotting_positions(n: int) -> NDArray[np.floating[Any]]:
    """Cumulative probabilities to compare `n` sorted observations against.

    Examples
    --------
    >>> plotting_positions(3).round(3)
    array([0.192, 0.5  , 0.808])
    """
    return (np.arange(1, n + 1) - _PLOTTING_OFFSET) / (n + 1 - 2 * _PLOTTING_OFFSET)
