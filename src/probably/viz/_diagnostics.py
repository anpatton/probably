"""Shared pieces of the distribution-diagnostic charts."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from probably._coerce import clean_vector

# Blom's plotting positions. Subtracting 3/8 keeps the extreme points off 0 and
# 1, where a normal quantile is infinite and the plot would have nothing to draw.
_PLOTTING_OFFSET = 0.375

# Both diagnostics compare against a normal fitted to the data itself, so the
# reference line is always 1:1 -- unlike simple_scatter, where the line could be
# one of several fits, there is nothing to caption.
REFERENCE_COLOR = "black"


def sorted_values(x: Any, name: str = "x") -> NDArray[np.floating[Any]]:
    """Coerce a vector and sort it ascending, rejecting one too short to diagnose.

    Examples
    --------
    >>> sorted_values([3.0, 1.0, 2.0])
    array([1., 2., 3.])
    """
    return np.sort(clean_vector(x, name=name, minimum=2))


def plotting_positions(n: int) -> NDArray[np.floating[Any]]:
    """Cumulative probabilities to compare `n` sorted observations against.

    Examples
    --------
    >>> plotting_positions(3).round(3)
    array([0.192, 0.5  , 0.808])
    """
    return (np.arange(1, n + 1) - _PLOTTING_OFFSET) / (n + 1 - 2 * _PLOTTING_OFFSET)
