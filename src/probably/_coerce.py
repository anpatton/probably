"""Input coercion shared across the package."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


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
