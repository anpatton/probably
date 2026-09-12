"""Input coercion shared across the package."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def coerce_to_1d_array(x: Any, name: str = "x") -> NDArray[np.floating[Any]]:
    """Coerce an array-like input to a 1-D numpy float array.

    Examples
    --------
    >>> coerce_to_1d_array([1, 2, 3])
    array([1., 2., 3.])
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a single (1-D) vector, got shape {arr.shape}")
    return arr


def clean_vector(x: Any, name: str = "x", minimum: int = 1) -> NDArray[np.floating[Any]]:
    """Coerce a vector and reject one too short or not finite to work with.

    `minimum` is whatever the caller needs to do its job: 1 where a single
    value still means something, 2 where a fit or a spread is involved.

    Examples
    --------
    >>> clean_vector([3, 1, 2])
    array([3., 1., 2.])
    """
    values = coerce_to_1d_array(x, name=name)
    if values.size < minimum:
        word = "value" if minimum == 1 else "values"
        raise ValueError(f"{name} must have at least {minimum} {word}, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be all finite values")
    return values


def holds_sub_vectors(x: Any) -> bool:
    """Is this several vectors rather than one?

    Examples
    --------
    >>> holds_sub_vectors([1, 2, 3])
    False
    >>> holds_sub_vectors([[1, 2], [3, 4]])
    True
    """
    if isinstance(x, np.ndarray):
        return x.ndim > 1
    try:
        first = next(iter(x))
    except (TypeError, StopIteration):
        return False
    return not isinstance(first, (str, bytes)) and hasattr(first, "__len__")
