"""Diagnostic tools for 1-D distributions."""

from probably.dx.describe import describe
from probably.dx.dists import (
    is_this_beta,
    is_this_exponential,
    is_this_gamma,
    is_this_lognormal,
    is_this_normal,
    is_this_uniform,
)

__all__ = [
    "describe",
    "is_this_beta",
    "is_this_exponential",
    "is_this_gamma",
    "is_this_lognormal",
    "is_this_normal",
    "is_this_uniform",
]
