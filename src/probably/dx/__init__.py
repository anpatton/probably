"""Diagnostic tools for 1-D distributions."""

from probably.dx.compare import is_this_different
from probably.dx.describe import describe
from probably.dx.dists import (
    fit_johnsonsu,
    fit_shash,
    is_this_beta,
    is_this_exponential,
    is_this_gamma,
    is_this_lognormal,
    is_this_normal,
    is_this_uniform,
    what_is_this,
)

__all__ = [
    "describe",
    "fit_johnsonsu",
    "fit_shash",
    "is_this_different",
    "is_this_beta",
    "is_this_exponential",
    "is_this_gamma",
    "is_this_lognormal",
    "is_this_normal",
    "is_this_uniform",
    "what_is_this",
]
