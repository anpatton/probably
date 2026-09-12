"""Distribution-fit tests that answer yes, no, or maybe."""

from probably.dx.dists.is_this_beta import is_this_beta
from probably.dx.dists.is_this_exponential import is_this_exponential
from probably.dx.dists.is_this_gamma import is_this_gamma
from probably.dx.dists.is_this_lognormal import is_this_lognormal
from probably.dx.dists.is_this_normal import is_this_normal
from probably.dx.dists.is_this_uniform import is_this_uniform

__all__ = [
    "is_this_beta",
    "is_this_exponential",
    "is_this_gamma",
    "is_this_lognormal",
    "is_this_normal",
    "is_this_uniform",
]
