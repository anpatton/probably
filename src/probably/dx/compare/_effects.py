"""Effect sizes and the verdict rule behind ``is_this_different``."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably.dx.dists._engine import CONFIDENT_N

_OVERLAP_GRID_POINTS = 400


def overlap(reference: NDArray[np.floating[Any]], other: NDArray[np.floating[Any]]) -> float:
    """Share of density the two samples have in common, from 0 to 1.

    The overlapping coefficient: the area under whichever density curve is
    lower at each point. 1 means the two are indistinguishable, 0 that they
    occupy separate ranges entirely. Unlike a p-value it does not grow more
    impressive with sample size -- it answers "how far apart", not "how sure".

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=500)
    >>> round(overlap(x, x))
    1
    """
    try:
        reference_density = stats.gaussian_kde(reference)
        other_density = stats.gaussian_kde(other)
    except (np.linalg.LinAlgError, ValueError):
        # A sample with no spread has no density to estimate.
        return float("nan")

    low = min(reference.min(), other.min())
    high = max(reference.max(), other.max())
    if low == high:
        return float("nan")

    # Pad the grid so tails that fall outside the observed range still count.
    pad = 0.1 * (high - low)
    grid = np.linspace(low - pad, high + pad, _OVERLAP_GRID_POINTS)
    shared = np.minimum(reference_density(grid), other_density(grid))
    return float(np.clip(np.trapezoid(shared, grid), 0.0, 1.0))


def prob_greater(reference: NDArray[np.floating[Any]], other: NDArray[np.floating[Any]]) -> float:
    """Chance a value drawn from `other` exceeds one drawn from `reference`.

    The common-language effect size: 0.5 means the two are interchangeable,
    0.8 means four times out of five a draw from `other` wins. This is the
    Mann-Whitney U statistic divided by the number of pairs, so it costs
    nothing beyond the test already being run.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> round(prob_greater(rng.normal(size=2000), rng.normal(loc=3, size=2000)), 1)
    1.0
    """
    if reference.size == 0 or other.size == 0:
        return float("nan")
    statistic = stats.mannwhitneyu(other, reference).statistic
    return float(statistic / (other.size * reference.size))


def verdict_for_difference(pvalues: list[float], smallest_n: int, alpha: float) -> tuple[str, str]:
    """Turn a battery of two-sample p-values into a yes/no/maybe plus its reason.

    The mirror image of
    :func:`probably.dx.dists._engine.verdict_for`, and deliberately not a reuse
    of it. There the null is "this matches the family", so nothing rejecting
    means yes. Here the null is "these two are the same", so rejecting is what
    means yes -- they differ. The two rules point in opposite directions and
    sharing one would silently invert the answer.

    Examples
    --------
    >>> verdict_for_difference([0.001, 0.002], 100, 0.05)[0]
    'yes'
    >>> verdict_for_difference([0.6, 0.8], 100, 0.05)[0]
    'no'
    >>> verdict_for_difference([0.6, 0.001], 100, 0.05)[0]
    'maybe'
    """
    usable = [p for p in pvalues if np.isfinite(p)]
    if not usable:
        return "maybe", "no test could be computed for these samples"

    rejected = sum(1 for p in usable if p < alpha)
    total = len(usable)

    if rejected == total:
        return "yes", f"all {total} tests find a difference at alpha={alpha}"
    if rejected > 0:
        return "maybe", f"{rejected} of {total} tests find a difference at alpha={alpha}"
    if smallest_n < CONFIDENT_N:
        return (
            "maybe",
            f"no test finds a difference, but n={smallest_n} is too small to be convincing",
        )
    return "no", f"no test finds a difference at alpha={alpha}"
