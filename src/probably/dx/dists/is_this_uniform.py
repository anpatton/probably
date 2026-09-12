"""Check whether one or more vectors look uniformly distributed."""

from typing import Any

from scipy import stats

from probably.dx.dists._engine import TestBattery, monte_carlo_test, run_battery

# Anderson-Darling is left out on purpose: weighting the tails as heavily as
# it does makes it unstable against a uniform, where it flags genuinely
# uniform samples often enough to be misleading.
UNIFORM_BATTERY: TestBattery = {
    "ks": monte_carlo_test(stats.uniform, "ks"),
    "cramer_von_mises": monte_carlo_test(stats.uniform, "cvm"),
}


def is_this_uniform(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for a uniform fit: yes, no, or maybe.

    Parameters
    ----------
    *x : array-like
        One or more vectors. Pass a single mapping instead to name them;
        otherwise they are named by position.
    alpha : float, default 0.05
        Significance level each test is judged against.

    Returns
    -------
    list[dict]
        One record per vector, shaped exactly like :func:`is_this_normal`'s.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> results = is_this_uniform(rng.uniform(size=200))
    >>> results[0]["verdict"]
    'yes'
    """
    return run_battery(x, alpha, "uniform", UNIFORM_BATTERY)
