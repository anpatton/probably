"""Check whether one or more vectors look gamma distributed."""

from typing import Any

from scipy import stats

from probably.dx.dists._engine import (
    TestBattery,
    monte_carlo_test,
    require_positive,
    run_battery,
)

GAMMA_BATTERY: TestBattery = {
    "ks": monte_carlo_test(stats.gamma, "ks"),
    "cramer_von_mises": monte_carlo_test(stats.gamma, "cvm"),
}


def is_this_gamma(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for a gamma fit: yes, no, or maybe.

    Slower than the other checks, for the same reason as
    :func:`is_this_beta` -- the simulated null refits the distribution for
    every resample.

    Parameters
    ----------
    *x : array-like
        One or more vectors of strictly positive values. Pass a single
        mapping instead to name them; otherwise they are named by position.
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
    >>> results = is_this_gamma(rng.gamma(2, size=100))
    >>> results[0]["verdict"]
    'yes'
    """
    return run_battery(x, alpha, "gamma", GAMMA_BATTERY, prepare=require_positive)
