"""Check whether one or more vectors look beta distributed."""

from typing import Any

from scipy import stats

from probably.dx.dists._engine import (
    TestBattery,
    monte_carlo_test,
    require_unit_interval,
    run_battery,
)

BETA_BATTERY: TestBattery = {
    "ks": monte_carlo_test(stats.beta, "ks"),
    "cramer_von_mises": monte_carlo_test(stats.beta, "cvm"),
}


def is_this_beta(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for a beta fit: yes, no, or maybe.

    Slower than the other checks -- a beta has no closed-form fit, so the
    simulated null refits one for every resample. Expect seconds, not
    milliseconds.

    Parameters
    ----------
    *x : array-like
        One or more vectors of values strictly between 0 and 1. Pass a
        single mapping instead to name them; otherwise they are named by
        position.
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
    >>> results = is_this_beta(rng.beta(2, 5, size=100))
    >>> results[0]["verdict"]
    'yes'
    """
    return run_battery(x, alpha, "beta", BETA_BATTERY, prepare=require_unit_interval)
