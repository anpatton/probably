"""Check whether one or more vectors look exponentially distributed."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably.dx.dists._engine import (
    TestBattery,
    monte_carlo_test,
    require_positive,
    run_battery,
)


def _anderson(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    # scipy-stubs mistypes this literal as "interpolated"; the runtime only
    # accepts "interpolate", so the call is right and the stub is not.
    result = stats.anderson(values, dist="expon", method="interpolate")  # type: ignore[call-overload]
    return float(result.statistic), float(result.pvalue)


EXPONENTIAL_BATTERY: TestBattery = {
    "anderson": _anderson,
    "ks": monte_carlo_test(stats.expon, "ks"),
    "cramer_von_mises": monte_carlo_test(stats.expon, "cvm"),
}


def is_this_exponential(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for an exponential fit: yes, no, or maybe.

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
    >>> results = is_this_exponential(rng.exponential(size=200))
    >>> results[0]["verdict"]
    'yes'
    """
    return run_battery(x, alpha, "exponential", EXPONENTIAL_BATTERY, prepare=require_positive)
