"""Check whether one or more vectors look normally distributed."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably.dx.dists._engine import TestBattery, run_battery


def _shapiro(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    result = stats.shapiro(values)
    return float(result.statistic), float(result.pvalue)


def _dagostino(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    result = stats.normaltest(values)
    return float(result.statistic), float(result.pvalue)


def _jarque_bera(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    result = stats.jarque_bera(values)
    return float(result.statistic), float(result.pvalue)


def _anderson(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
    # scipy-stubs mistypes this literal as "interpolated"; the runtime only
    # accepts "interpolate", so the call is right and the stub is not.
    result = stats.anderson(values, dist="norm", method="interpolate")  # type: ignore[call-overload]
    return float(result.statistic), float(result.pvalue)


# Four tests that look for different departures: Shapiro-Wilk is the most
# powerful overall, D'Agostino and Jarque-Bera key on skew and kurtosis, and
# Anderson-Darling weights the tails. Agreement across them is what makes a
# verdict worth trusting.
NORMAL_BATTERY: TestBattery = {
    "shapiro": _shapiro,
    "dagostino": _dagostino,
    "jarque_bera": _jarque_bera,
    "anderson": _anderson,
}


def is_this_normal(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for normality and answer yes, no, or maybe.

    Parameters
    ----------
    *x : array-like
        One or more vectors, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``. Pass a single mapping
        instead to name them; otherwise they are named by position.
    alpha : float, default 0.05
        Significance level each test is judged against.

    Returns
    -------
    list[dict]
        One record per vector, holding ``name``, ``distribution``, ``n``, a
        ``statistic`` and ``pvalue`` per test, ``alpha``, ``verdict``, and
        ``reason``.

    Examples
    --------
    >>> import numpy as np
    >>> bell = np.random.default_rng(0).normal(size=200)
    >>> skewed = np.random.default_rng(1).exponential(size=200)
    >>> [(r["name"], r["verdict"]) for r in is_this_normal(bell, skewed)]
    [('Series 1', 'yes'), ('Series 2', 'no')]
    >>> named = is_this_normal({"heights": bell})
    >>> named[0]["name"], named[0]["verdict"]
    ('heights', 'yes')
    """
    return run_battery(x, alpha, "normal", NORMAL_BATTERY)
