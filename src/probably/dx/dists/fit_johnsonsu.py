"""Fit a Johnson SU distribution to a vector."""

from typing import Any

from scipy import stats

from probably.dx.dists._engine import TestBattery, monte_carlo_test
from probably.dx.dists._families import Family, Identification, fit_and_check

JOHNSONSU_BATTERY: TestBattery = {
    "ks": monte_carlo_test(stats.johnsonsu, "ks"),
    "cramer_von_mises": monte_carlo_test(stats.johnsonsu, "cvm"),
}

JOHNSONSU_FAMILY = Family(
    name="johnsonsu",
    distribution=stats.johnsonsu,
    battery=JOHNSONSU_BATTERY,
    fit_kwargs={},
    parameter_names=("a", "b", "loc", "scale"),
)


def fit_johnsonsu(x: Any, alpha: float = 0.05, quick: bool = False) -> Identification:
    """Fit a Johnson SU distribution, and say whether it actually fits.

    Johnson SU is a four-parameter family covering the whole real line, with
    separate controls for skew and tail weight. Reach for it when a named
    family has already been ruled out.

    Parameters
    ----------
    x : array-like
        A single vector of values.
    alpha : float, default 0.05
        Significance level the goodness-of-fit tests are judged against.
    quick : bool, default False
        Skip the goodness-of-fit check and just fit. Roughly 500 times
        faster, because the check simulates its null. The verdict comes back
        as ``None``.

    Returns
    -------
    Identification
        Named tuple of ``distribution`` (a frozen scipy distribution) and
        ``diagnostics`` (a one-record list holding ``distribution``, ``n``,
        ``aic``, ``parameters``, ``verdict``, and ``reason``).

    Warns
    -----
    UserWarning
        When the fitted distribution does not pass its own tests.

    See Also
    --------
    probably.dx.what_is_this : search the named families instead.
    probably.dx.fit_shash : the other flexible four-parameter family.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> data = stats.johnsonsu.rvs(-2, 2, size=200, random_state=0)
    >>> distribution, diagnostics = fit_johnsonsu(data)
    >>> diagnostics[0]["verdict"]
    'yes'
    >>> round(float(distribution.ppf(0.95)), 3)
    2.941

    Skip the check when you only need the fitted distribution:

    >>> distribution, diagnostics = fit_johnsonsu(data, quick=True)
    >>> print(diagnostics[0]["verdict"])
    None
    """
    return fit_and_check(JOHNSONSU_FAMILY, x, alpha, quick)
