"""Fit a sinh-arcsinh distribution to a vector."""

from typing import Any

from probably.dx.dists._engine import TestBattery, monte_carlo_test
from probably.dx.dists._families import Family, Identification, fit_and_check
from probably.dx.dists._shash import shash

SHASH_BATTERY: TestBattery = {
    "ks": monte_carlo_test(shash, "ks"),
    "cramer_von_mises": monte_carlo_test(shash, "cvm"),
}

SHASH_FAMILY = Family(
    name="shash",
    distribution=shash,
    battery=SHASH_BATTERY,
    fit_kwargs={},
    parameter_names=("eps", "delta", "loc", "scale"),
)


def fit_shash(x: Any, alpha: float = 0.05, quick: bool = False) -> Identification:
    """Fit a sinh-arcsinh distribution, and say whether it actually fits.

    Sinh-arcsinh covers the whole real line and widens the normal rather than
    replacing it: ``eps`` adds skew, ``delta`` sets tail weight, and at
    ``eps=0, delta=1`` it is exactly the normal. scipy does not ship it, so
    this package defines it.

    Parameters
    ----------
    x : array-like
        A single vector of values.
    alpha : float, default 0.05
        Significance level the goodness-of-fit tests are judged against.
    quick : bool, default False
        Skip the goodness-of-fit check and just fit. Roughly 400 times
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
    probably.dx.fit_johnsonsu : the other flexible four-parameter family.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.default_rng(0).normal(size=200)
    >>> distribution, diagnostics = fit_shash(data, quick=True)
    >>> print(diagnostics[0]["verdict"])
    None

    The fitted object is a plain scipy distribution:

    >>> round(float(distribution.ppf(0.95)), 3)
    1.519
    """
    return fit_and_check(SHASH_FAMILY, x, alpha, quick)
