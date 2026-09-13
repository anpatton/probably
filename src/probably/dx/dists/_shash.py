"""The sinh-arcsinh distribution, which scipy does not ship.

Jones and Pewsey (2009). Two shape parameters sit on top of the usual
location and scale: ``eps`` sets the skew and ``delta`` the tail weight. At
``eps=0, delta=1`` it is exactly the standard normal, so it widens the normal
rather than replacing it.

Written as a ``scipy.stats.rv_continuous`` subclass so it behaves like any
other scipy distribution -- ``fit``, ``rvs``, ``ppf`` and
``scipy.stats.goodness_of_fit`` all work on it unchanged. scipy's extension
API passes the shape parameters through ``*args``, so each hook unpacks them
by name on its first line.
"""

from typing import Any

import numpy as np
from scipy import special, stats


def _inner(x: Any, eps: Any, delta: Any) -> Any:
    """The transform that takes the data back to a standard normal."""
    return delta * np.arcsinh(x) - eps


class _SinhArcsinh(stats.rv_continuous):
    """Sinh-arcsinh, parameterised by skew (``eps``) and tail weight (``delta``)."""

    def _argcheck(self, *args: Any) -> Any:
        eps, delta = args
        return np.isfinite(eps) & (delta > 0)

    def _pdf(self, x: Any, *args: Any) -> Any:
        return np.exp(self._logpdf(x, *args))

    def _logpdf(self, x: Any, *args: Any) -> Any:
        # Worked in logs because the fit maximises a log-likelihood, and the
        # exp(-sinh(.)**2 / 2) term underflows to zero well before the tail
        # stops mattering.
        eps, delta = args
        inner = _inner(x, eps, delta)
        return (
            np.log(np.cosh(inner))
            + np.log(delta)
            - 0.5 * np.log(2 * np.pi * (1 + x**2))
            - 0.5 * np.sinh(inner) ** 2
        )

    def _cdf(self, x: Any, *args: Any) -> Any:
        eps, delta = args
        return special.ndtr(np.sinh(_inner(x, eps, delta)))

    def _ppf(self, q: Any, *args: Any) -> Any:
        eps, delta = args
        return np.sinh((np.arcsinh(special.ndtri(q)) + eps) / delta)


shash = _SinhArcsinh(name="shash", shapes="eps, delta")
