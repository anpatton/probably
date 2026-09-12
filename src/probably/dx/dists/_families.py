"""One table of everything known about each candidate distribution family.

The ``is_this_*`` modules stay the single source of truth for *how a family is
tested* -- their batteries are imported here, not copied -- so
:func:`probably.dx.what_is_this` and ``is_this_<family>`` can never disagree
about whether a given family fits. What this module adds is the knowledge
needed to *rank* families against each other: how to fit one, how many free
parameters that costs, and which data a family could plausibly have produced.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably.dx.dists._engine import TestBattery
from probably.dx.dists.is_this_beta import BETA_BATTERY
from probably.dx.dists.is_this_exponential import EXPONENTIAL_BATTERY
from probably.dx.dists.is_this_gamma import GAMMA_BATTERY
from probably.dx.dists.is_this_normal import NORMAL_BATTERY
from probably.dx.dists.is_this_uniform import UNIFORM_BATTERY


@dataclass(frozen=True)
class Family:
    """How to fit, rank, and test one distribution family."""

    name: str
    distribution: Any
    battery: TestBattery
    # Parameters pinned during the fit rather than estimated. Pinning `loc` at 0
    # is not a detail: fit freely, a lognormal slides its location until it
    # mimics a normal, which both ruins the fit and makes its AIC incomparable.
    fit_kwargs: dict[str, float]
    parameter_names: tuple[str, ...]
    # Data this family could not have produced, and how to say so.
    support: str = "any"

    def excluded_reason(self, values: NDArray[np.floating[Any]]) -> str | None:
        """Say why this family cannot have produced `values`, or None if it could."""
        low, high = float(values.min()), float(values.max())
        if self.support == "positive" and low <= 0:
            return (
                f"{self.name} needs strictly positive values; "
                f"this sample runs {low:g} to {high:g}"
            )
        if self.support == "unit" and (low <= 0 or high >= 1):
            return (
                f"{self.name} needs values strictly between 0 and 1; "
                f"this sample runs {low:g} to {high:g}"
            )
        return None

    def freeze(self, parameters: dict[str, float]) -> Any:
        """Turn fitted parameters back into a frozen scipy distribution."""
        return self.distribution(*(parameters[name] for name in self.parameter_names))

    def fit(self, values: NDArray[np.floating[Any]]) -> tuple[dict[str, float], float]:
        """Fit the family and return its named parameters and AIC.

        Lower AIC is a better fit. Only the freely estimated parameters count
        toward the penalty -- a pinned ``loc`` was not estimated from the data,
        so charging for it would unfairly favour families with fewer pins.
        """
        estimates = self.distribution.fit(values, **self.fit_kwargs)
        parameters = dict(zip(self.parameter_names, (float(v) for v in estimates), strict=True))
        free = len(estimates) - len(self.fit_kwargs)
        log_likelihood = float(np.sum(self.distribution.logpdf(values, *estimates)))
        return parameters, 2 * free - 2 * log_likelihood


# Ordered as a practitioner would meet them, not alphabetically.
FAMILIES: tuple[Family, ...] = (
    Family(
        name="normal",
        distribution=stats.norm,
        battery=NORMAL_BATTERY,
        fit_kwargs={},
        parameter_names=("loc", "scale"),
    ),
    Family(
        name="lognormal",
        distribution=stats.lognorm,
        battery=NORMAL_BATTERY,  # applied to log(x); see LOG_FAMILIES below
        fit_kwargs={"floc": 0},
        parameter_names=("s", "loc", "scale"),
        support="positive",
    ),
    Family(
        name="exponential",
        distribution=stats.expon,
        battery=EXPONENTIAL_BATTERY,
        fit_kwargs={"floc": 0},
        parameter_names=("loc", "scale"),
        support="positive",
    ),
    Family(
        name="gamma",
        distribution=stats.gamma,
        battery=GAMMA_BATTERY,
        fit_kwargs={"floc": 0},
        parameter_names=("a", "loc", "scale"),
        support="positive",
    ),
    Family(
        name="beta",
        distribution=stats.beta,
        battery=BETA_BATTERY,
        fit_kwargs={"floc": 0, "fscale": 1},
        parameter_names=("a", "b", "loc", "scale"),
        support="unit",
    ),
    Family(
        name="uniform",
        distribution=stats.uniform,
        battery=UNIFORM_BATTERY,
        fit_kwargs={},
        parameter_names=("loc", "scale"),
    ),
)

# Families whose test battery runs on the logarithm of the data rather than the
# data itself, matching how `is_this_lognormal` works.
LOG_FAMILIES = frozenset({"lognormal"})

FAMILIES_BY_NAME = {family.name: family for family in FAMILIES}
