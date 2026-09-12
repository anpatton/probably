"""Identify which distribution a vector came from."""

import warnings
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from probably._coerce import clean_vector
from probably.dx.dists._engine import verdict_for
from probably.dx.dists._families import FAMILIES, FAMILIES_BY_NAME, LOG_FAMILIES

# How many candidates come back. Each one is fitted, tested, and reported --
# past the first few the families are too far behind to be worth reading.
TOP_N = 3

# Two fits within this much AIC of each other are not meaningfully different.
# Reported as `tied_with_best` so the reader need not know the rule.
TIE_THRESHOLD = 2.0


class Identification(NamedTuple):
    """The distribution :func:`what_is_this` found, and the evidence for it."""

    distribution: Any
    diagnostics: list[dict[str, Any]]


def what_is_this(x: Any, alpha: float = 0.05) -> Identification:
    """Identify which distribution a vector came from.

    Parameters
    ----------
    x : array-like
        A single vector of values.
    alpha : float, default 0.05
        Significance level the goodness-of-fit tests are judged against.

    Returns
    -------
    Identification
        Named tuple of ``distribution`` (a frozen scipy distribution) and
        ``diagnostics`` (the top candidates, best fit first). Each diagnostic
        record holds ``distribution``, ``n``, ``fit_rank``, ``aic``,
        ``delta_aic``, ``tied_with_best``, ``parameters``, ``verdict``, and
        ``reason``.

    Warns
    -----
    UserWarning
        When every test rejects the best-fitting family.

    Examples
    --------
    >>> import numpy as np
    >>> yields = np.random.default_rng(0).lognormal(0, 0.5, size=300)
    >>> found = what_is_this(yields)
    >>> found.distribution.dist.name
    'lognorm'
    >>> round(float(found.distribution.ppf(0.95)), 3)
    2.269
    >>> for row in found.diagnostics:
    ...     print(row["distribution"], row["verdict"])
    lognormal yes
    gamma maybe
    normal no
    >>> found.diagnostics[0]["reason"]
    'fits best of the 5 families that could apply; no test rejects at alpha=0.05'
    >>> distribution, diagnostics = what_is_this(yields)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    values = clean_vector(x, minimum=2)
    fitted = _fit_candidates(values)
    if not fitted:
        raise ValueError("no candidate distribution could be fitted to this data")

    fitted.sort(key=lambda record: record["aic"])
    best = fitted[0]
    diagnostics = fitted[:TOP_N]

    for position, record in enumerate(diagnostics, start=1):
        record["fit_rank"] = position
        record["delta_aic"] = record["aic"] - best["aic"]
        record["tied_with_best"] = record["delta_aic"] < TIE_THRESHOLD
        record["verdict"], note = _check(record["distribution"], values, alpha)
        record["reason"] = _explain(record, len(fitted), best["distribution"], note)

    if best["verdict"] == "no":
        warnings.warn(
            f"the closest family is {best['distribution']}, but it does not actually fit: "
            f"{best['reason']}. It is returned anyway; read the diagnostics before using it.",
            UserWarning,
            stacklevel=2,
        )

    return Identification(
        FAMILIES_BY_NAME[best["distribution"]].freeze(best["parameters"]), diagnostics
    )


def _fit_candidates(values: NDArray[np.floating[Any]]) -> list[dict[str, Any]]:
    """Fit every family the data could have come from.

    Families whose support excludes the data are skipped rather than
    reported: a beta cannot have produced values outside 0 to 1, so ranking
    one would be noise.
    """
    fitted: list[dict[str, Any]] = []
    for family in FAMILIES:
        if family.excluded_reason(values) is not None:
            continue
        try:
            parameters, aic = family.fit(values)
        except Exception:
            continue
        fitted.append(
            {
                "distribution": family.name,
                "n": values.size,
                "fit_rank": 0,
                "aic": aic,
                "delta_aic": 0.0,
                "tied_with_best": False,
                "parameters": parameters,
                "verdict": "",
                "reason": "",
            }
        )
    return fitted


def _explain(record: dict[str, Any], considered: int, best_name: str, note: str) -> str:
    """Say in one sentence why this family sits here, and whether it fits.

    The ranking is relative (which of these fits best) while the verdict is
    absolute (does this one fit at all), and a list sorted by fit quality
    invites the reader to assume it was sorted by p-value. So every row says
    which half is which.
    """
    if record["fit_rank"] == 1:
        fit = f"fits best of the {considered} families that could apply"
    elif record["tied_with_best"]:
        fit = f"fits about as well as {best_name} (AIC {record['delta_aic']:.1f} higher)"
    else:
        fit = f"fits worse than {best_name} (AIC {record['delta_aic']:.1f} higher)"
    return f"{fit}; {note}"


def _check(family_name: str, values: NDArray[np.floating[Any]], alpha: float) -> tuple[str, str]:
    """Run one family's own battery and reduce it to a verdict.

    The battery comes from the matching ``is_this_*`` module, so a verdict
    here and a verdict there are the same judgement made by the same code.
    """
    family = FAMILIES_BY_NAME[family_name]
    tested = np.log(values) if family.name in LOG_FAMILIES else values

    pvalues = []
    for test in family.battery.values():
        try:
            _, pvalue = test(tested)
        except Exception:
            pvalue = float("nan")
        pvalues.append(pvalue)

    return verdict_for(pvalues, len(values), alpha)
