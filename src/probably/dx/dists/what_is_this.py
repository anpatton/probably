"""Identify which distribution a vector came from."""

import warnings
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from probably._coerce import coerce_to_1d_array
from probably.dx.dists._engine import verdict_for
from probably.dx.dists._families import FAMILIES, FAMILIES_BY_NAME, LOG_FAMILIES

# How many candidates come back. Each one is fitted, tested, and reported --
# past the first few the families are too far behind to be worth reading.
TOP_N = 3

# Two fits within this much AIC of each other are not meaningfully different.
# Reported as `tied_with_best` so the reader need not know the rule.
TIE_THRESHOLD = 2.0


class Identification(NamedTuple):
    """What :func:`what_is_this` found: the distribution, and the evidence.

    Attributes
    ----------
    distribution : scipy.stats frozen distribution
        The best-fitting family, ready to use -- ``rvs`` to simulate, ``ppf``
        for quantiles, ``cdf`` for probabilities.
    diagnostics : list[dict]
        One record per candidate, best first, each with its fit statistics
        and its own goodness-of-fit verdict.
    """

    distribution: Any
    diagnostics: list[dict[str, Any]]


def what_is_this(x: Any, alpha: float = 0.05) -> Identification:
    """Identify which distribution a vector came from.

    Fits every distribution family the data could plausibly have come from,
    ranks them by how well they fit, and returns both halves of the answer:
    the winning distribution ready to use, and the evidence behind it.

    Two different questions are being answered, and keeping them apart
    matters.

    **Which family fits best?** That is the order of ``diagnostics``, and it
    is a *comparison* between candidates, by AIC. It is not ordered by
    p-value, and first place does not mean "significant" -- it means "closest
    of the families tried".

    **Does the winner actually fit?** That is each record's ``verdict``, and
    it is *absolute*. Something always ranks first even when nothing fits:
    bimodal data is none of these families, yet one still wins. A warning is
    raised when the winner fails its own tests, since in a script nobody
    reads the diagnostics.

    Parameters
    ----------
    x : array-like
        A single vector of values, e.g. a ``list``, ``numpy.ndarray``,
        ``pandas.Series``, or ``polars.Series``.
    alpha : float, default 0.05
        Significance level the goodness-of-fit tests are judged against.

    Returns
    -------
    Identification
        A named tuple of ``distribution`` and ``diagnostics``, so it can be
        read by name or unpacked. ``diagnostics`` holds the top candidates,
        best first, each carrying:

        ``distribution``, ``n``
            Which family this record is about, and the sample size.
        ``fit_rank``, ``aic``, ``delta_aic``, ``tied_with_best``
            Rank 1 is the best fit. ``delta_aic`` is how far behind the best
            this family is; ``tied_with_best`` marks anything within 2, which
            is not a meaningful gap.
        ``parameters``
            The fitted parameters, named.
        ``verdict``, ``reason``
            ``"yes"``, ``"no"``, or ``"maybe"`` -- does this family actually
            fit -- and one sentence covering both where it placed and what
            its tests said.

    Warns
    -----
    UserWarning
        When every test rejects the best-fitting family. It is still
        returned, since it remains the closest candidate, but it does not
        describe this data.

    Notes
    -----
    Testing a candidate simulates a null distribution, so a call takes a few
    seconds when beta or gamma rank highly.

    A family's verdict matches what the corresponding ``is_this_*`` function
    would say, because it is the same battery. Verdicts are *not* strictly
    comparable between families, though: they run different numbers of tests,
    so a family facing fewer tests has fewer chances to fail. ``delta_aic``
    is the like-for-like comparison.

    Examples
    --------
    >>> import numpy as np
    >>> yields = np.random.default_rng(0).lognormal(0, 0.5, size=300)
    >>> found = what_is_this(yields)

    The distribution is a plain scipy object, ready to use:

    >>> found.distribution.dist.name
    'lognorm'
    >>> round(float(found.distribution.ppf(0.95)), 3)
    2.269
    >>> draws = found.distribution.rvs(size=1000, random_state=0)

    The diagnostics say why, and whether to trust it:

    >>> for row in found.diagnostics:
    ...     print(row["distribution"], row["verdict"])
    lognormal yes
    gamma maybe
    normal no
    >>> found.diagnostics[0]["reason"]
    'fits best of the 5 families that could apply; no test rejects at alpha=0.05'

    It unpacks like any tuple:

    >>> distribution, diagnostics = what_is_this(yields)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    values = _clean(x)
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


def _clean(x: Any) -> NDArray[np.floating[Any]]:
    values = coerce_to_1d_array(x)
    if values.size < 2:
        raise ValueError(f"x must have at least 2 values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must be all finite values")
    return values


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
