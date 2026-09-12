"""Shared machinery behind the ``is_this_*`` distribution checks."""

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from probably._coerce import coerce_to_1d_array

# A goodness-of-fit test whose parameters were estimated from the same data
# has no analytic null, so the null is simulated. More samples buy p-value
# resolution at a real cost in time -- fitting a beta 199 times is seconds,
# not milliseconds -- and 199 already resolves far past the usual alpha.
MC_SAMPLES = 199

# Below this, a battery of normality tests has so little power that "nothing
# rejected" says almost nothing, so the verdict is capped at "maybe".
CONFIDENT_N = 20

VERDICTS = ("yes", "no", "maybe")

# Each entry maps a test name to a callable returning (statistic, pvalue).
TestBattery = dict[str, Callable[[NDArray[np.floating[Any]]], tuple[float, float]]]


def monte_carlo_test(
    distribution: Any, statistic: Literal["ad", "ks", "cvm", "filliben"]
) -> Callable[[NDArray[np.floating[Any]]], tuple[float, float]]:
    """Build a goodness-of-fit test against `distribution` with fitted parameters.

    Uses scipy's Monte Carlo null rather than comparing to a table, because
    estimating the parameters from the sample being tested makes the standard
    critical values anti-conservative -- they would call too many samples a
    good fit.
    """

    def run(values: NDArray[np.floating[Any]]) -> tuple[float, float]:
        # Fitting hundreds of simulated resamples makes scipy grumble about
        # its own optimizer on some of them. Those warnings are about the
        # simulated null, not the caller's data, so there is nothing they
        # could act on -- the p-value itself is still sound.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = stats.goodness_of_fit(
                distribution,
                values,
                statistic=statistic,
                n_mc_samples=MC_SAMPLES,
                rng=np.random.default_rng(0),
            )
        return float(result.statistic), float(result.pvalue)

    return run


def verdict_for(pvalues: list[float], n: int, alpha: float) -> tuple[str, str]:
    """Turn a battery of p-values into a yes/no/maybe plus its reason."""
    usable = [p for p in pvalues if np.isfinite(p)]
    if not usable:
        return "maybe", "no test could be computed for this sample"

    rejected = sum(1 for p in usable if p < alpha)
    total = len(usable)

    if rejected == total:
        return "no", f"all {total} tests reject at alpha={alpha}"
    if rejected > 0:
        return "maybe", f"{rejected} of {total} tests reject at alpha={alpha}"
    if n < CONFIDENT_N:
        return "maybe", f"no test rejects, but n={n} is too small to be convincing"
    return "yes", f"no test rejects at alpha={alpha}"


def labelled_inputs(x: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """Label the input vectors, taking names from a mapping when one is given.

    Examples
    --------
    >>> [name for name, _ in labelled_inputs(([1, 2], [3, 4]))]
    ['Series 1', 'Series 2']
    >>> [name for name, _ in labelled_inputs(({"a": [1, 2], "b": [3]},))]
    ['a', 'b']
    """
    if not x:
        raise ValueError("pass at least one vector")
    if len(x) == 1 and isinstance(x[0], Mapping):
        return [(str(key), value) for key, value in x[0].items()]
    return [(f"Series {i + 1}", item) for i, item in enumerate(x)]


def run_battery(
    x: tuple[Any, ...],
    alpha: float,
    distribution: str,
    battery: TestBattery,
    prepare: Callable[[NDArray[np.floating[Any]], str], NDArray[np.floating[Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Run `battery` over every input vector and return one record each."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")

    records = []
    for name, item in labelled_inputs(x):
        values = coerce_to_1d_array(item, name=name)
        if values.size == 0:
            raise ValueError(f"{name} must have at least 1 value, got an empty vector")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be all finite values")
        if prepare is not None:
            values = prepare(values, name)

        record: dict[str, Any] = {"name": name, "distribution": distribution, "n": values.size}
        pvalues = []
        for test_name, test in battery.items():
            try:
                statistic, pvalue = test(values)
            except Exception:
                # A test that cannot run on this sample (usually too few
                # points) reports nan rather than removing its own keys, so
                # every record from one call has the same shape.
                statistic, pvalue = float("nan"), float("nan")
            record[f"{test_name}_statistic"] = statistic
            record[f"{test_name}_pvalue"] = pvalue
            pvalues.append(pvalue)

        record["alpha"] = alpha
        record["verdict"], record["reason"] = verdict_for(pvalues, values.size, alpha)
        records.append(record)

    return records


def require_positive(values: NDArray[np.floating[Any]], name: str) -> NDArray[np.floating[Any]]:
    """Reject values a strictly-positive distribution cannot have produced."""
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive for this distribution")
    return values


def require_unit_interval(
    values: NDArray[np.floating[Any]], name: str
) -> NDArray[np.floating[Any]]:
    """Reject values outside the open unit interval a beta is defined on."""
    if np.any(values <= 0) or np.any(values >= 1):
        raise ValueError(f"{name} must lie strictly between 0 and 1 for a beta distribution")
    return values
