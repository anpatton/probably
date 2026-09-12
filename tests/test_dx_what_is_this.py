import warnings
from functools import cache

import numpy as np
import pytest
from scipy import stats

from probably.dx import (
    is_this_beta,
    is_this_exponential,
    is_this_gamma,
    is_this_lognormal,
    is_this_normal,
    is_this_uniform,
    what_is_this,
)
from probably.dx.dists._families import FAMILIES
from probably.dx.dists.what_is_this import TIE_THRESHOLD, TOP_N, Identification

# Each family, a sample from it, its scipy name, and the is_this_* that owns it.
SAMPLES = {
    "normal": (lambda r: r.normal(10, 2, size=200), "norm", is_this_normal),
    "lognormal": (lambda r: r.lognormal(0, 0.5, size=200), "lognorm", is_this_lognormal),
    "exponential": (lambda r: r.exponential(2, size=200), "expon", is_this_exponential),
    "gamma": (lambda r: r.gamma(3, size=200), "gamma", is_this_gamma),
    "beta": (lambda r: r.beta(2, 5, size=200), "beta", is_this_beta),
    "uniform": (lambda r: r.uniform(size=200), "uniform", is_this_uniform),
}
FAMILY_NAMES = [family.name for family in FAMILIES]


# Testing a candidate simulates its null, which costs seconds for beta and
# gamma. The samples are deterministic, so questions about the same data share
# one answer.
@cache
def _found(family):
    draw, _, _ = SAMPLES[family]
    return what_is_this(draw(np.random.default_rng(0)))


@cache
def _asked_directly(family):
    draw, _, is_this = SAMPLES[family]
    return is_this(draw(np.random.default_rng(0)))


def _by_name(diagnostics, distribution):
    return next(r for r in diagnostics if r["distribution"] == distribution)


# --- the distribution half -------------------------------------------------


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_finds_the_family_the_data_came_from(family):
    # the true family must be among the candidates offered; which of the top
    # few wins can legitimately shift, since some families nest others (a beta
    # with a = b = 1 *is* the uniform) and AIC separates them by very little
    found = _found(family)
    assert family in [r["distribution"] for r in found.diagnostics]


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_the_true_family_is_never_rejected(family):
    assert _by_name(_found(family).diagnostics, family)["verdict"] != "no"


def test_the_returned_object_is_always_the_top_ranked_family():
    for family in FAMILY_NAMES:
        found = _found(family)
        assert found.diagnostics[0]["fit_rank"] == 1
        winner = found.diagnostics[0]["parameters"]
        assert found.distribution.args == tuple(winner.values())


def test_the_distribution_is_a_usable_scipy_object():
    found = what_is_this(np.random.default_rng(0).normal(10, 2, size=500))
    assert isinstance(found.distribution, stats.distributions.rv_frozen)
    assert found.distribution.rvs(size=7, random_state=0).shape == (7,)
    assert found.distribution.mean() == pytest.approx(10, abs=0.5)
    assert found.distribution.std() == pytest.approx(2, abs=0.5)
    assert found.distribution.cdf(found.distribution.ppf(0.9)) == pytest.approx(0.9)


def test_the_returned_distribution_is_the_top_ranked_one():
    found = _found("gamma")
    assert found.diagnostics[0]["fit_rank"] == 1
    assert found.distribution.dist.name == "gamma"
    assert found.diagnostics[0]["distribution"] == "gamma"


def test_the_fitted_parameters_match_the_data():
    values = np.random.default_rng(0).normal(10, 2, size=1000)
    found = what_is_this(values)
    assert found.distribution.ppf(0.5) == pytest.approx(np.median(values), abs=0.2)


def test_it_unpacks_and_reads_by_name():
    distribution, diagnostics = _found("normal")
    assert distribution is _found("normal").distribution
    assert diagnostics is _found("normal").diagnostics
    assert isinstance(_found("normal"), Identification)


# --- the diagnostics half --------------------------------------------------


def test_returns_the_top_candidates_only():
    assert len(_found("normal").diagnostics) == TOP_N


def test_every_record_has_the_same_keys_and_a_real_verdict():
    diagnostics = _found("normal").diagnostics
    assert all(set(r) == set(diagnostics[0]) for r in diagnostics)
    # no None verdicts and no unranked rows -- everything returned was tested
    assert all(r["verdict"] in ("yes", "no", "maybe") for r in diagnostics)
    assert [r["fit_rank"] for r in diagnostics] == [1, 2, 3]


def test_ranked_by_fit_not_by_p_value():
    diagnostics = _found("lognormal").diagnostics
    aics = [r["aic"] for r in diagnostics]
    assert aics == sorted(aics)
    assert all(r["delta_aic"] == pytest.approx(r["aic"] - aics[0]) for r in diagnostics)


def test_the_reason_names_both_the_ranking_and_the_verdict():
    # a list sorted by fit quality invites the reader to assume it was sorted
    # by p-value, so every row has to say which half is which
    diagnostics = _found("lognormal").diagnostics
    assert diagnostics[0]["reason"].startswith("fits best of the")
    assert "AIC" in diagnostics[1]["reason"]
    assert all("alpha" in r["reason"] for r in diagnostics)


def test_tied_with_best_marks_only_close_fits():
    for record in _found("normal").diagnostics:
        assert record["tied_with_best"] == (record["delta_aic"] < TIE_THRESHOLD)
    assert _found("normal").diagnostics[0]["tied_with_best"]


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_agrees_with_the_matching_is_this_function(family):
    # both must call one implementation of "does this family fit", so the two
    # entry points can never drift apart
    found = _found(family)
    if family not in [r["distribution"] for r in found.diagnostics]:
        pytest.skip(f"{family} did not reach the top {TOP_N} for its own sample")
    assert _by_name(found.diagnostics, family)["verdict"] == _asked_directly(family)[0]["verdict"]


def test_families_the_data_cannot_come_from_are_not_offered():
    # a beta cannot produce values outside 0 to 1, so it is not a candidate
    values = np.random.default_rng(0).uniform(1, 7, size=200)
    assert "beta" not in [r["distribution"] for r in what_is_this(values).diagnostics]


def test_negative_data_drops_every_positive_only_family():
    found = what_is_this(np.random.default_rng(0).normal(-10, 1, size=200))
    offered = [r["distribution"] for r in found.diagnostics]
    assert not {"lognormal", "gamma", "exponential"} & set(offered)


# --- the safety net --------------------------------------------------------


def test_a_winner_that_does_not_fit_warns_and_is_still_returned():
    # the reason ranking alone is unsafe: bimodal data is none of these
    # families, but something always wins
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(2, 0.3, 200), rng.normal(8, 0.4, 200)])
    with pytest.warns(UserWarning, match="does not actually fit"):
        found = what_is_this(bimodal)
    assert isinstance(found.distribution, stats.distributions.rv_frozen)
    assert found.diagnostics[0]["verdict"] == "no"


def test_a_good_fit_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        what_is_this(np.random.default_rng(0).normal(size=300))


# --- input handling --------------------------------------------------------


def test_accepts_any_vector_type():
    values = np.random.default_rng(0).normal(size=200)
    assert what_is_this(list(values)).distribution.dist.name == (
        what_is_this(values).distribution.dist.name
    )


def test_pandas_and_polars_series_work():
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    values = np.random.default_rng(0).normal(size=200)
    expected = what_is_this(values).distribution.dist.name
    assert what_is_this(pd.Series(values)).distribution.dist.name == expected
    assert what_is_this(pl.Series(values)).distribution.dist.name == expected


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        what_is_this([1.0, 2.0, 3.0], alpha=alpha)


def test_rejects_tiny_and_non_finite_input():
    with pytest.raises(ValueError, match="at least 2 values"):
        what_is_this([1.0])
    with pytest.raises(ValueError, match="finite"):
        what_is_this([1.0, np.nan, 3.0])
