import math

import numpy as np
import pytest

from probably.dx import (
    is_this_beta,
    is_this_exponential,
    is_this_gamma,
    is_this_lognormal,
    is_this_normal,
    is_this_uniform,
)
from probably.dx.dists._engine import CONFIDENT_N, VERDICTS, verdict_for

# Each check paired with a sample it should accept and one it should reject.
# The rejecting sample is chosen to be genuinely the wrong shape, not merely
# a different parameterization of the same family.
CHECKS = [
    (is_this_normal, lambda r: r.normal(size=200), lambda r: r.exponential(size=200)),
    (is_this_lognormal, lambda r: r.lognormal(size=200), lambda r: r.uniform(size=200)),
    (is_this_exponential, lambda r: r.exponential(size=200), lambda r: r.uniform(size=200)),
    (is_this_uniform, lambda r: r.uniform(size=200), lambda r: r.exponential(size=200)),
]


@pytest.mark.parametrize("check,matching,mismatching", CHECKS)
def test_recognizes_its_own_distribution(check, matching, mismatching):
    assert check(matching(np.random.default_rng(0)))[0]["verdict"] == "yes"


@pytest.mark.parametrize("check,matching,mismatching", CHECKS)
def test_rejects_a_different_distribution(check, matching, mismatching):
    assert check(mismatching(np.random.default_rng(1)))[0]["verdict"] == "no"


def test_beta_and_gamma_recognize_their_own():
    # separated from CHECKS because the simulated null makes these slow
    rng = np.random.default_rng(0)
    assert is_this_beta(rng.beta(2, 5, size=100))[0]["verdict"] == "yes"
    assert is_this_gamma(rng.gamma(2, size=100))[0]["verdict"] == "yes"


def test_accepts_an_arbitrary_number_of_vectors():
    rng = np.random.default_rng(0)
    for count in (1, 2, 5):
        results = is_this_normal(*[rng.normal(size=50) for _ in range(count)])
        assert len(results) == count
        assert [r["name"] for r in results] == [f"Series {i + 1}" for i in range(count)]


def test_a_mapping_names_the_vectors():
    rng = np.random.default_rng(0)
    results = is_this_normal({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    assert [r["name"] for r in results] == ["a", "b"]


def test_every_record_has_the_same_keys():
    rng = np.random.default_rng(0)
    results = is_this_normal(rng.normal(size=50), rng.normal(size=5))
    assert set(results[0]) == set(results[1])


def test_record_reports_each_test_and_the_verdict():
    record = is_this_normal(np.random.default_rng(0).normal(size=100))[0]
    assert record["distribution"] == "normal"
    assert record["n"] == 100
    assert record["alpha"] == 0.05
    assert record["verdict"] in VERDICTS
    assert isinstance(record["reason"], str) and record["reason"]
    for name in ("shapiro", "dagostino", "jarque_bera", "anderson"):
        assert math.isfinite(record[f"{name}_statistic"])
        assert math.isfinite(record[f"{name}_pvalue"])


def test_small_samples_are_never_a_confident_yes():
    # too few points for "nothing rejected" to mean anything
    record = is_this_normal(np.random.default_rng(0).normal(size=CONFIDENT_N - 1))[0]
    assert record["verdict"] == "maybe"
    assert "too small" in record["reason"]


def test_a_test_that_cannot_run_reports_nan_without_dropping_keys():
    record = is_this_normal([1.0, 2.0, 3.0])[0]
    assert set(record) >= {"dagostino_pvalue", "shapiro_pvalue"}
    # D'Agostino needs at least 8 points; Shapiro runs from 3
    assert math.isnan(record["dagostino_pvalue"])
    assert math.isfinite(record["shapiro_pvalue"])


@pytest.mark.parametrize(
    "pvalues,n,expected",
    [
        ([0.9, 0.8, 0.7], 100, "yes"),
        ([0.001, 0.002, 0.003], 100, "no"),
        ([0.9, 0.001, 0.8], 100, "maybe"),
        ([0.9, 0.8, 0.7], 5, "maybe"),
        ([float("nan")], 100, "maybe"),
    ],
)
def test_verdict_rules(pvalues, n, expected):
    assert verdict_for(pvalues, n, 0.05)[0] == expected


def test_alpha_changes_the_verdict():
    values = np.random.default_rng(0).normal(size=200)
    assert is_this_normal(values, alpha=0.5)[0]["verdict"] != "yes"


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        is_this_normal([1.0, 2.0, 3.0], alpha=alpha)


def test_rejects_no_vectors():
    with pytest.raises(ValueError, match="at least one vector"):
        is_this_normal()


@pytest.mark.parametrize("check", [is_this_lognormal, is_this_exponential, is_this_gamma])
def test_positive_only_checks_reject_non_positive_values(check):
    with pytest.raises(ValueError, match="strictly positive"):
        check([1.0, 0.0, 2.0])


@pytest.mark.parametrize("bad", [[0.5, 0.0, 0.7], [0.5, 1.0, 0.7], [0.5, 1.2]])
def test_beta_rejects_values_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        is_this_beta(bad)


def test_rejects_non_finite_and_empty_input():
    with pytest.raises(ValueError, match="finite"):
        is_this_normal([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="at least 1 value"):
        is_this_normal([])


def test_pandas_and_polars_series_work():
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    values = np.random.default_rng(0).normal(size=50)
    expected = is_this_normal(values)[0]["verdict"]
    assert is_this_normal(pd.Series(values))[0]["verdict"] == expected
    assert is_this_normal(pl.Series(values))[0]["verdict"] == expected
