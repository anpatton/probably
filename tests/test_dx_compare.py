import math

import numpy as np
import pytest

from probably.dx import is_this_different
from probably.dx.compare._effects import overlap, prob_greater, verdict_for_difference
from probably.dx.compare.is_this_different import TEST_NAMES


def _samples(seed=0, n=300):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(loc=2, size=n)


def test_clearly_shifted_samples_are_different():
    control, treated = _samples()
    assert is_this_different(control, treated)[0]["verdict"] == "yes"


def test_samples_from_the_same_distribution_are_not_different():
    rng = np.random.default_rng(0)
    assert is_this_different(rng.normal(size=300), rng.normal(size=300))[0]["verdict"] == "no"


@pytest.mark.parametrize("seed", range(6))
def test_the_null_case_is_not_trigger_happy(seed):
    # two draws from one distribution should rarely be called different
    rng = np.random.default_rng(seed)
    verdict = is_this_different(rng.normal(size=300), rng.normal(size=300))[0]["verdict"]
    assert verdict != "yes"


def test_verdict_points_the_right_way():
    # the null here is "these are the same", the opposite of is_this_normal's,
    # so rejecting means "yes, different" -- guard against a silent inversion
    control, _ = _samples()
    assert is_this_different(control, control)[0]["verdict"] == "no"
    assert is_this_different(control, control + 100)[0]["verdict"] == "yes"


def test_small_samples_cannot_earn_a_confident_no():
    # nothing rejects, but eight points cannot show that two things match --
    # absence of evidence is not evidence of sameness
    record = is_this_different(list(range(1, 9)), [v + 0.1 for v in range(1, 9)])[0]
    assert record["verdict"] == "maybe"
    assert "too small" in record["reason"]


def test_small_samples_can_still_find_a_real_difference():
    # the small-n cap guards the "no" direction only; rejecting despite low
    # power is stronger evidence, not weaker
    record = is_this_different(list(range(1, 9)), [v + 100 for v in range(1, 9)])[0]
    assert record["verdict"] == "yes"


@pytest.mark.parametrize(
    "pvalues,n,expected",
    [
        ([0.001, 0.002, 0.003], 300, "yes"),
        ([0.6, 0.7, 0.8], 300, "no"),
        ([0.6, 0.001, 0.8], 300, "maybe"),
        ([0.6, 0.7, 0.8], 5, "maybe"),
        ([float("nan")], 300, "maybe"),
    ],
)
def test_verdict_rules(pvalues, n, expected):
    assert verdict_for_difference(pvalues, n, 0.05)[0] == expected


def test_record_carries_every_test_and_effect_size():
    control, treated = _samples()
    record = is_this_different(control, treated)[0]
    for name in TEST_NAMES:
        assert math.isfinite(record[f"{name}_statistic"])
        assert math.isfinite(record[f"{name}_pvalue"])
        assert math.isfinite(record[f"{name}_pvalue_adjusted"])
    for key in ("prob_greater", "overlap", "median_shift", "wasserstein"):
        assert math.isfinite(record[key])
    assert record["n_reference"] == 300
    assert record["n_other"] == 300
    assert record["alpha"] == 0.05


def test_prob_greater_reads_as_a_chance():
    control, treated = _samples()
    assert is_this_different(control, treated)[0]["prob_greater"] > 0.8
    assert is_this_different(treated, control)[0]["prob_greater"] < 0.2
    identical = is_this_different(control, control)[0]["prob_greater"]
    assert identical == pytest.approx(0.5, abs=0.01)


def test_overlap_runs_from_one_to_zero():
    control, _ = _samples()
    assert is_this_different(control, control)[0]["overlap"] == pytest.approx(1.0, abs=0.02)
    assert is_this_different(control, control + 100)[0]["overlap"] == pytest.approx(0.0, abs=0.02)


def test_overlap_and_prob_greater_are_usable_alone():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(loc=1, size=500)
    assert 0.5 < overlap(a, b) < 0.75  # theory is about 0.62 one sd apart
    assert prob_greater(a, b) > 0.5


def test_distances_are_in_data_units():
    control, treated = _samples()
    record = is_this_different(control, treated)[0]
    assert record["median_shift"] == pytest.approx(2, abs=0.3)
    assert record["wasserstein"] == pytest.approx(2, abs=0.3)


def test_median_shift_is_signed():
    control, treated = _samples()
    assert is_this_different(treated, control)[0]["median_shift"] < 0


def test_many_against_one_compares_each_to_the_reference():
    rng = np.random.default_rng(0)
    control = rng.normal(size=300)
    records = is_this_different(
        control, {"same": rng.normal(size=300), "shifted": rng.normal(loc=3, size=300)}
    )
    assert [r["name"] for r in records] == ["same", "shifted"]
    assert records[0]["verdict"] == "no"
    assert records[1]["verdict"] == "yes"


def test_positional_vectors_are_named_by_position():
    rng = np.random.default_rng(0)
    records = is_this_different(rng.normal(size=300), [rng.normal(size=300), rng.normal(size=300)])
    assert [r["name"] for r in records] == ["Series 1", "Series 2"]


def test_every_record_has_the_same_keys():
    rng = np.random.default_rng(0)
    records = is_this_different(rng.normal(size=300), {"a": rng.normal(size=300), "b": [1.0, 2.0]})
    assert set(records[0]) == set(records[1])


def test_a_single_comparison_has_nothing_to_correct():
    control, treated = _samples()
    record = is_this_different(control, treated)[0]
    for name in TEST_NAMES:
        assert record[f"{name}_pvalue_adjusted"] == record[f"{name}_pvalue"]


def test_a_batch_of_comparisons_is_corrected_upward():
    rng = np.random.default_rng(0)
    control = rng.normal(size=200)
    records = is_this_different(control, {f"s{i}": rng.normal(size=200) for i in range(6)})
    raw = [r["ks_pvalue"] for r in records]
    adjusted = [r["ks_pvalue_adjusted"] for r in records]
    assert all(a >= p - 1e-12 for a, p in zip(adjusted, raw, strict=True))
    assert adjusted != raw  # something actually moved


def test_correction_changes_a_borderline_verdict():
    # a comparison that squeaks past alpha alone should not survive a batch of
    # twenty; this is the trap the correction exists to close
    rng = np.random.default_rng(3)
    control = rng.normal(size=60)
    others = {f"s{i}": rng.normal(size=60) for i in range(20)}
    batch = is_this_different(control, others)
    assert all(r["verdict"] != "yes" for r in batch)


def test_accepts_any_vector_type():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(loc=2, size=200)
    expected = is_this_different(a, b)[0]["verdict"]
    assert is_this_different(list(a), list(b))[0]["verdict"] == expected


def test_pandas_and_polars_series_work():
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(loc=2, size=200)
    expected = is_this_different(a, b)[0]["verdict"]
    assert is_this_different(pd.Series(a), pd.Series(b))[0]["verdict"] == expected
    assert is_this_different(pl.Series(a), pl.Series(b))[0]["verdict"] == expected


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        is_this_different([1.0, 2.0], [3.0, 4.0], alpha=alpha)


def test_rejects_tiny_and_non_finite_input():
    with pytest.raises(ValueError, match="at least 2 values"):
        is_this_different([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        is_this_different([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="reference"):
        is_this_different([1.0], [1.0, 2.0])


def test_constant_samples_do_not_crash():
    # no spread means no density to estimate; overlap reports nan rather than
    # raising, and the keys stay put
    record = is_this_different([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])[0]
    assert math.isnan(record["overlap"])
    assert record["verdict"] in ("yes", "no", "maybe")
