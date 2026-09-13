import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from probably.dx import fit_johnsonsu, fit_shash
from probably.dx.dists._families import Identification
from probably.dx.dists._shash import shash

FITTERS = [fit_johnsonsu, fit_shash]

# Checking a fit simulates its null, which costs seconds per call -- about 7
# for Johnson SU and 23 for sinh-arcsinh. The cheaper of the two runs by
# default so the checked path is never wholly untested; the rest need
# --runslow.
CHECKED_FITTERS = [fit_johnsonsu, pytest.param(fit_shash, marks=pytest.mark.slow)]


def _sample(fitter, size=250, seed=0):
    """Draw from the family `fitter` fits, so a good fit is the expected answer."""
    rng = np.random.default_rng(seed)
    if fitter is fit_johnsonsu:
        return stats.johnsonsu.rvs(-1.5, 1.5, size=size, random_state=rng)
    return shash.rvs(1.2, 0.9, loc=3, scale=2, size=size, random_state=rng)


def _bimodal(size=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-4, 0.5, size // 2), rng.normal(4, 0.5, size // 2)])


# --- the sinh-arcsinh distribution itself ------------------------------------
# scipy does not ship it, so its maths is ours to prove.


def test_shash_density_integrates_to_one():
    for eps, delta in [(0.0, 1.0), (0.5, 1.2), (-1.0, 0.7), (2.0, 2.5)]:
        area, _ = quad(shash.pdf, -np.inf, np.inf, args=(eps, delta))
        assert area == pytest.approx(1.0, abs=1e-6)


def test_shash_is_exactly_normal_at_eps_zero_delta_one():
    # The family is a widening of the normal, not a different shape; this is
    # the identity that makes it safe to reach for when normality fails.
    grid = np.linspace(-4, 4, 17)
    assert np.allclose(shash.pdf(grid, 0.0, 1.0), stats.norm.pdf(grid))
    assert np.allclose(shash.cdf(grid, 0.0, 1.0), stats.norm.cdf(grid))


def test_shash_ppf_inverts_cdf():
    for q in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert shash.cdf(shash.ppf(q, 0.5, 1.2), 0.5, 1.2) == pytest.approx(q)


def test_shash_logpdf_agrees_with_pdf():
    grid = np.linspace(-4, 4, 17)
    assert np.allclose(shash.logpdf(grid, 0.5, 1.2), np.log(shash.pdf(grid, 0.5, 1.2)))


def test_shash_fit_recovers_its_own_parameters():
    values = shash.rvs(0.8, 1.3, loc=2, scale=1.5, size=4000, random_state=np.random.default_rng(0))
    eps, delta, loc, scale = shash.fit(values)
    assert eps == pytest.approx(0.8, abs=0.2)
    assert delta == pytest.approx(1.3, abs=0.2)
    assert loc == pytest.approx(2.0, abs=0.2)
    assert scale == pytest.approx(1.5, abs=0.2)


# --- shared behaviour of both fitters ----------------------------------------


@pytest.mark.parametrize("fitter", FITTERS)
def test_returns_an_identification_that_unpacks(fitter):
    found = fitter(_sample(fitter), quick=True)
    assert isinstance(found, Identification)
    distribution, diagnostics = found
    assert distribution is found.distribution
    assert diagnostics is found.diagnostics


@pytest.mark.parametrize("fitter", FITTERS)
def test_returns_a_usable_frozen_scipy_distribution(fitter):
    distribution, _ = fitter(_sample(fitter), quick=True)
    assert isinstance(distribution, stats.distributions.rv_frozen)
    assert np.isfinite(distribution.rvs(10, random_state=0)).all()
    assert distribution.cdf(distribution.ppf(0.3)) == pytest.approx(0.3)


@pytest.mark.parametrize("fitter", FITTERS)
def test_diagnostics_hold_one_record_with_the_documented_keys(fitter):
    _, diagnostics = fitter(_sample(fitter), quick=True)
    assert len(diagnostics) == 1
    assert set(diagnostics[0]) == {"distribution", "n", "aic", "parameters", "verdict", "reason"}
    assert diagnostics[0]["n"] == 250
    assert np.isfinite(diagnostics[0]["aic"])


@pytest.mark.parametrize(
    ("fitter", "expected"),
    [(fit_johnsonsu, ("a", "b", "loc", "scale")), (fit_shash, ("eps", "delta", "loc", "scale"))],
)
def test_parameters_are_named_and_plain_floats(fitter, expected):
    _, diagnostics = fitter(_sample(fitter), quick=True)
    parameters = diagnostics[0]["parameters"]
    assert tuple(parameters) == expected
    assert all(type(value) is float for value in parameters.values())


@pytest.mark.parametrize("fitter", FITTERS)
def test_the_returned_distribution_pickles(fitter):
    # shash is defined in this package rather than scipy, and a custom
    # rv_continuous is exactly the thing that fails to pickle -- which would
    # rule it out of multiprocessing and any saved-model workflow.
    import pickle

    distribution, _ = fitter(_sample(fitter), quick=True)
    restored = pickle.loads(pickle.dumps(distribution))
    assert restored.ppf(0.9) == pytest.approx(distribution.ppf(0.9))


@pytest.mark.parametrize("fitter", FITTERS)
def test_the_named_parameters_rebuild_the_returned_distribution(fitter):
    distribution, diagnostics = fitter(_sample(fitter), quick=True)
    assert list(distribution.args) == list(diagnostics[0]["parameters"].values())


# --- quick -------------------------------------------------------------------


@pytest.mark.parametrize("fitter", FITTERS)
def test_quick_skips_the_verdict_and_says_so(fitter):
    _, diagnostics = fitter(_sample(fitter), quick=True)
    assert diagnostics[0]["verdict"] is None
    assert "quick=True" in diagnostics[0]["reason"]


@pytest.mark.parametrize("fitter", FITTERS)
def test_quick_stays_silent_on_data_the_family_does_not_fit(fitter):
    # The whole point of quick is skipping the check, so it must not warn --
    # that is what makes it usable inside a resampling loop.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitter(_bimodal(), quick=True)


@pytest.mark.slow
@pytest.mark.parametrize("fitter", FITTERS)
def test_quick_and_checked_fit_identically(fitter):
    # quick changes only whether the fit is judged, never the fit itself.
    values = _sample(fitter)
    quick, _ = fitter(values, quick=True)
    checked, _ = fitter(values)
    assert quick.args == checked.args


# --- verdicts ----------------------------------------------------------------


@pytest.mark.parametrize("fitter", CHECKED_FITTERS)
def test_a_sample_from_the_family_is_not_rejected(fitter):
    _, diagnostics = fitter(_sample(fitter))
    assert diagnostics[0]["verdict"] == "yes"


@pytest.mark.slow
@pytest.mark.parametrize("fitter", FITTERS)
def test_bimodal_data_is_rejected_and_warns(fitter):
    # Four flexible parameters still cannot make one mode into two. Without
    # this, a practitioner reads a returned distribution as an endorsement.
    with pytest.warns(UserWarning, match="does not fit this data"):
        distribution, diagnostics = fitter(_bimodal())
    assert diagnostics[0]["verdict"] == "no"
    assert distribution is not None


# --- input handling ----------------------------------------------------------


@pytest.mark.parametrize("fitter", FITTERS)
@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_rejects_an_alpha_outside_the_unit_interval(fitter, alpha):
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        fitter(_sample(fitter), alpha=alpha, quick=True)


@pytest.mark.parametrize("fitter", FITTERS)
def test_rejects_a_vector_too_short_to_fit(fitter):
    with pytest.raises(ValueError, match="at least 2 values"):
        fitter([1.0], quick=True)


@pytest.mark.parametrize("fitter", FITTERS)
def test_rejects_non_finite_values(fitter):
    with pytest.raises(ValueError, match="finite"):
        fitter([1.0, 2.0, np.nan], quick=True)


@pytest.mark.parametrize("fitter", FITTERS)
def test_rejects_a_2d_input(fitter):
    with pytest.raises(ValueError, match="1-D"):
        fitter([[1.0, 2.0], [3.0, 4.0]], quick=True)


@pytest.mark.parametrize("fitter", FITTERS)
def test_accepts_a_dataframe_column(fitter):
    pd = pytest.importorskip("pandas")
    pl = pytest.importorskip("polars")
    values = _sample(fitter)
    expected = fitter(values, quick=True).distribution.args
    assert fitter(pd.Series(values), quick=True).distribution.args == expected
    assert fitter(pl.Series(values), quick=True).distribution.args == expected
