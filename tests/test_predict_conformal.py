import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from probably.predict import ProbablyConformalRegressor
from probably.predict.conformal import _conformal_quantile


def _linear_data(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 1))
    y = 3 * X[:, 0] + rng.normal(scale=2, size=n)
    return X[: n // 2], X[n // 2 :], y[: n // 2], y[n // 2 :]


# sklearn's checks use tiny datasets, so the small-calibration-set warning fires
# constantly; it is tested on its own below.
@pytest.mark.filterwarnings("ignore:coverage=.*calibration rows:UserWarning")
@parametrize_with_checks([ProbablyConformalRegressor()])
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_intervals_cover_at_the_requested_rate():
    X_train, X_test, y_train, y_test = _linear_data()
    model = ProbablyConformalRegressor(LinearRegression(), coverage=0.9, random_state=0)
    model.fit(X_train, y_train)
    low, high = model.predict_interval(X_test)
    observed = np.mean((low <= y_test) & (y_test <= high))
    # realized coverage varies with the calibration draw (sd ~0.01 at 1250 rows)
    # and the test draw (sd ~0.004 at 5000 rows); 0.03 is about 3 sd
    assert observed == pytest.approx(0.9, abs=0.03)


@pytest.mark.parametrize("coverage", [0.5, 0.8, 0.95])
def test_coverage_tracks_the_argument(coverage):
    X_train, X_test, y_train, y_test = _linear_data()
    model = ProbablyConformalRegressor(coverage=coverage, random_state=0).fit(X_train, y_train)
    low, high = model.predict_interval(X_test)
    assert np.mean((low <= y_test) & (y_test <= high)) == pytest.approx(coverage, abs=0.05)


def test_higher_coverage_means_wider_intervals():
    X_train, _, y_train, _ = _linear_data()
    narrow = ProbablyConformalRegressor(coverage=0.5, random_state=0).fit(X_train, y_train)
    wide = ProbablyConformalRegressor(coverage=0.95, random_state=0).fit(X_train, y_train)
    assert wide.quantile_ > narrow.quantile_


def test_interval_is_centred_on_the_prediction():
    X_train, X_test, y_train, _ = _linear_data()
    model = ProbablyConformalRegressor(random_state=0).fit(X_train, y_train)
    low, high = model.predict_interval(X_test)
    prediction = model.predict(X_test)
    assert np.allclose((low + high) / 2, prediction)
    assert np.allclose(high - low, 2 * model.quantile_)


def test_wraps_any_regressor():
    X_train, X_test, y_train, _ = _linear_data(n=400)
    forest = RandomForestRegressor(n_estimators=10, random_state=0)
    model = ProbablyConformalRegressor(forest, random_state=0).fit(X_train, y_train)
    assert isinstance(model.estimator_, RandomForestRegressor)
    assert model.estimator_ is not forest  # fit on a clone, the original is untouched
    assert model.predict(X_test).shape == (len(X_test),)


def test_works_inside_a_pipeline():
    X_train, X_test, y_train, _ = _linear_data(n=400)
    pipeline = make_pipeline(StandardScaler(), ProbablyConformalRegressor(random_state=0))
    pipeline.fit(X_train, y_train)
    low, high = pipeline[-1].predict_interval(pipeline[:-1].transform(X_test))
    assert np.all(low < high)


def test_defaults_to_linear_regression():
    X_train, _, y_train, _ = _linear_data(n=400)
    model = ProbablyConformalRegressor(random_state=0).fit(X_train, y_train)
    assert isinstance(model.estimator_, LinearRegression)


def test_random_state_makes_the_split_repeatable():
    X_train, _, y_train, _ = _linear_data(n=400)
    first = ProbablyConformalRegressor(random_state=7).fit(X_train, y_train)
    second = ProbablyConformalRegressor(random_state=7).fit(X_train, y_train)
    assert first.quantile_ == second.quantile_


def test_predict_interval_before_fit_raises():
    with pytest.raises(NotFittedError):
        ProbablyConformalRegressor().predict_interval([[1.0]])


@pytest.mark.parametrize("coverage", [0, 1, -0.1, 1.5])
def test_rejects_invalid_coverage(coverage):
    X_train, _, y_train, _ = _linear_data(n=100)
    with pytest.raises(ValueError, match="coverage must be between 0 and 1"):
        ProbablyConformalRegressor(coverage=coverage).fit(X_train, y_train)


@pytest.mark.parametrize("size", [0, 1, 1.5])
def test_rejects_invalid_calibration_size(size):
    X_train, _, y_train, _ = _linear_data(n=100)
    with pytest.raises(ValueError, match="calibration_size must be between 0 and 1"):
        ProbablyConformalRegressor(calibration_size=size).fit(X_train, y_train)


def test_too_few_calibration_rows_warns_and_uses_the_largest_error():
    errors = np.array([1.0, 5.0, 2.0])
    with pytest.warns(UserWarning, match="needs at least 10 calibration rows"):
        assert _conformal_quantile(errors, coverage=0.9) == 5.0


def test_conformal_quantile_uses_the_finite_sample_rank():
    # n=9, coverage=0.5: rank = ceil(10 * 0.5) = 5 -> the 5th smallest error
    errors = np.arange(1.0, 10.0)
    assert _conformal_quantile(errors, coverage=0.5) == 5.0
