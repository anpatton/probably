"""Split conformal prediction intervals around any scikit-learn regressor."""

import math
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data


class ProbablyConformalRegressor(MetaEstimatorMixin, RegressorMixin, BaseEstimator):
    """Wrap a regressor so every prediction comes with an interval that holds.

    Parameters
    ----------
    estimator : scikit-learn regressor, optional
        The model to wrap. Defaults to ``LinearRegression``.
    coverage : float, default 0.9
        Fraction of true values the intervals should contain, e.g. ``0.9``
        for a 90% interval.
    calibration_size : float, default 0.25
        Fraction of the training data held out to measure errors on. The
        rest fits ``estimator``.
    random_state : int, optional
        Seed for the split, so results repeat.


    Attributes
    ----------
    estimator_ : scikit-learn regressor
        The fitted copy of ``estimator``.
    quantile_ : float
        The interval half-width: every interval is ``prediction ± quantile_``.


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 10, size=(500, 1))
    >>> y = 3 * X[:, 0] + rng.normal(scale=2, size=500)
    >>> model = ProbablyConformalRegressor(
    ...     RandomForestRegressor(random_state=0), coverage=0.9, random_state=0
    ... )
    >>> _ = model.fit(X[:400], y[:400])
    >>> low, high = model.predict_interval(X[400:])
    >>> bool(np.mean((low <= y[400:]) & (y[400:] <= high)) >= 0.85)
    True
    """

    def __init__(
        self,
        estimator: Any = None,
        coverage: float = 0.9,
        calibration_size: float = 0.25,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.coverage = coverage
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "ProbablyConformalRegressor":
        """Fit the wrapped model on part of the data and calibrate on the rest."""
        if not 0 < self.coverage < 1:
            raise ValueError(f"coverage must be between 0 and 1, got {self.coverage!r}")
        if not 0 < self.calibration_size < 1:
            raise ValueError(
                f"calibration_size must be between 0 and 1, got {self.calibration_size!r}"
            )

        X, y = validate_data(self, X, y, y_numeric=True)

        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )

        estimator = self.estimator if self.estimator is not None else LinearRegression()
        self.estimator_ = clone(estimator).fit(X_fit, y_fit)

        errors = np.abs(y_cal - self.estimator_.predict(X_cal))
        self.quantile_ = _conformal_quantile(errors, self.coverage)
        return self

    def predict(self, X: Any) -> NDArray[np.floating[Any]]:
        """Point predictions from the wrapped model."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        prediction: NDArray[np.floating[Any]] = self.estimator_.predict(X)
        return prediction

    def predict_interval(
        self, X: Any
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Lower and upper bounds of the interval around each prediction."""
        prediction = self.predict(X)
        return prediction - self.quantile_, prediction + self.quantile_


def _conformal_quantile(errors: NDArray[np.floating[Any]], coverage: float) -> float:
    """The half-width that makes ``prediction ± width`` cover ``coverage`` of new rows.

    The rank is ``ceil((n + 1) * coverage)`` rather than ``n * coverage``: the
    extra ``+ 1`` is what turns a sample quantile into a guarantee that holds
    for the next observation, not just the ones already seen.
    """
    n = errors.size
    rank = math.ceil((n + 1) * coverage)
    if rank > n:
        warnings.warn(
            f"coverage={coverage} needs at least {math.ceil(coverage / (1 - coverage))} "
            f"calibration rows to hold exactly; got {n}, so the interval uses the largest "
            "observed error and covers slightly less than requested. Pass more training "
            "data or raise calibration_size.",
            UserWarning,
            stacklevel=3,
        )
        rank = n
    return float(np.sort(errors)[rank - 1])
