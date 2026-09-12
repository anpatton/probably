"""Check whether one or more vectors look lognormally distributed."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from probably.dx.dists._engine import require_positive, run_battery
from probably.dx.dists.is_this_normal import NORMAL_BATTERY


def _log_of_positive(values: NDArray[np.floating[Any]], name: str) -> NDArray[np.floating[Any]]:
    return np.log(require_positive(values, name))


def is_this_lognormal(*x: Any, alpha: float = 0.05) -> list[dict[str, Any]]:
    """Test one or more vectors for lognormality and answer yes, no, or maybe.

    Parameters
    ----------
    *x : array-like
        One or more vectors of strictly positive values. Pass a single
        mapping instead to name them; otherwise they are named by position.
    alpha : float, default 0.05
        Significance level each test is judged against.


    Returns
    -------
    list[dict]
        One record per vector, shaped like :func:`is_this_normal`'s. The
        statistics describe the logged values.


    Examples
    --------
    >>> import numpy as np
    >>> heavy_tailed = np.random.default_rng(0).lognormal(size=200)
    >>> flat = np.random.default_rng(1).uniform(size=200)
    >>> [(r["name"], r["verdict"]) for r in is_this_lognormal(heavy_tailed, flat)]
    [('Series 1', 'yes'), ('Series 2', 'no')]
    """
    return run_battery(x, alpha, "lognormal", NORMAL_BATTERY, prepare=_log_of_positive)
