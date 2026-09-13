import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked slow (the Monte Carlo goodness-of-fit batteries)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: simulates a null by refitting hundreds of samples; seconds to minutes",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the slow tests unless asked for, so the default suite stays quick.

    They are skipped rather than deselected so the count still shows up and
    nobody mistakes them for tests that do not exist.
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow Monte Carlo test; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
