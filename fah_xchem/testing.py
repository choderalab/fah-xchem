import pytest


def pytest_addoption(parser):
    """
    Additional PyTest CLI flags to add

    See `pytest_collection_modifyitems` for handling and `pytest_configure` for adding known in-line marks.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runexamples", action="store_true", default=False, help="run example tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Handle test triggers based on the CLI flags

    Use decorators:
    @pytest.mark.slow
    @pyrest.mark.example
    """
    runslow = config.getoption("--runslow")
    runexamples = config.getoption("--runexamples")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_example = pytest.mark.skip(reason="need --runexamples option to run")
    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "example" in item.keywords and not runexamples:
            item.add_marker(skip_example)


def pytest_configure(config):
    import sys

    sys._called_from_test = True
    config.addinivalue_line(
        "markers", "example: Mark a given test as an example which can be run"
    )
    config.addinivalue_line(
        "markers",
        "slow: Mark a given test as slower than most other tests, needing a special "
        "flag to run.",
    )
