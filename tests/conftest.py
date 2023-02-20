from pathlib import Path
import pytest
from click.testing import CliRunner

@pytest.fixture
def baseline():
    return Path(__file__).parent / 'data'

@pytest.fixture
def sourcetoml():
    return Path(__file__).parent.parent / 'sources.toml'

@pytest.fixture
def cosrvatoml():
    return Path(__file__).parent.parent / 'projects' / 'cosrva.toml'

@pytest.fixture
def cosrva_local_toml():
    return Path(__file__).parent.parent / 'projects' / 'cosrva_local.toml'

@pytest.fixture
def runner():
    return CliRunner()


def pytest_addoption(parser):
    parser.addoption("--plot",
                     action="store_true",
                     default=False,
                     help="show plots")

@pytest.fixture
def plot(pytestconfig):
    return pytestconfig.getoption('plot')
