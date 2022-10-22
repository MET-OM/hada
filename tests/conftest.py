from pathlib import Path
import pytest
from click.testing import CliRunner

@pytest.fixture
def sourcetoml():
    return Path(__file__).parent.parent / 'sources.toml'

@pytest.fixture
def runner():
    return CliRunner()
