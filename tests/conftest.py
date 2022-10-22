from pathlib import Path
import pytest

@pytest.fixture
def sourcetoml():
    return Path(__file__).parent.parent / 'sources.toml'

