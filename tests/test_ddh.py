import os
from pathlib import Path
from click.testing import CliRunner
from ddh.cli import ddh

def test_run(sourcetoml, tmpdir):
    ru = CliRunner()
    with ru.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = ru.invoke(ddh, ['--sources', str(sourcetoml), '--output', td / 'test.nc' ])
        assert r.exit_code == 0
