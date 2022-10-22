from pathlib import Path
from ddh.cli import ddh

def test_run(sourcetoml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(ddh, ['--sources', str(sourcetoml), '--output', td / 'test.nc' ])
        assert r.exit_code == 0

def test_custom_bbox(sourcetoml, runner):
    r = runner.invoke(ddh, ['--sources', str(sourcetoml), '--output', '/tmp/test.nc', '--bbox', '6,7,67,68'])
    assert r.exit_code == 0
