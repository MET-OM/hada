from pathlib import Path
from hada.cli import hada

def test_run(sourcetoml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, ['--sources', str(sourcetoml), '--output', td / 'test.nc' ])
        assert r.exit_code == 0

def test_custom_bbox(sourcetoml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, ['--sources', str(sourcetoml), '--output', td / 'test.nc', '--bbox', '6,7,67,68'])
        assert r.exit_code == 0
