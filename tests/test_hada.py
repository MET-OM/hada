from pathlib import Path
from hada.cli import hada
import xarray as xr

def test_run(sourcetoml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, ['--sources', str(sourcetoml), '--output', td / 'test.nc', '--freq', '1D' ])
        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'test.nc')
        print(ds)

        assert 'sea_temperature' in ds

def test_custom_bbox(sourcetoml, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, ['--sources', str(sourcetoml), '--output', td / 'test.nc', '--bbox-deg', '6,7,67,68'])
        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'test.nc')
        print(ds)

        assert 'sea_temperature' in ds
