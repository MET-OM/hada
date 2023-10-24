from pathlib import Path
from hada.cli import hada
import xarray as xr

def test_always_valid_hs(test_dir, tmpdir, runner):
    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        testd = test_dir / 'regressions' / '2023-10-24-always-valid'

        r = runner.invoke(hada, ['--sources', str(testd / 'scosrva23_local.toml'), '--grid', str(testd / 'GRID_SCOSRVA_Kyst_EPSG3575.csv'), '--output', td / 'test.nc', '--from', '2022-05-01', '--to', '2022-05-05', '--always-valid' ])
        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'test.nc')
        print(ds)

        assert 'sea_temperature' in ds

