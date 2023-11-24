import pytest, os
from hada.cli import hada
from pathlib import Path
from pyproj import CRS
from pytest import approx
from hada.sources import *
from hada.target import Target
import numpy as np


def test_gridfile_3575(tmpdir):
    d = Dataset(
        "barents",
        "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_zdepth_be",
        'X', 'Y', ['ice_concentration'])
    # t = Target.from_lonlat(15, 16, 65, 66, 100, 150, tmpdir)
    t = Target.from_gridfile(
        'projects/Svalbard_3km_Grid_EPSG3575_Hav_Kyst.csv', tmpdir, epsg=3575)
    # 59.725278, 3.872345
    # 60.081193, 4.574901

    time = pd.date_range("2022-11-06T02:00:00",
                         "2022-11-06T06:00:00",
                         freq='1H')
    vo = d.regrid(d.ds['ice_concentration'], t, time)
    print(vo)

    assert np.any(~np.isnan(vo.values))
    assert np.any(np.isnan(vo.values))
    print(np.nanmax(vo.values.ravel()))


def test_gridfile_csv_grid(test_dir, tmpdir, runner):
    sourcetoml = test_dir / ".." / "projects" / "grid_csv_test.toml"
    grid_file = test_dir / ".." / "projects" / "Svalbard_3km_Grid_EPSG3575_Hav_Kyst.csv"

    with runner.isolated_filesystem(temp_dir=tmpdir) as td:
        td = Path(td)
        r = runner.invoke(hada, [
            '--sources',
            str(sourcetoml), '--output', td / 'test.nc', '--grid', grid_file,
            '--target-epsg', 3575, '--from', '2022-11-06T02:00:00', '--to',
            '2022-11-06T06:00:00', '--freq', '1H',
            '--output-csv', td / 'test.csv'
        ])
        print(r.stdout_bytes.decode('utf-8'))

        assert r.exit_code == 0

        ds = xr.open_dataset(td / 'test.nc')
        print(ds)

        assert 'ice_concentration' in ds

        vo = ds['ice_concentration']

        assert np.any(~np.isnan(vo.values))
        assert np.any(np.isnan(vo.values))
        print(np.nanmax(vo.values.ravel()))

        dsc = pd.read_csv(td / 'test.csv').set_index(['time', 'grid_id'])
        print(dsc)

        assert np.any(~np.isnan(dsc['ice_concentration']))
        assert np.any(np.isnan(dsc['ice_concentration']))
        assert np.nanmax(dsc['ice_concentration']) == np.nanmax(vo.values.ravel())

