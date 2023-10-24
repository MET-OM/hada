import pytest, os
from pyproj import CRS
from pytest import approx
from hada.sources import *
from hada.target import Target
import numpy as np


def test_nearest_valid_ib(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', {'x_wind': 'Uwind'})
    # t = Target.from_lonlat(15, 16, 65, 66, 100, 150, tmpdir)
    t = Target.from_lonlat(3.872345, 4.574901, 59.725278, 60.081193, 100, 150, tmpdir)
    # 59.725278, 3.872345
    # 60.081193, 4.574901
    tx, ty, _, _, i = d.__interpolate_nearest_valid_grid__(t, 'Uwind')

    assert tx.shape == (150, 100)
    assert ty.shape == (150, 100)

    print(tx, ty)

    tcx, tcy, ib = d.__calculate_grid__(t)

    assert ib.ravel().any()
    assert ib.ravel().all()

    # This area should be completely within the domain so both methods should map to the same pixels
    txi, tyi = d.__map_to_index__(tx, ty)
    tcxi, tcyi = d.__map_to_index__(tcx, tcy)

    np.testing.assert_allclose(txi, tcxi, atol=1)
    np.testing.assert_allclose(tyi, tcyi, atol=1)

def test_nearest_valid_ib_with_nan(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', {'x_wind': 'Uwind'})
    # XXX: Norkyst doesn't have wind over land.
    t = Target.from_lonlat(3.872345, 10.574901, 59.725278, 60.081193, 100, 150, tmpdir)
    tx, ty, _, _, i = d.__interpolate_nearest_valid_grid__(t, 'Uwind')

    assert tx.shape == (150, 100)
    assert ty.shape == (150, 100)

    print(tx, ty)

    tcx, tcy, ib = d.__calculate_grid__(t)

    assert ib.ravel().any()
    assert ib.ravel().all()

    txi, tyi = d.__map_to_index__(tx, ty)
    tcxi, tcyi = d.__map_to_index__(tcx, tcy)

    # The area is within the bounds of the reader, but have nan values. so should _not_ map to the same pixels.
    assert not np.allclose(txi, tcxi, atol=1)
    assert not np.allclose(tyi, tcyi, atol=1)

def test_nearest_valid_oob(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', {'x_wind': 'Uwind'})
    # t = Target.from_lonlat(15, 16, 65, 66, 100, 150, tmpdir)
    t = Target.from_lonlat(-3.872345, 4.574901, 59.725278, 80.081193, 100, 150, tmpdir)
    # XXX: Completely in the ocean, but passing outside the end of the reader.
    # 59.725278, 3.872345
    # 60.081193, 4.574901
    tx, ty, _, _, i = d.__interpolate_nearest_valid_grid__(t, 'Uwind', max_dist=10000.e3)

    assert i.ravel().all()

    assert tx.shape == (150, 100)
    assert ty.shape == (150, 100)

    print(tx, ty)

    tcx, tcy, ib = d.__calculate_grid__(t)
    assert tcx.shape == (150, 100)
    assert tcy.shape == (150, 100)
    assert ib.shape == (150, 100)

    assert ib.ravel().any()
    assert not ib.ravel().all()

    txi, tyi = d.__map_to_index__(tx, ty)
    tcxi, tcyi = d.__map_to_index__(tcx[ib], tcy[ib])

    # Should not be the same
    assert not np.allclose(tyi[ib], tcyi, atol=1)
    assert not np.allclose(txi[ib], tcxi, atol=1)

    # But the pixels inside the reader should be the same
    # np.testing.assert_allclose(txi[ib], tcxi[ib], atol=1)
    # assert np.allclose(txi[ib], tcxi[ib], atol=1)
    # assert np.allclose(tyi[ib], tcyi[ib], atol=1)

def test_mywave_nans(tmpdir):
    d = Dataset(
        "mywave",
        "https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml",
        'rlon', 'rlat', {'hs': 'hs'})

    t = Target.from_gridfile('projects/Svalbard_3km_Grid_EPSG3575_Kyst.csv',
                             tmpdir)

    tcx, tcy, ibc = d.__calculate_grid__(t)
    assert ibc.ravel().all()

    tx, ty, tx_i, ty_i, ib = d.__interpolate_nearest_valid_grid__(t, 'hs')
    assert ib.ravel().all()

    np.testing.assert_array_equal(tx, d.ds.X.values[tx_i])
    np.testing.assert_array_equal(ty, d.ds.Y.values[ty_i])

    # find index of tx and ty
    txi, tyi = d.__map_to_index__(tx, ty)

    np.testing.assert_allclose(txi, tx_i, atol=8)
    np.testing.assert_allclose(tyi, ty_i, atol=1)

    tcxi, tcyi = d.__map_to_index__(tcx, tcy)

    np.testing.assert_allclose(txi, tcxi, atol=8)
    np.testing.assert_allclose(tyi, tcyi, atol=6)

    vo = d.regrid(d.ds['hs'], t, pd.to_datetime("2019-11-06T02:00:00"), always_nearest=True)
    assert not np.isnan(vo.values).any()

@pytest.mark.skipif(
    not os.path.exists('/lustre/storeB/project/fou/om/ERA/ERA5/atm'),
    reason="dataset not accessible, skipping dependent tests")
def test_era5_index(tmpdir):
    d = Dataset(
        "era5",
        "/lustre/storeB/project/fou/om/ERA/ERA5/atm/era5_sst_CDS_202205.nc",
        'longitude',
        'latitude', ['sst'],
        proj4='+proj=latlong')

    t = Target.from_gridfile('projects/Svalbard_3km_Grid_EPSG3575_Kyst.csv',
                             tmpdir)

    tcx, tcy, ibc = d.__calculate_grid__(t)
    assert ibc.ravel().all()

    tx, ty, tx_i, ty_i, ib = d.__interpolate_nearest_valid_grid__(t, 'sst')
    assert ib.ravel().all()

    np.testing.assert_array_equal(tx, d.ds.X.values[tx_i])
    np.testing.assert_array_equal(ty, d.ds.Y.values[ty_i])

    # find index of tx and ty
    txi, tyi = d.__map_to_index__(tx, ty)

    np.testing.assert_allclose(txi, tx_i, atol=8)
    np.testing.assert_allclose(tyi, ty_i, atol=1)

    tcxi, tcyi = d.__map_to_index__(tcx, tcy)

    np.testing.assert_allclose(txi, tcxi, atol=8)
    np.testing.assert_allclose(tyi, tcyi, atol=6)

    vo = d.regrid(d.ds['sst'], t, pd.to_datetime("2022-05-06T02:00:00"), always_nearest=True)
    assert not np.isnan(vo.values).any()

