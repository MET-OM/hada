import pytest
from hada.sources import *
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


def test_load_default_conf(sourcetoml):
    s = Sources.from_toml(sourcetoml)
    print(s)


@pytest.mark.parametrize("var_filter, exp_s, exp_v",
                         [(("wind", ), 0, 1), (("wind", "air_temp"), 1, 1)])
def test_filter_var(sourcetoml, var_filter, exp_s, exp_v):
    s = Sources.from_toml(sourcetoml, variable_filter=var_filter)
    assert len(s.scalar_variables) == exp_s
    assert len(s.vector_magnitude_variables) == exp_v


def test_filter_dataset(sourcetoml):
    s = Sources.from_toml(sourcetoml, dataset_filter=('norkyst', ))
    assert len(s.datasets) == 1


def test_load_norkyst():
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    assert d.ds.rio.crs is not None

    print(d.ds)
    print(d.ds.cf['x_wind'])
    print(d.ds.cf)

    # dd, i = d.kdtree.query(np.array([d.x[0], d.y[0]]))
    # print(i)

    # np.testing.assert_array_equal(i, [0, 0])

    # dd, i = d.kdtree.query(np.array([d.x[-1], d.y[-1]]))
    # print(i)
    # print(d.x.shape)
    # np.testing.assert_array_equal(i, [len(d.x)-1, len(d.y)-1])


def test_find_var(sourcetoml):
    s = Sources.from_toml(sourcetoml)
    _, v = s.find_dataset_for_var('x_wind')
    assert v is not None
    print(v)
    assert v.attrs['long_name'] == 'surface u-wind component'

    _, v = s.find_dataset_for_var('gibberish')
    assert v is None


def test_slice_barents():
    d = xr.decode_cf(
        xr.open_dataset(
            'https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_zdepth_be',
            decode_coords='all'))

    i = d['ice_concentration']
    # i = i.isel(time=slice(3000, -24*3))

    print(i.time.isel(time=slice(-30, None)))
    ii = np.argmax(
        i.time.values > pd.to_datetime("2022-12-01T00:00:00").to_datetime64())
    print(ii)
    print(i.isel(time=ii))

    tr = pd.date_range("2022-12-01T00:00:00", "2022-12-06T00:00:00", freq='1H')
    print(tr)
    # ic = i.sel(time=tr, method='nearest')

    time = i.time.values
    utime = np.unique(time)
    print(len(time), len(utime))

    # i = i.sel(time="2022-12-01T00:00:00", method='nearest')
    # print(i)
