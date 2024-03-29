import pytest, os
from hada.sources import *
from hada.target import Target
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box


def test_norkyst_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    assert d.ds.rio.crs is not None

    v = d.ds['Uwind'].sel(time="2022-11-06T02:00:00")
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)

    tbox = t.bbox
    t_crs = t.cartopy_crs

    time = pd.date_range("2022-11-06T02:00:00", "2022-11-06T06:00:00", freq='1H')
    # time = pd.to_datetime("2022-11-06T02:00:00")
    vo = d.regrid(d.ds['Uwind'], t, time)
    print(vo)

    # vo.to_netcdf(baseline / 'norkyst_w_baseline.nc')

    bvo = xr.open_dataset(baseline / 'norkyst_w_baseline.nc')

    if plot:
        ncrs = ccrs.Stereographic(true_scale_latitude=60,
                                  central_latitude=90,
                                  central_longitude=70,
                                  false_easting=3192800,
                                  false_northing=1784000)

        plt.figure()
        ax = plt.subplot(121, projection=ccrs.Mercator())
        v.plot(transform=ncrs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(122, projection=ccrs.Mercator())
        vo.isel(time=0).plot(transform=t.cartopy_crs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()

    np.testing.assert_array_equal(bvo.Uwind, vo)


def test_barents_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "barents",
        "https://thredds.met.no/thredds/dodsC/fou-hi/barents_eps_zdepth_be",
        'X', 'Y', ['ice_concentration'])
    assert d.ds.rio.crs is not None
    print(repr(d.crs))

    v = d.ds['ice_concentration'].sel(time="2022-11-06T02:00:00")
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(8, 20, 68, 73, 100, 100, tmpdir)
    tbox = t.bbox
    t_crs = t.cartopy_crs

    time = pd.date_range("2022-11-06T02:00:00", "2022-11-06T06:00:00", freq='1H')
    vo = d.regrid(d.ds['ice_concentration'], t, time)
    print(vo)
    # vo.to_netcdf(baseline / 'barents_ic_baseline.nc')

    bvo = xr.open_dataset(baseline / 'barents_ic_baseline.nc')
    np.testing.assert_array_equal(bvo.ice_concentration, vo)

    bcrs = ccrs.LambertConformal(
        central_longitude=-25,
        central_latitude=77.5,
        standard_parallels=(77.5, 77.5),
    )

    if plot:
        plt.figure()
        ax = plt.subplot(121, projection=ccrs.Mercator())
        v.plot(transform=bcrs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(122, projection=ccrs.Mercator())
        vo.plot(transform=t.cartopy_crs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()


def test_mywave_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "barents",
        'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_be',
        'rlon', 'rlat', ['hs'])
    assert d.ds.rio.crs is not None
    print(repr(d.crs))

    v = d.ds['hs'].sel(time="2022-11-06T02:00:00")
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)
    tbox = t.bbox
    t_crs = t.cartopy_crs

    time = pd.date_range("2022-11-06T02:00:00", "2022-11-06T06:00:00", freq='1H')
    vo = d.regrid(d.ds['hs'], t, time)
    print(vo)

    # vo.to_netcdf(baseline / 'mywave_hs_baseline.nc')

    bvo = xr.open_dataset(baseline / 'mywave_hs_baseline.nc')
    np.testing.assert_array_equal(bvo.hs, vo)

    import cartopy.feature as cfeature
    land = cfeature.GSHHSFeature(scale='auto',
                                 edgecolor='black',
                                 facecolor=cfeature.COLORS['land'])

    # import pyresample as pr
    # adef, _  = pr.utils.load_cf_area(d.ds)
    # print(adef)

    # mcrs = ccrs.RotatedPole(
    #     pole_longitude=140.,
    #     pole_latitude=22.,
    # )

    if plot:
        plt.figure()
        # ax = plt.subplot(121, projection=ccrs.Mercator())
        # v.plot(transform=mcrs)
        # ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        # ax.add_feature(land)
        # ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(111, projection=ccrs.Mercator())
        vo.plot(transform=t.cartopy_crs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.add_feature(land)
        # ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()


def test_regrid_ice_fallback_value(sourcetoml, tmpdir):
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)
    s = Sources.from_toml(sourcetoml,
                          dataset_filter=('barents', ),
                          variable_filter=('ice_concentration', ))

    ice = s.regrid('ice_concentration', t, pd.to_datetime("2022-11-06T02:00:00"))
    assert ice is not None
    np.testing.assert_array_equal(ice.values, 0.0)

    s.fallback = {}
    ice = s.regrid('ice_concentration', t, pd.to_datetime("2022-11-06T02:00:00"))
    np.testing.assert_array_equal(ice.values, np.nan)


def test_regrid_wind_fallback(sourcetoml, tmpdir):
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)
    s = Sources.from_toml(sourcetoml,
                          dataset_filter=(
                              'norkyst',
                              'meps',
                          ),
                          variable_filter=('wind', ))

    t0 = datetime.utcnow() + timedelta(hours=1)
    t1 = t0 + timedelta(hours=3)

    time = pd.date_range(t0, t1, freq="1H")

    nk = s.datasets[0]
    assert nk.name == 'norkyst'

    x_wind = nk.regrid(nk.ds['Uwind'], t, time)
    assert np.isnan(x_wind).any()
    assert not np.isnan(x_wind).all()

    x_wind = s.regrid('x_wind', t, time)
    y_wind = s.regrid('y_wind', t, time)

    assert not np.isnan(x_wind).any()
    assert not np.isnan(y_wind).any()


@pytest.mark.skipif(
    not os.path.exists('/lustre/storeB/project/fou/om/ERA/ERA5/atm'),
    reason="dataset not accessible, skipping dependent tests")
def test_era5_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "era5",
        "/lustre/storeB/project/fou/om/ERA/ERA5/atm/era5_sst_CDS_202205.nc",
        'longitude',
        'latitude', ['sst'],
        proj4='+proj=latlong')

    v = d.ds['sst'].sel(time="2022-05-06T02:00:00", method='nearest')
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)

    tbox = t.bbox
    t_crs = t.cartopy_crs

    time = pd.date_range("2022-05-06T02:00:00", "2022-05-06T06:00:00", freq='1H')
    vo = d.regrid(d.ds['sst'], t, time)
    print(vo)
    # vo.to_netcdf(baseline / 'era5_sst_baseline.nc')

    bvo = xr.open_dataset(baseline / 'era5_sst_baseline.nc')
    np.testing.assert_array_equal(bvo.sst, vo)

    if plot:
        ncrs = ccrs.Geodetic()

        import cartopy.feature as cfeature
        land = cfeature.GSHHSFeature(scale='auto',
                                     edgecolor='black',
                                     facecolor=cfeature.COLORS['land'])

        plt.figure()
        ax = plt.subplot(121, projection=ccrs.Mercator())
        # v.plot(transform=ncrs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.add_feature(land)
        ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(122, projection=ccrs.Mercator())
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.add_feature(land)
        vo.plot(transform=t.cartopy_crs)
        ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()

def test_arome_arctic_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "arome_arctic",
        "https://thredds.met.no/thredds/dodsC/aromearcticarchive/2023/02/22/arome_arctic_det_2_5km_20230222T00Z.nc",
        'x', 'y', ['x_wind_10m'])

    v = d.ds['x_wind_10m'].sel(time="2023-02-22T02:00:00", method='nearest').isel(height7=0)
    print(v)
    assert d.ds.rio.crs is not None
    assert np.any(np.isfinite(v))

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(10, 20, 67, 70, 100, 100, tmpdir)

    tbox = t.bbox
    t_crs = t.cartopy_crs

    time = pd.date_range("2023-02-22T02:00:00", "2023-02-22T06:00:00", freq='1H')
    vo = d.regrid(d.ds['x_wind_10m'], t, time)
    assert vo is not None
    print(vo)

    # vo.to_netcdf(baseline / 'arome_arctic_w_baseline.nc')

    bvo = xr.open_dataset(baseline / 'arome_arctic_w_baseline.nc')
    np.testing.assert_array_equal(bvo.x_wind_10m, vo)

    if plot:
        bcrs = ccrs.LambertConformal(
            central_longitude=-25,
            central_latitude=77.5,
            standard_parallels=(77.5, 77.5),
        )

        plt.figure()
        plt.imshow(v)

        plt.figure()
        ax = plt.subplot(121, projection=ccrs.Mercator())
        v.plot(transform=bcrs)
        ex = ax.get_extent(crs=ccrs.Mercator())
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        ax.set_extent(ex, crs=ccrs.Mercator())
        # ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(122, projection=ccrs.Mercator())
        vo[0,...].plot(transform=t.cartopy_crs)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')

        import cartopy.feature as cfeature
        land = cfeature.GSHHSFeature(scale='auto',
                                     edgecolor='black',
                                     facecolor=cfeature.COLORS['land'],
                                     alpha=.5)
        ax.add_feature(land)

       # ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()

def test_ww3_4km_transform_points(tmpdir, baseline, plot):
    d = Dataset(
        "ww3_4km",
        'https://thredds.met.no/thredds/dodsC/ww3_4km_agg',
        'rlon', 'rlat', ['hs'])
    assert d.ds.rio.crs is not None
    print(repr(d.crs))

    v = d.ds['hs'].sel(time="2023-08-11T03:00:00")
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)
    tbox = t.bbox
    t_crs = t.cartopy_crs

    # time = pd.date_range("2023-08-11T03:00:00", "2022-11-06T05:00:00", freq='1H')
    time = pd.to_datetime("2023-08-11T03:00:00")
    vo = d.regrid(d.ds['hs'], t, time)
    print(vo)

    # vo.to_netcdf(baseline / 'ww3_4km_hs_baseline.nc')

    bvo = xr.open_dataset(baseline / 'ww3_4km_hs_baseline.nc')

    import cartopy.feature as cfeature
    land = cfeature.GSHHSFeature(scale='auto',
                                 edgecolor='black',
                                 facecolor=cfeature.COLORS['land'])

    # import pyresample as pr
    # adef, _  = pr.utils.load_cf_area(d.ds)
    # print(adef)

    # mcrs = ccrs.RotatedPole(
    #     pole_longitude=140.,
    #     pole_latitude=22.,
    # )

    if plot:
        plt.figure()
        # ax = plt.subplot(121, projection=ccrs.Mercator())
        # v.plot(transform=mcrs)
        # ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        # ax.add_feature(land)
        # ex = ax.get_extent(crs=ccrs.Mercator())

        ax = plt.subplot(111, projection=ccrs.Mercator())
        vo.plot(transform=t.cartopy_crs)
        ax.add_feature(land)
        ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
        # ax.set_extent(ex, crs=ccrs.Mercator())

        plt.show()

    np.testing.assert_array_equal(bvo.hs, vo)
