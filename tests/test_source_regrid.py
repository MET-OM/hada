import pytest
from hada.sources import *
from hada.target import Target
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box


def test_norkyst_transform_points(tmpdir, plot):
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

    vo = d.regrid(d.ds['Uwind'], t, pd.to_datetime("2022-11-06T02:00:00"))
    print(vo)

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
    vo.plot(transform=t.cartopy_crs)
    ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
    ax.set_extent(ex, crs=ccrs.Mercator())

    if plot:
        plt.show()


def test_barents_transform_points(tmpdir, plot):
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

    vo = d.regrid(d.ds['ice_concentration'], t,
                  pd.to_datetime("2022-11-06T02:00:00"))
    print(vo)

    bcrs = ccrs.LambertConformal(
        central_longitude=-25,
        central_latitude=77.5,
        standard_parallels=(77.5, 77.5),
    )

    plt.figure()
    ax = plt.subplot(121, projection=ccrs.Mercator())
    v.plot(transform=bcrs)
    ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
    ex = ax.get_extent(crs=ccrs.Mercator())

    ax = plt.subplot(122, projection=ccrs.Mercator())
    vo.plot(transform=t.cartopy_crs)
    ax.plot(*tbox.exterior.xy, '-x', transform=t_crs, label='target box')
    ax.set_extent(ex, crs=ccrs.Mercator())

    if plot:
        plt.show()


def test_mywave_transform_points(tmpdir, plot):
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

    vo = d.regrid(d.ds['hs'], t, pd.to_datetime("2022-11-06T02:00:00"))
    print(vo)

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

    if plot:
        plt.show()
