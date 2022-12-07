import pytest
from hada.sources import *
from hada.target import Target
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def test_norkyst_transform_points(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    assert d.ds.rio.crs is not None

    ## Norkyst EPSG: 8901

    v = d.ds['Uwind'].sel(time="2022-11-06T02:00:00")
    print(v)

    ## Try to get some values and check if they end up where they should.
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)

    vo = d.regrid(d.ds['Uwind'], t, pd.to_datetime("2022-11-06T02:00:00"))
    print(vo)

    ncrs = ccrs.Stereographic(true_scale_latitude=60,
                              central_latitude=90,
                              central_longitude=70,
                              false_easting=3192800,
                              false_northing=1784000)

    ax = plt.axes(projection=ccrs.Mercator())
    v.plot(transform=ncrs)

    plt.figure()
    ax = plt.axes(projection=ccrs.Mercator())
    vo.plot(transform=t.cartopy_crs)

    plt.show()
