import pytest, os
from pyproj import CRS
from pytest import approx
from hada.sources import *
from hada.target import Target

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def test_init(tmpdir):
    t = Target.from_box(5, 10, 55, 60, 100, 100, tmpdir)


def test_proj_attr(tmpdir):
    t = Target.from_box(5, 10, 55, 60, 100, 100, tmpdir)
    print(t.proj_var)


def test_calculate_grid(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    t = Target.from_box(5, 10, 55, 60, 100, 100, tmpdir)

    tx, ty, _ = d.__calculate_grid__(t)

    assert tx.shape == (100, 100)
    assert ty.shape == (100, 100)

    print(tx, ty)

def test_map_index_norkyst(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    t = Target.from_box(5, 10, 55, 60, 100, 100, tmpdir)

    print("xb=", d.xmin, d.xmax)
    print("yb=", d.ymin, d.ymax)

    # Take a range from the coordinates and check if they match the indexes.

    xi = np.arange(30, 156, 1).astype(int)
    yi = np.arange(230, 354, 1).astype(int)

    tx = d.x[xi]
    ty = d.y[yi]

    print("tx=", tx.min(), tx.max())
    print("ty=", ty.min(), ty.max())

    ttx, tty = d.__map_to_index__(tx, ty)

    np.testing.assert_array_equal(xi, ttx)
    np.testing.assert_array_equal(yi, tty)

    # Negative range
    xi = np.arange(30, 156, 1).astype(int)
    yi = np.arange(354, 230, -1).astype(int)
    tx = d.x[xi]
    ty = d.y[yi]

    ttx, tty = d.__map_to_index__(tx, ty)

    np.testing.assert_array_equal(xi, ttx)
    np.testing.assert_array_equal(yi, tty)


@pytest.mark.skipif(
    not os.path.exists('/lustre/storeB/project/fou/om/ERA/ERA5/atm'),
    reason="dataset not accessible, skipping dependent tests")
def test_map_index_era5(tmpdir):
    d = Dataset(
        "era5",
        "/lustre/storeB/project/fou/om/ERA/ERA5/atm/era5_sst_CDS_202205.nc",
        'longitude',
        'latitude', ['sst'],
        proj4='+proj=latlong')
    t = Target.from_box(5, 10, 55, 60, 100, 100, tmpdir)

    print("xb=", d.xmin, d.xmax)
    print("yb=", d.ymin, d.ymax)

    # Take a range from the coordinates and check if they match the indexes.

    xi = np.arange(30, 156, 1).astype(int)
    yi = np.arange(230, 354, 1).astype(int)

    tx = d.x[xi]
    ty = d.y[yi]

    print("tx=", tx.min(), tx.max())
    print("ty=", ty.min(), ty.max())

    ttx, tty = d.__map_to_index__(tx, ty)

    np.testing.assert_array_equal(xi, ttx)
    np.testing.assert_array_equal(yi, tty)

    # Negative range
    xi = np.arange(30, 156, 1).astype(int)
    yi = np.arange(354, 230, -1).astype(int)
    tx = d.x[xi]
    ty = d.y[yi]

    ttx, tty = d.__map_to_index__(tx, ty)

    np.testing.assert_array_equal(xi, ttx)
    np.testing.assert_array_equal(yi, tty)

def test_init_lonlat(tmpdir, plot):
    t = Target.from_lonlat(5, 10, 55, 60, 1000, 1000, tmpdir)

    if plot:
        # plot bounding box
        from shapely.geometry import box
        llbox = box(5, 55, 10, 60)
        prbox = t.bbox

        p_crs = ccrs.epsg(t.epsg)

        ax = plt.axes(projection=ccrs.Mercator())
        plt.plot(*llbox.exterior.xy,
                 '-x',
                 transform=ccrs.PlateCarree(),
                 label='ll box')
        plt.plot(*prbox.exterior.xy, '-x', transform=p_crs, label='target box')
        ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())

        plt.legend()
        plt.show()


def test_transform_from_latlon():
    # 5E, 55N should be: x=-335060.5808799216, y=-3829759.9640511023
    crs = CRS.from_epsg(4326)  # lonlat
    print(repr(crs))

    x, y = Target.transform(crs, 5, 55)
    print(x, y)
    assert (x, y) == approx((-335060.5808799216, -3829759.9640511023))

    xx, yy = Target.itransform(crs, x, y)
    assert (xx, yy) == approx((5., 55.))

    # test target proj bounds
    x, y = Target.transform(crs, -180, 45)
    print(x, y)
    assert (x, y) == approx((849024.0785366141, 4815054.821022379))
    xx, yy = Target.itransform(crs, x, y)
    assert (Target.modulate_longitude(xx), yy) == approx((-180., 45))

    x, y = Target.transform(crs, 180, 90)
    print(x, y)
    assert (x, y) == approx((0., 0.))
    # xx, yy = Target.itransform(crs, x, y) # doesn't work, NP singularity?
    # assert (xx, yy) == approx((180., 90))


def test_from_gridfile(tmpdir):
    t = Target.from_gridfile(
        'projects/Svalbard_3km_Grid_EPSG3575_Hav_Kyst.csv', tmpdir)


def test_from_gridfile_coast(tmpdir):
    t = Target.from_gridfile('projects/Svalbard_3km_Grid_EPSG3575_Kyst.csv',
                             tmpdir)
