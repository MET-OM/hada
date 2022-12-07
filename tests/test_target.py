from pyproj import CRS
from pytest import approx
from hada.sources import *
from hada.target import Target


def test_init(tmpdir):
    t = Target(5, 10, 55, 60, 100, 100, tmpdir)

def test_proj_attr(tmpdir):
    t = Target(5, 10, 55, 60, 100, 100, tmpdir)
    print(t.proj_var)

def test_calculate_grid(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        'X', 'Y', ['x_wind'])
    t = Target(5, 10, 55, 60, 100, 100, tmpdir)

    tx, ty, _ = d.__calculate_grid__(t)

    assert tx.shape == (100, 100)
    assert ty.shape == (100, 100)

    print(tx, ty)

def test_init_lonlat(tmpdir):
    t = Target.from_lonlat(5, 10, 55, 60, 100, 100, tmpdir)

def test_transform_from_latlon():
    # 5E, 55N should be: x=-335060.5808799216, y=-3829759.9640511023
    crs = CRS.from_epsg(4326) # lonlat
    print(repr(crs))

    x, y = Target.transform(crs, 5, 55)
    print(x, y)
    assert (x, y) == (-335060.5808799216, -3829759.9640511023)

    xx, yy = Target.itransform(crs, x, y)
    assert (xx, yy) == approx((5., 55.))

    # test target proj bounds
    x, y = Target.transform(crs, -180, 45)
    print(x, y)
    assert (x, y) == (849024.0785366141, 4815054.821022379)
    xx, yy = Target.itransform(crs, x, y)
    assert (Target.modulate_longitude(xx), yy) == approx((-180., 45))

    x, y = Target.transform(crs, 180, 90)
    print(x, y)
    assert (x, y) == (0., 0.)
    # xx, yy = Target.itransform(crs, x, y) # doesn't work, NP singularity?
    # assert (xx, yy) == approx((180., 90))
