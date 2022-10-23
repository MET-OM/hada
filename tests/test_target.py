from ddh.sources import *
from ddh.target import Target


def test_init(tmpdir):
    t = Target(5, 10, 55, 60, 100, 100, tmpdir)


def test_calculate_grid(tmpdir):
    d = Dataset(
        "norkyst",
        "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be",
        ['x_wind'])
    t = Target(5, 10, 55, 60, 100, 100, tmpdir)

    tx, ty = d.calculate_grid(t)

    assert len(tx) == 100 * 100
    assert len(ty) == 100 * 100

    print(tx, ty)
