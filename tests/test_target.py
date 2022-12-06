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
