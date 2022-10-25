from hada.sources import *

def test_load_default_conf(sourcetoml):
    s = Sources.from_toml(sourcetoml)
    print(s)

def test_load_norkyst():
    d = Dataset("norkyst", "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be", ['x_wind'])
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

    _, v = s.find_dataset_for_var('gibberish')
    assert v is None
