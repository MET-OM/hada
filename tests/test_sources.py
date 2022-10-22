from ddh.sources import *

def test_load_default_conf():
    s = Sources.from_toml("./sources.toml")
    print(s)

def test_load_norkyst():
    d = Dataset("norkyst", "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be", ['x_wind'])
    assert d.ds.rio.crs is not None
