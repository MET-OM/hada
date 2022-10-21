from ddh.sources import *

def test_load_default_conf():
    s = Sources.from_toml("./sources.toml")
    print(s)
