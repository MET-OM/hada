from ddh.dataset import Dataset
from ddh.sources import Sources

def test_load_norkyst():
    s = Sources.from_toml('./sources.toml')
    d = Dataset(s.datasets[0].url)
    print(d.ds)
    print(d.ds.rio)
    print(d.ds.rio.crs)
